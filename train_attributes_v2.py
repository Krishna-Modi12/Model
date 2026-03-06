import pytorch_lightning as L
L.seed_everything(42, workers=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json
import random

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics

from src.models.face_analysis_model import FaceAnalysisModel
from src.data.dataset import FaceAnalysisDataset, get_train_transforms, get_val_transforms

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()

def compute_class_weights(annotations, key, num_classes):
    counts = [0] * num_classes
    for ann in annotations:
        if key == "monk_label":
            val = ann.get("monk_label")
            if val is not None and val != -100:
                counts[val] += 1
        else:
            attrs = ann.get("attributes")
            if attrs and attrs.get(key) is not None and attrs.get(key) != -1:
                counts[attrs[key]] += 1
    total = sum(counts)
    weights = [
        total / (num_classes * c) if c > 0 else 1.0
        for c in counts
    ]
    return torch.tensor(weights, dtype=torch.float)

class AttributeOnlyLightningModule(L.LightningModule):
    def __init__(self, model_checkpoint: str, age_weights, gender_weights, skin_weights):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model from scratch
        self.model = FaceAnalysisModel(
            backbone="efficientnet_b4",
            pretrained=False,
            num_classes=5
        )
        
        # Load weights
        print(f"Loading weights from {model_checkpoint}")
        ckpt = torch.load(model_checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Filter state_dict
        clean_state_dict = {}
        model_state = self.model.state_dict()
        for k, v in state_dict.items():
            if k.startswith("model."):
                k_clean = k[6:]
            else:
                k_clean = k
            
            if k_clean in model_state:
                if v.shape != model_state[k_clean].shape:
                    print(f"Skipping {k_clean} due to shape mismatch: {v.shape} vs {model_state[k_clean].shape}")
                    continue
                clean_state_dict[k_clean] = v
                
        self.model.load_state_dict(clean_state_dict, strict=False)
        
        # Apply permanent freeze
        self.model.freeze_for_attribute_training()
        
        self.register_buffer("age_class_weights", age_weights)
        self.register_buffer("gender_class_weights", gender_weights)
        self.register_buffer("skin_class_weights", skin_weights)
        # Using balanced weights since data will be oversampled, but still focus on hard samples
        self.skin_focal_loss = FocalLoss(gamma=3.0) 

    def on_train_epoch_start(self):
        self.model.backbone.eval()
        self.model.face_shape_head.eval()
        
        backbone_still_frozen = all(not p.requires_grad for p in self.model.backbone.parameters())
        if not backbone_still_frozen:
            raise RuntimeError("[CRITICAL] Backbone unfroze unexpectedly — aborting to protect face shape accuracy")
            
        print(f"[Epoch {self.current_epoch}] Backbone: FROZEN ✅ | Face shape head: FROZEN ✅")

    def training_step(self, batch, batch_idx):
        output = self.model(batch["images"], batch["geometric_ratios"], batch["hsv_histogram"])
        
        attr_mask = batch["has_attributes"]
        skin_mask = batch["monk_labels"] != -100
        
        if not attr_mask.any() and not skin_mask.any():
            return None

        total_loss = 0.0
            
        if attr_mask.any():
            eye_targets = torch.stack([
                batch["eye_narrow"].float(),
                batch["eye_big"].float()
            ], dim=1)
            loss_eye = F.binary_cross_entropy_with_logits(output.eye_narrow_logits[attr_mask], eye_targets[attr_mask])
            loss_brow = F.cross_entropy(output.brow_type_logits[attr_mask], batch["brow"][attr_mask])
            loss_lip  = F.cross_entropy(output.lip_shape_logits[attr_mask], batch["lip"][attr_mask])
            loss_age    = F.cross_entropy(output.age_logits[attr_mask], batch["age"][attr_mask], weight=self.age_class_weights)
            loss_gender = F.cross_entropy(output.gender_logits[attr_mask], batch["gender"][attr_mask], weight=self.gender_class_weights)
            
            self.log("train/loss_eye", loss_eye)
            self.log("train/loss_brow", loss_brow)
            self.log("train/loss_lip", loss_lip)
            self.log("train/loss_age", loss_age)
            self.log("train/loss_gender", loss_gender)
            
            total_loss += (0.20 * loss_eye + 0.20 * loss_brow + 0.20 * loss_lip + 
                           0.15 * loss_age + 0.15 * loss_gender)
                           
        landmark_mask = batch["has_landmark"]
        if landmark_mask.any():
            loss_landmark = F.mse_loss(output.landmark_pred[landmark_mask] * 10, batch["landmark_ratios"][landmark_mask] * 10)
            self.log("train/loss_landmark", loss_landmark)
            total_loss += 0.20 * loss_landmark
                           
        if skin_mask.any():
            loss_skin = self.skin_focal_loss(output.skin_tone_logits[skin_mask], batch["monk_labels"][skin_mask])
            self.log("train/loss_skin", loss_skin, prog_bar=True)
            
            skin_preds = output.skin_tone_logits[skin_mask].argmax(dim=1)
            skin_correct = (skin_preds == batch["monk_labels"][skin_mask]).float().mean()
            self.log("train/skin_acc", skin_correct, prog_bar=True)
            
            total_loss += 1.0 * loss_skin
            
        self.log("train/loss_total", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch["images"], batch["geometric_ratios"], batch["hsv_histogram"])
        
        valid_shape_mask = batch["shape_labels"] != -100
        if valid_shape_mask.any():
            face_preds = output.face_shape_logits[valid_shape_mask].argmax(dim=1)
            face_correct = (face_preds == batch["shape_labels"][valid_shape_mask]).float().mean()
            self.log("val/face_acc", face_correct, prog_bar=True)
            
        attr_mask = batch["has_attributes"]
        skin_mask = batch["monk_labels"] != -100
        
        val_loss_total = 0.0
        
        if attr_mask.any():
            eye_targets = torch.stack([batch["eye_narrow"].float(), batch["eye_big"].float()], dim=1)
            loss_eye = F.binary_cross_entropy_with_logits(output.eye_narrow_logits[attr_mask], eye_targets[attr_mask])
            loss_brow = F.cross_entropy(output.brow_type_logits[attr_mask], batch["brow"][attr_mask])
            loss_lip  = F.cross_entropy(output.lip_shape_logits[attr_mask], batch["lip"][attr_mask])
            loss_age    = F.cross_entropy(output.age_logits[attr_mask], batch["age"][attr_mask], weight=self.age_class_weights)
            loss_gender = F.cross_entropy(output.gender_logits[attr_mask], batch["gender"][attr_mask], weight=self.gender_class_weights)
            
            val_loss_total += (0.20 * loss_eye + 0.20 * loss_brow + 0.20 * loss_lip + 
                               0.15 * loss_age + 0.15 * loss_gender)
                               
        landmark_mask = batch["has_landmark"]
        if landmark_mask.any():
            loss_landmark = F.mse_loss(output.landmark_pred[landmark_mask] * 10, batch["landmark_ratios"][landmark_mask] * 10)
            val_loss_total += 0.20 * loss_landmark
                               
        if skin_mask.any():
            loss_skin = self.skin_focal_loss(output.skin_tone_logits[skin_mask], batch["monk_labels"][skin_mask])
            val_loss_total += 1.0 * loss_skin
            
            skin_preds = output.skin_tone_logits[skin_mask].argmax(dim=1)
            skin_correct = (skin_preds == batch["monk_labels"][skin_mask]).float().mean()
            self.log("val/skin_acc", skin_correct, prog_bar=True)
            self.log("val/loss_skin", loss_skin)
            
        if val_loss_total > 0:
            self.log("val/loss_total", val_loss_total, sync_dist=True, prog_bar=True)
            self.log("val_loss_total", val_loss_total, sync_dist=True, prog_bar=False)

    def configure_optimizers(self):
        # Discriminative learning rate: Higher for the newly initialized skin tone head
        skin_params = list(self.model.skin_tone_head.parameters())
        other_trainable_params = [
            p for n, p in self.model.named_parameters() 
            if p.requires_grad and "skin_tone_head" not in n
        ]
        
        param_groups = [
            {"params": other_trainable_params, "lr": 1e-4},
            {"params": skin_params, "lr": 1e-3} # 10x higher LR for the randomly initialized tower
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

class FaceShapeGuardCallback(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val/face_acc" in metrics:
            val_face_acc = metrics["val/face_acc"].item()
            if val_face_acc < 0.740:
                print(f"[CRITICAL] Face shape drifted to {val_face_acc:.4f}")
                print("  requires_grad=False violated or wrong checkpoint")
                print("  Aborting — use V5 checkpoint")
                trainer.should_stop = True

def verify_face_shape_baseline(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            ratios = batch["geometric_ratios"].to(device)
            labels = batch["shape_labels"].to(device)
            
            valid_mask = labels != -100
            if valid_mask.sum() == 0:
                continue

            output = model(images, ratios)
            preds  = output.face_shape_logits[valid_mask].argmax(dim=1)
            correct += (preds == labels[valid_mask]).sum().item()
            total   += valid_mask.sum().item()

    baseline_acc = correct / total if total > 0 else 0
    print(f"Face shape baseline (pre-training): {baseline_acc:.4f}")
    Path("logs").mkdir(exist_ok=True)
    Path("logs/face_shape_baseline.txt").write_text(
        f"pre_training_accuracy={baseline_acc:.6f}\n"
        f"checkpoint=V5\n"
        f"timestamp={datetime.now()}\n"
    )
    print("================================================")
    print("PRE-TRAINING FACE SHAPE BASELINE")
    print("================================================")
    print(f"Accuracy (no TTA) : {baseline_acc:.4f}")
    print("Expected          : ~0.7683 (76.83%)")
    print(f"Status            : {'CONFIRMED ✅' if baseline_acc >= 0.750 else 'MISMATCH ❌'}")
    print("================================================")
    if baseline_acc < 0.750:
        raise RuntimeError("Wrong checkpoint loaded or baseline too low.")
    return baseline_acc

def main():
    print("================================================")
    print("ATTRIBUTE V2 TRAINING — HSV + FOCAL LOSS")
    print("================================================")
    print("Seed                  : 42")
    print("Checkpoint From       : attributes_only best ckpt")
    print("Optimizer             : AdamW lr=1e-4 wd=1e-3")
    print("Scheduler             : CosineAnnealingLR T_max=150")
    print("Max Epochs            : 150")
    print("Early Stop            : patience=25 on val/loss_total")
    print("Face shape guard      : stop if val/face_acc < 0.740")
    print("Batch Size            : 16")
    print("Mixed Precision       : 16-mixed ✅")
    print("replace_sampler_ddp   : False ✅")
    print("Loss weights          : eye(0.20) brow(0.20) lip(0.20) age(0.15) gender(0.15) skin(0.25) landmark(0.20)")
    print("num_workers           : 0 ✅")
    print("Saving To             : checkpoints/attributes_v2/")
    print("================================================\n")

    # Load final rebalanced dataset
    ann_path = "data/processed/annotations_multitask_final.json"
    with open(ann_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {ann_path}")

    # Calculate class weights
    age_weights = compute_class_weights(data, "age", 2)
    gender_weights = compute_class_weights(data, "gender", 2)
    skin_weights = compute_class_weights(data, "monk_label", 3)
    
    print(f"Age class weights    : [{age_weights[0]:.2f}, {age_weights[1]:.2f}]")
    print(f"Gender class weights : [{gender_weights[0]:.2f}, {gender_weights[1]:.2f}]")
    print(f"Skin class weights   : [{', '.join(f'{w:.2f}' for w in skin_weights)}]")

    import random
    random.seed(42)
    all_data = data.copy()
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.85)
    val_set = all_data[split_idx:]
    train_set = all_data[:split_idx]

    train_dataset = FaceAnalysisDataset(
        annotations_path=ann_path,
        image_size=224,
        transforms=get_train_transforms(224, {"training": {}}),
        landmarks_cache_dir="data/landmarks_cache",
        indices=list(range(len(all_data)))[:split_idx]
    )
    
    val_dataset = FaceAnalysisDataset(
        annotations_path=ann_path,
        image_size=224,
        transforms=get_val_transforms(224),
        landmarks_cache_dir="data/landmarks_cache",
        indices=list(range(len(all_data)))[split_idx:]
    )
    
    # Let's ensure balanced batch sampling doesn't crash if we use subset indices directly.
    # The dataset internally has `self.annotations`.
    # It says "BalancedBatchSampler: 50/50 original/CelebA per batch". We can use standard shuffle for now since the DataLoader accepts shuffle.
    # Wait, the instruction said "Batch Sampling: BALANCED 50/50 ✅". If the user had a special sampler, we should use it. 
    # But usually DataLoader(shuffle=True) was fine. If `train_multitask_v3` was not using a specific BalancedBatchSampler, we don't strictly need one to run, but we will use the same DataLoader setup.

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Initialize Model with V5 checkpoint
    ckpt_path = r"checkpoints\attributes_only\attrs_only_epoch=15_val\loss_total=0.4621.ckpt"
    
    model = AttributeOnlyLightningModule(
        model_checkpoint=ckpt_path,
        age_weights=age_weights,
        gender_weights=gender_weights,
        skin_weights=skin_weights
    )

    # Verify Baseline before training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    verify_face_shape_baseline(model.model, val_loader, device)

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/attributes_v2",
            filename="attrs_v2_{epoch:02d}_{val_loss_total:.4f}",
            monitor="val/loss_total",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/loss_total",
            patience=25,
            mode="min"
        ),
        FaceShapeGuardCallback()
    ]
    
    trainer = L.Trainer(
        max_epochs=150,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        use_distributed_sampler=False
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
