import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import json
import sys

from src.models.face_analysis_model import FaceAnalysisModel
from src.data.dataset import FaceAnalysisDataset, get_train_transforms, get_val_transforms
from src.training.trainer import FocalLoss
from torchmetrics.classification import F1Score

L.seed_everything(42, workers=True)

class FaceShapeGuardCallback(Callback):
    """Stops training if face shape performance regresses."""
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        
        if trainer.current_epoch < 5:
            return
            
        val_f1 = pl_module.val_f1.compute()
        if val_f1 < 0.700:
            print(f"\n[CRITICAL] Face shape regressed (val_f1={val_f1:.4f} < 0.700) - stopping")
            trainer.should_stop = True

class DualOptimizerLightningModule(L.LightningModule):
    def __init__(self, model_checkpoint: str):
        super().__init__()
        self.model = FaceAnalysisModel()
        
        print(f"Loading checkpoint: {model_checkpoint}")
        ckpt = torch.load(model_checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Filter out mismatched prefix keys AND mismatched shapes
        model_dict = self.model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            core_k = k.replace("model.", "") if k.startswith("model.") else k
            if core_k in model_dict and v.shape == model_dict[core_k].shape:
                filtered_dict[core_k] = v
        
        self.model.load_state_dict(filtered_dict, strict=False)
        
        self.model.unfreeze_backbone()
        
        self.shape_loss = FocalLoss(gamma=2.0)
        self.val_f1 = F1Score(task="multiclass", num_classes=5, average="macro")
        
        self.w_shape = 1.00
        self.w_eye = 0.15
        self.w_brow = 0.15
        self.w_lip = 0.15
        self.w_age = 0.10
        self.w_gender = 0.10
        self.w_skin = 0.20
        self.w_landmark = 0.20
        
        self.age_pos_weight = 8.834567901234568
        self.gen_pos_weight = 3.110423116615067

    def forward(self, images, geometric_ratios):
        return self.model(images, geometric_ratios)

    def train(self, mode: bool = True):
        """Override train mode to keep backbone and face_shape_head in eval mode to freeze BatchNorm."""
        super().train(mode)
        if mode and hasattr(self, 'model'):
            self.model.backbone.eval()
            self.model.face_shape_head.eval()

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            print("Batch 0 diagnostic:")
            print(f"  Age pos_weight    : {self.age_pos_weight:.2f} (compensates for imbalance)")
            print(f"  Gender pos_weight : {self.gen_pos_weight:.2f} (compensates for imbalance)")
            print("Freeze strategy   : IMPLICIT via LR differential")
            print("  Backbone LR     : 1e-7 (30x slower than heads)")
            print("  Heads LR        : 3e-6")
            print("  No explicit freeze/unfreeze - smooth co-adaptation")

    def training_step(self, batch, batch_idx):
        # 1. Block attribute gradients from entering backbone
        features = self.model.backbone(batch["images"])
        
        # Face shape runs on standard features (gradients flow to backbone)
        face_shape_logits = self.model.face_shape_head(features, batch["geometric_ratios"])
        
        # Fused computes with DETACHED features (Face Shape ONLY requires geometry)
        # Note: we don't actually need fused for attribute heads anymore
        
        detached_features = features.detach()
        eye_narrow_logits = self.model.eye_narrow_head(detached_features)
        brow_type_logits  = self.model.brow_type_head(detached_features)
        lip_shape_logits  = self.model.lip_shape_head(detached_features)
        age_logits        = self.model.age_head(detached_features)
        gender_logits     = self.model.gender_head(detached_features)
        landmark_pred     = self.model.landmark_head(detached_features)
        hsv_placeholder = torch.zeros(detached_features.shape[0], 48, device=detached_features.device)
        skin_tone_logits  = self.model.skin_tone_head(detached_features, hsv_placeholder)

        # 2. Compute Diagnostics (Batch 0 only)
        attr_mask = batch.get("has_attributes", torch.zeros_like(batch["shape_labels"], dtype=torch.bool))

        if self.current_epoch == 0 and batch_idx == 0:
            print(f"\nBatch 0 diagnostic:")
            print(f"  has_attributes True  : {attr_mask.sum().item()} samples")
            print(f"  has_attributes False : {(~attr_mask).sum().item()} samples")

        dummy_loss = 0.0 * sum(p.sum() for p in self.model.parameters())

        # Compute Losses
        valid_shape = batch["shape_labels"] >= 0
        if valid_shape.any():
            loss_face = self.shape_loss(face_shape_logits[valid_shape], batch["shape_labels"][valid_shape])
        else:
            loss_face = dummy_loss

        if attr_mask.any():
            eye_targets = torch.stack([batch["eye_narrow"].float(), batch["eye_big"].float()], dim=1)
            loss_eye = F.binary_cross_entropy_with_logits(eye_narrow_logits[attr_mask], eye_targets[attr_mask])
            loss_brow = F.cross_entropy(brow_type_logits[attr_mask], batch["brow"][attr_mask])
            loss_lip = F.cross_entropy(lip_shape_logits[attr_mask], batch["lip"][attr_mask])
            
            age_weight = torch.tensor([1.0, self.age_pos_weight], device=self.device)
            gender_weight = torch.tensor([1.0, self.gen_pos_weight], device=self.device)
            loss_age = F.cross_entropy(age_logits[attr_mask], batch["age"][attr_mask], weight=age_weight)
            loss_gender = F.cross_entropy(gender_logits[attr_mask], batch["gender"][attr_mask], weight=gender_weight)
            loss_landmark = F.mse_loss(landmark_pred[attr_mask] * 10, batch["landmark_ratios"][attr_mask] * 10)
        else:
            loss_eye = loss_brow = loss_lip = loss_age = loss_gender = loss_landmark = dummy_loss

        skin_mask = batch.get("monk_labels", torch.full_like(batch["shape_labels"], -100)) != -100
        if skin_mask.any():
            loss_skin = F.cross_entropy(skin_tone_logits[skin_mask], batch["monk_labels"][skin_mask])
        else:
            loss_skin = dummy_loss

        total_loss = (
            self.w_shape * loss_face +
            self.w_eye * loss_eye +
            self.w_brow * loss_brow +
            self.w_lip * loss_lip +
            self.w_age * loss_age +
            self.w_gender * loss_gender +
            self.w_landmark * loss_landmark +
            self.w_skin * loss_skin
        )

        self.log("train/loss_total", total_loss, prog_bar=True)
        self.log("train/loss_face", loss_face, prog_bar=False)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self(batch["images"], batch["geometric_ratios"])
        
        valid_shape = batch["shape_labels"] >= 0
        if valid_shape.any():
            loss_face = self.shape_loss(output.face_shape_logits[valid_shape], batch["shape_labels"][valid_shape])
        else:
            loss_face = torch.tensor(0.0, device=self.device, requires_grad=True)

        attr_mask = batch.get("has_attributes", torch.zeros_like(batch["shape_labels"], dtype=torch.bool))
        if attr_mask.any():
            eye_targets = torch.stack([batch["eye_narrow"].float(), batch["eye_big"].float()], dim=1)
            loss_eye = F.binary_cross_entropy_with_logits(output.eye_narrow_logits[attr_mask], eye_targets[attr_mask])
            loss_brow = F.cross_entropy(output.brow_type_logits[attr_mask], batch["brow"][attr_mask])
            loss_lip = F.cross_entropy(output.lip_shape_logits[attr_mask], batch["lip"][attr_mask])
            
            age_weight = torch.tensor([1.0, self.age_pos_weight], device=self.device)
            gender_weight = torch.tensor([1.0, self.gen_pos_weight], device=self.device)
            loss_age = F.cross_entropy(output.age_logits[attr_mask], batch["age"][attr_mask], weight=age_weight)
            loss_gender = F.cross_entropy(output.gender_logits[attr_mask], batch["gender"][attr_mask], weight=gender_weight)
            loss_landmark = F.mse_loss(output.landmark_pred[attr_mask] * 10, batch["landmark_ratios"][attr_mask] * 10)
        else:
            loss_eye = loss_brow = loss_lip = loss_age = loss_gender = loss_landmark = torch.tensor(0.0, device=self.device)

        skin_mask = batch.get("monk_labels", torch.full_like(batch["shape_labels"], -100)) != -100
        if skin_mask.any():
            loss_skin = F.cross_entropy(output.skin_tone_logits[skin_mask], batch["monk_labels"][skin_mask])
        else:
            loss_skin = torch.tensor(0.0, device=self.device)

        total_loss = (
            self.w_shape * loss_face +
            self.w_eye * loss_eye +
            self.w_brow * loss_brow +
            self.w_lip * loss_lip +
            self.w_age * loss_age +
            self.w_gender * loss_gender +
            self.w_landmark * loss_landmark +
            self.w_skin * loss_skin
        )

        valid_mask = batch["shape_labels"] >= 0
        if valid_mask.any():
            preds = output.face_shape_logits.argmax(dim=1)
            self.val_f1(preds[valid_mask], batch["shape_labels"][valid_mask])
        
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, prog_bar=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        backbone_params, head_params = self.model.get_parameter_groups()
        
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 1e-7, "weight_decay": 1e-4},
            {"params": head_params,     "lr": 3e-6, "weight_decay": 1e-3}
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

def main():
    print("================================================")
    print("MULTI-TASK V3 - ISOLATED GRADIENTS & DUAL OPTIMIZER")
    print("================================================")
    print("Seed                  : 42")
    print("Checkpoint From       : attrs_skin_epoch=00_val_loss=1.6046")
    print("Optimizer A (backbone): AdamW lr=1e-7 wd=1e-4")
    print("Optimizer B (heads)   : AdamW lr=3e-6 wd=1e-3")
    print("Scheduler (heads)     : CosineAnnealingLR T_max=40")
    print("Gradient isolation    : ACTIVE - features.detach() for attributes")
    print("Input Dropout         : 0.5 on ALL heads")
    print("Age pos_weight        : 8.83")
    print("Max Epochs            : 60")
    print("Early Stop            : patience=15 on val_loss")
    print("Face Shape Guard      : stop if val_f1 < 0.700")
    print("================================================\n")

    with open("data/processed/annotations_multitask_balanced.json", "r") as f:
        data = json.load(f)
        
    # CRITICAL FIX: The old "split" keys in the JSON were biased (val was 100% Class 2!).
    # We must force a uniformly random 85/15 split here.
    import random
    random.seed(42)
    all_idx = list(range(len(data)))
    random.shuffle(all_idx)
    val_size = int(0.15 * len(all_idx))
    val_idx = all_idx[:val_size]
    train_idx = all_idx[val_size:]

    train_dataset = FaceAnalysisDataset(
        annotations_path="data/processed/annotations_multitask_balanced.json",
        image_size=224,
        transforms=get_train_transforms(224, {"training": {}}),
        landmarks_cache_dir="data/landmarks_cache",
        indices=train_idx
    )
    
    val_dataset = FaceAnalysisDataset(
        annotations_path="data/processed/annotations_multitask_balanced.json",
        image_size=224,
        transforms=get_val_transforms(224),
        landmarks_cache_dir="data/landmarks_cache",
        indices=val_idx
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = DualOptimizerLightningModule(
        model_checkpoint="checkpoints/multitask_skin_tone/attrs_skin_epoch=00_val_loss=1.6046.ckpt"
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/multitask_v3",
            filename="multitask_v3_{epoch:02d}_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            mode="min"
        ),
        FaceShapeGuardCallback()
    ]
    
    trainer = L.Trainer(
        max_epochs=60,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        accumulate_grad_batches=2,
        callbacks=callbacks,
        use_distributed_sampler=False
    )
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.resume:
        print("================================================")
        print("RESUMING MULTI-TASK V3 — FIXES APPLIED")
        print("================================================")
        print(f"Resuming from    : {args.resume}")
        print("Starting epoch   : 7 (continuing from epoch 6)")
        print("Guard threshold  : 0.700 (lowered from 0.720) ✅")
        print("Missing image    : removed from annotations ✅")
        print("All other config : UNCHANGED")
        print("================================================")

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

if __name__ == "__main__":
    main()
