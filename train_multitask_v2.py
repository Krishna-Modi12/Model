import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
L.seed_everything(42, workers=True)

from src.models.face_analysis_model import FaceAnalysisModel
from src.data.dataset import FaceAnalysisDataset, get_train_transforms, get_val_transforms
from src.training.trainer import FocalLoss

class FaceShapeGuardCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Prevent sanity checks from triggering early stop
        if trainer.sanity_checking:
            return
        # Don't trigger guard until epoch 5 — new dropout layers need warmup
        if trainer.current_epoch < 5:
            return
            
        val_f1 = trainer.callback_metrics.get("val_f1")
        if val_f1 is not None and val_f1 < 0.720:
            print(f"[CRITICAL] Face shape regressed (val_f1={val_f1:.4f} < 0.720) - stopping")
            trainer.should_stop = True

class MultiTaskV2LightningModule(L.LightningModule):
    def __init__(self, model_checkpoint: str):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = FaceAnalysisModel(
            backbone="efficientnet_b4",
            pretrained=False,
            dropout=0.4,
            geometric_features=15,
            num_classes=5,
            freeze_backbone=True
        )
        
        # Load pre-trained weights
        ckpt = torch.load(model_checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # Losses
        self.shape_loss = FocalLoss(gamma=2.0, label_smoothing=0.1)
        self.feature_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.landmark_loss = nn.MSELoss()
        
        from torchmetrics import F1Score
        self.val_f1 = F1Score(task="multiclass", num_classes=5, average="macro")
        
        self.w_shape = 1.00
        self.w_eye = 0.30
        self.w_brow = 0.30
        self.w_lip = 0.30
        self.w_age = 0.20
        self.w_gender = 0.20
        self.w_landmark = 0.40
        self.w_skin = 0.20
        
    def forward(self, images, geometric_ratios):
        return self.model(images, geometric_ratios)

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.model.freeze_backbone()
            print("[INFO] Epoch 0 — Backbone FROZEN | LR = 5e-8")
            print("  Heads initializing slowly — dropout active")
        elif self.current_epoch == 2:
            self.model.unfreeze_backbone()
            print("[INFO] Epoch 2 — Backbone UNFROZEN | LR → 3e-6")
            print("  End-to-end co-adaptation begins")
            print("  Expect temporary val_loss spike — this is normal")

    def _compute_loss(self, output, batch):
        losses = {}
        valid_shape = batch["shape_labels"] >= 0
        if valid_shape.any():
            losses["face"] = self.shape_loss(output.face_shape_logits[valid_shape], batch["shape_labels"][valid_shape])
        else:
            losses["face"] = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        attr_mask = batch.get("has_attributes", torch.zeros_like(batch["shape_labels"], dtype=torch.bool))
        
        if attr_mask.any():
            eye_targets = torch.stack([batch["eye_narrow"].float(), batch["eye_big"].float()], dim=1)
            losses["eye"] = F.binary_cross_entropy_with_logits(output.eye_narrow_logits[attr_mask], eye_targets[attr_mask])
            losses["brow"] = F.cross_entropy(output.brow_type_logits[attr_mask], batch["brow"][attr_mask])
            losses["lip"] = F.cross_entropy(output.lip_shape_logits[attr_mask], batch["lip"][attr_mask])
            losses["age"] = F.cross_entropy(output.age_logits[attr_mask], batch["age"][attr_mask])
            losses["gender"] = F.cross_entropy(output.gender_logits[attr_mask], batch["gender"][attr_mask])
            losses["landmark"] = F.mse_loss(output.landmark_pred[attr_mask] * 10, batch["landmark_ratios"][attr_mask] * 10)
        else:
            losses["eye"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["brow"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["lip"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["age"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["gender"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["landmark"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        skin_mask = batch.get("monk_labels", torch.full_like(batch["shape_labels"], -100)) != -100
        if skin_mask.any():
            losses["skin_tone"] = F.cross_entropy(output.skin_tone_logits[skin_mask], batch["monk_labels"][skin_mask])
        else:
            losses["skin_tone"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        total = (
            self.w_shape * losses["face"] +
            self.w_eye * losses["eye"] +
            self.w_brow * losses["brow"] +
            self.w_lip * losses["lip"] +
            self.w_age * losses["age"] +
            self.w_gender * losses["gender"] +
            self.w_landmark * losses["landmark"] +
            self.w_skin * losses["skin_tone"]
        )
        losses["total"] = total
        return losses

    def training_step(self, batch, batch_idx):
        output = self(batch["images"], batch["geometric_ratios"])
        losses = self._compute_loss(output, batch)
        self.log("train_loss", losses["total"], prog_bar=True)
        return losses["total"]
        
    def validation_step(self, batch, batch_idx):
        output = self(batch["images"], batch["geometric_ratios"])
        losses = self._compute_loss(output, batch)
        
        # Filter out invalid labels (-100) before F1 computation
        valid_mask = batch["shape_labels"] >= 0
        if valid_mask.any():
            preds = output.face_shape_logits.argmax(dim=1)
            self.val_f1(preds[valid_mask], batch["shape_labels"][valid_mask])
        
        self.log("val_loss", losses["total"], prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, prog_bar=True, sync_dist=True)
        return losses["total"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-8,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[2], # step up at epoch 2
            gamma=60        # 5e-8 -> 3e-6
        )
        return [optimizer], [scheduler]

def main():
    print("================================================")
    print("MULTI-TASK V2 - CONVERGENCE FIXED")
    print("================================================")
    print("Seed                 : 42")
    print("Checkpoint From      : attrs_skin_epoch=00_val_loss=1.6046.ckpt")
    print("Backbone Freeze      : epochs 0-1 (LR=5e-8)")
    print("Backbone Unfreeze    : epoch 2 (LR -> 3e-6)")
    print("Weight Decay         : 1e-4 (AdamW L2 regularization)")
    print("Input Dropout        : 0.5 on ALL heads [x]")
    print("Max Epochs           : 60")
    print("Early Stop           : patience=15 on val_loss")
    print("Face Shape Guard     : stop if val_f1 < 0.720")
    print("Batch Size           : 4 (effective 16 with accum=4)")
    print("Mixed Precision      : 16-mixed [x]")
    print("Replace Sampler DDP  : False [x]")
    print("Saving To            : checkpoints/multitask_v2/")
    print("================================================\n")

    # Load dataset
    # We will pass the full balanced json to a split mechanism since FaceAnalysisDataset natively doesn't have split string args unless we filter annotations json manually.
    import json
    with open("data/processed/annotations_multitask_balanced.json", "r") as f:
        data = json.load(f)
    train_idx = [i for i, d in enumerate(data) if d.get('split') in ['train', None]]
    val_idx = [i for i, d in enumerate(data) if d.get('split') == 'val']
    
    # safeguard, if no val split, take 10%
    if not val_idx:
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = MultiTaskV2LightningModule(
        model_checkpoint="checkpoints/multitask_skin_tone/attrs_skin_epoch=00_val_loss=1.6046.ckpt"
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/multitask_v2",
            filename="multitask_v2_{epoch:02d}_{val_loss:.4f}",
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
        accumulate_grad_batches=4,
        callbacks=callbacks,
        use_distributed_sampler=False
    )
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

if __name__ == "__main__":
    main()
