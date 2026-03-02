"""
train_attributes_only.py
========================
Priority 1C Fix: Retrains ONLY the 6 multi-task attribute heads 
using the new perfectly balanced annotations file. 
Freezes the entire backbone and the face_shape_head completely.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from loguru import logger
from sklearn.metrics import f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.face_analysis_model import FaceAnalysisModel, ModelOutput
from src.data.dataset import FaceAnalysisDataset, get_train_transforms, get_val_transforms
from src.config import LEARNING_RATE, IMAGE_SIZE_TRAIN
from src.config import FACE_SHAPES

VALIDATION_SPLIT = 0.15
NUM_WORKERS = 0

# ── Use balanced annotations ──
ANNOTATIONS_FILE = "data/processed/annotations_multitask_balanced.json"

class MultitaskAttributesFinetuner(pl.LightningModule):
    def __init__(self, model_checkpoint: str):
        super().__init__()
        logger.info(f"Loading PyTorch weights from: {model_checkpoint}")
        self.model = FaceAnalysisModel(num_classes=len(FACE_SHAPES))
        
        ckpt = torch.load(model_checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip the 'model.' prefix that Lightning adds
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded weights: {msg}")
        
        # COMPLETE FREEZE of Backbone and Face Shape Head
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.model.backbone.eval() # Freeze BatchNorm running stats
        
        for param in self.model.face_shape_head.parameters():
            param.requires_grad = False
        self.model.face_shape_head.eval()
        
        logger.info("FROZEN: Backbone and Face Shape Head")
        logger.info("UNFROZEN: Eye, Brow, Lip, Age, Gender, Landmark Heads")
        
        self.validation_step_outputs = []
        
        # We will use higher learning rate since we're only training heads
        self.lr = 1e-3

    def forward(self, images, geometric_ratios):
        # We must manually set backbone and face shape head to eval mode explicitly in nested structures
        self.model.backbone.eval()
        self.model.face_shape_head.eval()
        return self.model(images, geometric_ratios)

    def _compute_loss(self, output: ModelOutput, batch: dict) -> dict:
        losses = {}
        attr_mask = batch["has_attributes"]
        
        if attr_mask.any():
            # Class-weighted BCE for Eye (just in case of residual imbalance)
            eye_targets = torch.stack([batch["eye_narrow"].float(), batch["eye_big"].float()], dim=1)
            raw_eye_loss = F.binary_cross_entropy_with_logits(
                output.eye_narrow_logits[attr_mask], eye_targets[attr_mask], reduction="none"
            )
            eye_valid_mask = (eye_targets[attr_mask] != -1).float()
            losses["eye"] = (raw_eye_loss * eye_valid_mask).sum() / (eye_valid_mask.sum() + 1e-5)
            
            losses["brow"] = F.cross_entropy(output.brow_type_logits[attr_mask], batch["brow"][attr_mask], ignore_index=-1)
            losses["lip"] = F.cross_entropy(output.lip_shape_logits[attr_mask], batch["lip"][attr_mask], ignore_index=-1)
            losses["age"] = F.cross_entropy(output.age_logits[attr_mask], batch["age"][attr_mask], ignore_index=-1)
            losses["gender"] = F.cross_entropy(output.gender_logits[attr_mask], batch["gender"][attr_mask], ignore_index=-1)
            losses["landmark"] = F.mse_loss(
                output.landmark_pred[attr_mask] * 10, batch["landmark_ratios"][attr_mask] * 10
            )
        else:
            # Should not happen with 100% balanced labels, but safeguard
            losses["eye"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["brow"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["lip"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["age"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["gender"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            losses["landmark"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Skin Tone
        skin_mask = batch["monk_labels"] != -100
        if skin_mask.any():
            losses["skin_tone"] = F.cross_entropy(output.skin_tone_logits[skin_mask], batch["monk_labels"][skin_mask])
        else:
            losses["skin_tone"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        total = losses["eye"] + losses["brow"] + losses["lip"] + losses["age"] + losses["gender"] + losses["landmark"] + losses["skin_tone"]
        losses["total"] = total
        return losses

    def training_step(self, batch, batch_idx):
        output = self(batch["images"], batch["geometric_ratios"])
        losses = self._compute_loss(output, batch)
        
        # Log all losses neatly
        for k, v in losses.items():
            self.log(f"train_loss_{k}", v, on_step=True, on_epoch=True, prog_bar=(k=="total"), logger=True)
            
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        output = self(batch["images"], batch["geometric_ratios"])
        losses = self._compute_loss(output, batch)
        
        # Calculate attribute accuracies for verification
        attr_mask = batch["has_attributes"]
        if attr_mask.any():
            brow_preds = output.brow_type_logits[attr_mask].argmax(dim=1)
            lip_preds = output.lip_shape_logits[attr_mask].argmax(dim=1)
            age_preds = output.age_logits[attr_mask].argmax(dim=1)
            gender_preds = output.gender_logits[attr_mask].argmax(dim=1)
            
            brow_acc = (brow_preds == batch["brow"][attr_mask]).float().mean()
            lip_acc = (lip_preds == batch["lip"][attr_mask]).float().mean()
            
            # Save for epoch aggregation
            self.validation_step_outputs.append({
                "val_loss": losses["total"],
                "brow_acc": brow_acc,
                "lip_acc": lip_acc
            })
        
        return losses["total"]

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_brow_acc = torch.stack([x["brow_acc"] for x in self.validation_step_outputs if "brow_acc" in x]).mean()
        avg_lip_acc  = torch.stack([x["lip_acc"] for x in self.validation_step_outputs if "lip_acc" in x]).mean()
        
        # We track brow_acc as primary metric for early stopping now
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_brow_acc", avg_brow_acc, prog_bar=True, sync_dist=True)
        self.log("val_lip_acc", avg_lip_acc, prog_bar=True, sync_dist=True)
        
        logger.info(f"Epoch {self.current_epoch} - Val Loss: {avg_loss:.4f} | Brow Acc: {avg_brow_acc:.4f} | Lip Acc: {avg_lip_acc:.4f}")
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        parameters_to_update = [
            p for n, p in self.model.named_parameters() 
            if p.requires_grad and "face_shape_head" not in n and "backbone" not in n
        ]
        
        optimizer = torch.optim.AdamW(parameters_to_update, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    pl.seed_everything(42)

    from src.config import get_config_dict
    cfg = get_config_dict()

    logger.info("Initializing Fine-tuning Dataset with 100% Balanced Labels...")
    full_dataset = FaceAnalysisDataset(
        annotations_path=ANNOTATIONS_FILE,
        image_size=IMAGE_SIZE_TRAIN,
        transforms=get_train_transforms(IMAGE_SIZE_TRAIN, cfg),
        landmarks_cache_dir="data/landmarks_cache"
    )

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size

    # Split predictably
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Disable random augmentations for validation
    val_ds.dataset.transforms = get_val_transforms(IMAGE_SIZE_TRAIN)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=8,
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        persistent_workers=False,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=8, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        persistent_workers=False,
        pin_memory=True
    )

    # Base checkpoint containing the trained attribute heads
    base_ckpt = "checkpoints/multitask_balanced/attrs_epoch=11_val_brow_acc=0.8589.ckpt"
    model = MultitaskAttributesFinetuner(base_ckpt)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/multitask_skin_tone",
        filename="attrs_skin_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10, 
        mode="min",
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10
    )

    logger.info("Starting attribute-only fine-tuning...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
