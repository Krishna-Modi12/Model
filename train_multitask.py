"""
train_multitask.py — Multi-task Training Script
=============================================
Trains the extended model with multiple facial attribute heads.

Usage:
    python train_multitask.py
"""

import os
import sys
import json
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from pathlib import Path
from torchmetrics import Accuracy, F1Score
import numpy as np
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.face_analysis_model import FaceAnalysisModel, ModelOutput
from src.data.dataset import FaceAnalysisDataset, get_train_transforms, get_val_transforms
from src.config import get_config_dict


# Set seed FIRST
L.seed_everything(42, workers=True)


class BalancedBatchSampler(Sampler):
    """Ensures exactly 50% original + 50% CelebA in every batch."""
    
    def __init__(self, dataset, batch_size):
        self.orig = [
            i for i, a in enumerate(dataset.annotations)
            if a.get("attributes") is None
        ]
        self.celeba = [
            i for i, a in enumerate(dataset.annotations)
            if a.get("attributes") is not None
        ]
        self.half = batch_size // 2
        
        logger.info(f"BalancedBatchSampler: {len(self.orig)} original, {len(self.celeba)} CelebA")

    def __iter__(self):
        orig_idx = torch.randperm(len(self.orig))
        celeba_idx = torch.randperm(len(self.celeba))
        n = min(len(orig_idx), len(celeba_idx)) // self.half
        for i in range(n):
            o = [self.orig[j] for j in orig_idx[i*self.half:(i+1)*self.half]]
            c = [self.celeba[j] for j in celeba_idx[i*self.half:(i+1)*self.half]]
            yield o + c

    def __len__(self):
        return min(len(self.orig), len(self.celeba)) // self.half


class MultiTaskLightningModule(L.LightningModule):
    
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        self.model = FaceAnalysisModel(
            backbone=config["model"]["backbone"],
            pretrained=config["model"]["pretrained"],
            dropout=config["model"]["dropout"],
            geometric_features=config["model"]["geometric_features"],
            num_classes=config["model"]["num_face_shapes"],
            freeze_backbone=True,
        )
        
        # Focal loss for face shape
        self.face_loss = nn.CrossEntropyLoss()
        
        # Metrics
        num_shapes = config["model"]["num_face_shapes"]
        self.train_acc = Accuracy(task="multiclass", num_classes=num_shapes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_shapes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_shapes, average="macro")
        
        # Track epoch for backbone freezing schedule
        self._backbone_frozen = True
        
        # --- LOAD V5 EXTRACTED WEIGHTS ---
        ckpt_path = PROJECT_ROOT / "checkpoints/celeba_v5/celeba_v5_epoch=14_val_f1=0.7654.ckpt"
        if ckpt_path.exists():
            logger.info(f"Loading V5 weights from {ckpt_path}...")
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            if "state_dict" in ckpt:
                # We use strict=False because the new multi-task heads will be missing from the checkpoint
                missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=False)
                logger.info(f"V5 weights loaded. Missing keys (expected for new heads): {len(missing)}")
            else:
                logger.warning("V5 Checkpoint did not have state_dict!")
        else:
            logger.warning(f"V5 Checkpoint not found at {ckpt_path}!")
        
    def forward(self, images, geometric_ratios):
        return self.model(images, geometric_ratios)
    
    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self._backbone_frozen:
            self.model.freeze_backbone()
            # CRITICAL FIX: Protect V5 shared projections from random multi-task head gradients
            for param in self.model.face_shape_head.visual_proj.parameters():
                param.requires_grad = False
            for param in self.model.face_shape_head.geo_proj.parameters():
                param.requires_grad = False
                
            logger.info("[INFO] Epoch 0 - Backbone and Shared Projections FROZEN")
            logger.info("  New heads initializing - random gradients kept away from V5 weights")
            
        elif self.current_epoch == 5 and self._backbone_frozen:
            self.model.unfreeze_backbone()
            
            # Unfreeze shared projections
            for param in self.model.face_shape_head.visual_proj.parameters():
                param.requires_grad = True
            for param in self.model.face_shape_head.geo_proj.parameters():
                param.requires_grad = True
                
            self._backbone_frozen = False
            logger.info("[INFO] Epoch 5 - Backbone and Shared Projections UNFROZEN")
            logger.info("  End-to-end training begins")
            logger.info("  Expect temporary val_f1 dip (normal behavior)")
            logger.info("  Do NOT stop training during epochs 5-7")
    
    def _compute_loss(self, output: ModelOutput, batch: dict) -> dict:
        losses = {}
        
        # Primary task: Face shape (always computed)
        losses["face"] = self.face_loss(
            output.face_shape_logits, batch["shape_labels"])
        
        # Multi-task losses (masked by has_attributes)
        attr_mask = batch["has_attributes"]
        
        if attr_mask.any():
            # Eye - BCEWithLogitsLoss (multi-label binary)
            eye_targets = torch.stack([
                batch["eye_narrow"].float(),
                batch["eye_big"].float()
            ], dim=1)
            losses["eye"] = F.binary_cross_entropy_with_logits(
                output.eye_narrow_logits[attr_mask],
                eye_targets[attr_mask]
            )
            
            # Classification heads - CrossEntropyLoss
            losses["brow"] = F.cross_entropy(
                output.brow_type_logits[attr_mask], batch["brow"][attr_mask])
            losses["lip"] = F.cross_entropy(
                output.lip_shape_logits[attr_mask], batch["lip"][attr_mask])
            losses["age"] = F.cross_entropy(
                output.age_logits[attr_mask], batch["age"][attr_mask])
            losses["gender"] = F.cross_entropy(
                output.gender_logits[attr_mask], batch["gender"][attr_mask])
            
            # Landmark regression - MSELoss with x10 scaling
            losses["landmark"] = F.mse_loss(
                output.landmark_pred[attr_mask] * 10,
                batch["landmark_ratios"][attr_mask] * 10
            )
            
        else:
            losses["eye"] = torch.tensor(0.0, device=self.device, requires_grad=False)
            losses["brow"] = torch.tensor(0.0, device=self.device, requires_grad=False)
            losses["lip"] = torch.tensor(0.0, device=self.device, requires_grad=False)
            losses["age"] = torch.tensor(0.0, device=self.device, requires_grad=False)
            losses["gender"] = torch.tensor(0.0, device=self.device, requires_grad=False)
            losses["landmark"] = torch.tensor(0.0, device=self.device, requires_grad=False)
            
        # Skin Tone (Monk Scale 1-10 mapped to 0-9)
        skin_mask = batch["monk_labels"] != -100
        if skin_mask.any():
            losses["skin_tone"] = F.cross_entropy(
                output.skin_tone_logits[skin_mask], batch["monk_labels"][skin_mask]
            )
        else:
            losses["skin_tone"] = torch.tensor(0.0, device=self.device, requires_grad=False)
        
        # Weighted total
        total = (
            1.00 * losses["face"] +
            0.30 * losses["eye"] +
            0.30 * losses["brow"] +
            0.30 * losses["lip"] +
            0.20 * losses["age"] +
            0.20 * losses["gender"] +
            0.40 * losses["landmark"] +
            0.20 * losses["skin_tone"]
        )
        
        losses["total"] = total
        return losses
    
    def training_step(self, batch, batch_idx):
        output = self(batch["images"], batch["geometric_ratios"])
        losses = self._compute_loss(output, batch)
        
        preds = output.face_shape_logits.argmax(dim=1)
        self.train_acc(preds, batch["shape_labels"])
        
        self.log("train/loss", losses["total"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/face_loss", losses["face"], prog_bar=False, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_epoch=True)
        
        return losses["total"]
    
    def validation_step(self, batch, batch_idx):
        output = self(batch["images"], batch["geometric_ratios"])
        losses = self._compute_loss(output, batch)
        
        preds = output.face_shape_logits.argmax(dim=1)
        self.val_acc(preds, batch["shape_labels"])
        self.val_f1(preds, batch["shape_labels"])
        
        # Compute attribute accuracies for logging
        attr_mask = batch["has_attributes"]
        
        eye_narrow_acc = eye_big_acc = brow_acc = lip_acc = age_acc = gender_acc = 0.0
        landmark_mse = 0.0
        
        if attr_mask.any():
            # Eye accuracy
            eye_preds = (torch.sigmoid(output.eye_narrow_logits[attr_mask]) > 0.5).float()
            eye_targets = torch.stack([
                batch["eye_narrow"][attr_mask].float(),
                batch["eye_big"][attr_mask].float()
            ], dim=1)
            eye_narrow_acc = (eye_preds[:, 0] == eye_targets[:, 0]).float().mean()
            eye_big_acc = (eye_preds[:, 1] == eye_targets[:, 1]).float().mean()
            
            # Other attribute accuracies
            brow_acc = (output.brow_type_logits[attr_mask].argmax(1) == batch["brow"][attr_mask]).float().mean()
            lip_acc = (output.lip_shape_logits[attr_mask].argmax(1) == batch["lip"][attr_mask]).float().mean()
            age_acc = (output.age_logits[attr_mask].argmax(1) == batch["age"][attr_mask]).float().mean()
            gender_acc = (output.gender_logits[attr_mask].argmax(1) == batch["gender"][attr_mask]).float().mean()
            
            # Landmark MSE
            landmark_mse = F.mse_loss(
                output.landmark_pred[attr_mask], 
                batch["landmark_ratios"][attr_mask]
            )
        
        self.log("val/loss", losses["total"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        
        # Log all task metrics
        self.log("val_face_acc", self.val_acc)
        self.log("val_eye_narrow_acc", eye_narrow_acc)
        self.log("val_eye_big_acc", eye_big_acc)
        self.log("val_brow_acc", brow_acc)
        self.log("val_lip_acc", lip_acc)
        self.log("val_age_acc", age_acc)
        self.log("val_gender_acc", gender_acc)
        self.log("val_landmark_mse", landmark_mse)
        self.log("val_total_loss", losses["total"])
        
        return losses["total"]
    
    def on_validation_epoch_end(self):
        val_f1 = self.val_f1.compute()
        print(f"\nEPOCH {self.current_epoch} VALIDATION F1: {val_f1:.4f}")
        
        # Don't trigger regression guard during sanity checks or before unfreezing
        if not self.trainer.sanity_checking and self.current_epoch > 5:
            # Regression guard
            if val_f1 < 0.750:
                print(f"[WARNING] val_f1={val_f1:.4f} below 0.750")
                print("  If sustained 3+ epochs, halve attribute weights:")
                print("  eye:0.15 brow:0.15 lip:0.15 age:0.10")
                print("  gender:0.10 landmark:0.20")
            
            if val_f1 < 0.720:
                print("[CRITICAL] val_f1 below 0.720 - stopping")
                print("  Multi-task losses are hurting face shape accuracy")
                print("  Use V5 checkpoint as production model")
                self.trainer.should_stop = True
        
        self.val_f1.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.get_optimizer_param_groups(lr=3e-6),
            weight_decay=0.01,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def main():
    print("=" * 60)
    print("MULTI-TASK TRAINING - CONFIG SUMMARY")
    print("=" * 60)
    print("Seed                 : 42")
    print("Checkpoint From      : V5 (val_f1=0.7654, 79.21% TTA)")
    print("Backbone             : Frozen epochs 0-4, unfrozen epoch 5+")
    print("Learning Rate        : 3e-6")
    print("Max Epochs           : 50")
    print("Early Stop           : patience=8 on val_f1 (face shape)")
    print("Batch Size           : 8 (effective 16 with accum=2)")
    print("Gradient Accum       : 2")
    print("Mixed Precision      : 16-mixed")
    print("replace_sampler_ddp  : False")
    print("Batch Sampling       : BALANCED 50/50 original/CelebA")
    print("Loss Weights         : face(1.0) eye(0.3) brow(0.3)")
    print("                       lip(0.3) age(0.2) gender(0.2) landmark(0.4)")
    print("Eye Head             : BCEWithLogitsLoss (multi-label)")
    print("Landmark Scaling     : x10")
    print("Regression Guard     : stop if val_f1 < 0.720")
    print("Primary Monitor      : val_f1 (face shape F1 only)")
    print("num_workers          : 0 (Windows safe)")
    print("Saving To            : checkpoints/multitask/")
    print("=" * 60)
    
    # Paths
    ANNOTATIONS_PATH = PROJECT_ROOT / "data/processed/annotations_multitask.json"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/multitask"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"\nLoading annotations from {ANNOTATIONS_PATH}...")
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)
    print(f"Total annotations: {len(annotations)}")
    
    # Create datasets
    train_transforms = get_train_transforms(224, {})
    val_transforms = get_val_transforms(224)
    
    # The merged annotations file is structured as: [Original (0...5755), CelebA (5756...8911)]
    # To maintain a purely gold-standard validaton score, we evaluate ONLY on original data
    # that has human-verified True ground-truth labels, avoiding pseudo-label TTA-disagreement penalties.
    
    num_original = 5755  # Known count from Phase 1-4
    if len(annotations) < num_original:
        num_original = int(len(annotations) * 0.8)  # Fallback
        
    original_idx = list(range(num_original))
    celeba_idx = list(range(num_original, len(annotations)))
    
    train_orig = original_idx[:int(num_original * 0.8)]
    val_orig = original_idx[int(num_original * 0.8):int(num_original * 0.9)]
    test_orig = original_idx[int(num_original * 0.9):]
    
    # Train set includes full CelebA and 80% of original
    train_idx = train_orig + celeba_idx
    val_idx = val_orig
    test_idx = test_orig  # Optional test set if needed
    
    print(f"Dataset Split -> Train: {len(train_idx)} | Val: {len(val_idx)} (Pure Original)")
    
    train_dataset = FaceAnalysisDataset(
        annotations_path=str(ANNOTATIONS_PATH),
        image_size=224,
        landmarks_cache_dir="data/landmarks_cache",
        transforms=train_transforms,
        indices=train_idx,
    )
    
    val_dataset = FaceAnalysisDataset(
        annotations_path=str(ANNOTATIONS_PATH),
        image_size=224,
        landmarks_cache_dir="data/landmarks_cache",
        transforms=val_transforms,
        indices=val_idx,
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create balanced batch sampler
    batch_size = 8
    train_sampler = BalancedBatchSampler(train_dataset, batch_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model and trainer
    config = {
        "model": {
            "backbone": "efficientnet_b4",
            "pretrained": True,
            "dropout": 0.4,
            "geometric_features": 15,
            "num_face_shapes": 5,
        }
    }
    
    model = MultiTaskLightningModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="multitask_epoch={epoch}_val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop = EarlyStopping(
        monitor="val_f1",
        patience=8,
        mode="max",
    )
    
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, early_stop],
        use_distributed_sampler=False,
        log_every_n_steps=10,
    )
    
    # Train
    print("\nStarting training...\n")
    trainer.fit(model, train_loader, val_loader)
    
    print("\nTraining complete!")
    print(f"Best checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
