import os
import sys
import torch
import pytorch_lightning as L
from pathlib import Path
from loguru import logger
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import json

# Add current directory to path
sys.path.append(os.getcwd())

from src.config import get_config_dict, FACE_SHAPES
from src.data.dataset import FaceAnalysisDataset, get_val_transforms
from src.training.trainer import FaceAnalysisLightningModule
from torch.utils.data import DataLoader

# Seed for reproducibility
L.seed_everything(42, workers=True)

class TTALightningModule(FaceAnalysisLightningModule):
    def test_step(self, batch, batch_idx):
        images = batch["images"]
        geo_ratios = batch["geometric_ratios"]
        
        # 1. Original
        output_orig = self(images, geo_ratios)
        logits_orig = output_orig.face_shape_logits
        
        # 2. Horizontal Flip
        images_flip = TF.hflip(images)
        output_flip = self(images_flip, geo_ratios)
        logits_flip = output_flip.face_shape_logits
        
        # 3. Zoom (0.95 center crop)
        _, _, h, w = images.shape
        zh, zw = int(h * 0.95), int(w * 0.95)
        images_zoom = TF.center_crop(images, [zh, zw])
        images_zoom = TF.resize(images_zoom, [h, w], antialias=True)
        output_zoom = self(images_zoom, geo_ratios)
        logits_zoom = output_zoom.face_shape_logits
        
        # 4. Rotate +5
        images_rot_p = TF.rotate(images, 5.0)
        output_rot_p = self(images_rot_p, geo_ratios)
        logits_rot_p = output_rot_p.face_shape_logits
        
        # 5. Rotate -5
        images_rot_n = TF.rotate(images, -5.0)
        output_rot_n = self(images_rot_n, geo_ratios)
        logits_rot_n = output_rot_n.face_shape_logits
        
        # Ensemble Average
        avg_logits = (logits_orig + logits_flip + logits_zoom + logits_rot_p + logits_rot_n) / 5.0
        preds = avg_logits.argmax(dim=1)
        
        # Log metrics
        self.test_acc(preds, batch["shape_labels"])
        self.log("test/acc", self.test_acc, prog_bar=True, on_epoch=True)
        
        # Store for epoch end report
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(batch["shape_labels"].detach().cpu())

def main():
    checkpoint_path = Path("checkpoints/celeba_v5/celeba_v5_epoch=14_val_f1=0.7654.ckpt")
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    config = get_config_dict()
    image_size = config["data"]["image_size"]
    
    # Use the self-train annotations but ONLY the test indices to ensure we test on the same held-out data
    annotations_path = Path("data/processed/annotations_self_train_v3.json")
    meta_path = Path("data/processed/annotations_self_train_v3.meta.json")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    test_dataset = FaceAnalysisDataset(
        annotations_path=str(annotations_path),
        image_size=image_size,
        indices=meta["test_indices"],
        transforms=get_val_transforms(image_size)
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32",
        logger=False,
        enable_checkpointing=False
    )
    
    # --- 1. Standard Evaluation ---
    logger.info("--- RUNNING STANDARD EVALUATION ---")
    model_std = FaceAnalysisLightningModule.load_from_checkpoint(
        checkpoint_path, 
        config=config,
        strict=False
    )
    res_std = trainer.test(model_std, dataloaders=test_loader)[0]
    
    # --- 2. TTA Evaluation ---
    logger.info("--- RUNNING TTA EVALUATION ---")
    # Load same weights but into TTA wrapper
    model_tta = TTALightningModule(config=config)
    model_tta.load_state_dict(model_std.state_dict())
    res_tta = trainer.test(model_tta, dataloaders=test_loader)[0]
    
    print("\n" + "="*50)
    print("FINAL TEST SET PERFORMANCE (V5 - CELEBA FINE-TUNED)")
    print("="*50)
    print(f"Checkpoint: {checkpoint_path.name}")
    print("-" * 50)
    print(f"Standard Accuracy : {res_std['test/acc']*100:.2f}%")
    print(f"TTA Accuracy      : {res_tta['test/acc']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
