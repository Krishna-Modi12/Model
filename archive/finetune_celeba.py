import os
import sys
import json
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

# Add project root to path
sys.path.append(os.getcwd())

from src.config import get_config_dict, FACE_SHAPES
from src.data.dataset import FaceAnalysisDataset
from src.training.trainer import FaceAnalysisLightningModule
from torch.utils.data import DataLoader

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_train_transforms(image_size):
    """Conservative augmentations for CelebA fine-tuning."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def main():
    L.seed_everything(42)
    project_root = Path(os.getcwd())
    
    # 1. Load Integrated Dataset
    annotations_path = project_root / "data" / "processed" / "annotations_self_train_v3.json"
    meta_path = project_root / "data" / "processed" / "annotations_self_train_v3.meta.json"
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(annotations_path, "r") as f:
        all_data = json.load(f)

    # 2. Config & Hyperparameters
    config = get_config_dict()
    image_size = config["data"]["image_size"]
    batch_size = 16
    grad_accum = 16
    lr = 3e-6  # Conservative LR for breakthrough consistency
    max_epochs = 15
    unfreeze_epoch = 2
    
    # Calculate class weights for training set
    train_labels = [all_data[i]["shape_label"] for i in meta["train_indices"]]
    weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(FACE_SHAPES)),
        y=train_labels
    )
    config["training"]["class_weights"] = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights: {weights}")

    # 3. DataLoaders
    train_dataset = FaceAnalysisDataset(
        annotations_path=str(annotations_path),
        image_size=image_size,
        indices=meta["train_indices"],
        transforms=get_train_transforms(image_size)
    )
    val_dataset = FaceAnalysisDataset(
        annotations_path=str(annotations_path),
        image_size=image_size,
        indices=meta["val_indices"],
        transforms=get_val_transforms(image_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Load Model
    checkpoint_path = project_root / "checkpoints" / "finetune_matched_v3" / "finetune_v3_epoch=23_val_f1=0.7616.ckpt"
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    module = FaceAnalysisLightningModule.load_from_checkpoint(
        checkpoint_path, 
        config=config,
        strict=False
    )
    
    # 5. INITIAL VERIFICATION (Critical fix for session reliability)
    logger.info("Running pre-training verification on validation set...")
    module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    
    from torchmetrics.classification import MulticlassF1Score
    v_f1_metric = MulticlassF1Score(num_classes=5, average="macro").to(device)
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["images"].to(device)
            geos = batch["geometric_ratios"].to(device)
            targets = batch["shape_labels"].to(device)
            out = module(imgs, geos)
            preds = out.face_shape_logits.argmax(dim=1)
            v_f1_metric.update(preds, targets)
    
    start_f1 = v_f1_metric.compute().item()
    logger.info(f"Verification Success! Initial Val F1 (Macro): {start_f1:.4f}")
    
    if start_f1 < 0.70:
        logger.error(f"Model integrity check failed (Expected > 0.70, got {start_f1:.4f}). ABORTING.")
        return

    # 6. Trainer Setup
    config["training"]["phases"]["unfreeze_full_epoch"] = unfreeze_epoch
    
    mlflow_logger = MLFlowLogger(
        experiment_name="face_shape_celeba_v1",
        run_name="celeba_v5_balanced_adaptive_3e-6"
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=project_root / "checkpoints" / "celeba_v5",
            filename="celeba_v5_{epoch:02d}_{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=2
        ),
        EarlyStopping(monitor="val_f1", patience=5, mode="max"),
        LearningRateMonitor(logging_interval="step")
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=callbacks,
        accumulate_grad_batches=grad_accum,
        precision="32"
    )

    # 7. Start Training
    logger.info("Starting fine-tuning...")
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()
