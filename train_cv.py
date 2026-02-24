import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import mlflow
from tqdm import tqdm
from loguru import logger
import numpy as np
import json

from src.models.face_analysis_model import FaceAnalysisModel
from src.data.dataset import FaceAnalysisDataset, get_train_transforms, get_val_transforms
from src.config import get_config_dict

# ── Focal Loss ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        pt     = torch.exp(-ce)
        focal  = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

# ── Cross-Validation Trainer ────────────────────────────────

def train_cv(n_splits: int = 5):
    config = get_config_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting {n_splits}-Fold Cross-Validation on: {device}")

    # 1. Load FULL Dataset
    annotations_path = os.path.join(config["paths"]["processed_data"], "annotations.json")
    # We load one dataset instance to get the indices
    full_ds = FaceAnalysisDataset(
        annotations_path=annotations_path,
        image_size=config["data"]["image_size"],
        landmarks_cache_dir=config["paths"]["landmarks_cache"],
        transforms=None # Transforms applied later via wrapper or separate instances
    )
    
    # We need indices to split
    indices = np.arange(len(full_ds))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config["project"]["seed"])

    # MLflow Experiment
    mlflow.set_experiment("Face Analysis 5-Fold CV")
    
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        logger.info(f"── Fold {fold+1}/{n_splits} ──")
        logger.info(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

        # 2. Create Fold Datasets with transforms
        train_transforms = get_train_transforms(config["data"]["image_size"], config["augmentation"])
        val_transforms = get_val_transforms(config["data"]["image_size"])

        train_ds = FaceAnalysisDataset(
            annotations_path=annotations_path,
            image_size=config["data"]["image_size"],
            landmarks_cache_dir=config["paths"]["landmarks_cache"],
            transforms=train_transforms,
            indices=train_idx.tolist()
        )
        val_ds = FaceAnalysisDataset(
            annotations_path=annotations_path,
            image_size=config["data"]["image_size"],
            landmarks_cache_dir=config["paths"]["landmarks_cache"],
            transforms=val_transforms,
            indices=val_idx.tolist()
        )

        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

        # 3. Initialize Model (Reset for each fold)
        model = FaceAnalysisModel(
            backbone=config["model"]["backbone"],
            pretrained=True,
            dropout=0.5, # Enhanced dropout
            geometric_features=config["model"]["geometric_features"],
            freeze_backbone=True,
        ).to(device)

        # 4. Loss & Optimizer
        cfg_loss = config["loss"]
        shape_criterion = FocalLoss(gamma=cfg_loss["focal_gamma"], label_smoothing=cfg_loss["label_smoothing"])
        feature_criterion = nn.CrossEntropyLoss(label_smoothing=cfg_loss["label_smoothing"])
        symmetry_criterion = nn.MSELoss()

        cfg_opt = config["optimizer"]
        param_groups = model.get_optimizer_param_groups(lr=cfg_opt["lr"], backbone_lr_multiplier=cfg_opt["backbone_lr_multiplier"])
        optimizer = optim.AdamW(param_groups, weight_decay=cfg_opt["weight_decay"], betas=(cfg_opt["beta1"], cfg_opt["beta2"]), eps=cfg_opt["eps"])

        cfg_sch = config["scheduler"]
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6 / cfg_opt["lr"], end_factor=1.0, total_iters=cfg_sch["warmup_steps"])
        cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg_sch["T_0"], T_mult=cfg_sch["T_mult"], eta_min=cfg_sch["min_lr"])
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg_sch["warmup_steps"]])

        # 5. Training Loop for this Fold
        best_fold_acc = 0.0
        
        with mlflow.start_run(run_name=f"Fold_{fold+1}"):
            mlflow.log_params({
                "fold": fold + 1,
                "backbone": config["model"]["backbone"],
                "batch_size": config["training"]["batch_size"],
                "lr": cfg_opt["lr"]
            })

            # Reduced epochs for CV to save time (e.g., 50 epochs per fold)
            # Or use full epochs if dataset is small
            epochs_per_fold = 50 
            
            for epoch in range(1, epochs_per_fold + 1):
                # Unfreeze logic
                if epoch == 5: model.unfreeze_backbone(num_blocks=2)
                if epoch == 15: model.unfreeze_backbone(num_blocks=None)

                model.train()
                train_loss = 0.0
                
                # Small dataset: pbar might be just 1-2 batches
                for batch in train_loader:
                    images = batch["images"].to(device)
                    geo_ratios = batch["geometric_ratios"].to(device)
                    
                    optimizer.zero_grad()
                    output = model(images, geo_ratios)
                    
                    l_shape = shape_criterion(output.face_shape_logits, batch["shape_labels"].to(device))
                    
                    l_feat = 0.0
                    n_feat = 0
                    for task in ["eye_labels", "nose_labels", "lip_labels", "brow_labels", "jaw_labels"]:
                        if task in batch:
                            l_feat += feature_criterion(getattr(output, task.replace("labels", "logits")), batch[task].to(device))
                            n_feat += 1
                    if n_feat > 0: l_feat /= n_feat
                    
                    l_sym = 0.0
                    if "symmetry_scores" in batch:
                        l_sym = symmetry_criterion(output.symmetry_score.squeeze(), batch["symmetry_scores"].to(device))
                    
                    loss = (cfg_loss["face_shape_weight"] * l_shape) + 
                           (cfg_loss["features_weight"] * l_feat) + 
                           (0.05 * l_sym)
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch["images"].to(device)
                        geo_ratios = batch["geometric_ratios"].to(device)
                        output = model(images, geo_ratios)
                        preds = output.face_shape_logits.argmax(dim=1)
                        correct += (preds == batch["shape_labels"].to(device)).sum().item()
                        total += batch["shape_labels"].size(0)
                
                val_acc = correct / total if total > 0 else 0.0
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                
                mlflow.log_metrics({"train_loss": train_loss/len(train_loader), "val_acc": val_acc}, step=epoch)

            fold_metrics.append(best_fold_acc)
            logger.success(f"Fold {fold+1} Best Acc: {best_fold_acc:.4f}")

    avg_acc = sum(fold_metrics) / len(fold_metrics)
    logger.info(f"── 5-Fold CV Complete ──")
    logger.info(f"Average Accuracy: {avg_acc:.4f}")
    logger.info(f"Fold Results: {fold_metrics}")

if __name__ == "__main__":
    train_cv()
