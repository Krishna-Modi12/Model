import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm
from loguru import logger

from src.models.face_analysis_model import FaceAnalysisModel
from src.data.dataset import create_dataloaders
from src.config import get_config_dict

# ── Focal Loss (since trainer.py is broken) ──────────────────

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

# ── Training Script ─────────────────────────────────────────

def train():
    config = get_config_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting DIRECT Training on: {device}")

    # 1. Initialize Data
    logger.info("Initializing DataLoaders...")
    loaders = create_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 2. Initialize Model
    logger.info("Building FaceAnalysisModel...")
    model = FaceAnalysisModel(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        geometric_features=config["model"]["geometric_features"],
        freeze_backbone=True,
    ).to(device)

    # 3. Initialize Loss & Optimizer
    cfg_loss = config["loss"]
    shape_criterion = FocalLoss(gamma=cfg_loss["focal_gamma"], label_smoothing=cfg_loss["label_smoothing"])
    feature_criterion = nn.CrossEntropyLoss(label_smoothing=cfg_loss["label_smoothing"])
    symmetry_criterion = nn.MSELoss()

    cfg_opt = config["optimizer"]
    param_groups = model.get_optimizer_param_groups(lr=cfg_opt["lr"], backbone_lr_multiplier=cfg_opt["backbone_lr_multiplier"])
    optimizer = optim.AdamW(param_groups, weight_decay=cfg_opt["weight_decay"], betas=(cfg_opt["beta1"], cfg_opt["beta2"]), eps=cfg_opt["eps"])

    # 4. Scheduler (Warmup + Cosine)
    cfg_sch = config["scheduler"]
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6 / cfg_opt["lr"], end_factor=1.0, total_iters=cfg_sch["warmup_steps"])
    cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg_sch["T_0"], T_mult=cfg_sch["T_mult"], eta_min=cfg_sch["min_lr"])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg_sch["warmup_steps"]])

    # 5. MLflow Tracking
    mlflow.set_experiment("Face Analysis Direct")
    best_val_acc = 0.0
    checkpoint_dir = config["paths"]["checkpoints"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with mlflow.start_run():
        mlflow.log_params({
            "backbone": config["model"]["backbone"],
            "batch_size": config["training"]["batch_size"],
            "lr": cfg_opt["lr"],
            "epochs": config["training"]["epochs"]
        })

        for epoch in range(1, config["training"]["epochs"] + 1):
            # ── Unfreeze Schedule ───────────────────────────
            phase_cfg = config["training"]["phases"]
            if epoch == phase_cfg["unfreeze_partial_epoch"]:
                model.unfreeze_backbone(num_blocks=2)
            if epoch == phase_cfg["unfreeze_full_epoch"]:
                model.unfreeze_backbone(num_blocks=None)

            # ── Training Phase ──────────────────────────────
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} [Train]")
            
            for batch in pbar:
                images = batch["images"].to(device)
                geo_ratios = batch["geometric_ratios"].to(device)
                
                optimizer.zero_grad()
                output = model(images, geo_ratios)
                
                # Loss Calculation
                l_shape = shape_criterion(output.face_shape_logits, batch["shape_labels"].to(device))
                
                l_feat = 0.0
                n_feat = 0
                for task in ["eye_labels", "nose_labels", "lip_labels", "brow_labels", "jaw_labels"]:
                    if task in batch:
                        l_feat += feature_criterion(getattr(output, task.replace("labels", "logits")), batch[task].to(device))
                        n_feat += 1
                if n_feat > 0:
                    l_feat /= n_feat
                
                l_sym = 0.0
                if "symmetry_scores" in batch:
                    l_sym = symmetry_criterion(output.symmetry_score.squeeze(), batch["symmetry_scores"].to(device))
                
                total_loss = (cfg_loss["face_shape_weight"] * l_shape) + \
                             (cfg_loss["features_weight"] * l_feat) + \
                             (0.05 * l_sym) # Fixed weight for symmetry
                
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += total_loss.item()
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)

            # ── Validation Phase ────────────────────────────
            model.eval()
            val_loss = 0.0
            correct_shape = 0
            total_shape = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["images"].to(device)
                    geo_ratios = batch["geometric_ratios"].to(device)
                    output = model(images, geo_ratios)
                    
                    l_shape = shape_criterion(output.face_shape_logits, batch["shape_labels"].to(device))
                    val_loss += l_shape.item()
                    
                    preds = output.face_shape_logits.argmax(dim=1)
                    correct_shape += (preds == batch["shape_labels"].to(device)).sum().item()
                    total_shape += batch["shape_labels"].size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct_shape / total_shape

            # ── Logging & Metrics ───────────────────────────
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0]
            }, step=epoch)

            logger.info(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.4f}")

            # ── Save Checkpoint ─────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_acc_{val_acc:.4f}.pth")
                torch.save(model.state_dict(), save_path)
                logger.success(f"  --> Saved new best model to {save_path}")

if __name__ == "__main__":
    train()
