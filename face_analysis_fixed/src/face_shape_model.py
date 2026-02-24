"""
face_shape_model.py  (FIXED)
─────────────────────────────────────────────────────────────
Fixes applied:
  1. Duplicate lip_conf calculation removed from predict().
     Previously lip_probs[i].max(0) was called twice; lip_conf2
     was computed and silently discarded.
─────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger


FACE_SHAPE_CLASSES = ["oval", "round", "square", "heart", "diamond", "oblong", "triangle"]
EYE_CLASSES        = ["almond", "round", "hooded", "monolid", "downturned", "upturned"]
NOSE_CLASSES       = ["straight", "roman", "button", "upturned", "bulbous"]
LIP_CLASSES        = ["thin", "medium", "full", "cupid"]
BROW_CLASSES       = ["flat", "arched", "s-shaped"]
JAW_CLASSES        = ["pointed", "square", "rounded"]


@dataclass
class ModelOutput:
    face_shape_logits: torch.Tensor   # (B, 7)
    eye_logits:        torch.Tensor   # (B, 6)
    nose_logits:       torch.Tensor   # (B, 5)
    lip_logits:        torch.Tensor   # (B, 4)
    brow_logits:       torch.Tensor   # (B, 3)
    jaw_logits:        torch.Tensor   # (B, 3)
    symmetry_score:    torch.Tensor   # (B, 1)


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaceShapeHead(nn.Module):
    """
    Combines visual backbone features + 15 geometric landmark ratios
    for higher accuracy than visual features alone.
    """

    def __init__(self, backbone_features: int, geometric_features: int = 15,
                 num_classes: int = 7, dropout: float = 0.4):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.geo_proj = nn.Sequential(
            nn.Linear(geometric_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, visual: torch.Tensor,
                geometric: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            torch.cat([self.visual_proj(visual), self.geo_proj(geometric)], dim=1)
        )


class FaceAnalysisModel(nn.Module):
    """Multi-task face analysis model on EfficientNet-B4 backbone."""

    def __init__(self,
                 backbone: str = "efficientnet_b4",
                 pretrained: bool = True,
                 dropout: float = 0.4,
                 geometric_features: int = 15,
                 freeze_backbone: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        backbone_features = self.backbone.num_features
        logger.info(f"Backbone: {backbone} | Features: {backbone_features} | "
                    f"Pretrained: {pretrained}")

        if freeze_backbone:
            self.freeze_backbone()

        self.face_shape_head = FaceShapeHead(
            backbone_features, geometric_features, num_classes=7, dropout=dropout)
        self.eye_head    = ClassificationHead(backbone_features, 256, 6,  dropout)
        self.nose_head   = ClassificationHead(backbone_features, 256, 5,  dropout)
        self.lip_head    = ClassificationHead(backbone_features, 256, 4,  dropout)
        self.brow_head   = ClassificationHead(backbone_features, 128, 3,  dropout)
        self.jaw_head    = ClassificationHead(backbone_features, 128, 3,  dropout)
        self.symmetry_head = nn.Sequential(
            nn.Linear(backbone_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        logger.info("FaceAnalysisModel built successfully")

    def forward(self, images: torch.Tensor,
                geometric_ratios: torch.Tensor) -> ModelOutput:
        features = self.backbone(images)
        return ModelOutput(
            face_shape_logits = self.face_shape_head(features, geometric_ratios),
            eye_logits        = self.eye_head(features),
            nose_logits       = self.nose_head(features),
            lip_logits        = self.lip_head(features),
            brow_logits       = self.brow_head(features),
            jaw_logits        = self.jaw_head(features),
            symmetry_score    = self.symmetry_head(features),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone FROZEN")

    def unfreeze_backbone(self, num_blocks: Optional[int] = None):
        if num_blocks is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone fully UNFROZEN")
        else:
            blocks = list(self.backbone.children())
            for block in blocks[-num_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            logger.info(f"Backbone last {num_blocks} blocks UNFROZEN")

    def get_optimizer_param_groups(self, lr: float,
                                    backbone_lr_multiplier: float = 0.1):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.face_shape_head.parameters()) +
            list(self.eye_head.parameters()) +
            list(self.nose_head.parameters()) +
            list(self.lip_head.parameters()) +
            list(self.brow_head.parameters()) +
            list(self.jaw_head.parameters()) +
            list(self.symmetry_head.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr * backbone_lr_multiplier,
             "name": "backbone"},
            {"params": head_params,     "lr": lr,
             "name": "heads"},
        ]

    def predict(self, images: torch.Tensor,
                geometric_ratios: torch.Tensor,
                confidence_threshold: float = 0.60) -> list:
        """Returns human-readable predictions for a batch."""
        self.eval()
        with torch.no_grad():
            output = self.forward(images, geometric_ratios)

        shape_probs = F.softmax(output.face_shape_logits, dim=1)
        eye_probs   = F.softmax(output.eye_logits,        dim=1)
        nose_probs  = F.softmax(output.nose_logits,       dim=1)
        lip_probs   = F.softmax(output.lip_logits,        dim=1)
        brow_probs  = F.softmax(output.brow_logits,       dim=1)
        jaw_probs   = F.softmax(output.jaw_logits,        dim=1)

        results = []
        for i in range(images.shape[0]):
            # FIX: Each head gets exactly one .max(0) call — no duplicates
            shape_conf, shape_idx = shape_probs[i].max(0)
            eye_conf,   eye_idx   = eye_probs[i].max(0)
            nose_conf,  nose_idx  = nose_probs[i].max(0)
            lip_conf,   lip_idx   = lip_probs[i].max(0)   # was called twice before
            brow_conf,  brow_idx  = brow_probs[i].max(0)
            jaw_conf,   jaw_idx   = jaw_probs[i].max(0)

            top3_vals, top3_idx = shape_probs[i].topk(3)
            top3 = [{"shape": FACE_SHAPE_CLASSES[j], "confidence": float(v)}
                    for j, v in zip(top3_idx.tolist(), top3_vals.tolist())]

            def certain(label, conf):
                return label if float(conf) >= confidence_threshold else "uncertain"

            results.append({
                "face_shape": {
                    "label":      certain(FACE_SHAPE_CLASSES[shape_idx], shape_conf),
                    "confidence": float(shape_conf),
                    "top3":       top3,
                },
                "features": {
                    "eyes":     {"shape":    certain(EYE_CLASSES[eye_idx],   eye_conf),
                                 "confidence": float(eye_conf)},
                    "nose":     {"type":     certain(NOSE_CLASSES[nose_idx], nose_conf),
                                 "confidence": float(nose_conf)},
                    "lips":     {"fullness": certain(LIP_CLASSES[lip_idx],   lip_conf),
                                 "confidence": float(lip_conf)},
                    "eyebrows": {"shape":    certain(BROW_CLASSES[brow_idx], brow_conf),
                                 "confidence": float(brow_conf)},
                    "jawline":  {"type":     certain(JAW_CLASSES[jaw_idx],   jaw_conf),
                                 "confidence": float(jaw_conf)},
                },
                "symmetry_score": float(output.symmetry_score[i].item()),
            })

        return results

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Testing on {device}")

    model = FaceAnalysisModel(
        backbone="efficientnet_b4", pretrained=False, freeze_backbone=True
    ).to(device)

    params = model.count_parameters()
    logger.info(f"Params — Total: {params['total']:,} | "
                f"Trainable: {params['trainable']:,} | Frozen: {params['frozen']:,}")

    images = torch.randn(4, 3, 256, 256).to(device)
    ratios = torch.randn(4, 15).to(device)
    output = model(images, ratios)
    preds  = model.predict(images, ratios)

    logger.info(f"face_shape_logits: {output.face_shape_logits.shape}")
    logger.info(f"Sample prediction: {preds[0]['face_shape']}")
    logger.success("Model forward pass OK!")
