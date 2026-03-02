"""
ensemble_predict.py — Dual Model Inference Pipeline
=========================================================================
Predicts face shapes using an ensemble of two trained models:
1) EfficientNet-B4 (Baseline / Phase 2)
2) ConvNeXt-Small (New Baseline / Phase 4)

Usage:
  Single image:   python ensemble_predict.py --image "path/to/photo.jpg"
  Batch folder:   python ensemble_predict.py --folder "path/to/folder/"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FACE_SHAPES
from src.training.trainer import FaceAnalysisLightningModule
from src.utils.landmark_extractor import LandmarkExtractor

# ── Constants ───────────────────────────────────────────────
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

MODEL1_CHECKPOINT = str(PROJECT_ROOT / "checkpoints" / "finetuned" / "face_analysis_unfreeze_epoch=89_val_f1=0.7178.ckpt")
MODEL2_CHECKPOINT = str(PROJECT_ROOT / "checkpoints" / "convnext_small" / "best_convnext.ckpt") # Will be updated after training

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Preprocessing ───────────────────────────────────────────
def preprocess_image(image_bgr: np.ndarray, apply_flip: bool = False, scale: float = 1.0, rotate_deg: float = 0.0) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    if apply_flip:
        image_rgb = cv2.flip(image_rgb, 1)

    if scale != 1.0 or rotate_deg != 0.0:
        h, w = image_rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rotate_deg, scale)
        image_rgb = cv2.warpAffine(image_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    image_resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    image_float = image_resized.astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    image_normalized = (image_float - mean) / std

    tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
    return tensor


# ── Face Detection ──────────────────────────────────────────
def detect_and_crop_faces(image_bgr: np.ndarray, extractor: LandmarkExtractor, padding: float = 0.20) -> List[Dict]:
    result = extractor.extract(image_bgr)
    if not result.success: return []

    x, y, w, h = result.face_bbox
    ih, iw = image_bgr.shape[:2]
    pad_x, pad_y = int(w * padding), int(h * padding)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(iw, x + w + pad_x), min(ih, y + h + pad_y)
    
    face_crop = image_bgr[y1:y2, x1:x2]

    return [{
        "face_crop": face_crop,
        "geometric_ratios": result.geometric_ratios,
        "bbox": (x1, y1, x2 - x1, y2 - y1),
        "landmarks_px": result.landmarks_px,
    }]


# ── Ensemble Inference ─────────────────────────────────────
def predict_single_ensemble(
    image_path: str,
    models: List[torch.nn.Module],
    model_weights: List[float],
    extractor: LandmarkExtractor,
    device: torch.device,
) -> List[Dict]:
    """Run Test-Time Augmentation on BOTH models and average the resulting probabilities."""
    image_path = str(Path(image_path).resolve())
    if not os.path.exists(image_path): return [{"image": image_path, "error": "File not found"}]

    image_bgr = cv2.imread(image_path)
    if image_bgr is None: return [{"image": image_path, "error": "Could not read image"}]

    faces = detect_and_crop_faces(image_bgr, extractor)
    if not faces: return [{"image": image_path, "error": "No face detected"}]

    results = []
    
    # 5-Crop TTA
    tta_transforms = [
        {"apply_flip": False, "scale": 1.0, "rotate_deg": 0.0},
        {"apply_flip": True,  "scale": 1.0, "rotate_deg": 0.0},
        {"apply_flip": False, "scale": 1.05, "rotate_deg": 0.0},
        {"apply_flip": False, "scale": 1.0, "rotate_deg": 5.0},
        {"apply_flip": False, "scale": 1.0, "rotate_deg": -5.0},
    ]

    for face_idx, face_data in enumerate(faces):
        geo_ratios = torch.from_numpy(face_data["geometric_ratios"]).float().unsqueeze(0).to(device)
        
        ensemble_probs = np.zeros(len(FACE_SHAPES), dtype=np.float32)

        with torch.no_grad():
            for model_idx, model in enumerate(models):
                logits_list = []
                for tta_kwargs in tta_transforms:
                    image_tensor = preprocess_image(face_data["face_crop"], **tta_kwargs).unsqueeze(0).to(device)
                    output = model(image_tensor, geo_ratios)
                    logits_list.append(output.face_shape_logits)
                
                # Average TTA passes for THIS model
                avg_logits = torch.stack(logits_list).mean(dim=0)
                probs = F.softmax(avg_logits, dim=1).squeeze(0).cpu().numpy()
                
                # Add to ensemble pool
                ensemble_probs += probs * model_weights[model_idx]

        predicted_idx = int(np.argmax(ensemble_probs))
        
        results.append({
            "image": image_path,
            "face_index": face_idx,
            "predicted_class": FACE_SHAPES[predicted_idx],
            "confidence": round(float(ensemble_probs[predicted_idx]), 4),
            "all_scores": {FACE_SHAPES[i]: round(float(ensemble_probs[i]), 4) for i in range(len(FACE_SHAPES))},
            "bbox": face_data["bbox"],
        })

    return results


# ── Utilities ───────────────────────────────────────────────
def print_result(result: Dict) -> None:
    if "error" in result:
        print(f"\n  Image: {os.path.basename(result['image'])} | Status: {result['error']}\n")
        return

    print(f"\n{'=' * 48}\n  Image: {os.path.basename(result['image'])}")
    print(f"  Predicted Face Shape : {result['predicted_class']} ({result['confidence'] * 100:.2f}%)")
    for cls_name, score in result["all_scores"].items():
        bar = "#" * int(score * 30)
        marker = " <--" if cls_name == result["predicted_class"] else ""
        print(f"    {cls_name:<8}: {score * 100:6.2f}%  {bar}{marker}")
    print("=" * 48)

def load_model(ckpt: str, device: torch.device):
    print(f"Loading checkpoint: {ckpt}")
    lightning_module = FaceAnalysisLightningModule.load_from_checkpoint(ckpt, map_location=device)
    model = lightning_module.model.to(device)
    model.eval()
    return model


# ── Main Entry Point ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ensemble Inference")
    parser.add_argument("--image", type=str, help="Single image")
    parser.add_argument("--folder", type=str, help="Folder of images")
    parser.add_argument("--ckpt1", type=str, default=MODEL1_CHECKPOINT)
    parser.add_argument("--ckpt2", type=str, default=MODEL2_CHECKPOINT)
    parser.add_argument("--weight1", type=float, default=0.5, help="Weight for EfficientNetB4 model (default 0.5)")
    parser.add_argument("--weight2", type=float, default=0.5, help="Weight for ConvNeXt model (default 0.5)")
    args = parser.parse_args()

    if not args.image and not args.folder: parser.error("Must provide --image or --folder")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n== Ensemble Inference ({device}) ==")
    
    if not os.path.exists(args.ckpt1):
        print(f"WARNING: Checkpoint 1 missing: {args.ckpt1}")
        return
    if not os.path.exists(args.ckpt2):
        print(f"WARNING: Checkpoint 2 missing: {args.ckpt2}. Perhaps training isn't finished yet?")
        print("Please wait for Phase 4 to finish and pass --ckpt2 path/to/model.ckpt to test.")
        return

    models = [load_model(args.ckpt1, device), load_model(args.ckpt2, device)]
    weights = [args.weight1, args.weight2]
    
    # Normalize weights just in case
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    extractor = LandmarkExtractor(static_image_mode=True)

    try:
        if args.image:
            results = predict_single_ensemble(args.image, models, weights, extractor, device)
            for r in results: print_result(r)
        elif args.folder:
            folder = Path(args.folder)
            files = sorted([f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS])
            for f in files:
                for r in predict_single_ensemble(str(f), models, weights, extractor, device):
                    print_result(r)
    finally:
        extractor.close()

if __name__ == "__main__":
    main()
