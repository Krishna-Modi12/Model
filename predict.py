"""
predict.py — Production Inference Pipeline for Face Shape Classification
=========================================================================
Usage:
  Single image:   python predict.py --image "path/to/photo.jpg"
  Batch folder:   python predict.py --folder "path/to/folder/"
  With visual:    python predict.py --image "path/to/photo.jpg" --visualize
  Custom ckpt:    python predict.py --image "photo.jpg" --checkpoint "path/to/model.ckpt"
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
DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "checkpoints" / "celeba_v5"
    / "celeba_v5_epoch=14_val_f1=0.7654.ckpt"
)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Preprocessing (mirrors val_transforms from dataset.py + TTA) ──
def preprocess_image(image_bgr: np.ndarray, apply_flip: bool = False, scale: float = 1.0, rotate_deg: float = 0.0) -> torch.Tensor:
    """
    Apply preprocessing for test-time inference.
    If apply_flip=True, also performs a horizontal flip for Test-Time Augmentation.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    if apply_flip:
        image_rgb = cv2.flip(image_rgb, 1)

    # Optional affine transforms for TTA
    if scale != 1.0 or rotate_deg != 0.0:
        h, w = image_rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rotate_deg, scale)
        image_rgb = cv2.warpAffine(image_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    image_resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    image_float = image_resized.astype(np.float32) / 255.0

    # Normalize
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    image_normalized = (image_float - mean) / std

    # HWC → CHW
    tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
    return tensor


# ── Face Detection ──────────────────────────────────────────
def detect_and_crop_faces(
    image_bgr: np.ndarray,
    extractor: LandmarkExtractor,
    padding: float = 0.20
) -> List[Dict]:
    """
    Detect faces using MediaPipe and return cropped face regions
    with their geometric ratios and bounding boxes.

    Returns a list of dicts, one per detected face:
      {
        "face_crop": np.ndarray (BGR),
        "geometric_ratios": np.ndarray (15,),
        "bbox": (x, y, w, h),
        "landmarks_px": np.ndarray
      }
    """
    result = extractor.extract(image_bgr)

    if not result.success:
        return []

    # MediaPipe FaceMesh returns one face per call
    x, y, w, h = result.face_bbox
    ih, iw = image_bgr.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)

    face_crop = image_bgr[y1:y2, x1:x2]

    return [{
        "face_crop": face_crop,
        "geometric_ratios": result.geometric_ratios,
        "bbox": (x1, y1, x2 - x1, y2 - y1),
        "landmarks_px": result.landmarks_px,
    }]


# ── Model Loading ───────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load the trained model from a Lightning checkpoint.
    Returns the inner FaceAnalysisModel in eval mode.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    lightning_module = FaceAnalysisLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model = lightning_module.model
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    return model


# ── Single Image Prediction ────────────────────────────────
def predict_single(
    image_path: str,
    model: torch.nn.Module,
    extractor: LandmarkExtractor,
    device: torch.device,
) -> List[Dict]:
    """
    Run prediction on a single image file.
    Returns a list of result dicts (one per detected face).
    """
    image_path = str(Path(image_path).resolve())

    # Validate file
    if not os.path.exists(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        return [{"image": image_path, "error": "File not found"}]

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [ERROR] Could not read image (corrupt?): {image_path}")
        return [{"image": image_path, "error": "Could not read image"}]

    # Detect faces
    faces = detect_and_crop_faces(image_bgr, extractor)
    if not faces:
        print(f"  [WARNING] No face detected in: {os.path.basename(image_path)}")
        return [{"image": image_path, "error": "No face detected — try a front-facing photo with good lighting and no obstructions. Slight tilts are OK."}]

    results = []
    for face_idx, face_data in enumerate(faces):
        # 5-Crop equivalent TTA
        tta_transforms = [
            {"apply_flip": False, "scale": 1.0, "rotate_deg": 0.0},
            {"apply_flip": True,  "scale": 1.0, "rotate_deg": 0.0},
            {"apply_flip": False, "scale": 1.05, "rotate_deg": 0.0},    # Slight zoom
            {"apply_flip": False, "scale": 1.0, "rotate_deg": 5.0},     # Slight correct rotation
            {"apply_flip": False, "scale": 1.0, "rotate_deg": -5.0},    # Slight reverse rotation
        ]

        geo_ratios = torch.from_numpy(face_data["geometric_ratios"]).float().unsqueeze(0).to(device)

        logits_list = []
        with torch.no_grad():
            for tta_kwargs in tta_transforms:
                image_tensor = preprocess_image(face_data["face_crop"], **tta_kwargs).unsqueeze(0).to(device)
                output = model(image_tensor, geo_ratios)
                logits_list.append(output.face_shape_logits)
            
            # Simple average of logits from all robust passes
            avg_logits = torch.stack(logits_list).mean(dim=0)
            probs  = F.softmax(avg_logits, dim=1).squeeze(0).cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_class = FACE_SHAPES[predicted_idx]
        confidence = float(probs[predicted_idx])

        all_scores = {
            FACE_SHAPES[i]: round(float(probs[i]), 4)
            for i in range(len(FACE_SHAPES))
        }

        result = {
            "image": image_path,
            "face_index": face_idx,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "bbox": face_data["bbox"],
        }
        results.append(result)

    return results


# ── Pretty Console Output ──────────────────────────────────
def print_result(result: Dict) -> None:
    """Print a single prediction result in a human-readable format."""
    if "error" in result:
        print(f"\n  Image: {os.path.basename(result['image'])}")
        print(f"  Status: {result['error']}\n")
        return

    filename = os.path.basename(result["image"])
    print()
    print("=" * 48)
    print(f"  Image: {filename}")
    if result.get("face_index", 0) > 0:
        print(f"  Face #: {result['face_index'] + 1}")
    print("=" * 48)
    print(f"  Predicted Face Shape : {result['predicted_class']}")
    print(f"  Confidence           : {result['confidence'] * 100:.2f}%")
    print()
    print("  All Class Scores:")
    for cls_name, score in result["all_scores"].items():
        bar = "#" * int(score * 30)
        marker = " <--" if cls_name == result["predicted_class"] else ""
        print(f"    {cls_name:<8}: {score * 100:6.2f}%  {bar}{marker}")
    print("=" * 48)


# ── Save JSON Output ───────────────────────────────────────
def save_json(result: Dict, output_path: str) -> None:
    """Save prediction result as a JSON file."""
    # Remove bbox tuple (not JSON serializable) — convert to list
    save_data = dict(result)
    if "bbox" in save_data:
        save_data["bbox"] = list(save_data["bbox"])

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  JSON saved: {output_path}")


# ── Visualization ───────────────────────────────────────────
def visualize_result(
    image_path: str,
    results: List[Dict],
    output_path: str = None
) -> None:
    """
    Draw bounding boxes and predictions on the image.
    Saves the annotated image.
    """
    image = cv2.imread(image_path)
    if image is None:
        return

    for result in results:
        if "error" in result:
            continue

        bx, by, bw, bh = result["bbox"]
        conf = result["confidence"]
        cls_name = result["predicted_class"]

        # Draw bounding box
        color = (0, 255, 120)
        cv2.rectangle(image, (bx, by), (bx + bw, by + bh), color, 2)

        # Draw label background
        label = f"{cls_name} ({conf * 100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (bx, by - th - 10), (bx + tw + 6, by), color, -1)
        cv2.putText(image, label, (bx + 3, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if output_path is None:
        stem = Path(image_path).stem
        output_path = str(Path(image_path).parent / f"{stem}_result.jpg")

    cv2.imwrite(output_path, image)
    print(f"  Visualization saved: {output_path}")


# ── Batch Processing ───────────────────────────────────────
def process_folder(
    folder_path: str,
    model: torch.nn.Module,
    extractor: LandmarkExtractor,
    device: torch.device,
    visualize: bool = False,
) -> List[Dict]:
    """Process all supported images in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder_path}")
        return []

    image_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    ])

    if not image_files:
        print(f"[WARNING] No supported images found in: {folder_path}")
        return []

    print(f"\nProcessing {len(image_files)} images from: {folder_path}\n")

    all_results = []
    for img_file in image_files:
        results = predict_single(str(img_file), model, extractor, device)
        for r in results:
            print_result(r)
            all_results.append(r)

        if visualize and results and "error" not in results[0]:
            visualize_result(str(img_file), results)

    # Print summary table
    print("\n" + "=" * 60)
    print("  BATCH PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  {'Image':<30} {'Prediction':<12} {'Confidence':<10}")
    print("-" * 60)
    for r in all_results:
        name = os.path.basename(r.get("image", ""))[:28]
        if "error" in r:
            print(f"  {name:<30} {'SKIPPED':<12} {r['error']}")
        else:
            print(f"  {name:<30} {r['predicted_class']:<12} {r['confidence'] * 100:.1f}%")
    print("=" * 60)

    # Save combined JSON
    batch_json_path = str(folder / "batch_predictions.json")
    serializable = []
    for r in all_results:
        rd = dict(r)
        if "bbox" in rd:
            rd["bbox"] = list(rd["bbox"])
        serializable.append(rd)

    with open(batch_json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Batch JSON saved: {batch_json_path}")

    return all_results


# ── Main Entry Point ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Face Shape Classification — Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image "myphoto.jpg"
  python predict.py --image "myphoto.jpg" --visualize
  python predict.py --folder "photos/"
  python predict.py --image "myphoto.jpg" --checkpoint "path/to/model.ckpt"
        """,
    )
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint (default: best finetuned)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated image with bounding box and prediction")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("You must provide either --image or --folder")

    # ── Device Setup ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 48}")
    print(f"  Face Shape Classifier — 76.83% accuracy (V3)")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 48}\n")

    # ── Load Model ──────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint, device)

    # ── Initialize Landmark Extractor ───────────────────────
    extractor = LandmarkExtractor(static_image_mode=True)

    try:
        # ── Single Image Mode ──────────────────────────────
        if args.image:
            results = predict_single(args.image, model, extractor, device)
            for r in results:
                print_result(r)
                # Save individual JSON
                if "error" not in r:
                    stem = Path(args.image).stem
                    json_path = str(Path(args.image).parent / f"{stem}_prediction.json")
                    save_json(r, json_path)

            if args.visualize and results and "error" not in results[0]:
                visualize_result(args.image, results)

        # ── Batch Folder Mode ──────────────────────────────
        elif args.folder:
            process_folder(args.folder, model, extractor, device, args.visualize)

    finally:
        extractor.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
