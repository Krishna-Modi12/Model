import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import hashlib
from loguru import logger

# Add project root to path
import sys
sys.path.append(os.getcwd())

from src.utils.landmark_extractor import LandmarkExtractor

def crop_face(image_bgr, landmarks_px, padding=0.20):
    ih, iw = image_bgr.shape[:2]
    
    # Compute bbox from landmarks
    x_min, y_min = np.min(landmarks_px, axis=0)
    x_max, y_max = np.max(landmarks_px, axis=0)
    
    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
    
    # Apply padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(iw, int(x + w + pad_x))
    y2 = min(ih, int(y + h + pad_y))

    return image_bgr[y1:y2, x1:x2]

def main():
    # Use the BALANCED pool
    input_csv = "pseudo_labels/pseudo_labels_balanced_pool.csv"
    output_dir = Path("data/curated/celeba_faces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path("data/landmarks_cache")
    
    if not Path(input_csv).exists():
        logger.error(f"Input CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    logger.info(f"Preprocessing {len(df)} CelebA faces from BALANCED pool (with fallback extraction)...")
    
    extractor = LandmarkExtractor(static_image_mode=True)
    processed_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        orig_path = row["image_path"]
        md5 = row["md5"]
        
        # Load landmarks from cache if possible
        cache_key = hashlib.md5(orig_path.encode()).hexdigest() + ".npz"
        cache_path = cache_dir / cache_key
        
        landmarks_px = None
        geometric_ratios = None
        
        if cache_path.exists():
            try:
                cached = np.load(str(cache_path))
                landmarks_px = cached["landmarks_px"]
                geometric_ratios = cached["geometric_ratios"].tolist()
            except:
                pass
                
        if landmarks_px is None:
            # Fallback extraction
            image = cv2.imread(orig_path)
            if image is None: continue
            result = extractor.extract(image)
            if not result.success:
                continue
            landmarks_px = result.landmarks_px
            geometric_ratios = result.geometric_ratios.tolist()
            # Save back to cache
            np.savez_compressed(str(cache_path), landmarks_px=landmarks_px, geometric_ratios=result.geometric_ratios)
        
        try:
            new_filename = f"{md5}.jpg"
            new_path = output_dir / new_filename
            
            # Use original image for cropping
            image = cv2.imread(orig_path)
            if image is None: continue
            face_crop = crop_face(image, landmarks_px)
            cv2.imwrite(str(new_path), face_crop)
            
            entry = {
                "image_path": str(new_path.absolute()),
                "shape_label": int(row["shape_label"]),
                "confidence": float(row["confidence"]),
                "md5": md5,
                "geometric_ratios": geometric_ratios
            }
            processed_list.append(entry)
            
        except Exception as e:
            logger.error(f"Failed to process {orig_path}: {e}")
            
    output_json = "pseudo_labels/pseudo_labels_balanced_preprocessed.json"
    with open(output_json, "w") as f:
        json.dump(processed_list, f, indent=2)
    
    logger.success(f"Final BALANCED pool: {len(processed_list)} faces. Saved to {output_json}")

if __name__ == "__main__":
    main()
