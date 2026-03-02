import os
import json
import cv2
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.landmark_extractor import LandmarkExtractor

def main():
    # 1. Load annotations
    annotations_path = PROJECT_ROOT / "data" / "processed" / "annotations.json"
    if not annotations_path.exists():
        logger.error(f"Annotations not found at {annotations_path}")
        return

    with open(annotations_path) as f:
        annotations = json.load(f)

    # 2. Setup cache directory
    cache_dir = PROJECT_ROOT / "data" / "landmarks_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. Initialize MediaPipe Extractor
    # Note: We use static_image_mode for better accuracy on individual frames
    extractor = LandmarkExtractor(static_image_mode=True)
    
    logger.info(f"Starting pre-processing for {len(annotations)} images...")
    logger.info(f"Cache directory: {cache_dir}")

    success_count = 0
    corrupt_count = 0   # Image file unreadable
    no_face_count = 0   # MediaPipe couldn't detect a face
    skip_count = 0

    # 4. Process each image
    for ann in tqdm(annotations, desc="Pre-processing landmarks",
                    unit="img", dynamic_ncols=True):
        image_path_str = ann["image_path"]
        
        # Use MD5 hash for collision-proof cache key (matches dataset.py)
        cache_key = hashlib.md5(image_path_str.encode()).hexdigest() + ".npz"
        cache_path = cache_dir / cache_key

        # Skip if already cached
        if cache_path.exists():
            skip_count += 1
            continue

        # Load image
        # Support both absolute and relative paths
        full_path = Path(image_path_str)
        if not full_path.is_absolute():
            full_path = PROJECT_ROOT / image_path_str

        image = cv2.imread(str(full_path))
        if image is None:
            logger.warning(f"Could not load image: {full_path}")
            corrupt_count += 1
            continue

        # Extract landmarks
        result = extractor.extract(image)
        
        if result.success:
            # Save relevant data to compressed numpy file
            # We save landmarks_px (478, 2) and geometric_ratios (15,)
            np.savez_compressed(
                cache_path,
                landmarks_px=result.landmarks_px,
                geometric_ratios=result.geometric_ratios,
                face_bbox=np.array(result.face_bbox) if result.face_bbox else np.array([])
            )
            success_count += 1
        else:
            # We don't cache failures so we can retry them later if needed
            logger.debug(f"Extraction failed for {image_path_str}: {result.error}")
            no_face_count += 1

    extractor.close()
    
    logger.info("Pre-processing complete!")
    logger.info(f"Successfully cached: {success_count}")
    logger.info(f"Skipped (already cached): {skip_count}")
    logger.info(f"Corrupt/unreadable images: {corrupt_count}")
    logger.info(f"No face detected: {no_face_count}")
    logger.info(f"Total entries in annotations: {len(annotations)}")

if __name__ == "__main__":
    main()
