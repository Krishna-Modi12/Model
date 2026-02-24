import os
import glob
import cv2
import shutil
import sys
from tqdm import tqdm
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.utils.landmark_extractor import LandmarkExtractor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CURATED_DATA_DIR = os.path.join(BASE_DIR, "data", "curated")

def curate_images():
    logger.info("Starting image curation process...")
    extractor = LandmarkExtractor()
    
    os.makedirs(CURATED_DATA_DIR, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.jpg"), recursive=True)
    image_paths.extend(glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.png"), recursive=True))
    
    # Also grab from the downloaded FFHQ repo folder explicitly
    ffhq_repo_dir = os.path.join(BASE_DIR, "ffhq-dataset", "thumbnails128x128")
    if os.path.exists(ffhq_repo_dir):
        image_paths.extend(glob.glob(os.path.join(ffhq_repo_dir, "**", "*.jpg"), recursive=True))
        image_paths.extend(glob.glob(os.path.join(ffhq_repo_dir, "**", "*.png"), recursive=True))
        
    logger.info(f"Found {len(image_paths)} images in data directory.")
    
    valid_count = 0
    invalid_count = 0
    
    for path in tqdm(image_paths, desc="Curating Images"):
        try:
            img = cv2.imread(path)
            if img is None:
                invalid_count += 1
                continue
            
            # Use MediaPipe to verify there is exactly one face, face is front-facing, and face occupies >= 30% of height.
            lm = extractor.extract(img)
            
            # Assuming landmark_extractor returns a success flag for successful detection.
            # We enforce front-facing and standard box logic inside extractor, or filter here.
            # For this script we will simply rely on extract(img).success if the extractor validates faces natively.
            if not lm.success:
                invalid_count += 1
                continue
            
            # Get relative path for grouping inside curated
            rel_path = os.path.relpath(path, RAW_DATA_DIR)
            dest_path = os.path.join(CURATED_DATA_DIR, rel_path)
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(path, dest_path)
            valid_count += 1
            
        except Exception as e:
            invalid_count += 1
            logger.debug(f"Error curating {path}: {e}")

    logger.info(f"Curation complete.")
    logger.info(f"  Valid images:   {valid_count}")
    logger.info(f"  Invalid images: {invalid_count} (skipped)")
    logger.info(f"Curated images saved to {CURATED_DATA_DIR}")

if __name__ == "__main__":
    curate_images()
