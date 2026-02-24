import os
import glob
import pandas as pd
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURATED_DATA_DIR = os.path.join(BASE_DIR, "data", "curated")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# We expect classes to map exactly to the output of get_config_dict()
# e.g., ["oval", "round", "square", "heart", "diamond", "oblong", "triangle"]
VALID_SHAPES = ["oval", "round", "square", "heart", "diamond", "oblong", "triangle"]

def build_kaggle_labels():
    """
    Scans the curated data directories derived from Kaggle.
    If an image path contains a shape folder name (e.g. data/curated/face_shape/oval/img.jpg),
    it automatically labels it for the model. 
    Returns a dataframe of path -> face_shape mappings.
    """
    logger.info("Scanning curated Kaggle dataset directories for implicit labels...")
    
    rows = []
    
    # Grab all curated images
    all_images = glob.glob(os.path.join(CURATED_DATA_DIR, "**", "*.*"), recursive=True)
    
    for img_path in all_images:
        path_lower = img_path.lower().replace('\\', '/')
        
        # Determine face shape by looking at directory structure
        # E.g., .../face_shape/round/... or .../men_face_shape/oblong/...
        found_shape = None
        for shape in VALID_SHAPES:
            if f"/{shape}/" in path_lower or f"_{shape}_" in path_lower:
                found_shape = shape
                break
                
        if found_shape:
            rows.append({
                "image_path": img_path,
                "face_shape": found_shape,
                # We do not have eye/nose/lip pre-labeled, but model can handle missing or we just train shape
                "eye_shape": "almond",     # Mock fallback to prevent training crash
                "nose_type": "straight",   # Mock fallback to prevent training crash
                "lip_fullness": "medium"   # Mock fallback to prevent training crash
            })

    df = pd.DataFrame(rows)
    logger.info(f"Found {len(df)} images with Kaggle face shape labels.")
    return df

def merge_sets(auto_csv_path, output_csv_path):
    logger.info(f"Loading auto-labels from {auto_csv_path}")
    try:
        df_auto = pd.read_csv(auto_csv_path)
    except FileNotFoundError:
        logger.error(f"Auto-labels CSV not found: {auto_csv_path}")
        return

    df_kaggle = build_kaggle_labels()
    
    if len(df_kaggle) == 0:
        logger.error("No heavily labeled kaggle folders found. Aborting.")
        return

    logger.info("Merging auto labels with Kaggle structure labels...")
    # Merge on path
    df_merged = pd.merge(df_kaggle, df_auto, on="image_path", how="inner")
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_merged.to_csv(output_csv_path, index=False)
    
    logger.info("="*50)
    logger.info(f"SUCCESS: Merged {len(df_merged)} rows to {output_csv_path}.")
    logger.info("Distribution of Face Shapes:")
    logger.info("\n" + str(df_merged['face_shape'].value_counts()))
    logger.info("="*50)

if __name__ == "__main__":
    auto_csv = os.path.join(PROCESSED_DATA_DIR, "skin_tone_auto.csv")
    out_csv = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    merge_sets(auto_csv, out_csv)
