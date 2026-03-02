import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger
import sys

# Add project root to path
sys.path.append(os.getcwd())
from src.config import FACE_SHAPES

# Project Paths
PROJECT_ROOT = Path(os.getcwd())
ORIGINAL_ANNOTATIONS = PROJECT_ROOT / "data" / "processed" / "annotations.json"
PSEUDO_HIGH = Path("pseudo_labels/pseudo_labels_high.csv")
PSEUDO_MED = Path("pseudo_labels/pseudo_labels_medium.csv")
PREPROCESSED_JSON = Path("pseudo_labels/pseudo_labels_balanced_preprocessed.json")
OUTPUT_ANNOTATIONS = PROJECT_ROOT / "data" / "processed" / "annotations_self_train_v3.json"

def main():
    if not ORIGINAL_ANNOTATIONS.exists():
        logger.error(f"Original annotations not found at {ORIGINAL_ANNOTATIONS}")
        return

    # 1. Load Original Data
    with open(ORIGINAL_ANNOTATIONS, "r") as f:
        original_data = json.load(f)
    
    df_orig = pd.DataFrame(original_data)
    
    # Clean original data (Essential for valid JSON/Torch conversion)
    label_cols = ["shape_label", "eye_label", "nose_label", "lip_label", "brow_label", "jaw_label", "monk_label"]
    for col in label_cols:
        if col in df_orig.columns:
            df_orig[col] = df_orig[col].fillna(-100).astype(int)
        else:
            df_orig[col] = -100
    df_orig = df_orig.fillna(-100)

    # 2. Re-create original splits
    temp_size = 0.15 + 0.10
    train_orig, temp_df = train_test_split(
        df_orig, 
        test_size=temp_size, 
        stratify=df_orig["shape_label"], 
        random_state=42
    )
    relative_test_size = 0.10 / temp_size
    val_orig, test_orig = train_test_split(
        temp_df, 
        test_size=relative_test_size, 
        stratify=temp_df["shape_label"], 
        random_state=42
    )

    # 3. Load Pseudo-labeled Data from both High and Medium pools
    df_high = pd.read_csv(PSEUDO_HIGH)
    df_med = pd.read_csv(PSEUDO_MED)
    df_pseudo = pd.concat([df_high, df_med], ignore_index=True)
    
    # 4. Filter and Balance
    TARGET_PER_CLASS = 600
    # Thresholds: Low represented classes get a break
    THRESHOLDS = {
        0: 0.65, # Heart
        1: 0.85, # Oblong
        2: 0.65, # Oval
        3: 0.82, # Round
        4: 0.82  # Square
    }
    
    # Sort by confidence to keep the best samples
    df_pseudo = df_pseudo.sort_values("confidence", ascending=False)
    
    # Load ratios from preprocessed JSON (index by md5)
    with open(PREPROCESSED_JSON, "r") as f:
        preprocessed_data = json.load(f)
    ratios_map = {entry["md5"]: entry["geometric_ratios"] for entry in preprocessed_data}
    path_map = {entry["md5"]: entry["image_path"] for entry in preprocessed_data}

    pseudo_list = []
    class_counts = {i: 0 for i in range(5)}
    
    for _, row in df_pseudo.iterrows():
        label = int(row["shape_label"])
        conf = float(row["confidence"])
        md5 = row["md5"]
        
        if label < 5 and class_counts[label] < TARGET_PER_CLASS:
            if conf >= THRESHOLDS[label]:
                # We need the preprocessed image path and ratios
                if md5 in ratios_map:
                    pseudo_list.append({
                        "image_path": path_map[md5],
                        "shape_label": label,
                        "eye_label": -100,
                        "nose_label": -100,
                        "lip_label": -100,
                        "brow_label": -100,
                        "jaw_label": -100,
                        "monk_label": -100,
                        "geometric_ratios": ratios_map[md5],
                        "dataset_source": "celeba_pseudo"
                    })
                    class_counts[label] += 1
    
    logger.info(f"Pseudo-label Counts (with adaptive thresholds):")
    for i, name in enumerate(FACE_SHAPES):
        logger.info(f"  {name:<10}: {class_counts[i]} (Threshold: {THRESHOLDS[i]})")

    # 5. Combine
    combined_data = val_orig.to_dict('records') + test_orig.to_dict('records') + train_orig.to_dict('records') + pseudo_list
    
    metadata = {
        "val_indices": list(range(0, len(val_orig))),
        "test_indices": list(range(len(val_orig), len(val_orig) + len(test_orig))),
        "train_indices": list(range(len(val_orig) + len(test_orig), len(combined_data)))
    }
    
    with open(OUTPUT_ANNOTATIONS, "w") as f:
        json.dump(combined_data, f, indent=2)
    with open(OUTPUT_ANNOTATIONS.with_suffix(".meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Final training set: {len(metadata['train_indices'])} samples ({len(train_orig)} original, {len(pseudo_list)} pseudo).")

if __name__ == "__main__":
    main()
