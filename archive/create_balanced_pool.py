import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def main():
    high_csv = Path("pseudo_labels/pseudo_labels_high.csv")
    med_csv = Path("pseudo_labels/pseudo_labels_medium.csv")
    
    df_h = pd.read_csv(high_csv)
    df_m = pd.read_csv(med_csv)
    
    df_all = pd.concat([df_h, df_m], ignore_index=True)
    
    # Target: Balanced pool (Max 1000 per class)
    TARGET_PER_CLASS = 1000
    
    selected_dfs = []
    
    for label in range(5):
        class_df = df_all[df_all["shape_label"] == label]
        if len(class_df) > TARGET_PER_CLASS:
            # Sort by confidence and take top
            class_df = class_df.sort_values("confidence", ascending=False).head(TARGET_PER_CLASS)
        
        selected_dfs.append(class_df)
        print(f"Class {label}: Selected {len(class_df)} samples")
        
    df_balanced = pd.concat(selected_dfs, ignore_index=True)
    output_csv = Path("pseudo_labels/pseudo_labels_balanced_pool.csv")
    df_balanced.to_csv(output_csv, index=False)
    logger.success(f"Created balanced pseudo-labeled pool with {len(df_balanced)} samples. Saved to {output_csv}")

if __name__ == "__main__":
    main()
