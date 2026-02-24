import argparse
import pandas as pd
import os
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def merge_csvs(auto_csv, manual_csv, output_csv):
    logger.info(f"Loading auto-labels from {auto_csv}")
    try:
        df_auto = pd.read_csv(auto_csv)
    except FileNotFoundError:
        logger.error(f"Auto-labels CSV not found: {auto_csv}")
        return

    logger.info(f"Loading manual annotations from {manual_csv}")
    try:
        df_manual = pd.read_csv(manual_csv)
    except FileNotFoundError:
        logger.error(f"Manual annotations CSV not found: {manual_csv}")
        return

    # To merge safely, ensure both dataframes have 'image_path' to join on.
    # Label Studio exports might have a different column name for the image URL/path (e.g. 'image')
    # If the user's manual CSV uses a different column, they will need to map it before merging.
    if 'image_path' not in df_manual.columns:
        logger.warning("Column 'image_path' not found in manual CSV. Attempting to fall back to 'image'.")
        if 'image' in df_manual.columns:
            df_manual = df_manual.rename(columns={'image': 'image_path'})
        else:
            logger.error("Could not find a valid image path column in manual CSV to merge on.")
            return

    logger.info("Merging datasets on 'image_path'...")
    # Merge inner so we only get rows that have BOTH auto labels and manual labels.
    df_merged = pd.merge(df_manual, df_auto, on="image_path", how="inner")
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    logger.info(f"Successfully merged {len(df_merged)} annotated rows to {output_csv}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge manual Label Studio CSV with Auto-Labels CSV")
    parser.add_argument("--auto", type=str, default=os.path.join(PROCESSED_DATA_DIR, "skin_tone_auto.csv"),
                        help="Path to auto-generated skin tone CSV")
    parser.add_argument("--manual", type=str, required=True,
                        help="Path to manual annotations CSV exported from Label Studio")
    parser.add_argument("--output", type=str, default=os.path.join(PROCESSED_DATA_DIR, "train.csv"),
                        help="Path to save the final merged training CSV")
    
    args = parser.parse_args()
    merge_csvs(args.auto, args.manual, args.output)
