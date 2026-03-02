"""
scripts/build_fitzpatrick_annotations.py
======================================
Builds the clinical Fitzpatrick skin dataset pipeline.
Downloads images if missing, maps Fitzpatrick scale (I-VI) 
to our Monk Scale tensor format (1-10 -> 0-9 index), and appends
to our multi-task training dataset.
"""

import os
import sys
import json
import csv
import logging
import urllib.request
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FitzpatrickBuilder")

FITZPATRICK_CSV = PROJECT_ROOT / "fitzpatrick17k" / "fitzpatrick17k.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "fitzpatrick17k" / "images"
OUTPUT_JSON = PROJECT_ROOT / "data" / "processed" / "annotations_multitask_balanced.json"

def map_fitz_to_monk(fitz_val: int) -> int:
    """
    Clinically maps Fitzpatrick I-VI to approximate Monk 1-10.
    1-2 (Light) -> Monk 2
    3-4 (Medium) -> Monk 5
    5-6 (Dark) -> Monk 8
    (0-indexed for CrossEntropyLoss)
    """
    if fitz_val in [1, 2]:
        return 1  # Monk 2 (index 1)
    elif fitz_val in [3, 4]:
        return 4  # Monk 5 (index 4)
    elif fitz_val in [5, 6]:
        return 7  # Monk 8 (index 7)
    return -100

def download_image(url: str, save_path: Path) -> bool:
    if save_path.exists():
        return True
    try:
        response = requests.get(url, timeout=5, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        return False

def run():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not FITZPATRICK_CSV.exists():
        logger.error(f"Cannot find CSV at {FITZPATRICK_CSV}.")
        return

    # Load multi-task balanced dataset
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r") as f:
            master_data = json.load(f)
    else:
        logger.warning(f"Master JSON missing at {OUTPUT_JSON}. Creating new.")
        master_data = []

    logger.info(f"Loaded {len(master_data)} existing annotations.")
    
    # Parse CSV
    rows_to_process = []
    with open(FITZPATRICK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_to_process.append(row)

    logger.info(f"Found {len(rows_to_process)} entries in Fitzpatrick17k CSV.")
    
    # Check what is already processed to avoid duplicates
    existing_paths = {d["image_path"] for d in master_data}
    
    # Process batch
    added_count = 0
    download_tasks = []
    
    # Phase 1: Dispatch downloads in parallel (limit to a subset to avoid excessive time/bandwidth for now)
    SUBSET_LIMIT = 1000  # We grab 1000 clinical images to integrate effectively without hanging forever
    rows_to_process = [r for r in rows_to_process if r.get('fitzpatrick_scale', '').isdigit()][:SUBSET_LIMIT]
    
    logger.info(f"Attempting batch download/verification of {len(rows_to_process)} images...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {}
        for row in rows_to_process:
            md5hash = row["md5hash"]
            url = row["url"]
            img_filename = f"{md5hash}.jpg"
            save_path = IMAGE_DIR / img_filename
            rel_path = f"data/raw/fitzpatrick17k/images/{img_filename}"
            
            if rel_path in existing_paths:
                continue
                
            fitz_val = int(row["fitzpatrick_scale"])
            monk_index = map_fitz_to_monk(fitz_val)
            
            future = executor.submit(download_image, url, save_path)
            futures[future] = (row, save_path, rel_path, monk_index)
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Clinical Images"):
            row, save_path, rel_path, monk_index = futures[future]
            success = future.result()
            
            if success:
                # Add to master json
                master_data.append({
                    "image_path": rel_path,
                    "shape_label": -100, # Missing face shape
                    "monk_label": monk_index,
                    "split": "train", # Mix into training pool
                    "attributes": {
                        "eye_narrow": -1,
                        "eye_big": -1,
                        "brow": -1,
                        "lip": -1,
                        "age": -1,
                        "gender": -1,
                        "landmark_ratios": None
                    }
                })
                added_count += 1

    if added_count > 0:
        logger.info(f"Writing {added_count} new clinical images to dataset...")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(master_data, f, indent=4)
        logger.info(f"Successfully integrated Fitzpatrick17k. Total images: {len(master_data)}")
    else:
        logger.info("No new images were successfully downloaded or mapped.")

if __name__ == "__main__":
    run()
