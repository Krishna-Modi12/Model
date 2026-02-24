import cv2
import csv
import glob
import os
import sys
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.utils.landmark_extractor import LandmarkExtractor
from src.utils.skin_tone_analyzer import SkinToneAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURATED_DATA_DIR = os.path.join(BASE_DIR, "data", "curated")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
AUTO_CSV = os.path.join(PROCESSED_DATA_DIR, "skin_tone_auto.csv")

def run():
    extractor = LandmarkExtractor()
    analyzer  = SkinToneAnalyzer()
    rows = []

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(CURATED_DATA_DIR, "**", "*.jpg"), recursive=True)
    image_paths.extend(glob.glob(os.path.join(CURATED_DATA_DIR, "**", "*.png"), recursive=True))

    for path in tqdm(image_paths, desc="Auto-labeling Skin Tones"):
        img  = cv2.imread(path)
        if img is None:
            continue
        
        lm   = extractor.extract(img)
        if lm.success:
            tone = analyzer.analyze(lm.skin_pixels_lab)
            if tone:
                rows.append({
                    "image_path":  path,
                    "ita_value":   tone.ita_value,
                    "fitzpatrick": tone.fitzpatrick_type,
                    "monk":        tone.monk_scale,
                })

    if not rows:
        print("No faces found to auto-label.")
        return

    with open(AUTO_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done — {len(rows)} rows written to {AUTO_CSV}")

if __name__ == "__main__":
    run()
