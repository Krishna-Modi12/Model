"""
eval_skin_val.py
================
Evaluates the finalized model's skin tone prediction accuracy on the newly assigned validation set.
"""

import os
import sys
from pathlib import Path
import json
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from predict import load_model, predict_single, DEFAULT_CHECKPOINT
from src.utils.landmark_extractor import LandmarkExtractor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(DEFAULT_CHECKPOINT, device)
    extractor = LandmarkExtractor(static_image_mode=True)
    
    with open("data/processed/annotations_multitask_balanced.json", "r") as f:
        data = json.load(f)
        
    val_samples = [d for d in data if d.get("split") == "val" and d.get("monk_label") is not None]
    
    if not val_samples:
        print("No skin tone validation samples found.")
        return
        
    correct = 0
    total = len(val_samples)
    off_by_one = 0
    
    print(f"Evaluating {total} validation samples for Skin Tone...")
    
    for i, sample in enumerate(val_samples):
        img_path = sample["image_path"]
        true_label = sample["monk_label"] # 0 for Monk 1, 9 for Monk 10
        true_monk_string = f"Monk {true_label + 1}"
        
        try:
            results = predict_single(img_path, model, extractor, device)
            
            if results and "error" not in results[0]:
                pred = results[0].get("skin_tone", "")
                
                # Parse 'Monk X' to integer index matching true_label
                if pred.startswith("Monk "):
                    pred_int = int(pred.split(" ")[1]) - 1
                    
                    if pred_int == true_label:
                        correct += 1
                        off_by_one += 1
                    elif abs(pred_int - true_label) <= 1:
                        off_by_one += 1
        except Exception as e:
            pass
            
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{total}...")
            
    acc = correct / total
    near_acc = off_by_one / total
    print(f"\n✅ Final Skin Tone Evaluation:")
    print(f"Exact Match Accuracy  : {acc*100:.2f}%")
    print(f"Within 1 Shade (±1)   : {near_acc*100:.2f}%")
    print(f"Total Samples Tested  : {total}")

if __name__ == "__main__":
    main()
