import torch
import cv2
import numpy as np
from pathlib import Path
from src.data.dataset import FaceAnalysisDataset, get_train_transforms
from src.utils.landmark_extractor import LandmarkExtractor
import json

def visualize_dataset():
    extractor = LandmarkExtractor(static_image_mode=True)
    with open("data/processed/annotations_multitask_balanced.json", "r") as f:
        data = json.load(f)
    
    # Filter for skin tone images
    skin_data = [d for d in data if d.get("monk_label") is not None and d["monk_label"] != -100]
    print(f"Total skin samples: {len(skin_data)}")
    
    ds = FaceAnalysisDataset(skin_data, transforms=get_train_transforms(224))
    
    out_dir = Path("debug_samples")
    out_dir.mkdir(exist_ok=True)
    
    for i in range(20):
        sample = ds[i]
        img = sample["images"].permute(1, 2, 0).numpy()
        # Denormalize (transforms use 0-1)
        img = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        lbl = sample["monk_labels"].item()
        # Add text
        cv2.putText(img_bgr, f"Label: {lbl}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(str(out_dir / f"skin_{i}_label{lbl}.jpg"), img_bgr)
        print(f"Saved debug_samples/skin_{i}_label{lbl}.jpg")

if __name__ == "__main__":
    visualize_dataset()
