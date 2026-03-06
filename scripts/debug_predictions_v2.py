import torch
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from src.models.face_analysis_model import FaceAnalysisModel
from src.data.dataset import get_val_transforms, extract_hsv_histogram_np
from src.utils.landmark_extractor import LandmarkExtractor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path("checkpoints/attributes_v2/last.ckpt")
    
    model = FaceAnalysisModel(backbone="efficientnet_b4", num_classes=5, pretrained=False)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("model.", "")] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    extractor = LandmarkExtractor(static_image_mode=True)
    with open("data/processed/annotations_multitask_balanced.json", "r") as f:
        data = json.load(f)
    
    # Validation split logic (same as training)
    import random
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.85)
    val_set = data[split_idx:]
    
    skin_val = [d for d in val_set if d.get("monk_label") is not None and d["monk_label"] != -100]
    print(f"Total skin samples in val: {len(skin_val)}")
    
    preds = []
    trues = []
    
    for d in tqdm(skin_val[:100]): # Check first 100
        img = cv2.imread(d["image_path"])
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = get_val_transforms(224)(image=img_rgb)["image"].unsqueeze(0).to(device)
        
        result = extractor.extract(img)
        # Match Dataset logic: allow monk_label images even if MediaPipe fails
        if not result.success:
            geo = torch.zeros(1, 15).to(device)
        else:
            geo = torch.tensor(result.geometric_ratios, dtype=torch.float32).unsqueeze(0).to(device)
            
        hsv_hist = torch.tensor(extract_hsv_histogram_np(img_rgb), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            full_features = model.backbone(tensor_img)
            
            # Print stats for first 3 samples
            if len(preds) < 3:
                print(f"\nSample {len(preds)+1}:")
                print(f"  Backbone features norm: {full_features.norm().item():.2f}")
                print(f"  HSV Features sum: {hsv_hist.sum().item():.2f}")
                print(f"  HSV Features max: {hsv_hist.max().item():.2f}")
                print(f"  True Label: {d['monk_label']}")
                
            out = model(tensor_img, geo, hsv_hist)
            pred = out.skin_tone_logits.argmax(dim=1).item()
            preds.append(pred)
            trues.append(d["monk_label"])
            
    print("\nSkin Tone Prediction Distribution (Batch 100):")
    print(Counter(preds))
    print("\nTrue Label Distribution (Batch 100):")
    print(Counter(trues))
    
    acc = (np.array(preds) == np.array(trues)).mean()
    print(f"\nAccuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
