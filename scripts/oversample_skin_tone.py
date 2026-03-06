import json
from pathlib import Path
from collections import Counter
import random

def oversample():
    root = Path(".")
    ann_path = root / "data" / "processed" / "annotations_multitask_balanced.json"
    out_path = root / "data" / "processed" / "annotations_multitask_final.json"
    
    if not ann_path.exists():
        print(f"Error: {ann_path} not found")
        return

    with open(ann_path, "r") as f:
        data = json.load(f)
        
    print(f"Initial total samples: {len(data)}")
    
    # Split out skin vs others
    skin_data = [d for d in data if d.get("monk_label") is not None and d["monk_label"] != -100]
    other_data = [d for d in data if d.get("monk_label") is None or d["monk_label"] == -100]
    
    counts = Counter([d["monk_label"] for d in skin_data])
    print(f"Skin distribution: {counts}")
    
    if not skin_data:
        print("No skin data to oversample!")
        return

    # Targeting ~5000 samples per skin class to give it enough "voice" vs CelebA (10k)
    target_per_class = 5000
    
    balanced_skin = []
    for label in [0, 1, 2]:
        class_samples = [d for d in skin_data if d["monk_label"] == label]
        if not class_samples:
            continue
            
        num_existing = len(class_samples)
        multiplier = (target_per_class // num_existing) + 1
        
        # Multiply samples
        multiplied = (class_samples * multiplier)[:target_per_class]
        balanced_skin.extend(multiplied)
        print(f"  Class {label}: {num_existing} -> {len(multiplied)} (x{multiplier})")
        
    final_data = other_data + balanced_skin
    
    # Final shuffle
    random.seed(42)
    random.shuffle(final_data)
    
    print(f"Final total samples: {len(final_data)} (Skin: {len(balanced_skin)}, Other: {len(other_data)})")
    
    with open(out_path, "w") as f:
        json.dump(final_data, f, indent=4)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    oversample()
