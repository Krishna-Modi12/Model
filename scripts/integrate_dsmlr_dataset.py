import os
import glob
import json
import hashlib
import shutil
import random
from collections import defaultdict
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import tqdm


# Ensure project root in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.config import FACE_SHAPES, get_config_dict
from src.data.dataset import create_dataloaders
import torchvision.transforms as T
import torch.nn.functional as F

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def main():
    print("="*50)
    print("STEP 2: Auditing new data (faceshape_extra/)")
    print("="*50)
    
    extra_dir = os.path.join(BASE_DIR, "faceshape_extra")
    corrupt_dir = os.path.join(extra_dir, "corrupted")
    os.makedirs(corrupt_dir, exist_ok=True)
    
    # Valid class names mapping: lowercase suffix -> proper name
    valid_classes = {c.lower(): c for c in FACE_SHAPES}
    
    tally = defaultdict(lambda: {"rgb": 0, "gray": 0, "corrupt": 0})
    mismatched_folders = set()
    
    valid_images = [] # list of (path, proper_class_name)
    
    for root, dirs, files in os.walk(extra_dir):
        if "corrupted" in root: continue
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            
            parent_folder = os.path.basename(root)
            lower_parent = parent_folder.lower()
            
            if lower_parent not in valid_classes:
                continue # ignore files not in a class-like folder
                
            proper_class = valid_classes[lower_parent]
            if parent_folder != proper_class:
                mismatched_folders.add((root, os.path.join(os.path.dirname(root), proper_class)))
                
            path = os.path.join(root, file)
            
            # Check Image
            try:
                with Image.open(path) as img:
                    img.verify() # verify integrity
                
                with Image.open(path) as img:
                    w, h = img.size
                    if w == 0 or h == 0:
                        raise ValueError("Zero dimensions")
                    
                    mode = img.mode
                    if mode == 'L':
                        tally[proper_class]["gray"] += 1
                    else:
                        tally[proper_class]["rgb"] += 1
                        
                valid_images.append((path, proper_class))
            except Exception as e:
                tally[proper_class]["corrupt"] += 1
                try:
                    shutil.move(path, os.path.join(corrupt_dir, os.path.basename(path)))
                except:
                    pass

    print(f"{'Class':<10} | {'Total':<8} | {'Corrupt':<10} | {'Greyscale %'}")
    print("-" * 50)
    for c in FACE_SHAPES:
        stats = tally[c]
        total = stats['rgb'] + stats['gray']
        gray_pct = (stats['gray'] / total * 100) if total > 0 else 0
        warn = " <-- WARNING (>20%)" if gray_pct > 20 else ""
        print(f"{c:<10} | {total:<8} | {stats['corrupt']:<10} | {gray_pct:.1f}%{warn}")

    print("\n" + "="*50)
    print("STEP 3: Checking for class name mismatches")
    print("="*50)
    if not mismatched_folders:
        print("No folder casing mismatches found. All matched FACE_SHAPES exactly.")
    else:
        for old, new in mismatched_folders:
            print(f"Mismatch found: {old} -> should be {new} (Documented, continuing dynamically)")

    print("\n" + "="*50)
    print("STEP 4: Check for metadata manifest")
    print("="*50)
    manifest_path = os.path.join(BASE_DIR, "data", "processed", "annotations.json")
    manifest_exists = os.path.exists(manifest_path)
    if manifest_exists:
        print(f"Manifest found: {manifest_path}")
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        print(f"Total existing entries in manifest: {len(manifest_data)}")
    else:
        print("No annotations.json found!")
        manifest_data = []

    print("\n" + "="*50)
    print("STEP 5: Deduplicate using MD5 hashing")
    print("="*50)
    
    raw_data_dir = os.path.join(BASE_DIR, "data", "raw")
    existing_hashes = set()
    print("Hashing existing raw data...")
    for file in glob.glob(os.path.join(raw_data_dir, "**", "*.*"), recursive=True):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            h = get_md5(file)
            if h: existing_hashes.add(h)
    
    print(f"Indexed {len(existing_hashes)} existing files.")
    
    skipped_duplicates = defaultdict(int)
    unique_new_images = [] # (path, class, hash)
    
    for path, cls in tqdm.tqdm(valid_images, desc="Deduplicating"):
        h = get_md5(path)
        if h in existing_hashes:
            skipped_duplicates[cls] += 1
        else:
            unique_new_images.append((path, cls, h))
            existing_hashes.add(h) # prevent duplicates within the new dataset itself
            
    print("\nDuplicates skipped per class:")
    for c in FACE_SHAPES:
        print(f"  {c}: {skipped_duplicates[c]}")

    print("\n" + "="*50)
    print("STEP 6: Copy and normalize images")
    print("="*50)
    
    # Determine destination dir -> Let's use data/raw/face_shape_extra to not contaminate explicitly, or data/raw/face_shape
    dest_base = os.path.join(raw_data_dir, "face_shape")
    copied_counts = defaultdict(int)
    
    new_manifest_entries = []
    
    for src_path, cls, h in tqdm.tqdm(unique_new_images, desc="Copying images"):
        ext = os.path.splitext(src_path)[1].lower()
        if ext == '.jpeg': ext = '.jpg'
        
        new_filename = f"ext_{h[:12]}{ext}"
        dest_folder = os.path.join(dest_base, cls)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, new_filename)
        
        shutil.copy2(src_path, dest_path)
        copied_counts[cls] += 1
        
        # Prepare step 7
        new_manifest_entries.append({
            "image_path": dest_path,
            "shape_label": FACE_SHAPES.index(cls),
            "eye_label": 0,
            "nose_label": 0,
            "lip_label": 0,
            "ita_value": 0.0,
            "fitzpatrick": "1",
            "monk_label": 0,
            "source": "dsmlr/faceshape"
        })
        
    print("\nCopied valid images per class:")
    total_skipped = 0
    for c in FACE_SHAPES:
        skip_count = skipped_duplicates[c] + tally[c]["corrupt"]
        total_skipped += skip_count
        print(f"  {c}: copied {copied_counts[c]} | skipped {skip_count}")

    print("\n" + "="*50)
    print("STEP 7: Update the metadata manifest")
    print("="*50)
    if manifest_exists and new_manifest_entries:
        manifest_data.extend(new_manifest_entries)
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=4)
        print(f"Appended {len(new_manifest_entries)} new rows to {manifest_path}")
        print(f"New total manifest length: {len(manifest_data)}")
        
    print("\n" + "="*50)
    print("STEP 8: Regenerate dataset splits")
    print("="*50)
    
    # In dataset.py, splits are generated implicitly by sklearn train_test_split when create_dataloaders is called.
    # We will instantiate them and print the sizes.
    config = get_config_dict()
    loaders = create_dataloaders(config)
    
    # Count classes in splits
    def count_split(dset):
        counts = {i: 0 for i in range(len(FACE_SHAPES))}
        for ann in dset.annotations:
            counts[ann.get("shape_label", 0)] += 1
        return counts

    train_c = count_split(loaders["train"].dataset)
    val_c   = count_split(loaders["val"].dataset)
    test_c  = count_split(loaders["test"].dataset)
    
    print("\nNEW DATASET SPLIT COUNTS PER CLASS:")
    print(f"{'Class':<10} | {'Train':<7} | {'Val':<7} | {'Test':<7}")
    print("-" * 45)
    for i, c in enumerate(FACE_SHAPES):
        print(f"{c:<10} | {train_c[i]:<7} | {val_c[i]:<7} | {test_c[i]:<7}")
        
    print("\n" + "="*50)
    print("STEP 9: Dry-run validation (Do not start training)")
    print("="*50)
    
    try:
        batch = next(iter(loaders["train"]))
        images = batch["images"]
        # In multi-task setup this might be a dictionary depending on your dataset implementation.
        # Fallback to key checks if it's a dict
        if isinstance(batch, dict) and "shape_labels" in batch:
            labels = batch["shape_labels"]
        elif isinstance(batch, dict) and "face_shape" in batch:
            labels = batch["face_shape"]
        else:
            labels = batch[1] # standard tuple output
            
        print(f"Successfully loaded a batch!")
        print(f"Batch shape: {images.shape}")
        
        # Save a 2x2 grid of distinct classes if possible
        uniq_labels, uniq_idx = np.unique(labels.numpy(), return_index=True)
        sample_indices = uniq_idx[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(6,6))
        
        for i, ax in enumerate(axes.flatten()):
            if i < len(sample_indices):
                idx = sample_indices[i]
                img_tsr = images[idx].clone()
                
                # unnormalize mapping (approx per config)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img_tsr = img_tsr * std + mean
                
                img_np = img_tsr.permute(1,2,0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                
                lbl_name = FACE_SHAPES[labels[idx].item()]
                ax.imshow(img_np)
                ax.set_title(lbl_name)
            ax.axis("off")
            
        plt.tight_layout()
        grid_path = os.path.join(BASE_DIR, "debug_sample_grid.png")
        plt.savefig(grid_path)
        print(f"Saved 2x2 sample grid grid to {grid_path}")
        
    except Exception as e:
        print(f"Error during validation dry-run: {e}")

if __name__ == "__main__":
    main()
