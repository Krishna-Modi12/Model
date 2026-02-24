import os
import json
import shutil
from src.config import FACE_SHAPES

# 1. Paths
external_dir = "data/external_faceshape/published_dataset"
annotations_path = "data/processed/annotations.json"
output_image_dir = "data/processed/external_images"

# Create output dir for copied images
os.makedirs(output_image_dir, exist_ok=True)

# 2. Class Mapping
# External folder names are lowercase versions of FACE_SHAPES
label_map = {shape.lower(): i for i, shape in enumerate(FACE_SHAPES)}
print("Label Map:", label_map)

# 3. Load existing annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

print(f"Original annotations count: {len(annotations)}")
original_count = len(annotations)

# Keep track of existing filenames to prevent exact name duplicates
# (Though external images will be in a different folder, it's good practice)
existing_paths = set(ann["image_path"] for ann in annotations)

# 4. Integrate external images
new_annotations = []
for folder_name in os.listdir(external_dir):
    folder_path = os.path.join(external_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
        
    class_name = folder_name.lower()
    if class_name not in label_map:
        print(f"Skipping unknown class folder: {folder_name}")
        continue
        
    class_id = label_map[class_name]
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        src_path = os.path.join(folder_path, filename)
        
        # We prefix external images to avoid naming collisions
        dest_filename = f"ext_{folder_name}_{filename}"
        dest_path = os.path.join(output_image_dir, dest_filename).replace("\\", "/") # standardize slashes
        
        # Copy file
        shutil.copy2(src_path, dest_path)
        
        # Create annotation entry
        ann = {
            "image_path": dest_path,
            "shape_label": class_id
        }
        
        # Ensure we don't add duplicates (based on the new path)
        if dest_path not in existing_paths:
            new_annotations.append(ann)
            existing_paths.add(dest_path)

print(f"Added {len(new_annotations)} new external images.")

# Append to annotations
annotations.extend(new_annotations)

# 5. Save back
print("Saving merged annotations.json...")
with open(annotations_path, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"New total annotations count: {len(annotations)}")

# 6. Calculate new training distribution
# We need to simulate the train split from dataset.py to know the new distribution
# dataset.py uses seed=42, original_count=5755 (or whatever it actually was)
val_split = 0.15
test_split = 0.1
n_val = int(original_count * val_split)
n_test = int(original_count * test_split)
n_train_original = original_count - n_val - n_test

# The new images are purely appended to train_idx
new_train_count = n_train_original + len(new_annotations)

train_shape_counts = {shape: 0 for shape in FACE_SHAPES}

import numpy as np
rng = np.random.default_rng(42)
indices = rng.permutation(original_count).tolist()
train_idx = indices[:n_train_original]
# Append the new indices
train_idx.extend(list(range(original_count, len(annotations))))

for idx in train_idx:
    shape_id = annotations[idx]["shape_label"]
    train_shape_counts[FACE_SHAPES[shape_id]] += 1

print("\n--- NEW TRAINING SET CLASS DISTRIBUTION ---")
for shape, count in train_shape_counts.items():
    print(f"{shape}: {count}")
