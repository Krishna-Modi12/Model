import os
import shutil
import hashlib
import csv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import imagehash

# Config
ORIGINAL_TEST_MANIFEST = Path('c:/Users/krish/OneDrive/Desktop/Model/original_test_manifest.csv')
NITEN19_DIR = Path('c:/Users/krish/OneDrive/Desktop/Model/data/raw/niten19/FaceShape Dataset/training_set')
CURATED_TRAIN_DIR = Path('c:/Users/krish/OneDrive/Desktop/Model/data/curated/face_shape/FaceShape Dataset/training_set')

# The 5 valid classes
VALID_CLASSES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

def get_image_hashes(img_path: Path):
    try:
        md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
        with Image.open(img_path) as img:
            phash = str(imagehash.phash(img))
        return md5, phash
    except Exception:
        return None, None

def load_test_manifest():
    test_md5s = set()
    with open(ORIGINAL_TEST_MANIFEST, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_md5s.add(row['md5'])
    return test_md5s

def main():
    print("Loading test set manifest...")
    test_md5s = load_test_manifest()
    print(f"Loaded {len(test_md5s)} test set MD5 hashes.")
    
    # We will also compute pHashes for curated training set to avoid duplicates within train set
    print("Hashing existing curated training images...")
    existing_train_md5s = set()
    existing_train_phashes = set()
    
    for cls in VALID_CLASSES:
        cls_dir = CURATED_TRAIN_DIR / cls
        if not cls_dir.exists(): continue
        for img_name in tqdm(os.listdir(cls_dir), desc=f"Hashing curated {cls}"):
            img_path = cls_dir / img_name
            md5, phash = get_image_hashes(img_path)
            if md5:
                existing_train_md5s.add(md5)
                # Store phash to avoid near-duplicates
                existing_train_phashes.add(phash)
                
    print(f"Loaded {len(existing_train_md5s)} existing train MD5 hashes.")
    
    # Process Niten19
    added_count = 0
    duplicate_test_count = 0
    duplicate_train_count = 0
    error_count = 0
    
    for cls in VALID_CLASSES:
        src_dir = NITEN19_DIR / cls
        dst_dir = CURATED_TRAIN_DIR / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_dir.exists():
            print(f"Warning: {src_dir} (Niten19) does not exist.")
            continue
            
        img_files = os.listdir(src_dir)
        for img_name in tqdm(img_files, desc=f"Merging Niten19 {cls}"):
            img_path = src_dir / img_name
            # Skip non-images
            if not img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']: continue
            
            md5, phash = get_image_hashes(img_path)
            if not md5:
                error_count += 1
                continue
                
            # Check collisions
            if md5 in test_md5s:
                duplicate_test_count += 1
                continue
                
            # pHash gives us protection against slightly modified duplicates
            if md5 in existing_train_md5s or phash in existing_train_phashes:
                duplicate_train_count += 1
                continue
                
            # Safely copy to target with a deterministic new name
            new_name = f"niten19_{md5[:8]}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dst_dir / new_name)
            
            # Register to prevent intra-niten19 duplicates
            existing_train_md5s.add(md5)
            existing_train_phashes.add(phash)
            added_count += 1
            
    print("\n--- Merge Summary ---")
    print(f"Successfully added images: {added_count}")
    print(f"Skipped (Test Set Collision): {duplicate_test_count}")
    print(f"Skipped (Already in Train Set): {duplicate_train_count}")
    print(f"Errors (Corrupt image): {error_count}")
    print("---------------------")

if __name__ == "__main__":
    main()
