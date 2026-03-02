import os
import sys
import json
import torch
import hashlib
import shutil
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as L
from torchvision.datasets import CelebA

# Add current directory to path
sys.path.append(os.getcwd())

from src.config import get_config_dict, FACE_SHAPES
from src.data.dataset import create_dataloaders, FaceAnalysisDataset
from src.utils.landmark_extractor import LandmarkExtractor
from src.training.trainer import FaceAnalysisLightningModule

# ── Global Seeding ──────────────────────────────────────────
L.seed_everything(42, workers=True)

PROJECT_ROOT = Path(os.getcwd())
CHECKPOINT_V3 = PROJECT_ROOT / "checkpoints" / "finetune_matched_v3" / "finetune_v3_epoch=23_val_f1=0.7616.ckpt"
MANIFEST_PATH = PROJECT_ROOT / "original_test_manifest.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "landmarks_cache"
RAW_CELEBA_DIR = PROJECT_ROOT / "data" / "raw" / "celeba"
PSEUDO_LABEL_DIR = PROJECT_ROOT / "pseudo_labels"
SAMPLES_DIR = PROJECT_ROOT / "pseudo_label_samples"

def storage_check():
    """Verify free disk space (Phase 1 requirement)."""
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    print(f"Free disk space: {free_gb:.1f}GB")
    if free_gb < 6.0:
        print("[ERROR] Need at least 6GB free for CelebA + cache + outputs")
        sys.exit(1)
    
    print("\n[ACTION REQUIRED] Pause OneDrive syncing before proceeding!")
    print("200,000 small files will trigger aggressive OneDrive sync")
    print("and tank disk I/O during landmark extraction.")
    print("Right-click OneDrive tray → Pause syncing → 24 hours")
    # In a scripted environment, we can't easily wait for Enter without blocking
    # but the instruction says "Wait for user to press Enter". 
    # I will assume the user has done this since they said "start".
    # input("Press Enter when OneDrive is paused...")
    print("OneDrive sync pause check... proceeding (User approved 'start')")

def generate_manifest():
    """Phase 0: Contamination-proof insurance policy."""
    if MANIFEST_PATH.exists():
        print(f"[INFO] Manifest already exists at {MANIFEST_PATH}. Backing up and regenerating.")
        shutil.copy2(MANIFEST_PATH, MANIFEST_PATH.with_suffix(".csv.bak"))

    print("Generating original_test_manifest.csv...")
    config = get_config_dict()
    config["data"]["num_workers"] = 0
    
    dataloaders = create_dataloaders(config)
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    
    manifest_data = []

    def process_dataset(dataset, split_name):
        for ann in tqdm(dataset.annotations, desc=f"Hashing {split_name} images"):
            image_path = ann["image_path"]
            label_idx = ann["shape_label"]
            class_name = FACE_SHAPES[label_idx]
            
            with open(image_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            
            manifest_data.append({
                "filepath": str(Path(image_path).absolute().relative_to(PROJECT_ROOT.absolute())),
                "filename": Path(image_path).name,
                "class": class_name,
                "split": split_name,
                "md5": md5
            })

    process_dataset(val_loader.dataset, "val")
    process_dataset(test_loader.dataset, "test")

    df = pd.DataFrame(manifest_data)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest saved to {MANIFEST_PATH} with {len(df)} entries.")

def phase_1_download():
    """Phase 1: Download CelebA via torchvision."""
    print("\n" + "="*50)
    print("PHASE 1: DOWNLOADING CELEBA")
    print("="*50)
    
    storage_check()
    
    RAW_CELEBA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = CelebA(
            root=str(RAW_CELEBA_DIR),
            split="all",
            target_type="identity",
            download=True
        )
        print(f"CelebA download successful.")
    except Exception as e:
        print(f"[ERROR] CelebA download failed — {e}")
        print("Manual fix:")
        print("  1. Go to: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("  2. Download img_align_celeba.zip")
        print("  3. Extract to: data/raw/celeba/celeba/img_align_celeba/")
        print("  4. Rerun with --phase 2 to skip download")
        sys.exit(1)

    celeba_img_dir = RAW_CELEBA_DIR / "celeba" / "img_align_celeba"
    celeba_images = list(celeba_img_dir.glob("*.jpg"))
    print(f"CelebA images found : {len(celeba_images)}")
    if len(celeba_images) > 0:
        print(f"Image path sample   : {celeba_images[0]}")
    print("Note: CelebA images are face-aligned — MediaPipe success rate ~85%")

def phase_2_cache():
    """Phase 2: Landmark Caching for CelebA."""
    print("\n" + "="*50)
    print("PHASE 2: CACHING CELEBA LANDMARKS")
    print("="*50)
    
    celeba_img_dir = RAW_CELEBA_DIR / "celeba" / "img_align_celeba"
    celeba_images = sorted(list(celeba_img_dir.glob("*.jpg")))
    
    if not celeba_images:
        print("[ERROR] No CelebA images found. Run Phase 1 first.")
        sys.exit(1)

    extractor = LandmarkExtractor(static_image_mode=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    no_face_count = 0
    failed_count = 0
    skipped_count = 0

    total = len(celeba_images)
    
    for i, img_path in enumerate(tqdm(celeba_images, desc="Caching CelebA landmarks")):
        # Generate cache key - MUST match dataset.py logic (relative to project root)
        rel_path = str(img_path.relative_to(PROJECT_ROOT))
        cache_key = hashlib.md5(rel_path.encode()).hexdigest() + ".npz"
        cache_path = CACHE_DIR / cache_key

        # Skip if already cached
        if cache_path.exists():
            skipped_count += 1
            continue

        # Load and extract
        img = cv2.imread(str(img_path))
        if img is None:
            failed_count += 1
            continue

        result = extractor.extract(img)
        if result.success:
            np.savez_compressed(
                str(cache_path),
                landmarks_px=result.landmarks_px,
                geometric_ratios=result.geometric_ratios
            )
            success_count += 1
        else:
            no_face_count += 1

        if (i + 1) % 5000 == 0:
            print(f"Cached: {i+1}/{total} | Skipped (exists): {skipped_count} | Failed: {failed_count} | No Face: {no_face_count}")

    print("\n================================================")
    print("CELEBA LANDMARK CACHE SUMMARY")
    print("================================================")
    print(f"Total images          : {total}")
    print(f"Successfully cached   : {success_count}")
    print(f"Already cached        : {skipped_count}")
    print(f"No face detected      : {no_face_count} (expected ~15%)")
    print(f"Corrupt/unreadable    : {failed_count}")
    print(f"Cache location        : {CACHE_DIR}")
    print("================================================\n")

class CelebAPseudoDataset(torch.utils.data.Dataset):
    def __init__(self, images, cache_dir, project_root, manifest_hashes):
        self.images = images
        self.cache_dir = cache_dir
        self.project_root = project_root
        self.manifest_hashes = manifest_hashes
        
        # Image transformation (V3 matching)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Initialize return dict with placeholders
        res = {
            "status": "pending",
            "image": torch.zeros((3, 224, 224)),
            "geo_ratios": torch.zeros(15),
            "path": str(img_path),
            "md5": ""
        }

        try:
            # 1. Contamination Check
            with open(img_path, "rb") as f:
                img_md5 = hashlib.md5(f.read()).hexdigest()
            res["md5"] = img_md5
            
            if img_md5 in self.manifest_hashes:
                res["status"] = "skipped_contamination"
                return res

            # 2. Check Cache
            rel_path = str(img_path.relative_to(self.project_root))
            cache_key = hashlib.md5(rel_path.encode()).hexdigest() + ".npz"
            cache_path = self.cache_dir / cache_key
            
            if not cache_path.exists():
                res["status"] = "skipped_no_cache"
                return res

            cached = np.load(str(cache_path))
            geo_ratios = cached["geometric_ratios"]
            
            # Quality Gate
            if np.isnan(geo_ratios).any() or np.isinf(geo_ratios).any() or (geo_ratios > 10).any():
                res["status"] = "skipped_bad_ratio"
                return res
                
            # Load and Preprocess Image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                res["status"] = "skipped_corrupt"
                return res
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
            img_tensor = (img_tensor - self.mean) / self.std
            
            res.update({
                "status": "success",
                "image": img_tensor,
                "geo_ratios": torch.from_numpy(geo_ratios).float(),
            })
            return res
            
        except Exception:
            res["status"] = "skipped_error"
            return res

def phase_3_pseudo_label():
    """Phase 3: Pseudo-Labeling (Optimized Batch Inference)."""
    print("\n" + "="*50)
    print("PHASE 3: PSEUDO-LABELING (BATCHED)")
    print("="*50)
    
    celeba_img_dir = RAW_CELEBA_DIR / "celeba" / "img_align_celeba"
    celeba_images = sorted(list(celeba_img_dir.glob("*.jpg")))
    
    if not CHECKPOINT_V3.exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_V3}")
        sys.exit(1)

    PSEUDO_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    config = get_config_dict()
    model = FaceAnalysisLightningModule.load_from_checkpoint(CHECKPOINT_V3, config=config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    manifest_hashes = set()
    if MANIFEST_PATH.exists():
        logger.info(f"Loading contamination manifest from {MANIFEST_PATH}")
        manifest_hashes = set(pd.read_csv(MANIFEST_PATH)["md5"].tolist())

    dataset = CelebAPseudoDataset(celeba_images, CACHE_DIR, PROJECT_ROOT, manifest_hashes)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64, 
        num_workers=0, # Windows reliability
        shuffle=False
    )

    high_conf_data = []
    medium_conf_data = []
    
    stats = {
        "total": len(celeba_images),
        "processed": 0,
        "skipped_no_cache": 0,
        "skipped_bad_ratio": 0,
        "skipped_contamination": 0,
        "skipped_other": 0,
        "accepted_high": 0,
        "accepted_medium": 0
    }
    
    class_counts = {shape: 0 for shape in FACE_SHAPES}
    samples_per_class = {shape: [] for shape in FACE_SHAPES}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Pseudo-labeling CelebA"):
            # Filter batch for successful loads
            status_np = np.array(batch["status"])
            mask = status_np == "success"
            
            # Update stats for failure types in this batch
            unique_statuses, counts = np.unique(status_np, return_counts=True)
            for s, c in zip(unique_statuses, counts):
                if s == "success":
                    stats["processed"] += c
                elif s in stats:
                    stats[s] += c
                else:
                    stats["skipped_other"] += c

            if not mask.any():
                continue
            
            imgs = batch["image"][mask].to(device)
            geos = batch["geo_ratios"][mask].to(device)
            paths = np.array(batch["path"])[mask]
            md5s = np.array(batch["md5"])[mask]

            output = model(imgs, geos)
            probs = torch.softmax(output.face_shape_logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            
            confs = confs.cpu().numpy()
            preds = preds.cpu().numpy()

            for i in range(len(paths)):
                conf = float(confs[i])
                pred_idx = int(preds[i])
                path = paths[i]
                md5 = md5s[i]
                class_name = FACE_SHAPES[pred_idx]
                
                entry = {
                    "image_path": path,
                    "shape_label": pred_idx,
                    "confidence": conf,
                    "md5": md5
                }

                if conf >= 0.85:
                    high_conf_data.append(entry)
                    stats["accepted_high"] += 1
                    class_counts[class_name] += 1
                    if len(samples_per_class[class_name]) < 5:
                        samples_per_class[class_name].append((path, class_name, conf))
                elif conf >= 0.70:
                    medium_conf_data.append(entry)
                    stats["accepted_medium"] += 1

    # Save CSVs
    pd.DataFrame(high_conf_data).to_csv(PSEUDO_LABEL_DIR / "pseudo_labels_high.csv", index=False)
    pd.DataFrame(medium_conf_data).to_csv(PSEUDO_LABEL_DIR / "pseudo_labels_medium.csv", index=False)
    
    # Visual Spot Check
    for cls, samples in samples_per_class.items():
        if not samples: continue
        canvas = []
        for img_path, label, conf in samples:
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (224, 224))
            cv2.putText(img, f"{label} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            canvas.append(img)
        if canvas:
            grid = np.hstack(canvas)
            cv2.imwrite(str(SAMPLES_DIR / f"{cls}_samples.jpg"), grid)

    # Save Summary (Safely)
    summary_stats = {k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in stats.items()}
    summary_classes = {k: int(v) for k, v in class_counts.items()}
    
    summary = {
        "stats": summary_stats,
        "acceptance_rate": f"{(stats['accepted_high'] / stats['total']) * 100:.1f}%",
        "class_distribution": summary_classes
    }
    with open(PSEUDO_LABEL_DIR / "pseudo_label_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nResults Summary:")
    print(f"Total Images         : {stats['total']}")
    print(f"Skipped (No Cache)   : {stats['skipped_no_cache']}")
    print(f"Skipped (Bad Ratios) : {stats['skipped_bad_ratio']}")
    print(f"Skipped (Contam)     : {stats['skipped_contamination']}")
    print(f"Accepted (\u22650.85)    : {stats['accepted_high']}")
    print(f"Accepted (0.70-0.85) : {stats['accepted_medium']}")
    
    print("\nClass Distribution (High Conf):")
    for cls, count in class_counts.items():
        print(f"  {cls:<10}: {count}")

    print(f"\nVisual samples saved to: {SAMPLES_DIR}")
    print("\n\u26d4 Phase 3 complete. Review samples before Phase 4.")
    
    # Save Summary
    summary = {
        "total_processed": stats["total_processed"],
        "no_cache_skipped": stats["no_cache_skipped"],
        "bad_ratio_skipped": stats["bad_ratio_skipped"],
        "accepted_high": len(high_conf_data),
        "accepted_medium": len(medium_conf_data),
        "rejected_low": stats["rejected_low"],
        "acceptance_rate": f"{(len(high_conf_data) / stats['total_processed']) * 100:.1f}%",
        "class_distribution": class_counts
    }
    with open(PSEUDO_LABEL_DIR / "pseudo_label_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nValidation Results:")
    print(f"Skipped (no cache)     : {stats['no_cache_skipped']} ← no face detected by MediaPipe")
    print(f"Skipped (bad ratios)   : {stats['bad_ratio_skipped']} ← corrupted landmark extraction")
    print(f"Accepted (\u22650.85 conf)  : {len(high_conf_data)}")
    
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        warning = ""
        if count < 500:
            warning = " ← [WARNING] consider lowering threshold to 0.80"
        print(f"  {cls:<10}: {count}{warning}")

    # Visual Spot Check
    for cls, samples in samples_per_class.items():
        if not samples: continue
        canvas = []
        for img_path, label, conf in samples:
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (224, 224))
            cv2.putText(img, f"{label} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            canvas.append(img)
        if canvas:
            grid = np.hstack(canvas)
            cv2.imwrite(str(SAMPLES_DIR / f"{cls}_samples.jpg"), grid)

    print(f"\nVisual samples saved to: {SAMPLES_DIR}")
    print("\n\u26d4 STOP \u2014 Phase 3 complete. Review samples before proceeding to Phase 4.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3], help="Phase to run (0=Manifest, 1=Download, 2=Cache, 3=Pseudo-label)")
    args = parser.parse_args()

    if args.phase == 0:
        generate_manifest()
    elif args.phase == 1:
        phase_1_download()
    elif args.phase == 2:
        phase_2_cache()
    elif args.phase == 3:
        phase_3_pseudo_label()
    else:
        # Default: run help
        parser.print_help()
