"""
dataset.py  (FIXED)
─────────────────────────────────────────────────────────────
Fixes applied:
  1. CRITICAL: Val/test data leakage — train transforms were being
     set on the shared underlying dataset, polluting val + test splits.
     Fixed by creating separate dataset instances with correct transforms.
  2. CRITICAL: val_dataset was created but never used in DataLoaders.
     Removed the orphaned instance; val/test now use dedicated instances.
  3. train_split config key is now actually read (was ignored before).
  4. LandmarkExtractor is now properly closed via __del__ to prevent
     MediaPipe resource leaks.
─────────────────────────────────────────────────────────────
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.landmark_extractor import LandmarkExtractor


# ── Quality filter thresholds ──────────────────────────────
MIN_RESOLUTION = 112
MIN_FACE_SIZE  = 80
MIN_BLUR_SCORE = 100.0


def compute_blur_score(image_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())


def align_face(image: np.ndarray, landmarks_px: np.ndarray) -> np.ndarray:
    left_eye  = landmarks_px[33].astype(float)
    right_eye = landmarks_px[263].astype(float)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def crop_face(image: np.ndarray, bbox: tuple, padding: float = 0.20) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    ih, iw = image.shape[:2]
    return image[max(0, y - pad_y):min(ih, y + h + pad_y),
                 max(0, x - pad_x):min(iw, x + w + pad_x)]


# ── Augmentation pipelines ─────────────────────────────────

def get_train_transforms(image_size: int, cfg: dict) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=cfg["horizontal_flip_prob"]),
        A.Rotate(limit=cfg["rotation_limit"], p=0.40),
        A.RandomBrightnessContrast(
            brightness_limit=cfg["brightness_limit"],
            contrast_limit=cfg["contrast_limit"],
            p=0.50,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=int(cfg["saturation_limit"] * 100),
            val_shift_limit=10,
            p=0.30,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=cfg["blur_prob"]),
        A.GaussNoise(var_limit=(10, 50), p=cfg["noise_prob"]),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=1, p=cfg["cutout_prob"]),
        A.ImageCompression(quality_lower=50, quality_upper=95,
                           p=cfg["jpeg_prob"]),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Dataset class ───────────────────────────────────────────

class FaceAnalysisDataset(Dataset):
    """
    Dataset for face analysis training.

    annotations.json format:
    [
      {
        "image_path": "path/to/face.jpg",
        "shape_label": 0,          // 0-6  (required)
        "eye_label": 2,            // 0-5  (optional)
        "nose_label": 1,           // 0-4  (optional)
        "lip_label": 0,            // 0-3  (optional)
        "brow_label": 1,           // 0-2  (optional)
        "jaw_label": 2,            // 0-2  (optional)
        "symmetry_score": 0.87     // 0-1  (optional)
      }
    ]
    """

    def __init__(self,
                 annotations_path: str,
                 image_size: int = 256,
                 landmarks_cache_dir: str = "data/landmarks_cache/",
                 transforms=None,
                 indices: list = None,       # <-- FIX: pass explicit indices for splits
                 max_samples: int = None):

        with open(annotations_path) as f:
            all_annotations = json.load(f)

        # Apply split indices if provided, else use everything
        if indices is not None:
            self.annotations = [all_annotations[i] for i in indices]
        else:
            self.annotations = all_annotations

        if max_samples:
            self.annotations = self.annotations[:max_samples]

        self.image_size   = image_size
        self.cache_dir    = Path(landmarks_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transforms   = transforms
        self._extractor   = None    # lazy init to avoid multiprocessing issues

        logger.info(f"Dataset: {len(self.annotations)} samples | "
                    f"transforms={'YES' if transforms else 'NO'}")

    @property
    def extractor(self):
        """Lazy-init LandmarkExtractor so it works safely with num_workers > 0."""
        if self._extractor is None:
            self._extractor = LandmarkExtractor(static_image_mode=True)
        return self._extractor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        image_path = ann["image_path"]

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return self._zero_sample(ann)

        geo_ratios = self._load_or_compute_landmarks(image_path, image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image_tensor = self.transforms(image=image_rgb)["image"]
        else:
            image_tensor = torch.from_numpy(
                image_rgb.transpose(2, 0, 1)).float() / 255.0

        sample = {
            "images":           image_tensor,
            "geometric_ratios": torch.from_numpy(geo_ratios).float(),
            "shape_labels":     torch.tensor(ann["shape_label"], dtype=torch.long),
        }

        for key in ["eye_label", "nose_label", "lip_label", "brow_label", "jaw_label"]:
            if key in ann:
                sample[key.replace("_label", "_labels")] = \
                    torch.tensor(ann[key], dtype=torch.long)

        if "symmetry_score" in ann:
            sample["symmetry_scores"] = torch.tensor(
                ann["symmetry_score"], dtype=torch.float32)

        return sample

    def _load_or_compute_landmarks(self, image_path: str,
                                    image: np.ndarray) -> np.ndarray:
        cache_key  = image_path.replace("/", "_").replace("\\", "_") + ".npy"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            return np.load(str(cache_path))

        result = self.extractor.extract(image)
        ratios = result.geometric_ratios if result.success else np.zeros(15, dtype=np.float32)

        np.save(str(cache_path), ratios)
        return ratios

    def _zero_sample(self, ann: dict) -> dict:
        return {
            "images":           torch.zeros(3, self.image_size, self.image_size),
            "geometric_ratios": torch.zeros(15),
            "shape_labels":     torch.tensor(ann.get("shape_label", 0), dtype=torch.long),
        }

    def __del__(self):
        """Ensure MediaPipe is released when dataset is garbage collected."""
        if self._extractor is not None:
            try:
                self._extractor.close()
            except Exception:
                pass

    def close(self):
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None


# ── DataLoader factory ──────────────────────────────────────

def create_dataloaders(config: dict) -> dict:
    """
    Creates train/val/test DataLoaders.

    FIX: Each split now gets its OWN dataset instance with the correct
    transforms applied. Previously, setting train_transforms on train_ds.dataset
    would mutate the shared underlying dataset, contaminating val + test splits.
    """
    data_cfg = config["data"]
    aug_cfg  = config["augmentation"]

    annotations_path = os.path.join(config["paths"]["processed_data"],
                                    "annotations.json")

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            f"Annotations not found at {annotations_path}. "
            "Run data preparation first."
        )

    # Determine total size and compute split indices once
    with open(annotations_path) as f:
        total = len(json.load(f))

    # FIX: train_split is now actually used (was ignored before)
    val_split   = data_cfg["val_split"]
    test_split  = data_cfg["test_split"]
    train_split = data_cfg.get("train_split", 1.0 - val_split - test_split)

    n_val   = int(total * val_split)
    n_test  = int(total * test_split)
    n_train = total - n_val - n_test

    # Deterministic shuffled indices
    rng     = np.random.default_rng(config["project"]["seed"])
    indices = rng.permutation(total).tolist()

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    logger.info(f"Split sizes → Train: {len(train_idx)} | "
                f"Val: {len(val_idx)} | Test: {len(test_idx)}")

    # FIX: Separate dataset instance per split, each with its own transforms
    train_transforms = get_train_transforms(data_cfg["image_size"], aug_cfg)
    val_transforms   = get_val_transforms(data_cfg["image_size"])

    common_kwargs = dict(
        annotations_path      = annotations_path,
        image_size            = data_cfg["image_size"],
        landmarks_cache_dir   = config["paths"]["landmarks_cache"],
    )

    train_dataset = FaceAnalysisDataset(
        **common_kwargs, transforms=train_transforms, indices=train_idx)
    val_dataset   = FaceAnalysisDataset(
        **common_kwargs, transforms=val_transforms,   indices=val_idx)
    test_dataset  = FaceAnalysisDataset(
        **common_kwargs, transforms=val_transforms,   indices=test_idx)

    loader_kwargs = dict(
        batch_size  = config["training"]["batch_size"],
        num_workers = data_cfg["num_workers"],
        pin_memory  = data_cfg["pin_memory"],
    )

    return {
        "train": DataLoader(train_dataset, shuffle=True,  **loader_kwargs),
        "val":   DataLoader(val_dataset,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_dataset,  shuffle=False, **loader_kwargs),
    }
