"""
src/data/dataset.py — Multi-task face analysis dataset
=======================================================
Provides the FaceAnalysisDataset for Phase 3 multi-task training and
evaluation.  The annotation JSON is produced by build_multitask_annotations.py
and has the following per-record schema:

  {
    "image_path":      "path/to/face.jpg",
    "shape_label":     0,          // 0-4 face shape class; -100 = unknown
    "split":           "train",    // optional split hint
    "attributes":      {           // null if no CelebA attributes
      "eye_narrow":      0,        // 0/1 binary
      "eye_big":         1,        // 0/1 binary
      "brow":            2,        // class index
      "lip":             1,        // class index
      "age":             0,        // 0/1 binary
      "gender":          1,        // 0/1 binary
      "landmark_ratios": [...]     // 15 floats (may duplicate geometric_ratios)
    },
    "geometric_ratios": [...],     // 15 landmark-derived geometric ratios
    "monk_scale":       3          // optional int 1-10; omit or null = missing
  }

Also exports ``extract_hsv_histogram_np`` for building the 48-dim HSV color
feature used by SkinTower.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from torch.utils.data import Dataset


# ── HSV histogram ──────────────────────────────────────────────────────────

def extract_hsv_histogram_np(image_rgb: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """
    Compute a normalised HSV colour histogram from an RGB image.

    Returns a 1-D float32 array of length ``3 * n_bins`` (default 48).
    Each channel histogram is independently L1-normalised so that the
    resulting vector is invariant to image brightness differences.

    Args:
        image_rgb: uint8 or float32 RGB image (any resolution).
        n_bins:    Number of histogram bins per channel (default 16).

    Returns:
        np.ndarray of shape (3 * n_bins,), dtype float32.
    """
    if image_rgb.dtype != np.uint8:
        image_rgb = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    ranges = [(0, 180), (0, 256), (0, 256)]  # H: 0-179, S: 0-255, V: 0-255
    histograms = []
    for ch, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist(
            [image_hsv], [ch], None, [n_bins], [lo, hi]
        ).flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        histograms.append(hist)

    return np.concatenate(histograms)  # shape: (3 * n_bins,)


# ── Augmentation pipelines ─────────────────────────────────────────────────

def get_train_transforms(image_size: int, cfg: Optional[dict] = None) -> A.Compose:
    """Return training augmentation pipeline.

    ``cfg`` is accepted for API compatibility but defaults are used when
    the key is absent, so passing ``{"training": {}}`` is safe.
    """
    cfg = cfg or {}
    return A.Compose([
        A.HorizontalFlip(p=cfg.get("horizontal_flip_prob", 0.5)),
        A.Rotate(limit=cfg.get("rotation_limit", 10), p=0.40),
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.2),
            contrast_limit=cfg.get("contrast_limit", 0.2),
            p=0.50,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=int(cfg.get("saturation_limit", 0.2) * 100),
            val_shift_limit=10,
            p=0.30,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=cfg.get("blur_prob", 0.2)),
        A.GaussNoise(var_limit=(10, 50), p=cfg.get("noise_prob", 0.2)),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=1, p=cfg.get("cutout_prob", 0.1)),
        A.ImageCompression(quality_lower=50, quality_upper=95,
                           p=cfg.get("jpeg_prob", 0.1)),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int) -> A.Compose:
    """Return validation / test augmentation pipeline (no random augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────

_MISSING_LABEL = -100   # sentinel value for missing class labels


class FaceAnalysisDataset(Dataset):
    """
    Multi-task face analysis dataset for Phase 3 training and evaluation.

    Reads a JSON annotations file produced by ``build_multitask_annotations.py``
    and returns per-sample dicts with all available supervision signals.

    Batch keys
    ----------
    Required:
      images           (C, H, W) float tensor — normalised RGB
      geometric_ratios (15,)     float tensor — landmark-derived ratios
      shape_labels     ()        long  tensor — face shape class (0-4), or -100

    Optional (present when ``attributes`` is not null in the annotation):
      has_attributes   ()   bool  tensor
      eye_narrow       ()   long  tensor — 0/1
      eye_big          ()   long  tensor — 0/1
      brow             ()   long  tensor — brow type class index
      lip              ()   long  tensor — lip shape class index
      age              ()   long  tensor — 0/1
      gender           ()   long  tensor — 0/1
      landmark_ratios  (15,) float tensor — may duplicate geometric_ratios

    Optional (present when ``monk_scale`` is in the annotation):
      monk_labels      ()   long  tensor — Monk scale 0-9 (raw value minus 1)
    """

    def __init__(
        self,
        annotations_path: str,
        image_size: int = 224,
        transforms: Optional[A.Compose] = None,
        landmarks_cache_dir: str = "data/landmarks_cache",
        indices: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
    ):
        with open(annotations_path) as f:
            all_anns = json.load(f)

        if indices is not None:
            self.annotations = [all_anns[i] for i in indices]
        else:
            self.annotations = all_anns

        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]

        self.image_size = image_size
        self.transforms = transforms
        self.cache_dir = Path(landmarks_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        n_with_attrs = sum(
            1 for a in self.annotations if a.get("attributes") is not None
        )
        logger.info(
            f"FaceAnalysisDataset: {len(self.annotations)} samples | "
            f"with_attributes={n_with_attrs} | "
            f"transforms={'YES' if transforms else 'NO'}"
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        image_path = ann["image_path"]

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Cannot load image: {image_path}")
            return self._zero_sample(ann)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image_tensor = self.transforms(image=image_rgb)["image"]
        else:
            image_tensor = torch.from_numpy(
                image_rgb.transpose(2, 0, 1)
            ).float() / 255.0

        geo = self._load_geometric_ratios(ann, image_path, image)

        shape_label = int(ann.get("shape_label", _MISSING_LABEL))
        if shape_label < 0:
            shape_label = _MISSING_LABEL

        sample: dict = {
            "images":           image_tensor,
            "geometric_ratios": torch.tensor(geo, dtype=torch.float32),
            "shape_labels":     torch.tensor(shape_label, dtype=torch.long),
        }

        attrs = ann.get("attributes")
        if attrs is not None:
            sample["has_attributes"] = torch.tensor(True)
            sample["eye_narrow"]     = torch.tensor(int(attrs["eye_narrow"]),  dtype=torch.long)
            sample["eye_big"]        = torch.tensor(int(attrs["eye_big"]),     dtype=torch.long)
            sample["brow"]           = torch.tensor(int(attrs["brow"]),        dtype=torch.long)
            sample["lip"]            = torch.tensor(int(attrs["lip"]),         dtype=torch.long)
            sample["age"]            = torch.tensor(int(attrs["age"]),         dtype=torch.long)
            sample["gender"]         = torch.tensor(int(attrs["gender"]),      dtype=torch.long)
            lmk = attrs.get("landmark_ratios", geo)
            sample["landmark_ratios"] = torch.tensor(lmk, dtype=torch.float32)
        else:
            sample["has_attributes"] = torch.tensor(False)

        monk = ann.get("monk_scale")
        if monk is not None and monk > 0:
            sample["monk_labels"] = torch.tensor(int(monk) - 1, dtype=torch.long)

        return sample

    def _load_geometric_ratios(
        self,
        ann: dict,
        image_path: str,
        image: np.ndarray,
    ) -> np.ndarray:
        """Return pre-computed geometric ratios from the annotation, or zeros."""
        geo = ann.get("geometric_ratios")
        if geo is not None:
            arr = np.asarray(geo, dtype=np.float32)
            if arr.shape == (15,):
                return arr

        cache_key  = image_path.replace("/", "_").replace("\\", "_") + ".npy"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            return np.load(str(cache_path))

        return np.zeros(15, dtype=np.float32)

    def _zero_sample(self, ann: dict) -> dict:
        """Return a zero-filled sample for images that cannot be loaded."""
        shape_label = int(ann.get("shape_label", _MISSING_LABEL))
        if shape_label < 0:
            shape_label = _MISSING_LABEL
        return {
            "images":           torch.zeros(3, self.image_size, self.image_size),
            "geometric_ratios": torch.zeros(15),
            "shape_labels":     torch.tensor(shape_label, dtype=torch.long),
            "has_attributes":   torch.tensor(False),
        }
