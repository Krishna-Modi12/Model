# Codebase Analysis — Face Analysis AI V6

## 1. Project Purpose

This repository implements a **multi-task facial analysis system** that performs comprehensive face analysis from a single image. Using an EfficientNet-B4 backbone shared across 8+ prediction heads, the model predicts face shape, skin tone, age group, gender, eye shape, brow type, lip shape, and geometric landmarks in a single forward pass.

**Live demo:** https://huggingface.co/spaces/Krishna1205/face-analysis

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Deep learning | PyTorch 2.0, PyTorch Lightning |
| Backbone model | EfficientNet-B4 (via `timm`) |
| Face detection / landmarks | MediaPipe |
| Image processing | OpenCV, Albumentations |
| Web UI | Gradio |
| Inference optimization | ONNX Runtime (11.6ms CPU) |
| Experiment tracking | MLflow |
| Logging | Loguru |
| Metrics | TorchMetrics, scikit-learn |

---

## 3. Directory Structure

```
Model/
├── app.py                           # Gradio web demo (586 lines)
├── predict.py                       # CLI & Python API inference (549 lines)
├── train_multitask_v3.py            # Latest multi-task training script (319 lines)
├── train_attributes_v2.py           # Attribute-head-only training (376 lines)
├── eval_multitask_proper.py         # Full multi-task evaluation with TTA (385 lines)
├── build_multitask_annotations.py   # Annotation pipeline (408 lines)
├── export_onnx.py                   # PyTorch → ONNX export (91 lines)
├── original_test_manifest.csv       # Held-out test set manifest
├── COLAB_GUIDE.md                   # Google Colab training guide
│
├── src/                             # Core library
│   ├── config.py                    # All hyperparameters and class definitions
│   ├── dataset.py                   # Top-level dataset re-export (unused stub)
│   ├── landmark_extractor.py        # Top-level landmark extractor re-export
│   ├── skin_tone_analyzer.py        # HSV-based skin tone utilities
│   ├── trainer.py                   # Top-level trainer re-export
│   ├── models/
│   │   ├── face_analysis_model.py   # PRIMARY model: FaceAnalysisModel (471 lines)
│   │   └── multi_task_model.py      # LEGACY model: MultiTaskFaceModel (77 lines)
│   ├── training/
│   │   ├── trainer.py               # FaceAnalysisLightningModule (385 lines)
│   │   └── losses.py                # MultiTaskLoss for legacy model (62 lines)
│   └── utils/
│       ├── landmark_extractor.py    # MediaPipe wrapper + geometric ratio extraction
│       ├── geometric_math.py        # Pure-math geometric ratio computations
│       ├── ethics_guard.py          # Bias/ethics safeguards
│       ├── report_generator.py      # Human-readable result formatting
│       ├── setup_dashboard.py       # MLflow dashboard helpers
│       └── skin_tone_analyzer.py    # HSV color space skin tone analysis
│
├── scripts/                         # Data pipeline utilities
│   ├── build_fitzpatrick_annotations.py
│   ├── curate_data.py
│   ├── download_datasets.py / download_kaggle_datasets.py
│   ├── generate_algorithmic_labels.py / generate_auto_labels.py
│   ├── merge_csvs.py / merge_dataset.py / merge_kaggle_csvs.py
│   ├── oversample_skin_tone.py
│   ├── preprocess_landmarks.py
│   ├── sanity_check_splits.py
│   └── visualize_dataset_skin.py
│
├── face_analysis_fixed/             # Isolated copy of core src with hot-fixes
│   ├── verify_setup.py
│   └── src/
│       ├── dataset.py / face_shape_model.py
│       ├── landmark_extractor.py / trainer.py
│
├── hf_space/                        # HuggingFace Spaces deployment (empty)
├── archive/                         # Legacy / experimental scripts (not active)
│   └── train.py, train_multitask.py, train_multitask_v2.py, ...
│
├── demo_examples/                   # Sample images for the Gradio UI
└── install.ps1                      # Windows setup script
```

---

## 4. Model Architecture

### 4.1 Backbone

`EfficientNet-B4` from `timm` with `global_pool='avg'`, producing a **1792-dimensional** feature vector per image.

### 4.2 Prediction Heads

All heads branch from the same frozen backbone features (except face shape which also uses geometry):

| Head | Module | Input | Output | Purpose |
|---|---|---|---|---|
| `face_shape_head` | `FaceShapeHead` | 1792 visual + 15 geometric | 5 classes | Primary: Heart/Oblong/Oval/Round/Square |
| `eye_head` | `ClassificationHead` | 1792 | 6 classes | Legacy eye shape |
| `nose_head` | `ClassificationHead` | 1792 | 5 classes | Legacy nose type |
| `lip_head` | `ClassificationHead` | 1792 | 4 classes | Legacy lip fullness |
| `brow_head` | `ClassificationHead` | 1792 | 3 classes | Legacy brow shape |
| `jaw_head` | `ClassificationHead` | 1792 | 3 classes | Legacy jaw type |
| `symmetry_head` | `nn.Sequential` | 1792 | 1 scalar | Symmetry regression (0–1) |
| `eye_narrow_head` | `nn.Sequential` | 1792 | 2 binary logits | Multi-task: narrow/big eye |
| `brow_type_head` | `nn.Sequential` | 1792 | 4 classes | Multi-task: brow type |
| `lip_shape_head` | `nn.Sequential` | 1792 | 2 classes | Multi-task: lip shape |
| `age_head` | `nn.Sequential` | 1792 | 2 classes | Multi-task: age group |
| `gender_head` | `nn.Sequential` | 1792 | 2 classes | Multi-task: gender |
| `landmark_head` | `nn.Sequential` | 1792 | 15 ratios | Geometric ratio regression |
| `skin_tone_head` | `SkinTower` | 1792 + 48 HSV | 3 classes | Light/Medium/Dark |

### 4.3 FaceShapeHead (fusion architecture)

```
Visual features (1792) → Linear(512) → BN → ReLU → Dropout(0.4) ──┐
                                                                     ├─ cat → Linear(256) → BN → ReLU → Linear(5)
Geometric ratios (15)  → Linear(64)  → BN → ReLU ──────────────────┘
```

### 4.4 SkinTower (two-tower fusion)

```
Backbone features (1792) → Linear(256) → ReLU → BN → Dropout ──┐
                                                                   ├─ cat → Linear(128) → ReLU → BN → Linear(3)
HSV histogram (48-dim)   → Linear(128) → ReLU → BN → Dropout ──┘
```

The HSV tower bypasses the frozen backbone to provide raw color information for skin tone, solving the "shape-only feature space" problem.

---

## 5. Data Flow

```
Input: Raw Photo (any resolution)
        │
        ▼
[Face Detection via MediaPipe]  →  Bounding box (x, y, w, h)
        │
        ▼
[Face Crop + Alignment]         →  Eye-aligned, padded face region
        │
        ▼
[Resize to 224×224]
        │
        ├──────────────────────────────────────────────┐
        ▼                                               ▼
[ImageNet Normalization]              [HSV Histogram extraction]
[shape: (1, 3, 224, 224)]            [48-dim: 16 bins × 3 channels]
        │                                               │
        ▼                                               │
[EfficientNet-B4 Backbone]                             │
[output: 1792-dim features]                            │
        │                                               │
        ├── [FaceShapeHead] ← [geometric_ratios (15)] │
        │       → 5-class face shape                   │
        ├── [eye_head]  → 6-class eye shape            │
        ├── [nose_head] → 5-class nose type            │
        ├── [lip_head]  → 4-class lip fullness         │
        ├── [brow_head] → 3-class brow shape           │
        ├── [jaw_head]  → 3-class jaw type             │
        ├── [symmetry_head] → 1 regression score       │
        ├── [eye_narrow_head] → 2 binary               │
        ├── [brow_type_head]  → 4 classes              │
        ├── [lip_shape_head]  → 2 classes              │
        ├── [age_head]    → 2 classes (young/adult)    │
        ├── [gender_head] → 2 classes                  │
        ├── [landmark_head] → 15 geometric ratios      │
        └── [SkinTower] ←──────────────────────────────┘
                → 3 classes (Light/Medium/Dark)
```

---

## 6. Training Pipeline

### 6.1 Data

- **Primary dataset:** FaceShape Dataset (~4,000 images) for face shape labels
- **CelebA:** 202,599 images used for self-training pseudo-labels and multi-task attributes (age, gender, eye/brow/lip)
- **Fitzpatrick17k:** 16,577 images for skin tone labels
- **Niten19:** ~1,000 additional face shape samples
- **Total annotated:** ~5,755 training images

Annotation format: JSON with per-image records containing shape label, attribute dict (nullable), HSV histogram, geometric ratios, Monk scale skin tone.

### 6.2 Training Phases

**Phase 1 — Face Shape Training** (`train_attributes_v2.py` / `FaceAnalysisLightningModule`):
- Backbone **frozen** (BatchNorm stats locked)
- Only face shape head and legacy attribute heads trained
- FocalLoss (γ=2.0) + label smoothing (0.2) for face shape
- Mixed precision (fp16), gradient accumulation ×16

**Phase 2 — Backbone Unfreeze** (epoch 5+):
- Full backbone unfrozen gradually
- Differential LR: backbone=1e-5, heads=1e-4
- CosineAnnealingLR with linear warmup

**Phase 3 — Multi-Task Training** (`train_multitask_v3.py` / `DualOptimizerLightningModule`):
- Backbone + face_shape_head kept in **eval() mode** (implicit freeze via tiny LR=1e-7)
- Attribute heads use **detached** backbone features to prevent gradient interference
- Weighted multi-task loss: shape(1.0) + eye(0.15) + brow(0.15) + lip(0.15) + age(0.10) + gender(0.10) + landmark(0.20) + skin(0.20)
- `FaceShapeGuardCallback` stops training if val_f1 < 0.700 (regression guard)

### 6.3 Loss Functions

| Task | Loss | Notes |
|---|---|---|
| Face shape | `FocalLoss(γ=2.0, ls=0.2)` | Class imbalance handling |
| Eye/nose/lip/brow/jaw | `CrossEntropyLoss(ls=0.2)` | Standard CE with label smoothing |
| Eye narrow (binary) | `BCEWithLogitsLoss` | Multi-label |
| Age / gender | `CrossEntropyLoss` + class weights | Compensates severe imbalance (age_pw=8.83) |
| Skin tone | `CrossEntropyLoss` | On 3-class Monk scale |
| Landmark ratios | `MSELoss` (×10 scaled) | Geometric ratio regression |
| Symmetry | `MSELoss` | 0–1 scalar regression |

### 6.4 Key Engineering Decisions

**MediaPipe caching:** All geometric ratios are pre-computed and cached to `.npz` files using MD5 hash keys, preventing floating-point non-determinism across runs.

**BatchNorm protection:** `backbone.eval()` is forced at every `on_train_epoch_start()` during multi-task training to lock BatchNorm running statistics — preventing drift that would corrupt the face shape head.

**HSV two-tower:** Skin tone head receives both backbone features (shape context) and an HSV color histogram (48-dim, raw color) — this was the fix for "Medium class collapse" where the frozen backbone had no color signal.

**Pseudo-labeling:** 2,787 CelebA images were added via self-training with >85% confidence filtering, yielding +1.75% accuracy (77.62% → 79.21%).

---

## 7. Inference Pipeline (`predict.py`)

```python
from predict import predict_single

result = predict_single("photo.jpg")
print(result["predicted_class"])          # "Oval"
print(result["confidence"])               # 0.87
print(result["attributes"]["skin_tone"])  # "Light"
```

The pipeline:
1. Load image → MediaPipe face detection → crop
2. Extract geometric ratios (15 values) via `LandmarkExtractor`
3. Compute HSV histogram (48-dim) from face crop
4. Apply 5-augmentation TTA: original + hflip + zoom(0.95) + rotate(+5°) + rotate(-5°)
5. Average logits → argmax → confidence threshold (default 0.60)
6. Return structured JSON with all predictions

**Checkpoint path:** `checkpoints/attributes_v2/last-v5.ckpt` (app.py) or `checkpoints/final/model_v6_multitask_skin.ckpt` (predict.py).

---

## 8. Evaluation (`eval_multitask_proper.py`)

Uses `TTALightningModule` (inherits from `FaceAnalysisLightningModule`) which overrides `test_step` to perform 5-augmentation TTA. Reports:
- Per-class F1, precision, recall (scikit-learn classification_report)
- Confusion matrix
- Per-task accuracy logged to console

---

## 9. Deployment

### Gradio Web UI (`app.py`)
- Dark glassmorphism theme with CSS custom styling
- Drag-and-drop image upload
- Visualizes bounding box overlay, face shape icon, confidence bars
- Accepts optional checkpoint path via `--checkpoint` flag

### ONNX Export (`export_onnx.py`)
```python
import onnxruntime as ort

session = ort.InferenceSession("exported/model_v6.onnx")
outputs = session.run(None, {
    "input_image": image_tensor,  # [1, 3, 224, 224]
    "input_geo":   ratios,         # [1, 15]
    "input_hsv":   hsv             # [1, 48]
})
```
Achieves **11.6ms CPU inference** after ONNX graph optimization.

### HuggingFace Spaces
The model is deployed at https://huggingface.co/spaces/Krishna1205/face-analysis using `requirements.txt` for dependency management.

---

## 10. Configuration (`src/config.py`)

All hyperparameters live in `src/config.py`:

```python
FACE_SHAPES     = ["Heart", "Oblong", "Oval", "Round", "Square"]  # 5 classes
IMAGE_SIZE_TRAIN = 224
BATCH_SIZE       = 16
LEARNING_RATE    = 1e-4
EPOCHS           = 250
LABEL_SMOOTHING  = 0.2
LOSS_WEIGHTS     = {
    "face_shape_weight": 0.35,
    "landmark_weight":   0.30,
    "features_weight":   0.15,
    "skin_tone_weight":  0.20,
    "focal_gamma":       2.0,
    "label_smoothing":   0.2
}
```

`get_config_dict()` returns the full nested config dict consumed by `FaceAnalysisLightningModule`.

---

## 11. Bugs Found and Fixed

### Bug 1 — `train_multitask_v3.py`: `SkinTower` called with 1 argument (TypeError)

**File:** `train_multitask_v3.py`, training step  
**Severity:** Critical — crashes training at runtime  
**Root cause:** `SkinTower.forward(bb_features, hsv_histogram)` requires 2 positional arguments, but was called as `self.model.skin_tone_head(detached_features)` with only 1.

```python
# Before (broken):
skin_tone_logits = self.model.skin_tone_head(detached_features)

# After (fixed):
hsv_placeholder = torch.zeros(detached_features.shape[0], 48, device=detached_features.device)
skin_tone_logits = self.model.skin_tone_head(detached_features, hsv_placeholder)
```

### Bug 2 — `face_analysis_model.py`: Unused `fused` variable in `forward()`

**File:** `src/models/face_analysis_model.py`, `FaceAnalysisModel.forward()`  
**Severity:** Low — wasted computation on every forward pass  
**Root cause:** `fused = self._get_fused_features(features, geometric_ratios)` was computed but never referenced. The face shape head invokes `_get_fused_features` internally.

```python
# Before (wasteful):
features = self.backbone(images)
fused = self._get_fused_features(features, geometric_ratios)  # computed, never used
return ModelOutput(...)

# After (cleaned up):
features = self.backbone(images)
return ModelOutput(...)
```

### Bug 3 — `face_analysis_model.py`: `skin_tone_head` excluded from optimizer parameter groups

**File:** `src/models/face_analysis_model.py`, `get_optimizer_param_groups()`  
**Severity:** High — skin tone head parameters never receive gradient updates when using `FaceAnalysisLightningModule`  
**Root cause:** `skin_tone_head` was missing from the `head_params` list in `get_optimizer_param_groups()`, even though it was present in `get_parameter_groups()`.

```python
# Before (missing skin_tone_head):
head_params = (
    ...
    list(self.landmark_head.parameters())
    # skin_tone_head NOT included
)

# After (fixed):
head_params = (
    ...
    list(self.landmark_head.parameters()) +
    list(self.skin_tone_head.parameters())  # added
)
```

### Bug 4 — `src/training/losses.py`: Wrong LOSS_WEIGHTS key names (KeyError)

**File:** `src/training/losses.py`, `MultiTaskLoss.forward()`  
**Severity:** High — raises `KeyError` at runtime if `MultiTaskLoss` is used with the default `LOSS_WEIGHTS` from `config.py`  
**Root cause:** `LOSS_WEIGHTS` in `config.py` uses keys `face_shape_weight`, `features_weight`, `skin_tone_weight`, but `MultiTaskLoss.forward()` looked up `'shape'`, `'features'`, `'skin'`.

```python
# Before (KeyError):
total_loss = (self.weights['shape'] * l_shape) + \
             (self.weights['features'] * l_features) + \
             (self.weights['skin'] * l_skin)

# After (fixed):
total_loss = (self.weights['face_shape_weight'] * l_shape) + \
             (self.weights['features_weight'] * l_features) + \
             (self.weights['skin_tone_weight'] * l_skin)
```

---

## 12. Performance Results

| Task | Accuracy | Baseline |
|---|---|---|
| Face Shape (5-aug TTA) | **79.21%** | Academic: 70.33% |
| Skin Tone (3-class Monk) | **82.29%** | Random: 33.3% |
| Age Group (binary) | **82.59%** | Random: 50.0% |
| Gender (binary) | **80.60%** | Random: 50.0% |
| Eye Shape | **78.11%** | Random: 50.0% |
| Brow Type | **76.12%** | Random: 50.0% |
| Lip Shape | **70.65%** | Random: 50.0% |
| Landmark MAE | **0.12 (r=0.914)** | — |

---

## 13. Known Limitations

1. **No real-time HSV in multi-task training (`train_multitask_v3.py`):** The skin tone head receives a zero-filled HSV placeholder during training (the batch does not include pre-computed HSV histograms). The HSV tower thus learns only from the backbone features and the zero vector, limiting its colour discrimination ability. Wiring `batch.get("hsv_histogram")` from the dataset would improve skin tone accuracy in future training runs.

2. **`skin_tone_loss` placeholder in `FaceAnalysisLightningModule`:** The `_compute_loss` method in `src/training/trainer.py` computes `skin_tone` loss as a constant 0.0 tensor even when `monk_labels` are present. The `skin_tone_logits` from the model are never used here (only used in `train_multitask_v3.py`).

3. **`LIP_FULLNESS` has 3 entries but `lip_head` outputs 4 classes:** `lip_head` outputs 4 logits but `LIP_FULLNESS = ["Thin", "Medium", "Full"]` only has 3 entries. The 4th class is unlabeled. The `predict()` method uses `lip_idx` to index `LIP_CLASSES`, which could raise `IndexError` if the 4th class is predicted.

4. **`multi_task_model.py` / `losses.py` are dead code:** `MultiTaskFaceModel` and `MultiTaskLoss` are not imported anywhere in the active codebase. They represent an earlier iteration and should either be deleted or moved to `archive/`.

5. **`hf_space/` directory is empty:** Deployment artifacts for HuggingFace Spaces are not checked in.

6. **Geometric ratio caching path is hardcoded:** `data/landmarks_cache` and `data/processed/` are hardcoded relative paths throughout the codebase, making it fragile when run from different working directories.

---

## 14. Additional Fix: Created Missing `src/data/` Package

`predict.py`, `eval_multitask_proper.py`, `train_multitask_v3.py`, and `scripts/debug_predictions_v2.py` all import from `src.data.dataset`, but this package did not exist in the repository. Running any of these scripts would immediately raise `ModuleNotFoundError: No module named 'src.data'`.

The package was created at `src/data/dataset.py` with:

- `extract_hsv_histogram_np(image_rgb, n_bins=16) -> np.ndarray` — normalised 48-dim HSV colour histogram used by `SkinTower`
- `FaceAnalysisDataset` — multi-task aware `Dataset` that reads the annotation format produced by `build_multitask_annotations.py` and returns all supervision signals (shape, eye, brow, lip, age, gender, landmarks, Monk skin scale)
- `get_train_transforms(image_size, cfg=None) -> A.Compose` — training augmentation pipeline (backward-compatible with legacy `{"training": {}}` call pattern)
- `get_val_transforms(image_size) -> A.Compose` — deterministic val/test preprocessing
