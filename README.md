---

# 🔍 Face Analysis AI — V6 Multi-Task Model

> Complete facial analysis from a single photo.
> EfficientNet-B4 backbone · 8 prediction heads · 11.6ms CPU inference

**[🚀 Live Demo](https://huggingface.co/spaces/Krishna1205/face-analysis)**

---

## Results

| Task | Accuracy | Baseline |
|---|---|---|
| Face Shape (TTA) | **79.21%** | Academic: 70.33% |
| Skin Tone | **82.29%** | Random: 33.3% |
| Age Group | **82.59%** | Random: 50.0% |
| Gender | **80.60%** | Random: 50.0% |
| Eye Shape | **78.11%** | Random: 50.0% |
| Brow Type | **76.12%** | Random: 50.0% |
| Lip Shape | **70.65%** | Random: 50.0% |
| Landmark MAE | **0.12 (r=0.914)** | — |

---

## What It Does

Upload a photo → complete facial profile in one forward pass:

```python
{
  "predicted_class" : "Oval",
  "confidence"      : 0.847,
  "attributes": {
      "skin_tone"  : "Light",
      "eye_shape"  : "Almond",
      "brow_type"  : "Thick/Arched",
      "lip_shape"  : "Full",
      "age_group"  : "Young",
      "gender"     : "Female",
      "face_ratios": [0.82, 0.71, ...]
  }
}
```

---

## Architecture

Input Photo (224×224)
↓
[EfficientNet-B4 Backbone] → 1792-dim features
↓                           ↓
[Face Shape Head]          [6 Attribute Heads]
5-class softmax            Eye  · Brow  · Lip
79.21% TTA                Age  · Gender · Landmark
↓
[HSV Color Tower] ← bypasses frozen backbone
48-dim histogram            ↓
→→→→→→→→→→→→→ [Skin Tone Head]
3-class (Light/Medium/Dark)
82.29% accuracy

**Key architectural decisions:**

- `requires_grad=False` on backbone — attribute heads read
  frozen features without destroying face shape accuracy
- HSV histogram (48-dim) fed directly to skin head —
  bypasses backbone's shape-only feature space to access
  raw color information
- Separate optimization trajectory for backbone vs heads
- Focal loss + class oversampling for minority Dark class

---

## Training Journey

| Version | Accuracy | Key Change |
|---|---|---|
| V1 Baseline | 62.15% | EfficientNet-B4 from scratch |
| V2 | 69.84% | External dataset integration |
| V3 | 76.20% | TTA + augmentation |
| V4 | 77.62% | MediaPipe drift fix |
| V5 | **79.21%** | CelebA self-training (+2787 pseudo-labels) |
| V6 | **79.21%** + 7 new heads | Multi-task attribute heads |

---

## Hard Problems Solved

### 1 — MediaPipe Geometric Drift

MediaPipe landmark extraction produces slightly different
results each run due to floating point non-determinism.
This caused silent accuracy degradation during evaluation.

**Fix:** Cache all geometric ratios to `.npz` files
using MD5 hash keys. All training and inference uses
cached values — MediaPipe never runs twice on the same image.

### 2 — CelebA Self-Training Pipeline

Collected 2,787 high-confidence pseudo-labeled CelebA images
to augment the training set. Required:

- Confidence threshold filtering (>85%)
- Quality filtering (no blurry, no heavy makeup)
- Balanced sampling to prevent distribution shift

**Result:** +1.75% TTA accuracy gain (77.62% → 79.21%)

### 3 — BatchNorm Destruction During Multi-Task Training

When the backbone was set to `.train()` mode during
attribute head training, BatchNorm running statistics
drifted with batch_size=8, destroying the face shape
features the backbone spent weeks learning.

**Fix:** Override `on_train_epoch_start()` to permanently
force `backbone.eval()` — parameters still update via
`lr=1e-7` but batch statistics are locked.

### 4 — Skin Tone Medium Class Collapse

Initial skin tone head predicted "Medium" for every image
(achieving 100% near-match but 23% exact match) because:

- Class imbalance: Light=569, Medium=330, Dark=93
- Frozen backbone features encode shape, not color

**Fix:** HSV histogram tower (48-dim) feeds raw color
directly to skin head, bypassing the shape-only backbone.
Focal loss (γ=2.0) + 5× oversampling for Dark class.

**Result:** 22.94% → 82.29% exact match accuracy

### 5 — Silent Dataset Label Poisoning

Original face shape images were accidentally populated with
all-zero attribute dictionaries. 60% of training data had
fake attribute labels, causing Age/Gender heads to predict
majority class for every input (F1=1.0 collapse).

**Fix:** Strict `has_attributes` mask requiring real
CelebA-sourced labels. Face shape images always get
`attributes=null` — never fake zeros.

---

## Inference

**Python:**

```python
from predict import predict_single
result = predict_single("photo.jpg")
print(result["predicted_class"])   # "Oval"
print(result["attributes"]["skin_tone"])  # "Light"
```

**ONNX (11.6ms CPU):**

```python
import onnxruntime as ort
session = ort.InferenceSession("exported/model_v6.onnx")
outputs = session.run(None, {
    "image": image_tensor,      # [1, 3, 224, 224]
    "geometric_ratios": ratios  # [1, 15]
})
```

---

## Setup

```bash
git clone https://github.com/Krishna1205/face-analysis
cd face-analysis
pip install -r requirements_colab_match.txt

# Run Gradio demo
python app.py

# Run inference
python predict.py --image photo.jpg \
  --checkpoint checkpoints/attributes_v2/last-v5.ckpt
```

**Environment:** Python 3.10 · PyTorch 2.0 · Windows 11
**Hardware used:** RTX 4060 Laptop 8GB · ~40 hours training

---

## Project Structure

```text
├── predict.py                  # Production inference
├── app.py                      # Gradio web demo
├── train_multitask_v3.py       # Multi-task training
├── train_attributes_v2.py      # Attribute head training
├── evaluate_v3.py              # Face shape evaluation
├── eval_multitask_proper.py    # Multi-task evaluation
├── export_onnx.py              # ONNX export
├── build_multitask_annotations.py  # Data pipeline
├── src/
│   ├── models/face_analysis_model.py  # Architecture
│   ├── data/dataset.py                # Dataset + sampler
│   └── utils/landmark_extractor.py    # MediaPipe wrapper
├── exported/
│   └── model_v6.onnx           # ONNX model (CPU optimized)
└── archive/                    # Experimental scripts
```

---

## Dataset Sources

| Dataset | Images | Used For |
|---|---|---|
| FaceShape Dataset | 4,000 | Primary face shape training |
| CelebA | 202,599 | Self-training pseudo-labels + attributes |
| Fitzpatrick17k | 16,577 | Skin tone evaluation |
| Niten19 | ~1,000 | Additional face shape samples |

---

*Built with PyTorch Lightning · EfficientNet-B4 via timm*
*MediaPipe · ONNX Runtime · Gradio*

---
