"""
verify_setup.py  (FIXED)
─────────────────────────────────────────────────────────────
Fixes applied:
  1. cv2.cuda check replaced with a safe fallback. The previous check
     called cv2.cuda.getCudaEnabledDeviceCount() which throws AttributeError
     on the standard pip install of opencv-python (only available in
     opencv-contrib-python built with CUDA). This caused every environment
     to show a false failure, even fully working ones.
     Now: reports CUDA-OpenCV status as informational, not a pass/fail gate.
─────────────────────────────────────────────────────────────
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.getcwd())


def check(label, fn, critical=True):
    global all_ok
    try:
        result = fn()
        print(f"  ✅  {label}: {result}")
        return True
    except Exception as e:
        marker = "❌" if critical else "⚠️ "
        print(f"  {marker}  {label}: {'FAILED' if critical else 'SKIPPED'} — {e}")
        if critical:
            all_ok = False
        return False


print("\n" + "=" * 60)
print("  Face Analysis Model — Environment Verification")
print("=" * 60)

all_ok = True

# ── Python ──
print("\n[1] Python")
check("Version", lambda: sys.version.split()[0])

# ── PyTorch + CUDA ──
print("\n[2] PyTorch + CUDA")
import torch
check("PyTorch version",   lambda: torch.__version__)
check("CUDA available",    lambda: str(torch.cuda.is_available()))
check("GPU name",          lambda: torch.cuda.get_device_name(0))
check("CUDA version",      lambda: torch.version.cuda)
check("cuDNN version",     lambda: str(torch.backends.cudnn.version()))
check("GPU memory (GB)",   lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")

try:
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = (a @ b).sum()
    print(f"  ✅  GPU matrix multiply: {float(c):.2f} (OK)")
except Exception as e:
    print(f"  ❌  GPU matrix multiply: FAILED — {e}")
    all_ok = False

# ── timm ──
print("\n[3] timm (Model Backbones)")
import timm
check("timm version",       lambda: timm.__version__)
check("EfficientNet-B4",    lambda: str(timm.create_model(
    "efficientnet_b4", pretrained=False).num_features) + " features")
check("ConvNeXt-Base",      lambda: str(timm.create_model(
    "convnext_base", pretrained=False).num_features) + " features")

# ── MediaPipe ──
print("\n[4] MediaPipe")
import mediapipe as mp
check("MediaPipe version", lambda: mp.__version__)
check("FaceMesh init",     lambda: (
    mp.solutions.face_mesh.FaceMesh(static_image_mode=True).__class__.__name__))

# ── OpenCV ──
print("\n[5] OpenCV")
import cv2
check("OpenCV version", lambda: cv2.__version__)

# FIX: cv2.cuda is only available in CUDA-built opencv-contrib, not standard pip.
# Report as informational (⚠️) not a hard failure.
print("  " + "-" * 54)
try:
    n = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"  ✅  OpenCV CUDA devices: {n}")
except AttributeError:
    print(f"  ℹ️   OpenCV CUDA: not available in standard install "
          f"(this is NORMAL — PyTorch CUDA is what matters for training)")

# ── Albumentations ──
print("\n[6] Albumentations")
import albumentations as A
import numpy as np
check("Version", lambda: A.__version__)
check("Transform pipeline", lambda: str(
    A.Compose([A.HorizontalFlip(), A.Resize(256, 256)])(
        image=np.zeros((300, 300, 3), dtype=np.uint8))["image"].shape))

# ── PyTorch Lightning ──
print("\n[7] PyTorch Lightning")
import pytorch_lightning as pl
check("Version", lambda: pl.__version__)

# ── scikit-learn ──
print("\n[8] Scikit-learn")
import sklearn
check("Version", lambda: sklearn.__version__)

# ── FastAPI ──
print("\n[9] FastAPI")
import fastapi
check("Version", lambda: fastapi.__version__)

# ── skin tone math ──
print("\n[10] Skin Tone Dependencies")
check("numpy LAB conversion",
      lambda: str(np.degrees(np.arctan2(75 - 50, 18))) + "° (ITA test)")

# ── Full model forward pass ──
print("\n[11] End-to-end Model Test")
try:
    from src.models.face_analysis_model import FaceAnalysisModel

    model = FaceAnalysisModel(
        backbone="efficientnet_b4", pretrained=False).cuda()
    model.eval()

    with torch.no_grad():
        imgs   = torch.randn(2, 3, 256, 256).cuda()
        ratios = torch.randn(2, 15).cuda()
        out    = model(imgs, ratios)

    check("Model forward pass", lambda: str(out.face_shape_logits.shape))
    preds = model.predict(imgs, ratios)
    check("Model predict()",    lambda: preds[0]["face_shape"]["label"])

except Exception as e:
    print(f"  ❌  Model test: FAILED — {e}")
    all_ok = False

# ── Summary ──
print("\n" + "=" * 60)
if all_ok:
    print("  🎉  ALL CHECKS PASSED — Ready to train!")
else:
    print("  ⚠️   SOME CHECKS FAILED — Fix issues above before training")
print("=" * 60 + "\n")
