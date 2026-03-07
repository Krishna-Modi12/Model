"""Phase 0 — VRAM Check for Multi-Task Model"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models.face_analysis_model import FaceAnalysisModel

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

device = torch.device("cuda")
model = FaceAnalysisModel(
    backbone="efficientnet_b4", pretrained=False, freeze_backbone=False
).to(device)

dummy_images = torch.randn(16, 3, 224, 224).to(device)
dummy_ratios = torch.randn(16, 15).to(device)

with torch.cuda.amp.autocast():
    output = model(dummy_images, dummy_ratios)

peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak VRAM (batch=16): {peak:.2f}GB / 8.00GB")

if peak > 7.5:
    print("[CRITICAL] Reduce batch_size=4, accumulate_grad_batches=4")
elif peak > 7.0:
    print("[WARNING] Use batch_size=8, accumulate_grad_batches=2")
else:
    print("[OK] batch_size=8 with accumulate_grad_batches=2 safe")

print()
print(f"face_shape_logits: {output.face_shape_logits.shape}")
print(f"eye_narrow_logits: {output.eye_narrow_logits.shape}")
print(f"brow_type_logits:  {output.brow_type_logits.shape}")
print(f"lip_shape_logits:  {output.lip_shape_logits.shape}")
print(f"age_logits:        {output.age_logits.shape}")
print(f"gender_logits:     {output.gender_logits.shape}")
print(f"landmark_pred:     {output.landmark_pred.shape}")
print()
params = model.count_parameters()
print(f"Total params:     {params['total']:,}")
print(f"Trainable params: {params['trainable']:,}")
