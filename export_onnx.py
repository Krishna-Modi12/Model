"""
export_onnx.py
==============
Exports the multi-task PyTorch Lightning model to an ONNX graph for faster CPU inference.
"""

import os
import torch
import torch.nn as nn
from loguru import logger
from src.models.face_analysis_model import FaceAnalysisModel
from src.config import FACE_SHAPES

class ONNXWrapper(nn.Module):
    """Wraps the FaceAnalysisModel to return a tuple instead of a Dataclass (which ONNX tracing doesn't support well)."""
    def __init__(self, base_model: FaceAnalysisModel):
        super().__init__()
        self.base = base_model
        
    def forward(self, img, geo, hsv):
        out = self.base(img, geo, hsv)
        return (
            out.face_shape_logits,
            out.eye_logits,
            out.nose_logits,
            out.lip_logits,
            out.brow_logits,
            out.jaw_logits,
            out.symmetry_score,
            out.eye_narrow_logits,
            out.brow_type_logits,
            out.lip_shape_logits,
            out.age_logits,
            out.gender_logits,
            out.landmark_pred,
            out.skin_tone_logits
        )

from predict import load_model

def export_to_onnx(checkpoint_path: str, output_path: str):
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    device = torch.device('cpu')
    model = load_model(checkpoint_path, device)
    model.eval()

    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()
    
    # Create dummy inputs for ONNX tracing
    # EfficientNet-B4 input is (1, 3, 380, 380), but our pipeline uses 224x224
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_geo = torch.randn(1, 15)
    dummy_hsv = torch.randn(1, 48) # 3 channels * 16 bins = 48 bins

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info("Exporting to ONNX...")
    torch.onnx.export(
        wrapped_model,
        (dummy_image, dummy_geo, dummy_hsv),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image", "input_geo", "input_hsv"],
        output_names=[
            "face_shape_logits", "eye_logits", "nose_logits", "lip_logits", 
            "brow_logits", "jaw_logits", "symmetry_score", 
            "eye_narrow_logits", "brow_type_logits", "lip_shape_logits", 
            "age_logits", "gender_logits", "landmark_pred", "skin_tone_logits"
        ],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "input_geo": {0: "batch_size"},
            "input_hsv": {0: "batch_size"},
            "face_shape_logits": {0: "batch_size"}
        }
    )
    
    logger.info(f"✨ ONNX export successful! Saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best PyTorch .ckpt")
    parser.add_argument("--out", type=str, default="checkpoints/multitask_skin_tone/model.onnx", help="Output .onnx path")
    args = parser.parse_args()
    
    export_to_onnx(args.ckpt, args.out)
