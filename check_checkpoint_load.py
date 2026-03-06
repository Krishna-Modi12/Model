import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
import json
from pathlib import Path

from src.models.face_analysis_model import FaceAnalysisModel
from src.data.dataset import FaceAnalysisDataset, get_val_transforms
from src.config import get_config_dict
from src.training.trainer import FaceAnalysisLightningModule
from torchmetrics.classification import F1Score

def main():
    model = FaceAnalysisModel()
    
    ckpt_path = "checkpoints/multitask_skin_tone/attrs_skin_epoch=00_val_loss=1.6046.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    
    model_dict = model.state_dict()
    filtered_dict = {}
    matched_keys = []
    dropped_keys = []
    
    for k, v in state_dict.items():
        core_k = k.replace("model.", "") if k.startswith("model.") else k
        if core_k in model_dict:
            if v.shape == model_dict[core_k].shape:
                filtered_dict[core_k] = v
                matched_keys.append(core_k)
            else:
                dropped_keys.append(f"{core_k} (shape mismatch: {v.shape} vs {model_dict[core_k].shape})")
        else:
            dropped_keys.append(f"{core_k} (not in model)")
            
    print(f"Matched {len(matched_keys)} keys out of {len(state_dict)}.")
    print("Checking Face Shape Head weights:")
    fs_keys = [k for k in matched_keys if 'face_shape_head' in k]
    print(f"  -> Found {len(fs_keys)} face_shape_head keys.")
    if len(fs_keys) == 0:
        print("  !!! FATAL: No face_shape_head weights were loaded !!!")

    print("\nFirst 10 Dropped keys:")
    for k in dropped_keys[:10]:
        print(f"  {k}")

if __name__ == "__main__":
    main()
