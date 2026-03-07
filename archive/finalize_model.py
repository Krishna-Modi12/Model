"""
finalize_model.py
=================
Finds the best checkpoint from `checkpoints/multitask_skin_tone/`,
copies it to a stable `checkpoints/final_model.ckpt`,
writes metadata out, and exports the final ONNX model.
"""

import os
import glob
import json
import shutil
from datetime import datetime
from PIL import Image

def finalize_all():
    # 1. Find the best checkpoint in checkpoints/multitask_skin_tone/
    # The filename format is attrs_skin_epoch_{val_loss}.ckpt
    ckpts = glob.glob("checkpoints/multitask_skin_tone/*.ckpt")
    if not ckpts:
        print("❌ No checkpoints found in checkpoints/multitask_skin_tone/")
        return
        
    # Ignore 'last.ckpt' to ensure we only get the best val_loss
    valid_ckpts = [c for c in ckpts if "last.ckpt" not in c]
    
    if not valid_ckpts:
        # Fallback to last.ckpt if it's the only one
        best_ckpt = "checkpoints/multitask_skin_tone/last.ckpt"
    else:
        # Since 'val_loss' is smaller = better, we want to sort alphabetically
        # wait, the filename format might just have the val loss as a float string at the end.
        # Let's extract the loss directly
        def extract_loss(filepath):
            filename = os.path.basename(filepath)
            # e.g., 'attrs_skin_epoch=08_val_loss=1.5830.ckpt'
            parts = filename.replace('.ckpt', '').split('val_loss=')
            if len(parts) > 1:
                return float(parts[1])
            return 9999.0
            
        best_ckpt = sorted(valid_ckpts, key=extract_loss)[0]

    print(f"✨ Selected Best Checkpoint: {os.path.basename(best_ckpt)}")
    
    # 2. Copy and rename for stable downstream usage
    os.makedirs("checkpoints/final", exist_ok=True)
    final_ckpt_path = "checkpoints/final/model_v6_multitask_skin.ckpt"
    shutil.copy2(best_ckpt, final_ckpt_path)
    print(f"✅ Copied to: {final_ckpt_path}")
    
    # 3. Create JSON metadata
    metadata = {
        "version": "6.0",
        "description": "Multi-task Unified Model (Face Shape, Eye, Brow, Lip, Age, Gender, Landmarks, Skin Tone)",
        "build_date": datetime.now().isoformat(),
        "source_checkpoint": os.path.basename(best_ckpt),
        "backbone": "efficientnet_b4 (frozen)",
        "datasets": [
            "LFW (Face Shape base)",
            "CelebA (Multi-task Attributes)",
            "Fitzpatrick17k (Skin Tone / Monk Scale)"
        ],
        "input_resolution": "224x224",
        "geometric_features": 15
    }
    
    with open("checkpoints/final/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("✅ Created Version Metadata JSON.")
    
    # 4. Export to ONNX automatically
    from export_onnx import export_to_onnx
    onnx_path = "checkpoints/final/model_v6_multitask_skin.onnx"
    print(f"🚀 Exporting to ONNX...")
    export_to_onnx(final_ckpt_path, onnx_path)
    
if __name__ == "__main__":
    finalize_all()
