import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch, sys
sys.path.insert(0, '.')
from src.data.dataset import FaceAnalysisDataset, get_train_transforms
from src.models.face_analysis_model import FaceAnalysisModel
from src.training.trainer import FocalLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F

torch.manual_seed(42)

# Load data
ds = FaceAnalysisDataset(
    annotations_path='data/processed/annotations_multitask_balanced.json',
    image_size=224,
    transforms=get_train_transforms(224, {'training': {}}),
    landmarks_cache_dir='data/landmarks_cache',
)
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

# Build model on GPU
model = FaceAnalysisModel(
    backbone="efficientnet_b4",
    pretrained=False,
    dropout=0.4,
    geometric_features=15,
    num_classes=5,
    freeze_backbone=True
)

# Load checkpoint weights
ckpt = torch.load("checkpoints/multitask_skin_tone/attrs_skin_epoch=00_val_loss=1.6046.ckpt", map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
new_sd = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
model.load_state_dict(new_sd, strict=False)
model = model.cuda()
model.train()

shape_loss = FocalLoss(gamma=2.0, label_smoothing=0.1)

for i, batch in enumerate(dl):
    images = batch["images"].cuda()
    geo = batch["geometric_ratios"].cuda()
    
    output = model(images, geo)
    
    # Face shape
    labels = batch["shape_labels"].cuda()
    loss_face = shape_loss(output.face_shape_logits, labels)
    print(f"Batch {i}: face_loss={loss_face.item():.4f}")
    
    # Attributes
    attr_mask = batch.get("has_attributes", torch.zeros(8, dtype=torch.bool))
    if attr_mask.any():
        # Eye
        eye_narrow = batch["eye_narrow"].cuda()
        eye_big = batch["eye_big"].cuda()
        eye_targets = torch.stack([eye_narrow.float(), eye_big.float()], dim=1)
        loss_eye = F.binary_cross_entropy_with_logits(
            output.eye_narrow_logits[attr_mask], eye_targets[attr_mask])
        print(f"  eye_loss={loss_eye.item():.4f}")
        
        # Brow
        brow_labels = batch["brow"].cuda()
        print(f"  brow labels (masked): {brow_labels[attr_mask].tolist()}")
        print(f"  brow_logits shape: {output.brow_type_logits[attr_mask].shape}")
        loss_brow = F.cross_entropy(output.brow_type_logits[attr_mask], brow_labels[attr_mask])
        print(f"  brow_loss={loss_brow.item():.4f}")
        
        # Lip
        lip_labels = batch["lip"].cuda()
        print(f"  lip labels (masked): {lip_labels[attr_mask].tolist()}")
        print(f"  lip_logits shape: {output.lip_shape_logits[attr_mask].shape}")
        loss_lip = F.cross_entropy(output.lip_shape_logits[attr_mask], lip_labels[attr_mask])
        print(f"  lip_loss={loss_lip.item():.4f}")
        
        # Age
        age_labels = batch["age"].cuda()
        print(f"  age labels (masked): {age_labels[attr_mask].tolist()}")
        print(f"  age_logits shape: {output.age_logits[attr_mask].shape}")
        loss_age = F.cross_entropy(output.age_logits[attr_mask], age_labels[attr_mask])
        print(f"  age_loss={loss_age.item():.4f}")
        
        # Gender
        gender_labels = batch["gender"].cuda()
        print(f"  gender labels (masked): {gender_labels[attr_mask].tolist()}")
        print(f"  gender_logits shape: {output.gender_logits[attr_mask].shape}")
        loss_gender = F.cross_entropy(output.gender_logits[attr_mask], gender_labels[attr_mask])
        print(f"  gender_loss={loss_gender.item():.4f}")
    
    if i >= 10:
        print("Completed 10 batches successfully!")
        break
