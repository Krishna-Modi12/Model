import torch, json, sys
sys.path.insert(0, '.')
from src.data.dataset import FaceAnalysisDataset, get_train_transforms
from torch.utils.data import DataLoader
from src.models.face_analysis_model import FaceAnalysisModel

# Check data ranges
ds = FaceAnalysisDataset(
    annotations_path='data/processed/annotations_multitask_balanced.json',
    image_size=224,
    transforms=get_train_transforms(224, {'training': {}}),
    landmarks_cache_dir='data/landmarks_cache',
)
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

print("--- DATA VALIDATION ---")
for i, batch in enumerate(dl):
    mask = batch.get('has_attributes', torch.zeros(8, dtype=torch.bool))
    if mask.any():
        for key in ['eye_narrow', 'eye_big', 'brow', 'lip', 'age', 'gender']:
            vals = batch[key][mask]
            lo, hi = vals.min().item(), vals.max().item()
            print(f"  {key}: min={lo} max={hi}")
        monk = batch.get('monk_labels', torch.full((8,), -100))
        skin_mask = monk != -100
        if skin_mask.any():
            print(f"  monk_labels: min={monk[skin_mask].min().item()} max={monk[skin_mask].max().item()}")
        break
    if i > 50:
        print('No attr batches in first 50')
        break

# Check model head output sizes (use batch_size=2 to avoid BatchNorm issue)
print("\n--- MODEL HEAD SIZES ---")
model = FaceAnalysisModel(
    backbone="efficientnet_b4",
    pretrained=False,
    dropout=0.4,
    geometric_features=15,
    num_classes=5,
    freeze_backbone=True
)
model.eval()  # eval mode to avoid batchnorm issues
dummy_img = torch.randn(2, 3, 224, 224)
dummy_geo = torch.randn(2, 15)
with torch.no_grad():
    out = model(dummy_img, dummy_geo)
print(f"  face_shape_logits:  {out.face_shape_logits.shape}")
print(f"  eye_narrow_logits:  {out.eye_narrow_logits.shape}")
print(f"  brow_type_logits:   {out.brow_type_logits.shape}")
print(f"  lip_shape_logits:   {out.lip_shape_logits.shape}")
print(f"  age_logits:         {out.age_logits.shape}")
print(f"  gender_logits:      {out.gender_logits.shape}")
print(f"  skin_tone_logits:   {out.skin_tone_logits.shape}")
print(f"  landmark_pred:      {out.landmark_pred.shape}")

# Test CE for each head
print("\n--- CROSS ENTROPY TEST ---")
for name, logits_shape_1 in [
    ('brow', out.brow_type_logits.shape[1]),
    ('lip', out.lip_shape_logits.shape[1]),
    ('age', out.age_logits.shape[1]),
    ('gender', out.gender_logits.shape[1]),
    ('skin', out.skin_tone_logits.shape[1]),
]:
    print(f"  {name}: num_classes={logits_shape_1}")
    for label_val in [0, 1]:
        target = torch.tensor([label_val, label_val])
        logits_dummy = torch.randn(2, logits_shape_1)
        try:
            loss = torch.nn.functional.cross_entropy(logits_dummy, target)
            print(f"    label={label_val}: loss={loss.item():.4f} OK")
        except Exception as e:
            print(f"    label={label_val}: FAILED - {e}")

# Test with actual batch on CPU
print("\n--- FULL FORWARD PASS TEST (CPU) ---")
for i, batch in enumerate(dl):
    mask = batch.get('has_attributes', torch.zeros(8, dtype=torch.bool))
    if mask.any():
        with torch.no_grad():
            out = model(batch['images'], batch['geometric_ratios'])
        
        # Test each loss
        try:
            eye_targets = torch.stack([batch["eye_narrow"].float(), batch["eye_big"].float()], dim=1)
            loss_eye = torch.nn.functional.binary_cross_entropy_with_logits(
                out.eye_narrow_logits[mask], eye_targets[mask])
            print(f"  eye loss: {loss_eye.item():.4f} OK")
        except Exception as e:
            print(f"  eye loss FAILED: {e}")
            
        try:
            loss_brow = torch.nn.functional.cross_entropy(out.brow_type_logits[mask], batch["brow"][mask])
            print(f"  brow loss: {loss_brow.item():.4f} OK")
        except Exception as e:
            print(f"  brow loss FAILED: {e}")
            
        try:
            loss_lip = torch.nn.functional.cross_entropy(out.lip_shape_logits[mask], batch["lip"][mask])
            print(f"  lip loss: {loss_lip.item():.4f} OK")
        except Exception as e:
            print(f"  lip loss FAILED: {e}")
            
        try:
            loss_age = torch.nn.functional.cross_entropy(out.age_logits[mask], batch["age"][mask])
            print(f"  age loss: {loss_age.item():.4f} OK")
        except Exception as e:
            print(f"  age loss FAILED: {e}")
            
        try:
            loss_gender = torch.nn.functional.cross_entropy(out.gender_logits[mask], batch["gender"][mask])
            print(f"  gender loss: {loss_gender.item():.4f} OK")
        except Exception as e:
            print(f"  gender loss FAILED: {e}")

        print("\nAll losses computed successfully on CPU!")
        break
    if i > 50:
        print('No attr batch found')
        break
