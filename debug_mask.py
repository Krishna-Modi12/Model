import torch, sys
sys.path.insert(0,'.')
from src.data.dataset import FaceAnalysisDataset, get_train_transforms
from torch.utils.data import DataLoader

ds = FaceAnalysisDataset(
    annotations_path='data/processed/annotations_multitask_balanced.json',
    image_size=224,
    transforms=get_train_transforms(224, {'training': {}}),
    landmarks_cache_dir='data/landmarks_cache',
)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

for i, batch in enumerate(dl):
    mask = batch.get('has_attributes', torch.zeros(8, dtype=torch.bool))
    print(f"Batch {i}: has_attributes type={type(mask)}, value={mask}")
    
    if mask.any():
        # Check that masked values are valid
        for key in ['brow', 'lip', 'age', 'gender']:
            masked_vals = batch[key][mask]
            unmasked_vals = batch[key][~mask]
            print(f"  {key} masked (should be >=0): {masked_vals.tolist()}")
            print(f"  {key} unmasked (can be -1): {unmasked_vals.tolist()}")
        break
    
    if i > 5:
        print("No mixed batch found in first 5")
        break
