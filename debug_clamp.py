import sys; sys.path.insert(0,'.')
from src.data.dataset import FaceAnalysisDataset, get_train_transforms

ds = FaceAnalysisDataset(
    annotations_path='data/processed/annotations_multitask_balanced.json',
    image_size=224,
    transforms=get_train_transforms(224, {'training': {}}),
    landmarks_cache_dir='data/landmarks_cache',
)

neg_found = 0
for i in range(min(500, len(ds))):
    s = ds[i]
    if s['has_attributes']:
        for k in ['brow', 'lip', 'age', 'gender']:
            v = s[k].item()
            if v < 0:
                neg_found += 1
                print(f"Sample {i}: {k}={v} NEGATIVE!")

if neg_found == 0:
    print("All attribute labels >= 0. Clamping works correctly.")
else:
    print(f"Found {neg_found} negative labels!")
