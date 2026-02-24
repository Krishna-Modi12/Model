import os, sys, torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import FACE_SHAPES, get_config_dict
from src.data.dataset import create_dataloaders

print('Loading Config...')
config = get_config_dict()

print('Creating Loaders...')
loaders = create_dataloaders(config)

def count_split(dset):
    counts = {i: 0 for i in range(len(FACE_SHAPES))}
    for ann in dset.annotations:
        counts[ann.get("shape_label", 0)] += 1
    return counts

train_c = count_split(loaders['train'].dataset)
val_c = count_split(loaders['val'].dataset)
test_c = count_split(loaders['test'].dataset)

print('\nNEW DATASET SPLITS:')
for i, c in enumerate(FACE_SHAPES):
    print(f'{c:<10} | Train: {train_c[i]:<5} | Val: {val_c[i]:<5} | Test: {test_c[i]:<5}')

print('\nTesting Batch...')
batch = next(iter(loaders['train']))
images = batch['images']
if "shape_labels" in batch:
    labels = batch["shape_labels"]
elif "face_shape" in batch:
    labels = batch["face_shape"]
else:
    labels = batch[1]

print(f'Batch shape: {images.shape}')

uniq_labels, uniq_idx = np.unique(labels.numpy(), return_index=True)
sample_indices = uniq_idx[:4]
fig, axes = plt.subplots(2, 2, figsize=(6,6))

mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

for i, ax in enumerate(axes.flatten()):
    if i < len(sample_indices):
        idx = sample_indices[i]
        img_tsr = images[idx].clone() * std + mean
        img_np = np.clip(img_tsr.permute(1,2,0).cpu().numpy(), 0, 1)
        ax.imshow(img_np)
        ax.set_title(FACE_SHAPES[labels[idx].item()])
    ax.axis('off')

plt.tight_layout()
plt.savefig('debug_sample_grid.png')
print('Saved debug_sample_grid.png')
