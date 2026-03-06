import json
from collections import Counter

with open('data/processed/annotations_multitask_balanced.json') as f:
    anns = json.load(f)

skin_counts = Counter(a.get('monk_label') for a in anns if a.get('monk_label') is not None)
print(f'Total samples with skin labels: {sum(skin_counts.values())}')
print(f'Skin distribution (0=L, 1=M, 2=D): {dict(skin_counts)}')

# Check some samples
print('\nSample entries with skin labels:')
samples = [a for a in anns if a.get('monk_label') is not None][:10]
for s in samples:
    print(f"  Label: {s.get('monk_label')}, ImagePath: {s.get('image_path')}, ProcessedPath: {s.get('processed_path')}")
