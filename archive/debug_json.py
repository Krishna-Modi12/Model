import json

with open('data/processed/annotations_multitask_balanced.json','r') as f:
    data=json.load(f)

neg_count = {}
for d in data:
    a = d.get('attributes')
    if a:
        for key in ['eye_narrow', 'eye_big', 'brow', 'lip', 'age', 'gender']:
            v = a.get(key)
            if v is not None and v < 0:
                neg_count[key] = neg_count.get(key, 0) + 1

print("Negative values in annotations JSON (before clamping):")
for k, v in neg_count.items():
    print(f"  {k}: {v} entries with -1")

# Test clamping  
print("\nmax(0, -1) =", max(0, -1))  # Should be 0
print("max(0, None) would error")
