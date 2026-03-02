import os
total = 0
count = 0
for dirpath in ['data/curated', 'data/processed/external_images']:
    if os.path.exists(dirpath):
        for root, dirs, files in os.walk(dirpath):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
                count += 1
print(f"Image data: {count} files, {total / 1024 / 1024:.0f} MB")
