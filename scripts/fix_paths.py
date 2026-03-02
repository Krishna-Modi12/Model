"""
Fix annotations.json paths for cross-platform compatibility.
Converts absolute Windows paths to relative paths that work on both
Windows and Linux (Google Colab).

Run this ONCE on your local machine before uploading to Google Drive.
"""
import json
import os
from pathlib import Path

def fix_annotations():
    annotations_path = os.path.join("data", "processed", "annotations.json")
    
    with open(annotations_path, "r") as f:
        data = json.load(f)
    
    # The Windows base path to strip
    win_base = r"C:\Users\krish\OneDrive\Desktop\Model" + "\\"
    
    fixed_count = 0
    for entry in data:
        path = entry["image_path"]
        if win_base in path:
            # Convert absolute Windows path to relative with forward slashes
            relative = path.replace(win_base, "").replace("\\", "/")
            entry["image_path"] = relative
            fixed_count += 1
        elif "\\" in path:
            # Fix any remaining backslashes
            entry["image_path"] = path.replace("\\", "/")
            fixed_count += 1
    
    with open(annotations_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Fixed {fixed_count}/{len(data)} paths in {annotations_path}")
    print(f"Sample paths after fix:")
    for entry in data[:5]:
        print(f"  {entry['image_path']}")

if __name__ == "__main__":
    fix_annotations()
