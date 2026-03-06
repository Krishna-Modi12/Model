import json
from collections import Counter
import sys

def main():
    json_path = 'data/processed/annotations_multitask_balanced.json'
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        anns = json.load(f)

    print("Fixing labels...")
    for a in anns:
        if a.get('monk_label') == 1: a['monk_label'] = 0
        elif a.get('monk_label') == 4: a['monk_label'] = 1
        elif a.get('monk_label') == 7: a['monk_label'] = 2

    print("Saving...")
    with open(json_path, 'w') as f:
        json.dump(anns, f)

    with_skin = [a for a in anns if a.get('monk_label') is not None]
    skin_vals = [a['monk_label'] for a in with_skin]
    
    with_attrs    = [a for a in anns if a.get("attributes") is not None and any(a["attributes"].get(k) is not None and a["attributes"].get(k) != -1 for k in ["eye_narrow", "eye_big", "brow", "lip", "age", "gender"])]
    with_landmark = [a for a in anns if a.get("attributes") is not None and a["attributes"].get("landmark_ratios") is not None]
    
    print("================================================")
    print("ANNOTATION REBUILD VERIFICATION")
    print("================================================")
    print(f"Total annotations    : {len(anns)}")
    print(f"With attributes      : {len(with_attrs)}")
    print(f"With landmark_ratios : {len(with_landmark)}")
    print(f"With monk_label      : {len(with_skin)}")
    print(f"Skin distribution    : {Counter(skin_vals)}")
    print(f"Skin unique values   : {sorted(set(skin_vals))}")
    print("================================================")
    
    if sorted(set(skin_vals)) != [0, 1, 2]:
        print("[CRITICAL] Skin labels still wrong — check Fix 1")
        sys.exit(1)
        
    if len(with_attrs) < 5000:
        print("[CRITICAL] Still too few attribute samples")
        print("  has_attributes decoupling may not have worked")
        sys.exit(1)
        
    print("All annotation checks PASSED ✅")

if __name__ == '__main__':
    main()
