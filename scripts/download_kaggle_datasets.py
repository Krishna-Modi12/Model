import subprocess
import os
import sys

# Change directory to root of project if executing from /scripts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = [
    ("niten19/face-shape-dataset",                        os.path.join(BASE_DIR, "data/raw/face_shape")),
    ("lucifierx/face-shape-classification",               os.path.join(BASE_DIR, "data/raw/face_shape_cnn")),
    ("hanakb/men-face-shape",                             os.path.join(BASE_DIR, "data/raw/men_face_shape")),
    ("jangedoo/utkface-new",                              os.path.join(BASE_DIR, "data/raw/utkface")),
    ("moritzm00/utkface-cropped",                         os.path.join(BASE_DIR, "data/raw/utkface_cropped")),
    ("usamarana/skin-tone-classification-dataset",        os.path.join(BASE_DIR, "data/raw/skin_tone")),
    ("osmankagankurnaz/facial-feature-extraction-dataset",os.path.join(BASE_DIR, "data/raw/facial_features")),
]

def main():
    print("="*60)
    print("Initiating kaggle dataset downloads...")
    print("="*60)
    for slug, out_dir in datasets:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Downloading {slug}...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", out_dir, "--unzip"],
                check=True
            )
            print(f"  Done -> {out_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {slug}. Make sure you have your API token configured: ~/.kaggle/kaggle.json")
            sys.exit(1)
            
    print("\nAll downloads complete.")

if __name__ == "__main__":
    main()
