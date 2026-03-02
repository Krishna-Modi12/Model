"""
Create a FULL zip for Google Colab that includes code + image data.
This produces a larger file (~1GB) but is ready to train immediately.
"""
import zipfile
import os
from pathlib import Path

def zip_full_project(output_filename="FaceShapeAI_Colab.zip"):
    root_dir = Path(os.getcwd())
    
    # Only exclude truly unwanted root-level directories
    exclude_root_dirs = {
        '.git', 'venv', 'checkpoints', 'logs', 'node_modules', '.ipynb_checkpoints',
        'ffhq-dataset', 'fitzpatrick17k', 'design_previews', 'face_analysis_fixed',
        'faceshape_extra', 'exported'
    }
    exclude_extensions = {'.zip', '.pyc', '.pyo', '.tmp', '.pt', '.onnx'}
    exclude_files = {'.DS_Store', '.gitignore'}

    print(f"📦 Creating {output_filename} (this will take a few minutes)...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        count = 0
        for root, dirs, files in os.walk(root_dir):
            rel_path = Path(root).relative_to(root_dir)
            
            # Skip excluded ROOT-LEVEL directories only
            if rel_path.parts and rel_path.parts[0] in exclude_root_dirs:
                continue
            
            # Skip __pycache__ at any level
            if '__pycache__' in rel_path.parts:
                continue

            for file in files:
                if any(file.endswith(ext) for ext in exclude_extensions):
                    continue
                if file in exclude_files:
                    continue
                
                full_path = Path(root) / file
                zipf.write(full_path, rel_path / file)
                count += 1
                if count % 5000 == 0:
                    print(f"  Added {count} files...")

    size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"✅ Created {output_filename} ({count} files, {size_mb:.1f} MB)")

if __name__ == "__main__":
    zip_full_project()
