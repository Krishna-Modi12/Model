import os
import subprocess
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def run_cmd(cmd, cwd=None):
    """Run a shell command and stream output"""
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

def download_ffhq():
    ffhq_dir = os.path.join(RAW_DATA_DIR, "ffhq")
    repo_dir = os.path.join(BASE_DIR, "ffhq-dataset")
    
    # 1. Clone repository if not exists
    if not os.path.exists(repo_dir):
        logger.info("Cloning FFHQ repository...")
        run_cmd(["git", "clone", "https://github.com/NVlabs/ffhq-dataset.git", repo_dir])
    
    # Verify clone succeeded
    if not os.path.exists(os.path.join(repo_dir, "download_ffhq.py")):
        logger.error("FFHQ repository clone failed or missing download_ffhq.py.")
        sys.exit(1)
        
    os.makedirs(ffhq_dir, exist_ok=True)
    
    # 2. Run the download script inside the repo dir
    logger.info("Downloading FFHQ thumbnails...")
    run_cmd([sys.executable, "download_ffhq.py", "--thumbs"], cwd=repo_dir)
    
    # The output will be in the repo's thumbnails folder or similar, 
    # the user may need to move it to data/raw/ffhq/ manually depending on the NVlabs script structure.
    logger.info("FFHQ download script completed. Note: verify output path of thumbnails.")

def download_fitzpatrick17k():
    fitz_repo_dir = os.path.join(BASE_DIR, "fitzpatrick17k")
    output_dir = os.path.join(RAW_DATA_DIR, "fitzpatrick17k", "images")
    
    # 1. Clone repository
    if not os.path.exists(fitz_repo_dir):
        logger.info("Cloning Fitzpatrick17k repository...")
        run_cmd(["git", "clone", "https://github.com/mattgroh/fitzpatrick17k.git", fitz_repo_dir])
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Run the download script inside the repo dir
    if os.path.exists(os.path.join(fitz_repo_dir, "scripts", "download_fitzpatrick_images.py")):
        # The prompt mentioned scripts/download_fitzpatrick_images.py
        script_path = os.path.join("scripts", "download_fitzpatrick_images.py")
    elif os.path.exists(os.path.join(fitz_repo_dir, "download_fitzpatrick_images.py")):
        # Fallback if it's in root
        script_path = "download_fitzpatrick_images.py"
    else:
        logger.error("Could not find download_fitzpatrick_images.py in fitzpatrick17k repo.")
        sys.exit(1)
        
    logger.info(f"Downloading Fitzpatrick17k images to {output_dir}...")
    run_cmd([sys.executable, script_path, "--output", output_dir], cwd=fitz_repo_dir)

if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    print("="*50)
    print("Datasets Download Script")
    print("="*50)
    
    download_ffhq()
    download_fitzpatrick17k()
    
    print("="*50)
    print("Downloads Complete.")
    print("ATTENTION: UTKFace and SCIN datasets must be downloaded manually according to their respective licenses.")
    print("  - UTKFace: https://susanqq.github.io/UTKFace/ -> extract to data/raw/utkface/")
    print("="*50)
