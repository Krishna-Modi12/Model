import os
import cv2
import json
import uuid
import shutil
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import mediapipe as mp
import numpy as np

# Use the existing LandmarkExtractor logic for consistent alignment
from src.utils.landmark_extractor import LandmarkExtractor

def ingest_images(source_dir: str, output_dir: str = "data/images", annotations_path: str = "data/processed/annotations.json"):
    """
    Ingests raw images from a directory, aligns/crops them using MediaPipe,
    and adds them to the dataset registry.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing annotations
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = []
    
    extractor = LandmarkExtractor(static_image_mode=True)
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in source_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    logger.info(f"Found {len(files)} images in {source_dir}")
    
    new_count = 0
    
    for file_path in tqdm(files, desc="Ingesting"):
        try:
            # Read Image
            img = cv2.imread(str(file_path))
            if img is None:
                continue
            
            # Extract Landmarks & Align
            result = extractor.extract(img)
            
            if not result.success:
                logger.warning(f"No face detected in {file_path.name}")
                continue
                
            # Alignment Logic (Simple rotation based on eyes)
            # (Re-using the logic from dataset.py would be ideal, but implementing a simple version here)
            landmarks_px = result.landmarks_px
            left_eye = landmarks_px[33]
            right_eye = landmarks_px[263]
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            aligned = cv2.warpAffine(img, M, (w, h))
            
            # Crop (Basic center crop around face center)
            # In production, use the bounding box from extractor
            bbox = result.face_bbox
            if bbox:
                x, y, bw, bh = bbox
                # Add padding
                pad = 0.2
                x = max(0, int(x - bw*pad))
                y = max(0, int(y - bh*pad))
                bw = int(bw * (1 + 2*pad))
                bh = int(bh * (1 + 2*pad))
                aligned = aligned[y:y+bh, x:x+bw]
            
            if aligned.size == 0: continue
            
            # Resize to standard size (optional, but good for storage)
            aligned = cv2.resize(aligned, (512, 512))
            
            # Save
            unique_name = f"ingest_{uuid.uuid4().hex[:8]}.jpg"
            save_path = output_path / unique_name
            cv2.imwrite(str(save_path), aligned)
            
            # Add to Registry
            entry = {
                "image_path": str(save_path).replace("", "/"),
                "shape_label": -1, # Needs Labeling
                "status": "unlabeled",
                "original_source": file_path.name
            }
            annotations.append(entry)
            new_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            
    # Save updated annotations
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=2)
        
    logger.success(f"Successfully ingested {new_count} new images.")
    logger.info("Please open 'data/processed/annotations.json' and add labels for entries with 'shape_label': -1")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="Directory containing raw images")
    args = parser.parse_args()
    
    ingest_images(args.source_dir)
