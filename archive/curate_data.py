import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

from src.landmark_extractor import FaceLandmarkProcessor
from src.skin_tone_analyzer import SkinToneUtility

def compute_blur_score(image):
    """Compute the Laplacian variance to measure blurriness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def create_skin_mask(crop_shape, landmarks, img_w, img_h, x_min, y_min):
    """
    Create a polygon mask covering the face using the convex hull of the landmarks.
    This ensures background pixels don't pollute the skin tone calculation.
    """
    mask = np.zeros(crop_shape[:2], dtype=np.uint8)
    points = []
    for lm in landmarks:
        px = int(lm[0] * img_w) - x_min
        py = int(lm[1] * img_h) - y_min
        points.append([px, py])
    
    if len(points) == 0:
        return None
        
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Exclude eyes and mouth from the skin mask (rough regions)
    # Left eye: 33, Right eye: 263, Mouth: 13, 14
    # For a robust approach, we can just use the hull, as ITA calculation
    # uses the mean, which is somewhat resilient to small non-skin regions.
    return mask

def curate_dataset(input_dir, output_dir, output_csv, blur_threshold=100.0):
    """
    Process raw images, detect faces, filter blurry/invalid images,
    calculate skin tone, and generate a labeling CSV.
    """
    print(f"Initializing data curation pipeline...")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"Blur Threshold: {blur_threshold}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    processor = FaceLandmarkProcessor(min_detection_confidence=0.5)
    skin_utility = SkinToneUtility()
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found.")
        return
        
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(files)} target images. Beginning processing...\n")
    
    results = []
    stats = {
        'processed': 0,
        'blurry': 0,
        'no_face': 0,
        'error': 0
    }
    
    for filename in tqdm(files, desc="Curating Images"):
        img_path = os.path.join(input_dir, filename)
        
        try:
            # 1. Load image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                stats['error'] += 1
                continue
                
            img_h, img_w = img_bgr.shape[:2]
            
            # 2. Blur detection
            blur_score = compute_blur_score(img_bgr)
            if blur_score < blur_threshold:
                stats['blurry'] += 1
                continue
                
            # 3. Detect Landmarks
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            landmarks = processor.get_landmarks(img_rgb)
            
            if landmarks is None or len(landmarks) < 478:
                stats['no_face'] += 1
                continue
                
            # 4. Crop Face
            bbox = processor.get_bounding_box(landmarks, img_w, img_h, padding=0.1)
            x_min, y_min, x_max, y_max = bbox
            
            if x_max <= x_min or y_max <= y_min:
                stats['error'] += 1
                continue
                
            face_crop = img_bgr[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                stats['error'] += 1
                continue
                
            # 5. Calculate Skin Tone (ITA)
            lab_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            skin_mask = create_skin_mask(face_crop.shape, landmarks, img_w, img_h, x_min, y_min)
            
            ita = skin_utility.calculate_ita(lab_crop, mask=skin_mask)
            fitz, monk = skin_utility.get_scales(ita)
            
            # 6. Save curated image
            output_img_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_img_path, img_bgr) # Save the original high-res image, not the crop
            
            # 7. Record metadata
            results.append({
                "image_path": filename,
                "face_shape": -1,        # Placeholder for human label (0-6)
                "eye_shape": -1,         # Placeholder for human label
                "nose_type": -1,         # Placeholder for human label
                "lip_fullness": -1,      # Placeholder for human label
                "ita_value": round(ita, 2),
                "fitzpatrick": fitz,
                "monk": monk,
                "blur_score": round(blur_score, 2)
            })
            
            stats['processed'] += 1
            
        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            stats['error'] += 1
    
    # 8. Generate CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
    print("\n" + "="*40)
    print("📋 CURATION SUMMARY")
    print("="*40)
    print(f"Total Initial Images  : {len(files)}")
    print(f"Successfully Curated  : {stats['processed']}")
    print(f"Rejected (Blurry)     : {stats['blurry']}")
    print(f"Rejected (No Face)    : {stats['no_face']}")
    print(f"Errors                : {stats['error']}")
    print("="*40)
    if results:
        print(f"Labeling template saved to: {output_csv}")
        print("Ready for human annotation phase.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter raw images and generate baseline annotations.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing raw images to curate")
    parser.add_argument("--out_dir", type=str, default="data/images", help="Directory to save accepted images")
    parser.add_argument("--output_csv", type=str, default="data/labels.csv", help="Path to save the generated CSV")
    parser.add_argument("--blur_thresh", type=float, default=100.0, help="Minimum Laplacian variance (higher = sharper)")
    
    args = parser.parse_args()
    
    curate_dataset(args.img_dir, args.out_dir, args.output_csv, args.blur_thresh)
