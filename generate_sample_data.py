import cv2
import numpy as np
import os
import pandas as pd

def generate_sample_images(output_dir="data/images", num_images=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generating {num_images} sample images...")
    
    image_paths = []
    for i in range(num_images):
        # Create a basic colorful face-like silhouette
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img, (256, 256), 180, (200, 200, 200), -1) # Head
        cv2.circle(img, (200, 200), 20, (50, 50, 50), -1)    # Eye
        cv2.circle(img, (312, 200), 20, (50, 50, 50), -1)    # Eye
        cv2.ellipse(img, (256, 350), (60, 20), 0, 0, 180, (0, 0, 255), 2) # Mouth
        
        filename = f"sample_{i}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, img)
        image_paths.append(filename)
        
    return image_paths

def generate_sample_csv(image_paths, output_path="data/train.csv"):
    data = []
    for path in image_paths:
        data.append({
            "image_path": path,
            "face_shape": np.random.randint(0, 7),
            "eye_shape": np.random.randint(0, 6),
            "nose_type": np.random.randint(0, 5),
            "lip_fullness": np.random.randint(0, 3),
            "ita_value": np.random.uniform(-50, 70),
            "fitzpatrick": np.random.randint(0, 6),
            "monk": np.random.randint(1, 11)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample CSV saved to {output_path}")

if __name__ == "__main__":
    if not os.path.exists("data"): os.makedirs("data")
    imgs = generate_sample_images()
    generate_sample_csv(imgs, "data/train.csv")
    generate_sample_csv(imgs[:2], "data/val.csv")
