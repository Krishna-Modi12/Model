import cv2
import numpy as np
import mediapipe as mp
import os

# New MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class FaceLandmarkProcessor:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        # We need the local model asset for the tasks API
        model_path = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please download it.")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        
    def get_landmarks(self, image_rgb):
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the image
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        # Get the first face detected
        face_landmarks = detection_result.face_landmarks[0]
        
        # Convert NormalizedLandmark objects to numpy array
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks
        ])
        return landmarks

    @staticmethod
    def get_eye_centers(landmarks, img_w, img_h):
        if landmarks is None or len(landmarks) < 478:
            return (img_w//3, img_h//3), (2*img_w//3, img_h//3)
            
        # MediaPipe iris landmarks remain the same: 468, 473
        left_eye = (int(landmarks[468][0] * img_w), int(landmarks[468][1] * img_h))
        right_eye = (int(landmarks[473][0] * img_w), int(landmarks[473][1] * img_h))
        return left_eye, right_eye

    @staticmethod
    def get_bounding_box(landmarks, img_w, img_h, padding=0.2):
        if landmarks is None:
            return [0, 0, img_w, img_h]
            
        x_coords = landmarks[:, 0] * img_w
        y_coords = landmarks[:, 1] * img_h
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        w = x_max - x_min
        h = y_max - y_min
        
        x_min -= w * padding
        y_min -= h * padding
        x_max += w * padding
        y_max += h * padding
        
        return [int(max(0, x_min)), int(max(0, y_min)), int(min(img_w, x_max)), int(min(img_h, y_max))]
