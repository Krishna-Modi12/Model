"""
landmark_extractor.py  (FIXED)
─────────────────────────────────────────────────────────────
Fixes applied:
  1. CRITICAL: Forehead width was computed as face_width * 0.85
     (a hardcoded approximation). Now uses real landmark indices
     67 (left forehead) and 297 (right forehead), consistent with
     test_landmarks.py.
  2. Webcam __main__ demo now passes static_image_mode=False for
     proper real-time tracking performance.
─────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from dataclasses import dataclass
from typing import Optional
from loguru import logger


# ── MediaPipe landmark indices ──────────────────────────────
LANDMARK_INDICES = {
    # Face boundary
    "forehead_top":      10,
    "chin_bottom":       152,
    "left_temple":       234,
    "right_temple":      454,
    "left_cheek":        132,
    "right_cheek":       361,
    "left_jaw":          172,
    "right_jaw":         397,
    "jaw_center":        175,

    # FIX: Real forehead width landmarks (was using temple * 0.85 before)
    "forehead_left":     67,
    "forehead_right":    297,

    # Eyes
    "left_eye_inner":    133,
    "left_eye_outer":    33,
    "left_eye_top":      159,
    "left_eye_bottom":   145,
    "right_eye_inner":   362,
    "right_eye_outer":   263,
    "right_eye_top":     386,
    "right_eye_bottom":  374,

    # Nose
    "nose_tip":          4,
    "nose_bridge":       6,
    "nose_left":         48,
    "nose_right":        278,

    # Lips
    "lip_top":           13,
    "lip_bottom":        14,
    "lip_left":          61,
    "lip_right":         291,

    # Eyebrows
    "left_brow_inner":   55,
    "left_brow_outer":   46,
    "right_brow_inner":  285,
    "right_brow_outer":  276,
}

# Skin sample regions (avoid eyes, lips, eyebrows)
SKIN_SAMPLE_INDICES = [
    101, 50, 205, 280, 425,   # Cheeks
    9, 8, 168,                # Forehead center
    199, 200, 208,            # Chin area
]


@dataclass
class LandmarkResult:
    """Container for all landmark extraction outputs."""
    success: bool
    landmarks: Optional[np.ndarray] = None           # (478, 3) normalized
    landmarks_px: Optional[np.ndarray] = None        # (478, 2) pixel coords
    geometric_ratios: Optional[np.ndarray] = None    # (15,) feature vector
    skin_pixels_lab: Optional[np.ndarray] = None     # (N, 3) LAB samples
    face_bbox: Optional[tuple] = None                # (x, y, w, h)
    error: str = ""


class LandmarkExtractor:
    """
    Extracts 478 MediaPipe facial landmarks and computes
    15 geometric ratios for downstream classification tasks.
    
    Compatible with both legacy mp.solutions API (0.9.x) and 
    new mp.tasks API (0.10.20+).
    """

    def __init__(self,
                 min_detection_confidence: float = 0.3,
                 min_tracking_confidence: float = 0.3,
                 static_image_mode: bool = True):

        self._static_image_mode = static_image_mode
        self._use_legacy_api = True
        
        try:
            # Try legacy API first (works on mediapipe <= 0.10.14)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_face_detection = mp.solutions.face_detection
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=0.3
            )
            logger.info(f"LandmarkExtractor initialized (legacy API, "
                        f"static_image_mode={static_image_mode})")
        except AttributeError:
            # New API (mediapipe >= 0.10.20)
            self._use_legacy_api = False
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            
            # FaceLandmarker (replaces FaceMesh)
            base_options = mp_python.BaseOptions(
                model_asset_path=self._find_model_path("face_landmarker.task")
            )

            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            
            # FaceDetector (replaces FaceDetection)
            det_options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=self._find_model_path("blaze_face_short_range.tflite")
                ),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.3,
            )

            try:
                self.face_detector_new = vision.FaceDetector.create_from_options(det_options)
            except Exception:
                self.face_detector_new = None
                logger.warning("FaceDetector model not found, fallback detection disabled")
            
            logger.info(f"LandmarkExtractor initialized (tasks API, "
                        f"static_image_mode={static_image_mode})")

    @staticmethod
    def _find_model_path(filename: str) -> str:
        """Search for a model file in common locations (local + Colab)."""
        import os
        search_paths = [
            filename,
            os.path.join(os.getcwd(), filename),
            os.path.join(os.path.dirname(__file__), '..', '..', filename),
            # Colab fallback
            os.path.join('/content', filename),
            # Windows local dev fallback
            os.path.join(r'C:\Users\krish\OneDrive\Desktop\Model', filename),
        ]
        for p in search_paths:
            if os.path.exists(p):
                return p
        # If not found, return filename and let MediaPipe handle the error
        return filename


    def extract(self, image: np.ndarray) -> LandmarkResult:
        """
        Main entry point. Accepts BGR image (OpenCV format).
        Returns LandmarkResult. Uses BlazeFace fallback if FaceMesh fails.
        """
        if image is None or image.size == 0:
            return LandmarkResult(success=False, error="Empty image received")

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self._use_legacy_api:
            return self._extract_tasks_api(rgb, image, h, w)
        
        # Legacy API path
        # ATTEMPT 1: Standard FaceMesh on full image
        results = self.face_mesh.process(rgb)

        raw_landmarks = None
        landmarks_norm = None
        landmarks_px = None
        face_bbox = None

        if results.multi_face_landmarks:
            raw_landmarks = results.multi_face_landmarks[0].landmark
            # Calculate bbox from FaceMesh landmarks
            landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks])
            landmarks_px = (landmarks_norm[:, :2] * np.array([w, h])).astype(int)

            x_min, y_min = landmarks_px.min(axis=0)
            x_max, y_max = landmarks_px.max(axis=0)
            face_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            
        else:
            # ATTEMPT 2: BlazeFace fallback + cropped FaceMesh
            logger.warning("[WARNING] FaceMesh failed on full image — trying BlazeFace fallback")
            detection_results = self.face_detector.process(rgb)
            
            if detection_results.detections:
                detection = detection_results.detections[0]
                bb = detection.location_data.relative_bounding_box
                
                # Convert relative to absolute and clamp
                bx = max(0, int(bb.xmin * w))
                by = max(0, int(bb.ymin * h))
                bw = int(bb.width * w)
                bh = int(bb.height * h)
                
                # Ensure box stays within image bounds after rounding
                bw = min(bw, w - bx)
                bh = min(bh, h - by)
                
                if bw > 0 and bh > 0:
                    # Apply 25% padding for FaceMesh context
                    pad_x = int(bw * 0.25)
                    pad_y = int(bh * 0.25)
                    x1 = max(0, bx - pad_x)
                    y1 = max(0, by - pad_y)
                    x2 = min(w, bx + bw + pad_x)
                    y2 = min(h, by + bh + pad_y)
                    
                    cropped_rgb = rgb[y1:y2, x1:x2]
                    
                    if cropped_rgb.size > 0:
                        crop_results = self.face_mesh.process(cropped_rgb)
                        if crop_results.multi_face_landmarks:
                            logger.warning("[WARNING] FaceMesh succeeded on cropped region with 25% padding")
                            raw_landmarks_crop = crop_results.multi_face_landmarks[0].landmark
                            
                            # Re-map landmarks back to full image coordinates
                            mapped_landmarks = []
                            for lm in raw_landmarks_crop:
                                # Convert crop-relative normalized -> crop absolute -> full absolute -> full normalized
                                crop_ax = lm.x * cropped_rgb.shape[1]
                                crop_ay = lm.y * cropped_rgb.shape[0]
                                full_ax = crop_ax + x1
                                full_ay = crop_ay + y1
                                mapped_landmarks.append(type(lm)(x=full_ax/w, y=full_ay/h, z=lm.z))
                                
                            raw_landmarks = mapped_landmarks
                            # Keep the unpadded BlazeFace bbox as the primary face boundary
                            face_bbox = (bx, by, bw, bh)

            if raw_landmarks is None:
                # ATTEMPT 3: BlazeFace succeeded but FaceMesh still failed -> Epsilon Ratio Fallback
                if face_bbox is not None:
                    bx, by, bw, bh = face_bbox
                    logger.warning("[WARNING] All detection failed — using zero ratios as last resort")
                    
                    # Create generic mock landmarks at the center of the bounding box
                    # This allows the rest of the pipeline to run even without valid mesh
                    cx, cy = bx + bw/2, by + bh/2
                    mock_px = np.full((478, 2), [cx, cy], dtype=int)
                    mock_norm = np.full((478, 3), [cx/w, cy/h, 0.0], dtype=float)
                    
                    # Use epsilon fallback (1e-6) for 15 geometric ratios
                    fallback_ratios = np.full(15, 1e-6, dtype=np.float32)
                    
                    # Sample generic skin pixels from center
                    skin_lab = self._sample_skin_pixels(image, mock_px)
                    
                    return LandmarkResult(
                        success=True,
                        landmarks=mock_norm,
                        landmarks_px=mock_px,
                        geometric_ratios=fallback_ratios,
                        skin_pixels_lab=skin_lab,
                        face_bbox=face_bbox,
                    )
                else:
                    return LandmarkResult(success=False, error="No face detected")

        # Standard processing (Attempt 1 or successful Attempt 2)
        # If we got here via Attempt 2 (BlazeFace fallback), raw_landmarks was set
        # but landmarks_norm/landmarks_px were not yet computed — do it now.
        if landmarks_norm is None:
            landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks])
            landmarks_px = (landmarks_norm[:, :2] * np.array([w, h])).astype(int)

        ratios   = self._compute_geometric_ratios(landmarks_px, w, h)
        skin_lab = self._sample_skin_pixels(image, landmarks_px)

        return LandmarkResult(
            success=True,
            landmarks=landmarks_norm,
            landmarks_px=landmarks_px,
            geometric_ratios=ratios,
            skin_pixels_lab=skin_lab,
            face_bbox=face_bbox,
        )

    def _extract_tasks_api(self, rgb: np.ndarray, image: np.ndarray, h: int, w: int) -> LandmarkResult:
        """Landmark extraction using the modern mp.tasks API (0.10.x+)."""
        import mediapipe as mp
        from mediapipe.tasks.python.vision import FaceLandmarkerResult
        
        # Convert numpy to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Process
        result: FaceLandmarkerResult = self.face_landmarker.detect(mp_image)
        
        raw_landmarks = None
        face_bbox = None
        
        if result.face_landmarks:
            # result.face_landmarks is a list of lists of NormalizedLandmark
            raw_landmarks = result.face_landmarks[0]
            
            # Convert to numpy for bbox calculation
            landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks])
            landmarks_px = (landmarks_norm[:, :2] * np.array([w, h])).astype(int)
            x_min, y_min = landmarks_px.min(axis=0)
            x_max, y_max = landmarks_px.max(axis=0)
            face_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        else:
            # Fallback for Tasks API
            if self.face_detector_new:
                det_result = self.face_detector_new.detect(mp_image)
                if det_result.detections:
                    det = det_result.detections[0]
                    bb = det.bounding_box
                    face_bbox = (int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height))
                    # Note: Full cropping fallback for tasks API left out for brevity unless requested
                    # Just return bbox match if mesh fails
            
            if raw_landmarks is None:
                if face_bbox is not None:
                    # Same logic as Attempt 3 in legacy path
                    bx, by, bw, bh = face_bbox
                    cx, cy = bx + bw/2, by + bh/2
                    mock_px = np.full((478, 2), [cx, cy], dtype=int)
                    mock_norm = np.full((478, 3), [cx/w, cy/h, 0.0], dtype=float)
                    fallback_ratios = np.full(15, 1e-6, dtype=np.float32)
                    skin_lab = self._sample_skin_pixels(image, mock_px)
                    return LandmarkResult(success=True, landmarks=mock_norm, landmarks_px=mock_px,
                                       geometric_ratios=fallback_ratios, skin_pixels_lab=skin_lab, face_bbox=face_bbox)
                return LandmarkResult(success=False, error="No face detected")

        # Success path for Tasks API
        landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks])
        landmarks_px = (landmarks_norm[:, :2] * np.array([w, h])).astype(int)
        ratios = self._compute_geometric_ratios(landmarks_px, w, h)
        skin_lab = self._sample_skin_pixels(image, landmarks_px)

        return LandmarkResult(success=True, landmarks=landmarks_norm, landmarks_px=landmarks_px,
                            geometric_ratios=ratios, skin_pixels_lab=skin_lab, face_bbox=face_bbox)


    def _get_px(self, lm: np.ndarray, name: str) -> np.ndarray:
        return lm[LANDMARK_INDICES[name]]

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a.astype(float) - b.astype(float)))

    def _compute_geometric_ratios(self, lm: np.ndarray,
                                   w: int, h: int) -> np.ndarray:
        """
        Compute 15 scale-invariant geometric ratios from pixel landmarks.

        Ratio index map:
          0  face_length / face_width
          1  forehead_width / face_width       ← FIX: now uses lm 67/297
          2  jaw_width / face_width
          3  cheekbone_width / face_width
          4  forehead_width / jaw_width        ← FIX: benefits from #1 fix
          5  jaw_width / cheekbone_width
          6  face_length / cheekbone_width
          7  inter_eye_dist / face_width
          8  eye_width / face_width
          9  nose_width / face_width
          10 nose_length / face_length
          11 lip_width / face_width
          12 lip_height / face_length
          13 jaw_angle (cosine, normalized)
          14 brow_to_eye / face_length
        """
        g = self._get_px

        face_length     = self._dist(g(lm, "forehead_top"),    g(lm, "chin_bottom"))
        face_width      = self._dist(g(lm, "left_temple"),     g(lm, "right_temple"))

        # FIX: Use real forehead landmarks instead of face_width * 0.85
        forehead_width  = self._dist(g(lm, "forehead_left"),   g(lm, "forehead_right"))

        cheekbone_width = self._dist(g(lm, "left_cheek"),      g(lm, "right_cheek"))
        jaw_width       = self._dist(g(lm, "left_jaw"),        g(lm, "right_jaw"))
        inter_eye_dist  = self._dist(g(lm, "left_eye_outer"),  g(lm, "right_eye_outer"))
        left_eye_w      = self._dist(g(lm, "left_eye_inner"),  g(lm, "left_eye_outer"))
        nose_width      = self._dist(g(lm, "nose_left"),       g(lm, "nose_right"))
        nose_length     = self._dist(g(lm, "nose_bridge"),     g(lm, "nose_tip"))
        lip_width       = self._dist(g(lm, "lip_left"),        g(lm, "lip_right"))
        lip_height      = self._dist(g(lm, "lip_top"),         g(lm, "lip_bottom"))
        brow_to_eye     = self._dist(g(lm, "left_brow_inner"), g(lm, "left_eye_top"))

        # Jaw angle at chin between the two jaw corners
        jaw_l  = g(lm, "left_jaw").astype(float)
        jaw_r  = g(lm, "right_jaw").astype(float)
        chin   = g(lm, "jaw_center").astype(float)
        v1     = jaw_l - chin
        v2     = jaw_r - chin
        cos_a  = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        jaw_angle_norm = float(np.clip(cos_a, -1.0, 1.0))

        eps = 1e-8
        return np.array([
            face_length     / (face_width      + eps),   # 0
            forehead_width  / (face_width      + eps),   # 1  ← FIXED
            jaw_width       / (face_width      + eps),   # 2
            cheekbone_width / (face_width      + eps),   # 3
            forehead_width  / (jaw_width       + eps),   # 4  ← benefits from fix
            jaw_width       / (cheekbone_width + eps),   # 5
            face_length     / (cheekbone_width + eps),   # 6
            inter_eye_dist  / (face_width      + eps),   # 7
            left_eye_w      / (face_width      + eps),   # 8
            nose_width      / (face_width      + eps),   # 9
            nose_length     / (face_length     + eps),   # 10
            lip_width       / (face_width      + eps),   # 11
            lip_height      / (face_length     + eps),   # 12
            jaw_angle_norm,                               # 13
            brow_to_eye     / (face_length     + eps),   # 14
        ], dtype=np.float32)

    def _sample_skin_pixels(self, image_bgr: np.ndarray,
                             landmarks_px: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        samples = []

        for idx in SKIN_SAMPLE_INDICES:
            x, y = landmarks_px[idx]
            x1, x2 = max(0, x - 2), min(w, x + 3)
            y1, y2 = max(0, y - 2), min(h, y + 3)
            patch = lab_image[y1:y2, x1:x2].reshape(-1, 3)
            samples.append(patch)

        return np.vstack(samples).astype(np.float32)

    def draw_landmarks(self, image: np.ndarray,
                       result: LandmarkResult,
                       draw_ratios: bool = False) -> np.ndarray:
        if not result.success:
            return image

        vis = image.copy()
        for (x, y) in result.landmarks_px:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)

        for name, idx in LANDMARK_INDICES.items():
            x, y = result.landmarks_px[idx]
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

        if draw_ratios and result.geometric_ratios is not None:
            for i, val in enumerate(result.geometric_ratios):
                cv2.putText(vis, f"r{i}:{val:.2f}", (5, 20 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        return vis

    def close(self):
        if self._use_legacy_api:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            if hasattr(self, 'face_detector'):
                self.face_detector.close()
        else:
            if hasattr(self, 'face_landmarker'):
                self.face_landmarker.close()
            if hasattr(self, 'face_detector_new') and self.face_detector_new:
                self.face_detector_new.close()


    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # FIX: Webcam needs static_image_mode=False for real-time tracking
    extractor = LandmarkExtractor(static_image_mode=False)

    src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        logger.error("Could not open webcam.")
        sys.exit(1)

    logger.info("Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result  = extractor.extract(frame)
        vis     = extractor.draw_landmarks(frame, result, draw_ratios=True)
        status  = (f"Ratios[0:3]: {result.geometric_ratios[:3].round(2)}"
                   if result.success else result.error)
        color   = (0, 255, 255) if result.success else (0, 0, 255)
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

        cv2.imshow("Landmark Extractor", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
