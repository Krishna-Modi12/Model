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
    """

    def __init__(self,
                 min_detection_confidence: float = 0.85,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = True):

        self._static_image_mode = static_image_mode
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(f"LandmarkExtractor initialized "
                    f"(static_image_mode={static_image_mode})")

    def extract(self, image: np.ndarray) -> LandmarkResult:
        """
        Main entry point. Accepts BGR image (OpenCV format).
        Returns LandmarkResult.
        """
        if image is None or image.size == 0:
            return LandmarkResult(success=False, error="Empty image received")

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return LandmarkResult(success=False, error="No face detected")

        raw = results.multi_face_landmarks[0].landmark

        landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in raw])
        landmarks_px   = (landmarks_norm[:, :2] * np.array([w, h])).astype(int)

        x_min, y_min = landmarks_px.min(axis=0)
        x_max, y_max = landmarks_px.max(axis=0)
        face_bbox = (int(x_min), int(y_min),
                     int(x_max - x_min), int(y_max - y_min))

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
        self.face_mesh.close()

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
