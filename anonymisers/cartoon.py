import cv2
import numpy as np
from .base_anon import BaseAnonymiser
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def indices_from_connections(connections):
    """Return sorted unique landmark indices from a set of connections."""
    idx = set()
    for a, b in connections:
        idx.add(a); idx.add(b)
    return sorted(idx)


# ---------- Feature sets from MediaPipe Face Mesh ----------
FEATURE_CONNECTIONS = {
    "lips": mp_face_mesh.FACEMESH_LIPS,
    "left_eye": mp_face_mesh.FACEMESH_LEFT_EYE,
    "right_eye": mp_face_mesh.FACEMESH_RIGHT_EYE,
    "left_eyebrow": mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    "right_eyebrow": mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    "face_oval": mp_face_mesh.FACEMESH_FACE_OVAL,
}
# irises are available in newer versions
FEATURE_CONNECTIONS["irises"] = getattr(mp_face_mesh, "FACEMESH_IRISES", set())

FEATURE_INDICES = {k: indices_from_connections(v) for k, v in FEATURE_CONNECTIONS.items()}

class CartoonAnonymiser(BaseAnonymiser):
    def apply(self, frame, faces):
        """
        Apply cartoon anonymisation ONLY when full MediaPipe
        face_landmarks objects are provided.

        If landmarks come from YOLO or from NumPy, this method
        will raise a clear error to prevent silent failures.
        """

        out = frame.copy()

        for f in faces:
            lm = f.get("landmarks", None)

            # No landmarks at all,  NOT MediaPipe
            if lm is None:
                raise ValueError(
                    "CartoonAnonymiser: No landmarks provided. "
                    "This anonymiser works ONLY with MediaPipe FaceMesh detector."
                )

            # Case 1: Correct MediaPipe face_landmarks object
            if hasattr(lm, "landmark"):
                draw_face_landmarks_filled(out, lm)
                continue

            # Case 2: NumPy array (from mp_mesh_detector or YOLO) → ERROR
            if isinstance(lm, np.ndarray):
                raise TypeError(
                    "CartoonAnonymiser: Received NumPy landmark array. "
                    "Cartoon anonymisation requires MediaPipe face_landmarks objects. "
                    "Use MediaPipeMeshDetector() instead of YOLOFaceDetector() "
                    "for cartoon anonymisation."
                )

            # Unknown format → ERROR
            raise TypeError(
                f"CartoonAnonymiser: Unsupported landmark format: {type(lm)}. "
                "This anonymiser only accepts MediaPipe face_landmarks objects."
            )

        return out

# ---------- Draw helpers ----------
def draw_face_landmarks(image_bgr, face_landmarks):
    mp_drawing.draw_landmarks(
        image=image_bgr,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=image_bgr,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    if hasattr(mp_face_mesh, "FACEMESH_IRISES"):
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )


def draw_face_landmarks_filled(image_bgr, face_landmarks):
    """
    Draw opaque, solid-colored filled polygons for all main face regions:
    full face skin, lips, eyes, eyebrows, irises.
    """
    h, w, _ = image_bgr.shape
    points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], np.int32)

    # Distinct colors (B, G, R)
    region_colors = {
        "face_skin": (180, 180, 230),     # light skin tone
        "lips": (0, 0, 255),              # red
        "left_eye": (0, 255, 0),          # green
        "right_eye": (0, 255, 0),         # green
        "left_eyebrow": (255, 200, 0),    # light blue/cyan
        "right_eyebrow": (255, 200, 0),
        "irises": (255, 255, 0),          # yellow
    }

    # Fill full face region using tessellation points
    tess_points = np.array(
        [[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], np.int32
    )
    if len(tess_points) > 3:
        hull = cv2.convexHull(tess_points)
        cv2.fillPoly(image_bgr, [hull], region_colors["face_skin"])
        cv2.polylines(image_bgr, [hull], isClosed=True, color=(0, 0, 0), thickness=1)

    # Draw individual feature regions (on top of face fill)
    for region_name, idxs in FEATURE_INDICES.items():
        if not idxs:
            continue
        region_pts = points[idxs]
        if len(region_pts) < 3:
            continue

        hull = cv2.convexHull(region_pts)
        color = region_colors.get(region_name, None)
        if color is not None:
            cv2.fillPoly(image_bgr, [hull], color)
            cv2.polylines(image_bgr, [hull], isClosed=True, color=(0, 0, 0), thickness=1)


def collect_feature_points(face_landmarks, w, h, feature_key):
    idxs = FEATURE_INDICES.get(feature_key, [])
    pts = []
    for i in idxs:
        lm = face_landmarks.landmark[i]
        pts.append((lm.x * w, lm.y * h))
    return pts


