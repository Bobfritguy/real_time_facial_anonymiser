import cv2
from .base_anon import BaseAnonymiser

class CartoonAnonymiser(BaseAnonymiser):
    def apply(self, frame, faces):
        out = frame.copy()
        for f in faces:
            if f["landmarks"] is not None:
                draw_face_landmarks_filled(out, f["landmarks"])
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


