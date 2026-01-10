import numpy as np
import cv2


# ---- Face landmark indices (MediaPipe FaceMesh topology) ----
# These are "vertex indices" used to pick landmark points and draw convex hulls.
# Works with Tasks FaceLandmarker outputs (typically 468 or 478 landmarks).
FEATURE_INDICES = {
    # Face oval (approx boundary)
    "face_oval": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ],

    # Lips (outer + inner; robust enough for a filled mouth region)
    "lips": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        191, 80, 81, 82, 13, 312, 311, 310, 415, 308
    ],

    # Eyes (regions)
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ],
    "right_eye": [
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466
    ],

    # Eyebrows
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],

    # Irises (only present in models that output 478 landmarks)
    # Left iris indices 468..472, Right iris indices 473..477
    "irises": [468, 469, 470, 471, 472, 473, 474, 475, 476, 477],
}


def _safe_region_points(points: np.ndarray, idxs: list[int]) -> np.ndarray | None:
    """Return points for valid indices only; None if insufficient points."""
    n = len(points)
    valid = [i for i in idxs if 0 <= i < n]
    if len(valid) < 3:
        return None
    return points[valid]


class CartoonAnonymiser:
    """
    Tasks-only cartoon anonymiser.
    Expects faces entries to contain:
      f["landmarks"] = list of NormalizedLandmark with .x/.y in [0,1]
    """

    def apply(self, frame, faces):
        out = frame.copy()
        h, w = out.shape[:2]

        for f in faces:
            lms = f.get("landmarks", None)

            # Enforce Tasks FaceLandmarker format
            if lms is None:
                raise ValueError(
                    "CartoonAnonymiser (Tasks): No landmarks provided. "
                    "Use MediaPipe Tasks FaceLandmarker detector for cartoonisation."
                )
            if not isinstance(lms, (list, tuple)) or len(lms) == 0 or not hasattr(lms[0], "x"):
                raise TypeError(
                    "CartoonAnonymiser (Tasks): Unsupported landmarks format. "
                    "Expected list of NormalizedLandmark with .x/.y fields."
                )

            # Convert normalized landmarks -> pixel points
            points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms], dtype=np.int32)

            self._draw_cartoon_regions(out, points)

        return out


    def _draw_cartoon_regions(self, image_bgr, points: np.ndarray):
        # Region colors (BGR)
        region_colors = {
            "face_skin": (180, 180, 230),
            "lips": (0, 0, 255),
            "left_eye": (0, 255, 0),
            "right_eye": (0, 255, 0),
            "left_eyebrow": (255, 200, 0),
            "right_eyebrow": (255, 200, 0),
            "irises": (255, 255, 0),
        }

        # ---- helpers ----
        def _fill_hull(idxs, fill_color, outline=True):
            region_pts = _safe_region_points(points, idxs)
            if region_pts is None:
                return
            hull = cv2.convexHull(region_pts)
            cv2.fillPoly(image_bgr, [hull], fill_color)
            if outline:
                cv2.polylines(image_bgr, [hull], isClosed=True, color=(0, 0, 0), thickness=1)

        def _fill_contour(idxs, fill_color, outline=True):
            """
            Fill using the landmark order as a closed contour (avoids convex hull bridging).
            """
            region_pts = _safe_region_points(points, idxs)
            if region_pts is None:
                return
            contour = region_pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(image_bgr, [contour], fill_color)
            if outline:
                cv2.polylines(image_bgr, [contour], isClosed=True, color=(0, 0, 0), thickness=1)

        # 1) Face skin (convex hull is fine)
        oval_pts = _safe_region_points(points, FEATURE_INDICES["face_oval"])
        base_pts = oval_pts if oval_pts is not None else points
        if len(base_pts) >= 3:
            hull = cv2.convexHull(base_pts)
            cv2.fillPoly(image_bgr, [hull], region_colors["face_skin"])
            cv2.polylines(image_bgr, [hull], isClosed=True, color=(0, 0, 0), thickness=1)

        # 2) Lips (hull is OK, but contour also works; keep hull for robustness)
        _fill_hull(FEATURE_INDICES["lips"], region_colors["lips"], outline=True)

        # 3) Eyebrows (hull OK)
        _fill_hull(FEATURE_INDICES["left_eyebrow"], region_colors["left_eyebrow"], outline=True)
        _fill_hull(FEATURE_INDICES["right_eyebrow"], region_colors["right_eyebrow"], outline=True)

        # 4) Eyes: use contour (NOT convex hull) to prevent the nose-bridge bar
        _fill_contour(FEATURE_INDICES["left_eye"], region_colors["left_eye"], outline=True)
        _fill_contour(FEATURE_INDICES["right_eye"], region_colors["right_eye"], outline=True)

        # 5) Irises: draw as circles if present (looks cleaner than hull/contour)
        iris_idxs = FEATURE_INDICES.get("irises", [])
        if iris_idxs:
            iris_pts = _safe_region_points(points, iris_idxs)
            if iris_pts is not None and iris_pts.shape[0] >= 10:
                # left iris (468..472) and right iris (473..477) for 478-landmark models
                left = _safe_region_points(points, [468, 469, 470, 471, 472])
                right = _safe_region_points(points, [473, 474, 475, 476, 477])

                for side in (left, right):
                    if side is None:
                        continue
                    cx = int(np.mean(side[:, 0]))
                    cy = int(np.mean(side[:, 1]))
                    # radius based on spread
                    dists = np.linalg.norm(
                        (side.astype(np.float32) - np.array([cx, cy], dtype=np.float32)),
                        axis=1
                    )
                    r = int(max(2, np.mean(dists)))
                    cv2.circle(image_bgr, (cx, cy), r, region_colors["irises"], -1)
                    cv2.circle(image_bgr, (cx, cy), r, (0, 0, 0), 1)

