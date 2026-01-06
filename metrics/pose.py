# metrics/pose.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import math


# FaceMesh landmark indices we will use as 2D points (MediaPipe topology).
# These are commonly used stable points:
# nose tip (1), chin (152), left eye outer (33), right eye outer (263),
# left mouth corner (61), right mouth corner (291)
POSE_IDXS = [1, 152, 33, 263, 61, 291]


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _std(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    m = _mean(xs)
    return float((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)


def _rvec_to_euler_degrees(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation vector to Euler angles in degrees (yaw, pitch, roll).
    We use a standard decomposition from rotation matrix.
    """
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])      # roll
        y = math.atan2(-R[2, 0], sy)          # pitch
        z = math.atan2(R[1, 0], R[0, 0])      # yaw
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    roll = math.degrees(x)
    pitch = math.degrees(y)
    yaw = math.degrees(z)
    return yaw, pitch, roll


def estimate_head_pose_solvepnp(
    landmarks: Sequence,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[float, float, float]]:
    """
    Estimate (yaw, pitch, roll) from MediaPipe-style landmarks (list with .x,.y normalized)
    using solvePnP with a simple canonical 3D face model.

    Returns (yaw, pitch, roll) in degrees, or None if pose can't be estimated.
    """
    if landmarks is None:
        return None
    if len(landmarks) <= max(POSE_IDXS):
        return None

    # 2D image points from landmarks
    image_points = np.array(
        [[landmarks[i].x * image_w, landmarks[i].y * image_h] for i in POSE_IDXS],
        dtype=np.float64,
    )

    # Canonical 3D model points (approx, in arbitrary units).
    # This does NOT need to be perfect because you only care about relative consistency
    # between clear vs anonymised.
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye outer corner
            (225.0, 170.0, -135.0),  # Right eye outer corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0), # Right mouth corner
        ],
        dtype=np.float64,
    )

    # Camera matrix approximation
    focal_length = float(image_w)
    center = (image_w / 2.0, image_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    yaw, pitch, roll = _rvec_to_euler_degrees(rvec)
    return yaw, pitch, roll


@dataclass
class PoseMAEMetric:
    """
    Accumulates absolute errors between clear vs anonymised head pose estimates.
    """
    yaw_abs: List[float]
    pitch_abs: List[float]
    roll_abs: List[float]
    n: int

    def __init__(self):
        self.yaw_abs = []
        self.pitch_abs = []
        self.roll_abs = []
        self.n = 0

    def update(self, pose_clear: Optional[Tuple[float, float, float]],
               pose_anon: Optional[Tuple[float, float, float]]):
        if pose_clear is None or pose_anon is None:
            return
        yc, pc, rc = pose_clear
        ya, pa, ra = pose_anon

        self.yaw_abs.append(abs(yc - ya))
        self.pitch_abs.append(abs(pc - pa))
        self.roll_abs.append(abs(rc - ra))
        self.n += 1

    def finalise(self) -> Dict[str, Any]:
        return {
            "pose_pairs": self.n,
            "yaw_mae_deg": _mean(self.yaw_abs),
            "pitch_mae_deg": _mean(self.pitch_abs),
            "roll_mae_deg": _mean(self.roll_abs),
            "yaw_mae_std_deg": _std(self.yaw_abs),
            "pitch_mae_std_deg": _std(self.pitch_abs),
            "roll_mae_std_deg": _std(self.roll_abs),
        }
