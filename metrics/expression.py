# metrics/expression.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any, List, Tuple
import math
import numpy as np


def _nan() -> float:
    return float("nan")


def _isfinite(x: float) -> bool:
    return x == x and math.isfinite(x)


def _corr(a: List[float], b: List[float]) -> Optional[float]:
    """Pearson correlation, returns None if insufficient or constant."""
    if len(a) < 2 or len(b) < 2:
        return None
    aa = np.array(a, dtype=np.float64)
    bb = np.array(b, dtype=np.float64)

    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 2:
        return None

    # Avoid divide-by-zero when constant
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return None

    return float(np.corrcoef(aa, bb)[0, 1])


def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p, dtype=np.float64) - np.array(q, dtype=np.float64)))


def _lm_xy_px(landmarks: Sequence, idx: int, w: int, h: int) -> Tuple[float, float]:
    lm = landmarks[idx]
    return (float(lm.x) * w, float(lm.y) * h)


# --- Expression proxies (FaceMesh topology indices) ---
# Mouth corners (stable)
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Eye indices for classic EAR calculation (6 points per eye)
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]   # p1,p2,p3,p4,p5,p6
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380] # p1,p2,p3,p4,p5,p6


def smile_ratio(landmarks: Sequence, w: int, h: int) -> float:
    """
    Smile proxy: (mouth width) / (face width)
    - mouth width: distance between mouth corners
    - face width: max_x - min_x across all landmarks (robust, avoids hardcoded face-oval indices)
    """
    if landmarks is None or len(landmarks) <= max(MOUTH_LEFT, MOUTH_RIGHT):
        return _nan()

    # Mouth width
    pL = _lm_xy_px(landmarks, MOUTH_LEFT, w, h)
    pR = _lm_xy_px(landmarks, MOUTH_RIGHT, w, h)
    mouth_w = _dist(pL, pR)

    # Face width from all landmarks (in pixels)
    xs = [float(lm.x) * w for lm in landmarks]
    if not xs:
        return _nan()
    face_w = (max(xs) - min(xs)) + 1e-9

    return float(mouth_w / face_w)


def eye_aspect_ratio(landmarks: Sequence, eye_idxs: List[int], w: int, h: int) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    if landmarks is None or len(landmarks) <= max(eye_idxs):
        return _nan()

    p1 = _lm_xy_px(landmarks, eye_idxs[0], w, h)
    p2 = _lm_xy_px(landmarks, eye_idxs[1], w, h)
    p3 = _lm_xy_px(landmarks, eye_idxs[2], w, h)
    p4 = _lm_xy_px(landmarks, eye_idxs[3], w, h)
    p5 = _lm_xy_px(landmarks, eye_idxs[4], w, h)
    p6 = _lm_xy_px(landmarks, eye_idxs[5], w, h)

    denom = 2.0 * _dist(p1, p4) + 1e-9
    return float((_dist(p2, p6) + _dist(p3, p5)) / denom)


def eye_openness_proxy(landmarks: Sequence, w: int, h: int) -> float:
    """
    Eye openness proxy: mean EAR across left & right eyes.
    """
    ear_l = eye_aspect_ratio(landmarks, LEFT_EYE_IDXS, w, h)
    ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE_IDXS, w, h)
    if not _isfinite(ear_l) and not _isfinite(ear_r):
        return _nan()
    if not _isfinite(ear_l):
        return float(ear_r)
    if not _isfinite(ear_r):
        return float(ear_l)
    return float((ear_l + ear_r) / 2.0)


@dataclass
class ExpressionProxyMetric:
    """
    Expression utility (image-based) using landmark proxies:
      - smile_ratio: mouth-width / face-width
      - eye_openness: mean EAR
    Reports Pearson correlation clear vs anonymised for each proxy.

    This is intentionally not a full emotion classifier; it is a geometry-based utility measure.
    """
    smile_clear: List[float]
    smile_anon: List[float]
    eye_clear: List[float]
    eye_anon: List[float]
    pairs: int

    def __init__(self):
        self.smile_clear = []
        self.smile_anon = []
        self.eye_clear = []
        self.eye_anon = []
        self.pairs = 0

    def update(self, lm_clear: Optional[Sequence], lm_anon: Optional[Sequence], w: int, h: int):
        if lm_clear is None or lm_anon is None:
            return

        s_c = smile_ratio(lm_clear, w, h)
        s_a = smile_ratio(lm_anon, w, h)
        e_c = eye_openness_proxy(lm_clear, w, h)
        e_a = eye_openness_proxy(lm_anon, w, h)

        # Store even if one proxy is NaN; correlation will mask invalid pairs
        self.smile_clear.append(s_c)
        self.smile_anon.append(s_a)
        self.eye_clear.append(e_c)
        self.eye_anon.append(e_a)
        self.pairs += 1

    def finalise(self) -> Dict[str, Any]:
        return {
            "expression_pairs": self.pairs,
            "smile_ratio_corr": _corr(self.smile_clear, self.smile_anon),
            "eye_openness_corr": _corr(self.eye_clear, self.eye_anon),
        }
