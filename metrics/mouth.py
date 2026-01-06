# metrics/mouth.py
from __future__ import annotations
from typing import Sequence, Optional, Dict, Any, List
import numpy as np
import math


# Stable lip points (FaceMesh topology)
# left corner (61), right corner (291), upper lip inner-ish (13), lower lip inner-ish (14)
LIP_IDXS = [61, 291, 13, 14]


def mouth_aspect_ratio(landmarks: Sequence, w: int, h: int) -> float:
    if landmarks is None or len(landmarks) <= max(LIP_IDXS):
        return float("nan")

    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LIP_IDXS], dtype=np.float32)
    left, right, top, bottom = pts[0], pts[1], pts[2], pts[3]
    width = np.linalg.norm(left - right) + 1e-9
    height = np.linalg.norm(top - bottom)
    return float(height / width)


def is_open(mar: float, threshold: float) -> bool:
    return bool(mar == mar and mar > threshold)


class MouthAgreementMetric:
    def __init__(self, threshold: float = 0.35):
        self.threshold = float(threshold)
        self.total = 0
        self.agree = 0

    def update(self, mar_clear: float, mar_anon: float):
        if not (mar_clear == mar_clear and mar_anon == mar_anon):  # not NaN
            return
        self.total += 1
        if is_open(mar_clear, self.threshold) == is_open(mar_anon, self.threshold):
            self.agree += 1

    def finalise(self) -> Dict[str, Any]:
        return {
            "mouth_pairs": self.total,
            "mouth_open_threshold": self.threshold,
            "mouth_state_agreement": (self.agree / self.total) if self.total > 0 else None,
        }

