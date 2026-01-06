# metrics/robustness.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


def _isfinite(x: float) -> bool:
    return x == x and math.isfinite(x)


@dataclass
class RobustnessMetric:
    """
    Robustness & failure-mode metrics.

    Tracks:
      - Landmark retention rate (anon landmark success relative to clear landmark success)
      - Landmark failure rate on anon (given clear had landmarks)
      - Distortion outlier rate based on bbox stability signals:
          IoU < iou_thresh OR center_shift > shift_thresh OR scale_ratio outside [scale_min, scale_max]
    """
    iou_thresh: float = 0.30
    shift_thresh_px: float = 100.0
    scale_min: float = 0.5
    scale_max: float = 2.0

    clear_landmark_ok: int = 0
    anon_landmark_ok: int = 0
    both_landmark_ok: int = 0

    # Distortion checks only meaningful when we have a clear-vs-anon matched pair
    pair_checked: int = 0
    distortion_outliers: int = 0

    def update_landmarks(self, has_clear_landmarks: bool, has_anon_landmarks: bool):
        if has_clear_landmarks:
            self.clear_landmark_ok += 1
        if has_anon_landmarks:
            self.anon_landmark_ok += 1
        if has_clear_landmarks and has_anon_landmarks:
            self.both_landmark_ok += 1

    def update_distortion(
        self,
        iou: Optional[float] = None,
        center_shift_px: Optional[float] = None,
        scale_ratio: Optional[float] = None,
    ):
        """
        Update distortion outlier counts for a matched clear-vs-anon bbox pair.
        Pass values as floats; any missing/NaN will simply skip that test.
        """
        self.pair_checked += 1

        is_outlier = False

        if iou is not None and _isfinite(float(iou)):
            if float(iou) < self.iou_thresh:
                is_outlier = True

        if center_shift_px is not None and _isfinite(float(center_shift_px)):
            if float(center_shift_px) > self.shift_thresh_px:
                is_outlier = True

        if scale_ratio is not None and _isfinite(float(scale_ratio)):
            sr = float(scale_ratio)
            if sr < self.scale_min or sr > self.scale_max:
                is_outlier = True

        if is_outlier:
            self.distortion_outliers += 1

    def finalise(self) -> Dict[str, Any]:
        # landmark retention: "how often do we retain landmark ability after anonymisation"
        if self.clear_landmark_ok > 0:
            landmark_retention = self.anon_landmark_ok / self.clear_landmark_ok
            landmark_failure_on_anon = (self.clear_landmark_ok - self.both_landmark_ok) / self.clear_landmark_ok
        else:
            landmark_retention = None
            landmark_failure_on_anon = None

        if self.pair_checked > 0:
            distortion_outlier_rate = self.distortion_outliers / self.pair_checked
        else:
            distortion_outlier_rate = None

        return {
            "clear_landmark_ok": self.clear_landmark_ok,
            "anon_landmark_ok": self.anon_landmark_ok,
            "both_landmark_ok": self.both_landmark_ok,
            "landmark_retention_rate": landmark_retention,
            "landmark_failure_rate_on_anon": landmark_failure_on_anon,
            "distortion_pairs_checked": self.pair_checked,
            "distortion_outliers": self.distortion_outliers,
            "distortion_outlier_rate": distortion_outlier_rate,
            "distortion_iou_thresh": self.iou_thresh,
            "distortion_shift_thresh_px": self.shift_thresh_px,
            "distortion_scale_min": self.scale_min,
            "distortion_scale_max": self.scale_max,
        }
