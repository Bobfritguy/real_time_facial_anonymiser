# metrics/facing.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any


def yaw_to_class(yaw_deg: float, center_thresh: float = 15.0) -> str:
    """
    Coarse facing direction from head yaw.
      - yaw < -center_thresh => LEFT
      - yaw > +center_thresh => RIGHT
      - otherwise => CENTER
    Sign convention depends on pose extraction; agreement still works even if sign flips,
    but you can adjust if you notice inverted classes.
    """
    if yaw_deg < -center_thresh:
        return "LEFT"
    if yaw_deg > center_thresh:
        return "RIGHT"
    return "CENTER"


class FacingDirectionAgreementMetric:
    def __init__(self, center_thresh: float = 15.0):
        self.center_thresh = float(center_thresh)
        self.total = 0
        self.agree = 0

    def update(self, pose_clear: Optional[Tuple[float, float, float]],
               pose_anon: Optional[Tuple[float, float, float]]):
        if pose_clear is None or pose_anon is None:
            return
        yaw_c = pose_clear[0]
        yaw_a = pose_anon[0]
        self.total += 1
        if yaw_to_class(yaw_c, self.center_thresh) == yaw_to_class(yaw_a, self.center_thresh):
            self.agree += 1

    def finalise(self) -> Dict[str, Any]:
        return {
            "facing_pairs": self.total,
            "facing_center_thresh_deg": self.center_thresh,
            "facing_direction_agreement": (self.agree / self.total) if self.total > 0 else None,
        }
