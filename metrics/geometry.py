# metrics/geometry.py
from __future__ import annotations
from typing import List, Optional, Tuple
import math

Box = List[int]  # [x1,y1,x2,y2]

def box_area(b: Box) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_center(b: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    ua = box_area(a)
    ub = box_area(b)
    union = ua + ub - inter + 1e-9
    return inter / union

def best_iou_match(
    clear_boxes: List[Box],
    anon_boxes: List[Box],
    min_iou: float = 0.0
) -> List[Tuple[int, Optional[int], float]]:
    """
    Greedy matching:
      For each clear box, pick the anon box with max IoU (one-to-one).
    Returns list of (clear_idx, anon_idx_or_None, iou)
    """
    matches: List[Tuple[int, Optional[int], float]] = []
    used_anon = set()

    for ci, cb in enumerate(clear_boxes):
        best_j = None
        best = -1.0
        for aj, ab in enumerate(anon_boxes):
            if aj in used_anon:
                continue
            iou = box_iou(cb, ab)
            if iou > best:
                best = iou
                best_j = aj
        if best_j is not None and best >= min_iou:
            used_anon.add(best_j)
            matches.append((ci, best_j, float(best)))
        else:
            matches.append((ci, None, 0.0))
    return matches

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else math.nan
