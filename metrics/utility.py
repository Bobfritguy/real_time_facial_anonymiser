import math
from metrics.geometry import box_iou, box_center, box_area

def _mean(xs):
    return sum(xs) / len(xs) if xs else None

def _std(xs):
    if not xs:
        return None
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

def _best_match_iou(box, candidates):
    if not candidates:
        return None, 0.0
    best_j, best = None, -1.0
    for j, cb in enumerate(candidates):
        v = box_iou(box, cb)
        if v > best:
            best = v
            best_j = j
    return best_j, float(best)

class BBoxStabilityMetric:
    """
    Compares detector predictions on clear frame vs anonymised frame.
    Produces:
      - mean IoU between matched boxes
      - mean center shift (dx, dy)
      - mean scale ratio (area_anon / area_clear)
    """
    def __init__(self, min_match_iou=0.0):
        self.min_match_iou = float(min_match_iou)
        self.ious = []
        self.dxs = []
        self.dys = []
        self.scales = []

    def update(self, clear_pred_boxes, anon_pred_boxes):
        # Greedy: for each clear box, match best anon box (no one-to-one constraint for simplicity)
        for b in clear_pred_boxes:
            j, iou = _best_match_iou(b, anon_pred_boxes)
            if j is None or iou < self.min_match_iou:
                continue

            self.ious.append(iou)

            cxc, cyc = box_center(b)
            axc, ayc = box_center(anon_pred_boxes[j])
            self.dxs.append(axc - cxc)
            self.dys.append(ayc - cyc)

            a_clear = box_area(b)
            a_anon = box_area(anon_pred_boxes[j])
            self.scales.append((a_anon / a_clear) if a_clear > 0 else math.nan)

    def finalise(self):
        return {
            "bbox_iou_mean": _mean(self.ious),
            "bbox_iou_std": _std(self.ious),
            "center_dx_mean_px": _mean(self.dxs),
            "center_dx_std_px": _std(self.dxs),
            "center_dy_mean_px": _mean(self.dys),
            "center_dy_std_px": _std(self.dys),
            "scale_ratio_mean": _mean(self.scales),
            "scale_ratio_std": _std(self.scales),
            "pairs_measured": len(self.ious),
        }
