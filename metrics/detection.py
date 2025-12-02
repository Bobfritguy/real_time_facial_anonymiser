import numpy as np

def iou(a, b):
    # a,b = [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    union = areaA + areaB - inter + 1e-9
    return inter / union


class DetectionRecallMetric:
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.total_faces = 0
        self.detected = 0

    def update(self, gt_boxes, pred_boxes):
        """
        gt_boxes: list of [x1,y1,x2,y2] from CelebA GT
        pred_boxes: list of [x1,y1,x2,y2] from detector
        """

        self.total_faces += len(gt_boxes)

        matched = set()
        for g in gt_boxes:
            best_iou = 0
            best_pred = None
            for i, p in enumerate(pred_boxes):
                if i in matched: 
                    continue
                iou_val = iou(g, p)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred = i

            if best_iou >= self.iou_thresh:
                matched.add(best_pred)
                self.detected += 1

    def finalise(self):
        if self.total_faces == 0:
            return 0.0
        return self.detected / self.total_faces
