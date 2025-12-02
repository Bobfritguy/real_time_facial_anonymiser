## 1. Detection Recall (IoU-based)

**File:** `metrics/detection.py`  
**Class:** `DetectionRecallMetric`

### Purpose

Measures how well a face detector on anonymised frames preserves the ability to detect the original faces.  
This metric compares ground-truth face bounding boxes (from the clear/original frames) with predicted boxes (from anonymised frames).


### Definition

For each ground-truth face box `g` and each predicted box `p`, compute the Intersection over Union (IoU):

```
IoU(g, p) = Area(g ∩ p) / Area(g ∪ p)
```


A predicted face is considered a match if:

```
IoU(g, p) ≥ iou_thresh
```


(Default: **0.5**, following common detection benchmarks.)

The recall over a dataset is:

```

(Default: **0.5**, following common detection benchmarks.)

The recall over a dataset is:i

```
Recall = matched_ground_truth_faces / total_ground_truth_faces
```
### What It Captures

- Whether faces remain **detectable** after anonymisation.  
- Whether anonymisation significantly **shifts or distorts** face localisation.  
- A primary indicator of **utility preservation** for downstream tasks.

### Implementation Summary

```python
class DetectionRecallMetric:
    def update(self, gt_boxes, pred_boxes):
        # gt_boxes: [[x1, y1, x2, y2], ...] from clear video
        # pred_boxes: same format, from anonymised video
        # Matches each GT box to the best IoU prediction.
```


## Usage

```
metric = DetectionRecallMetric(iou_thresh=0.5)

for clear_frame, anon_frame in dataset:
    gt_boxes = clear_detector(clear_frame)
    pred_boxes = anon_detector(anon_frame)
    metric.update(gt_boxes, pred_boxes)

recall = metric.finalise()
print("Detection recall:", recall)
```
