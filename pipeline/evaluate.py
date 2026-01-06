import os
import cv2
import time
import json
import math

from metrics.detection import DetectionRecallMetric
from datasets.celebA_loader import load_celeba_bboxes

from metrics.pose import estimate_head_pose_solvepnp, PoseMAEMetric
from metrics.mouth import mouth_aspect_ratio, MouthAgreementMetric
from metrics.facing import FacingDirectionAgreementMetric

from metrics.utility import BBoxStabilityMetric
from metrics.geometry import box_iou, box_center, box_area  # used for robustness per-pair signals

from metrics.expression import ExpressionProxyMetric
from metrics.robustness import RobustnessMetric


def evaluate_video(detector, anonymiser, metrics, video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces_clear = detector.detect(frame)
        anon_frame = anonymiser.apply(frame, faces_clear) if anonymiser is not None else frame
        faces_anon = detector.detect(anon_frame)

        # Compute any metrics you’ve plugged in:
        metrics.update(frame, faces_clear, anon_frame, faces_anon)

    cap.release()
    return metrics.finalise()


def _best_match_iou(clear_box, anon_boxes):
    """Return (best_box, best_iou) for a clear_box vs list of anon boxes."""
    best = None
    best_iou = -1.0
    for b in anon_boxes:
        v = float(box_iou(clear_box, b))
        if v > best_iou:
            best_iou = v
            best = b
    return best, best_iou


def evaluate_celeba(detector, celebA_root, limit=None, anonymiser=None, save=True, landmarker=None):
    """
    Computes on CelebA images:
      - Detection recall (GT vs detections on eval_frame)
      - BBox stability (clear detections vs eval detections): IoU, center shift, scale ratio
      - Pose MAE (yaw/pitch/roll) via solvePnP (requires landmarker)
      - Mouth open/closed agreement via MAR threshold (requires landmarker)
      - Facing direction agreement (LEFT/CENTER/RIGHT) from yaw (requires landmarker)
      - Expression proxies (smile ratio + eye openness) correlation (requires landmarker)
      - Robustness: landmark retention/failure rates + distortion outlier rate

    landmarker:
      A detector that returns faces with:
        {"bbox": [...], "landmarks": <list of objects with .x/.y normalized>}
      Typically your MediaPipeMeshDetector (Tasks FaceLandmarker wrapper).
    """
    bbox_dict = load_celeba_bboxes(celebA_root)
    img_dir = os.path.join(celebA_root, "Img/img_celeba")

    recall_metric = DetectionRecallMetric()
    stability_metric = BBoxStabilityMetric(min_match_iou=0.0)

    # Landmark-based metrics (only used if landmarker is provided)
    pose_metric = PoseMAEMetric()
    mouth_metric = MouthAgreementMetric(threshold=0.35)
    facing_metric = FacingDirectionAgreementMetric(center_thresh=15.0)

    # NEW: Expression + Robustness
    expr_metric = ExpressionProxyMetric()
    robust_metric = RobustnessMetric(
        iou_thresh=0.30,
        shift_thresh_px=100.0,
        scale_min=0.5,
        scale_max=2.0,
    )

    images = sorted(bbox_dict.keys())
    if limit:
        images = images[:limit]

    processed = 0

    for img_name in images:
        print("Processing", img_name)
        path = os.path.join(img_dir, img_name)
        frame = cv2.imread(path)
        if frame is None:
            continue
        processed += 1

        h, w = frame.shape[:2]

        # 1) Ground truth (CelebA)
        gt = [bbox_dict[img_name]]

        # 2) Always run detector on CLEAR frame (for stability metrics)
        faces_clear = detector.detect(frame)
        pred_clear_boxes = [f["bbox"] for f in faces_clear]

        # 3) Build eval_frame (clear or anonymised)
        if anonymiser is None:
            eval_frame = frame
        else:
            # Some anonymisers (e.g. CartoonAnonymiser) require landmarks.
            if landmarker is not None and anonymiser.__class__.__name__ == "CartoonAnonymiser":
                faces_for_anon = landmarker.detect(frame)  # must include "landmarks"
            else:
                faces_for_anon = faces_clear  # bbox-only is fine for blur etc.

            eval_frame = anonymiser.apply(frame, faces_for_anon)

        # 4) Detect on eval_frame (clear or anonymised)
        faces_pred = detector.detect(eval_frame)
        pred_eval_boxes = [f["bbox"] for f in faces_pred]

        # 5) Recall is GT vs eval predictions
        recall_metric.update(gt, pred_eval_boxes)

        # 6) Stability is clear predictions vs eval predictions
        stability_metric.update(pred_clear_boxes, pred_eval_boxes)

        # 6b) Robustness distortion (per-image best match between first clear face and eval faces)
        # We only measure distortion when we have at least one clear and one eval detection.
        if pred_clear_boxes and pred_eval_boxes:
            clear_box = pred_clear_boxes[0]
            best_box, best_iou = _best_match_iou(clear_box, pred_eval_boxes)

            if best_box is not None:
                cxc, cyc = box_center(clear_box)
                axc, ayc = box_center(best_box)
                shift = math.sqrt((axc - cxc) ** 2 + (ayc - cyc) ** 2)

                a_clear = box_area(clear_box)
                a_anon = box_area(best_box)
                sr = (a_anon / a_clear) if a_clear > 0 else float("nan")

                robust_metric.update_distortion(iou=best_iou, center_shift_px=shift, scale_ratio=sr)

        # 7) Landmark-based metrics (pose/mouth/facing/expression/robustness) using landmarker
        if landmarker is not None:
            faces_lm_clear = None
            faces_lm_eval = None
            try:
                faces_lm_clear = landmarker.detect(frame)
                faces_lm_eval = landmarker.detect(eval_frame)
            except Exception:
                # Landmarker crashed on this frame => treat as failure for robustness
                robust_metric.update_landmarks(has_clear_landmarks=False, has_anon_landmarks=False)
                continue

            has_clear_lm = bool(faces_lm_clear) and faces_lm_clear[0].get("landmarks") is not None
            has_eval_lm = bool(faces_lm_eval) and faces_lm_eval[0].get("landmarks") is not None
            robust_metric.update_landmarks(has_clear_landmarks=has_clear_lm, has_anon_landmarks=has_eval_lm)

            if not (has_clear_lm and has_eval_lm):
                continue

            lm_clear = faces_lm_clear[0]["landmarks"]
            lm_eval = faces_lm_eval[0]["landmarks"]

            # Pose (yaw/pitch/roll) via solvePnP
            pose_clear = estimate_head_pose_solvepnp(lm_clear, w, h)
            pose_eval = estimate_head_pose_solvepnp(lm_eval, w, h)
            pose_metric.update(pose_clear, pose_eval)

            # Mouth agreement (open vs closed) via MAR
            mar_c = mouth_aspect_ratio(lm_clear, w, h)
            mar_e = mouth_aspect_ratio(lm_eval, w, h)
            mouth_metric.update(mar_c, mar_e)

            # Facing direction agreement (LEFT/CENTER/RIGHT) from yaw
            facing_metric.update(pose_clear, pose_eval)

            # Expression proxies (smile ratio + eye openness) correlation
            expr_metric.update(lm_clear, lm_eval, w, h)

    # Finalise metrics
    recall = recall_metric.finalise()
    stability = stability_metric.finalise()
    pose = pose_metric.finalise()
    mouth = mouth_metric.finalise()
    facing = facing_metric.finalise()
    expr = expr_metric.finalise()
    robust = robust_metric.finalise()

    print(f"Detection Recall: {recall:.4f}")
    if stability.get("bbox_iou_mean") is not None:
        print(f"BBox IoU mean: {stability['bbox_iou_mean']:.4f} (n={stability.get('pairs_measured', 0)})")

    if landmarker is not None:
        if pose.get("pose_pairs"):
            print(
                "Pose MAE (deg): "
                f"yaw={pose.get('yaw_mae_deg')}, pitch={pose.get('pitch_mae_deg')}, roll={pose.get('roll_mae_deg')} "
                f"(n={pose.get('pose_pairs')})"
            )
        if mouth.get("mouth_pairs"):
            print(
                f"Mouth agreement: {mouth.get('mouth_state_agreement')} "
                f"(n={mouth.get('mouth_pairs')}, thr={mouth.get('mouth_open_threshold')})"
            )
        if facing.get("facing_pairs"):
            print(
                f"Facing agreement: {facing.get('facing_direction_agreement')} "
                f"(n={facing.get('facing_pairs')}, thr={facing.get('facing_center_thresh_deg')})"
            )
        if expr.get("expression_pairs"):
            print(
                f"Expression corr: smile={expr.get('smile_ratio_corr')}, eye={expr.get('eye_openness_corr')} "
                f"(n={expr.get('expression_pairs')})"
            )
        if robust.get("landmark_retention_rate") is not None:
            print(
                f"Landmark retention: {robust.get('landmark_retention_rate')}, "
                f"Landmark failure on anon: {robust.get('landmark_failure_rate_on_anon')}"
            )
        if robust.get("distortion_outlier_rate") is not None:
            print(
                f"Distortion outliers: {robust.get('distortion_outlier_rate')} "
                f"(n={robust.get('distortion_pairs_checked')})"
            )

    # Save JSON summary
    out = {
        "detector": detector.__class__.__name__,
        "anonymiser": anonymiser.__class__.__name__ if anonymiser else "None",
        "dataset": "CelebA",
        "images_requested": len(images),
        "images_evaluated": processed,
        "recall": recall,
        **stability,
        **pose,
        **mouth,
        **facing,
        **expr,
        **robust,
        "timestamp": time.time(),
    }

    if save:
        os.makedirs("results", exist_ok=True)
        fname = f"results/celeba_metrics_{out['detector']}_{out['anonymiser']}.json"
        with open(fname, "w") as f:
            json.dump(out, f, indent=4)
        print(f"[saved] {fname}")

    return out

