import os
import cv2
import time
import json
import math
import random

from metrics.detection import DetectionRecallMetric
from datasets.celebA_loader import load_celeba_bboxes

from metrics.pose import estimate_head_pose_solvepnp, PoseMAEMetric
from metrics.mouth import mouth_aspect_ratio, MouthAgreementMetric
from metrics.facing import FacingDirectionAgreementMetric

from metrics.utility import BBoxStabilityMetric
from metrics.geometry import box_iou, box_center, box_area  # used for robustness per-pair signals

from metrics.expression import ExpressionProxyMetric
from metrics.robustness import RobustnessMetric

from metrics.perf import summarise_seconds

import matplotlib.pyplot as plt

from metrics.reid import (
    InsightFaceEmbedder,
    cosine_similarity,
    crop_face,
    summarise_floats,
)


def load_celeba_identities(celebA_root: str) -> dict[str, int]:
    """
    Reads Anno/identity_CelebA.txt -> dict {image_name: identity_id}
    """
    path = os.path.join(celebA_root, "Anno", "identity_CelebA.txt")
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img, ident = line.split()
            out[img] = int(ident)
    return out


def evaluate_celeba_reid_similarity_drop(
    detector,
    celebA_root: str,
    anonymiser=None,
    landmarker=None,
    limit_identities: int = 500,
    seed: int = 123,
    save: bool = True,
    out_dir: str = "results",
    plots_dir: str = "results",
    device: str = "cpu",
    pad: float = 0.15,
):
    """
    Option B: Pairwise similarity drop.

    For each identity:
      - pick 2 images: ref, probe
      - embed(ref_clear)
      - embed(probe_clear)
      - embed(probe_anon)
      - sim_cc = cos(ref, probe_clear)
      - sim_ca = cos(ref, probe_anon)
      - drop = sim_cc - sim_ca

    Requires CelebA identity file and an embedding model (InsightFace).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    bbox_dict = load_celeba_bboxes(celebA_root)
    id_map = load_celeba_identities(celebA_root)
    img_dir = os.path.join(celebA_root, "Img", "img_celeba")

    # Group images by identity, but only keep ones that have bbox annotations
    by_id = {}
    for img, ident in id_map.items():
        if img in bbox_dict:
            by_id.setdefault(ident, []).append(img)

    # Filter to identities that have >= 2 images
    candidates = [ident for ident, imgs in by_id.items() if len(imgs) >= 2]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates = candidates[:limit_identities]

    embedder = InsightFaceEmbedder(device=device)

    sims_cc = []
    sims_ca = []
    drops = []

    processed_ids = 0
    skipped_ids = 0

    anon_name = anonymiser.__class__.__name__ if anonymiser else "None"
    det_name = detector.__class__.__name__

    for ident in candidates:
        imgs = by_id[ident]
        if len(imgs) < 2:
            continue
        ref_img, probe_img = rng.sample(imgs, 2)

        ref_path = os.path.join(img_dir, ref_img)
        probe_path = os.path.join(img_dir, probe_img)

        ref_frame = cv2.imread(ref_path)
        probe_frame = cv2.imread(probe_path)
        if ref_frame is None or probe_frame is None:
            skipped_ids += 1
            continue

        # Use CelebA GT bbox for crop (single-face by design)
        ref_bbox = bbox_dict[ref_img]
        probe_bbox = bbox_dict[probe_img]

        ref_crop = crop_face(ref_frame, ref_bbox, pad=pad)
        probe_crop_clear = crop_face(probe_frame, probe_bbox, pad=pad)
        if ref_crop is None or probe_crop_clear is None:
            skipped_ids += 1
            continue

        # Create anonymised probe image (apply on full image using detected faces or landmarker when needed)
        if anonymiser is None:
            probe_eval = probe_frame
        else:
            faces_clear = detector.detect(probe_frame)
            if landmarker is not None and anonymiser.__class__.__name__ == "CartoonAnonymiser":
                faces_for_anon = landmarker.detect(probe_frame)
            else:
                faces_for_anon = faces_clear
            probe_eval = anonymiser.apply(probe_frame, faces_for_anon)

        probe_crop_anon = crop_face(probe_eval, probe_bbox, pad=pad)
        if probe_crop_anon is None:
            skipped_ids += 1
            continue

        # Compute embeddings
        e_ref = embedder.embed_bgr(ref_crop)
        e_probe_clear = embedder.embed_bgr(probe_crop_clear)
        e_probe_anon = embedder.embed_bgr(probe_crop_anon)

        if e_ref is None or e_probe_clear is None or e_probe_anon is None:
            skipped_ids += 1
            continue

        sim_cc = cosine_similarity(e_ref, e_probe_clear)
        sim_ca = cosine_similarity(e_ref, e_probe_anon)
        drop = sim_cc - sim_ca

        sims_cc.append(sim_cc)
        sims_ca.append(sim_ca)
        drops.append(drop)
        processed_ids += 1

    out = {
        "detector": det_name,
        "anonymiser": anon_name,
        "dataset": "CelebA",
        "reid_method": "pairwise_similarity_drop",
        "identities_requested": len(candidates),
        "identities_evaluated": processed_ids,
        "identities_skipped": skipped_ids,
        "sim_clear_clear": summarise_floats(sims_cc),
        "sim_clear_anon": summarise_floats(sims_ca),
        "drop_clearclear_minus_clearanon": summarise_floats(drops),
        "timestamp": time.time(),
    }

    print(json.dumps(out, indent=2))

    # Save JSON
    if save:
        fname = os.path.join(out_dir, f"celeba_reid_drop_{det_name}_{anon_name}.json")
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {fname}")

    # Plots
    if sims_cc and sims_ca:
        # Histogram: similarity distributions
        plt.figure()
        plt.hist(sims_cc, bins=40, alpha=0.6, label="clear-clear")
        plt.hist(sims_ca, bins=40, alpha=0.6, label="clear-anon")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.title(f"CelebA re-ID similarity: {det_name} | {anon_name}")
        plt.legend()
        plt.tight_layout()
        p1 = os.path.join(plots_dir, f"reid_similarity_{det_name}_{anon_name}.png")
        plt.savefig(p1, dpi=200)
        plt.close()
        print(f"[saved] {p1}")

        # Histogram: drop distribution
        plt.figure()
        plt.hist(drops, bins=40, alpha=0.85)
        plt.xlabel("Similarity drop (clear-clear minus clear-anon)")
        plt.ylabel("Count")
        plt.title(f"CelebA re-ID similarity drop: {det_name} | {anon_name}")
        plt.tight_layout()
        p2 = os.path.join(plots_dir, f"reid_drop_{det_name}_{anon_name}.png")
        plt.savefig(p2, dpi=200)
        plt.close()
        print(f"[saved] {p2}")

    return out


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


def _draw_boxes(img_bgr, faces, color=(0, 255, 0), thickness=2):
    out = img_bgr.copy()
    for f in faces or []:
        x1, y1, x2, y2 = f["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def evaluate_celeba(
    detector,
    celebA_root,
    limit=None,
    anonymiser=None,
    save=True,
    landmarker=None,
    sample_image_id=None,          # e.g. "000123.jpg"
    sample_seed=123,               # deterministic random choice
    save_samples=True,
    samples_dir="results/samples",
):
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

    # Expression + Robustness
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

    # ---- Choose a sample image for this run (utility metrics only) ----
    chosen_sample = None
    if save_samples and images:
        if sample_image_id is not None:
            if sample_image_id in bbox_dict:
                chosen_sample = sample_image_id
            else:
                print(f"[warn] sample_image_id '{sample_image_id}' not found in CelebA annotations.")
        else:
            rng = random.Random(sample_seed)
            chosen_sample = rng.choice(images)

    processed = 0
    sample_saved = False

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
            faces_for_anon = faces_clear
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

        # ---- Save sample images once per run ----
        if save_samples and (not sample_saved) and (chosen_sample is not None) and (img_name == chosen_sample):
            os.makedirs(samples_dir, exist_ok=True)

            det_name = detector.__class__.__name__
            anon_name = anonymiser.__class__.__name__ if anonymiser else "None"
            stem = img_name.rsplit(".", 1)[0]
            base = f"{stem}_{det_name}_{anon_name}"

            # Save raw frames
            clear_path = os.path.join(samples_dir, f"{base}_clear.jpg")
            anon_path = os.path.join(samples_dir, f"{base}_anon.jpg")
            cv2.imwrite(clear_path, frame)
            cv2.imwrite(anon_path, eval_frame)

            # Save bbox visualisations (green=clear boxes, red=eval boxes)
            clear_boxes_img = _draw_boxes(frame, faces_clear, color=(0, 255, 0))
            anon_boxes_img = _draw_boxes(eval_frame, faces_pred, color=(0, 0, 255))
            cv2.imwrite(os.path.join(samples_dir, f"{base}_clear_boxes.jpg"), clear_boxes_img)
            cv2.imwrite(os.path.join(samples_dir, f"{base}_anon_boxes.jpg"), anon_boxes_img)

            # Optional side-by-side montage (clear | anon)
            try:
                # ensure same height
                if frame.shape[:2] == eval_frame.shape[:2]:
                    montage = cv2.hconcat([frame, eval_frame])
                    cv2.imwrite(os.path.join(samples_dir, f"{base}_montage.jpg"), montage)
            except Exception:
                pass

            print(f"[sample saved] {clear_path}")
            print(f"[sample saved] {anon_path}")
            sample_saved = True

        # 5) Recall is GT vs eval predictions
        recall_metric.update(gt, pred_eval_boxes)

        # 6) Stability is clear predictions vs eval predictions
        stability_metric.update(pred_clear_boxes, pred_eval_boxes)

        # 6b) Robustness distortion (per-image best match between first clear face and eval faces)
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
            try:
                faces_lm_clear = landmarker.detect(frame)
                faces_lm_eval = landmarker.detect(eval_frame)
            except Exception:
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

            # Facing direction agreement
            facing_metric.update(pose_clear, pose_eval)

            # Expression proxies correlation
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
        "sample_image_id": chosen_sample,
        "samples_dir": samples_dir if save_samples else None,
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



def benchmark_celeba_latency(
    detector,
    celebA_root,
    limit=500,
    anonymiser=None,
    landmarker=None,
    warmup=20,
    save=True,
):
    """
    Camera-independent processing latency on CelebA images.
    Measures wall time for:
      - detect(frame)
      - anonymiser.apply(frame, faces_for_anon)  (optional)
    Excludes: disk read time (mostly), camera capture, display.

    Notes:
      - We still read images from disk, but timing starts AFTER cv2.imread().
      - CartoonAnonymiser uses landmarker to supply landmarks for anonymiser input.
    """
    bbox_dict = load_celeba_bboxes(celebA_root)
    img_dir = os.path.join(celebA_root, "Img/img_celeba")
    images = sorted(bbox_dict.keys())[:limit]

    # Warmup on first few frames (YOLO/GPU/caches)
    for img_name in images[:min(warmup, len(images))]:
        frame = cv2.imread(os.path.join(img_dir, img_name))
        if frame is None:
            continue
        _ = detector.detect(frame)

    detect_dts = []
    anon_dts = []
    total_dts = []

    processed = 0
    for img_name in images:
        path = os.path.join(img_dir, img_name)
        frame = cv2.imread(path)
        if frame is None:
            continue

        t0 = time.perf_counter()
        faces = detector.detect(frame)
        t1 = time.perf_counter()

        if anonymiser is not None:
            if landmarker is not None and anonymiser.__class__.__name__ == "CartoonAnonymiser":
                faces_for_anon = landmarker.detect(frame)
            else:
                faces_for_anon = faces
            _ = anonymiser.apply(frame, faces_for_anon)

        t2 = time.perf_counter()

        detect_dts.append(t1 - t0)
        anon_dts.append(t2 - t1)
        total_dts.append(t2 - t0)
        processed += 1

    out = {
        "detector": detector.__class__.__name__,
        "anonymiser": anonymiser.__class__.__name__ if anonymiser else "None",
        "dataset": "CelebA",
        "images_requested": len(images),
        "images_evaluated": processed,
        "latency_detect": summarise_seconds(detect_dts),
        "latency_anonymise": summarise_seconds(anon_dts),
        "latency_total_process": summarise_seconds(total_dts),
        "timestamp": time.time(),
    }

    print(json.dumps(out, indent=2))

    if save:
        os.makedirs("results", exist_ok=True)
        fname = f"results/celeba_latency_{out['detector']}_{out['anonymiser']}.json"
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {fname}")

    return out

