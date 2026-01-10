import os
import json
import math
from glob import glob
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default)


def _safe_float(x, default=None):
    return float(x) if _is_num(x) else default


def _label(run: Dict[str, Any]) -> str:
    return f"{run.get('detector','?')} | {run.get('anonymiser','?')}"

def _filter_numeric(labels, values):
    """
    Returns (labels2, values2) keeping only finite numeric values.
    """
    out_l, out_v = [], []
    for lab, v in zip(labels, values):
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out_l.append(lab)
            out_v.append(float(v))
    return out_l, out_v


def plot_results_from_dir(
    results_dir: str = "results",
    out_dir: str = "results",
    show: bool = False,
) -> Dict[str, str]:
    """
    Reads your results/*.json files and outputs a set of comparison charts using matplotlib.

    Expected files (as produced by your pipeline):
      - celeba_latency_<Detector>_<Anonymiser>.json
      - celeba_metrics_<Detector>_<Anonymiser>.json

    Outputs PNGs into out_dir and returns a dict {plot_name: filepath}.
    """
    os.makedirs(out_dir, exist_ok=True)

    latency_paths = sorted(glob(os.path.join(results_dir, "celeba_latency_*.json")))
    metrics_paths = sorted(glob(os.path.join(results_dir, "celeba_metrics_*.json")))

    latency_runs = [_load_json(p) for p in latency_paths]
    metrics_runs = [_load_json(p) for p in metrics_paths]

    outputs: Dict[str, str] = {}

    # -----------------------------
    # 1) Latency: Total p50 / p95
    # -----------------------------
    if latency_runs:
        labels = [_label(r) for r in latency_runs]
        p50 = [_safe_float(_get(_get(r, "latency_total_process", {}), "p50_ms")) for r in latency_runs]
        p95 = [_safe_float(_get(_get(r, "latency_total_process", {}), "p95_ms")) for r in latency_runs]
        mean = [_safe_float(_get(_get(r, "latency_total_process", {}), "mean_ms")) for r in latency_runs]

        x = list(range(len(labels)))

        plt.figure()
        plt.plot(x, mean, marker="o")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Mean total processing latency (ms)")
        plt.title("CelebA: Mean total processing latency")
        plt.tight_layout()
        path = os.path.join(out_dir, "latency_total_mean.png")
        plt.savefig(path, dpi=200)
        plt.close()
        outputs["latency_total_mean"] = path

        plt.figure()
        plt.plot(x, p50, marker="o", label="p50")
        plt.plot(x, p95, marker="o", label="p95")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Total processing latency (ms)")
        plt.title("CelebA: Total processing latency percentiles")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, "latency_total_p50_p95.png")
        plt.savefig(path, dpi=200)
        plt.close()
        outputs["latency_total_p50_p95"] = path

    # ---------------------------------------
    # 2) Latency breakdown: detect vs anonym
    # ---------------------------------------
    if latency_runs:
        labels = [_label(r) for r in latency_runs]
        det_mean = [_safe_float(_get(_get(r, "latency_detect", {}), "mean_ms"), 0.0) or 0.0 for r in latency_runs]
        anon_mean = [_safe_float(_get(_get(r, "latency_anonymise", {}), "mean_ms"), 0.0) or 0.0 for r in latency_runs]

        x = list(range(len(labels)))

        plt.figure()
        # Stacked bars: detect + anonymise (no explicit colors)
        plt.bar(x, det_mean, label="detect (mean)")
        plt.bar(x, anon_mean, bottom=det_mean, label="anonymise (mean)")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Latency (ms)")
        plt.title("CelebA: Mean latency breakdown")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, "latency_breakdown_mean.png")
        plt.savefig(path, dpi=200)
        plt.close()
        outputs["latency_breakdown_mean"] = path

    # ---------------------------------------
    # 3) Utility: Recall comparison (bar)
    # ---------------------------------------
    if metrics_runs:
        # Sort to keep output stable
        metrics_runs_sorted = sorted(metrics_runs, key=lambda r: (_get(r, "detector", ""), _get(r, "anonymiser", "")))
        labels = [_label(r) for r in metrics_runs_sorted]
        recall = [_safe_float(_get(r, "recall")) for r in metrics_runs_sorted]

        x = list(range(len(labels)))

        plt.figure()
        plt.bar(x, recall)
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Detection recall (IoU≥0.5 vs CelebA GT)")
        plt.title("CelebA: Detection recall by method")
        plt.tight_layout()
        path = os.path.join(out_dir, "recall_comparison.png")
        plt.savefig(path, dpi=200)
        plt.close()
        outputs["recall_comparison"] = path

    # ---------------------------------------
    # 4) Utility: BBox stability (IoU mean)
    # ---------------------------------------
    if metrics_runs:
        metrics_runs_sorted = sorted(metrics_runs, key=lambda r: (_get(r, "detector", ""), _get(r, "anonymiser", "")))
        labels = [_label(r) for r in metrics_runs_sorted]
        iou_mean = [_safe_float(_get(r, "bbox_iou_mean")) for r in metrics_runs_sorted]

        x = list(range(len(labels)))

        plt.figure()
        plt.bar(x, iou_mean)
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("BBox IoU mean (clear vs eval)")
        plt.title("CelebA: Bounding-box stability (IoU mean)")
        plt.tight_layout()
        path = os.path.join(out_dir, "bbox_iou_mean.png")
        plt.savefig(path, dpi=200)
        plt.close()
        outputs["bbox_iou_mean"] = path

    # ---------------------------------------------------
    # 5) Landmark robustness: retention + outlier rate
    # ---------------------------------------------------
    if metrics_runs:
        metrics_runs_sorted = sorted(metrics_runs, key=lambda r: (_get(r, "detector", ""), _get(r, "anonymiser", "")))
        labels = [_label(r) for r in metrics_runs_sorted]

        retention = [_safe_float(_get(r, "landmark_retention_rate")) for r in metrics_runs_sorted]
        outlier = [_safe_float(_get(r, "distortion_outlier_rate")) for r in metrics_runs_sorted]

        # Landmark retention
        labels_r, retention_r = _filter_numeric(labels, retention)
        if retention_r:
            x = list(range(len(labels_r)))
            plt.figure()
            plt.bar(x, retention_r)
            plt.xticks(x, labels_r, rotation=30, ha="right")
            plt.ylim(0, 1)
            plt.ylabel("Landmark retention rate")
            plt.title("CelebA: Landmark retention after anonymisation")
            plt.tight_layout()
            path = os.path.join(out_dir, "landmark_retention_rate.png")
            plt.savefig(path, dpi=200)
            plt.close()
            outputs["landmark_retention_rate"] = path
        else:
            print("[warn] No numeric landmark_retention_rate values found; skipping plot.")

        # Distortion outlier rate
        labels_o, outlier_o = _filter_numeric(labels, outlier)
        if outlier_o:
            x = list(range(len(labels_o)))
            plt.figure()
            plt.bar(x, outlier_o)
            plt.xticks(x, labels_o, rotation=30, ha="right")
            plt.ylim(0, 1)
            plt.ylabel("Distortion outlier rate")
            plt.title("CelebA: Distortion outlier rate (IoU/shift/scale thresholds)")
            plt.tight_layout()
            path = os.path.join(out_dir, "distortion_outlier_rate.png")
            plt.savefig(path, dpi=200)
            plt.close()
            outputs["distortion_outlier_rate"] = path
        else:
            print("[warn] No numeric distortion_outlier_rate values found; skipping plot.")


    # ---------------------------------------------------
    # 6) Expression proxy correlations (if present)
    # ---------------------------------------------------
    if metrics_runs:
        expr_runs = [r for r in metrics_runs if _get(r, "expression_pairs") and _get(r, "expression_pairs") > 0]
        if expr_runs:
            expr_runs_sorted = sorted(expr_runs, key=lambda r: (_get(r, "detector", ""), _get(r, "anonymiser", "")))
            labels = [_label(r) for r in expr_runs_sorted]
            smile = [_safe_float(_get(r, "smile_ratio_corr")) for r in expr_runs_sorted]
            eye = [_safe_float(_get(r, "eye_openness_corr")) for r in expr_runs_sorted]
            x = list(range(len(labels)))

            plt.figure()
            plt.bar(x, smile)
            plt.xticks(x, labels, rotation=30, ha="right")
            plt.ylim(-1, 1)
            plt.ylabel("Pearson correlation")
            plt.title("CelebA: Smile proxy correlation (clear vs anonymised)")
            plt.tight_layout()
            path = os.path.join(out_dir, "expression_smile_corr.png")
            plt.savefig(path, dpi=200)
            plt.close()
            outputs["expression_smile_corr"] = path

            plt.figure()
            plt.bar(x, eye)
            plt.xticks(x, labels, rotation=30, ha="right")
            plt.ylim(-1, 1)
            plt.ylabel("Pearson correlation")
            plt.title("CelebA: Eye-openness proxy correlation (clear vs anonymised)")
            plt.tight_layout()
            path = os.path.join(out_dir, "expression_eye_corr.png")
            plt.savefig(path, dpi=200)
            plt.close()
            outputs["expression_eye_corr"] = path

    if show:
        # If you want interactive viewing, call this after generating the files.
        # (On headless systems, keep show=False.)
        for _ in range(1):
            plt.show()

    return outputs


if __name__ == "__main__":
    outputs = plot_results_from_dir(
        results_dir="results",
        out_dir="results",
        show=False,
    )

    print("Saved plots:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
