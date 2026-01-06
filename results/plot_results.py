import os
import glob
import json
import math
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_PATH = os.path.join(RESULTS_DIR, "metrics_dashboard.png")

# What we consider a "fair comparison" set:
# keep only this detector family (change if you want)
DETECTOR_FILTER = "YOLOFaceDetector"


def as_float(x):
    if x is None:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def load_latest_by_key(results_dir):
    """
    Load all json files, keep only latest run for each (detector, anonymiser).
    """
    latest = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path, "r") as f:
            d = json.load(f)

        det = d.get("detector", "Unknown")
        anon = d.get("anonymiser", "None")
        ts = as_float(d.get("timestamp"))

        key = (det, anon)
        if key not in latest or ts > as_float(latest[key].get("timestamp")):
            latest[key] = d

    return list(latest.values())


def pick_and_sort(rows, detector_filter):
    rows = [r for r in rows if r.get("detector") == detector_filter]
    # Put baselines first if present
    order = {"None": 0, "BlurAnonymiser": 1, "CartoonAnonymiser": 2}
    rows.sort(key=lambda r: order.get(r.get("anonymiser", "None"), 99))
    return rows


rows = load_latest_by_key(RESULTS_DIR)
rows = pick_and_sort(rows, DETECTOR_FILTER)

if not rows:
    raise SystemExit(f"No results found for detector={DETECTOR_FILTER} in {RESULTS_DIR}/")

labels = [r.get("anonymiser", "None") for r in rows]
x = list(range(len(rows)))

# --- Metrics we will plot ---
recall = [as_float(r.get("recall")) for r in rows]
iou = [as_float(r.get("bbox_iou_mean")) for r in rows]
yaw_mae = [as_float(r.get("yaw_mae_deg")) for r in rows]
facing = [as_float(r.get("facing_direction_agreement")) for r in rows]
mouth = [as_float(r.get("mouth_state_agreement")) for r in rows]

smile_corr = [as_float(r.get("smile_ratio_corr")) for r in rows]
eye_corr = [as_float(r.get("eye_openness_corr")) for r in rows]

landmark_ret = [as_float(r.get("landmark_retention_rate")) for r in rows]
dist_out = [as_float(r.get("distortion_outlier_rate")) for r in rows]

# ---------------- Figure layout ----------------
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# 1) Detection recall
axs[0, 0].bar(x, recall)
axs[0, 0].set_title("Face Detection Recall")
axs[0, 0].set_ylim(0, 1.0)
axs[0, 0].set_ylabel("Recall")
axs[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

# 2) BBox IoU (stability)
axs[0, 1].bar(x, iou)
axs[0, 1].set_title("BBox Stability (Mean IoU)")
axs[0, 1].set_ylim(0, 1.0)
axs[0, 1].set_ylabel("Mean IoU")
axs[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

# 3) Yaw MAE (lower is better)
axs[0, 2].bar(x, yaw_mae)
axs[0, 2].set_title("Head Pose Degradation (Yaw MAE)")
axs[0, 2].set_ylabel("Yaw MAE (deg)")
axs[0, 2].grid(axis="y", linestyle="--", alpha=0.5)

# 4) Agreements: mouth vs facing (side-by-side)
bar_w = 0.38
axs[1, 0].bar([i - bar_w/2 for i in x], mouth, width=bar_w, label="Mouth agreement")
axs[1, 0].bar([i + bar_w/2 for i in x], facing, width=bar_w, label="Facing agreement")
axs[1, 0].set_title("Utility Agreements")
axs[1, 0].set_ylim(0, 1.0)
axs[1, 0].set_ylabel("Agreement")
axs[1, 0].legend()
axs[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

# 5) Expression proxy correlations
axs[1, 1].bar([i - bar_w/2 for i in x], smile_corr, width=bar_w, label="Smile proxy corr")
axs[1, 1].bar([i + bar_w/2 for i in x], eye_corr, width=bar_w, label="Eye openness corr")
axs[1, 1].set_title("Expression Proxy Preservation (Correlation)")
axs[1, 1].set_ylim(-1.0, 1.0)
axs[1, 1].set_ylabel("Pearson r")
axs[1, 1].legend()
axs[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

# 6) Robustness: landmark retention + distortion outliers
axs[1, 2].bar([i - bar_w/2 for i in x], landmark_ret, width=bar_w, label="Landmark retention")
axs[1, 2].bar([i + bar_w/2 for i in x], dist_out, width=bar_w, label="Distortion outlier rate")
axs[1, 2].set_title("Robustness")
axs[1, 2].set_ylim(0, 1.0)
axs[1, 2].set_ylabel("Rate")
axs[1, 2].legend()
axs[1, 2].grid(axis="y", linestyle="--", alpha=0.5)

# X ticks
for ax in axs.flat:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

# Figure title + save
n_images = rows[0].get("images_requested", "?")
fig.suptitle(f"CelebA Utility & Robustness Metrics (n={n_images}, detector={DETECTOR_FILTER})", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200)
print(f"Saved plot to {OUT_PATH}")
