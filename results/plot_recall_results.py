import json
import glob
import os
import math
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_PATH = "results/utility_metrics_comparison.png"

def as_float(x):
    """Convert JSON numbers/None to float/NaN for matplotlib."""
    if x is None:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan

labels = []

recall = []
iou = []
center_shift = []
scale_ratio = []

yaw_mae = []
facing_agreement = []
mouth_agreement = []

for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    with open(path, "r") as f:
        d = json.load(f)

    det = d.get("detector", "Unknown")
    anon = d.get("anonymiser", "None")
    labels.append(f"{det}\n{anon}")

    # Detection
    recall.append(as_float(d.get("recall", 0.0)))

    # Geometry
    iou.append(as_float(d.get("bbox_iou_mean")))
    dx = as_float(d.get("center_dx_mean_px"))
    dy = as_float(d.get("center_dy_mean_px"))
    if math.isnan(dx) or math.isnan(dy):
        center_shift.append(math.nan)
    else:
        center_shift.append(math.sqrt(dx * dx + dy * dy))

    scale_ratio.append(as_float(d.get("scale_ratio_mean")))

    # Pose / Facing / Mouth
    yaw_mae.append(as_float(d.get("yaw_mae_deg")))
    facing_agreement.append(as_float(d.get("facing_direction_agreement")))
    mouth_agreement.append(as_float(d.get("mouth_state_agreement")))

x = list(range(len(labels)))

fig, axs = plt.subplots(2, 2, figsize=(13, 8))

# 1) Recall
axs[0, 0].bar(x, recall)
axs[0, 0].set_title("Detection Recall")
axs[0, 0].set_ylim(0, 1.0)
axs[0, 0].set_ylabel("Recall")
axs[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

# 2) IoU
axs[0, 1].bar(x, iou)
axs[0, 1].set_title("BBox Stability (Mean IoU)")
axs[0, 1].set_ylim(0, 1.0)
axs[0, 1].set_ylabel("Mean IoU")
axs[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

# 3) Yaw MAE (degrees)
axs[1, 0].bar(x, yaw_mae)
axs[1, 0].set_title("Head Pose Degradation (Yaw MAE)")
axs[1, 0].set_ylabel("Yaw MAE (deg)")
axs[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

# 4) Agreements 
bar_w = 0.4
x_left = [i - bar_w/2 for i in x]
x_right = [i + bar_w/2 for i in x]

axs[1, 1].bar(x_left, mouth_agreement, width=bar_w, label="Mouth agreement")
axs[1, 1].bar(x_right, facing_agreement, width=bar_w, label="Facing agreement")
axs[1, 1].set_title("Utility Agreements")
axs[1, 1].set_ylim(0, 1.0)
axs[1, 1].set_ylabel("Agreement")
axs[1, 1].legend()
axs[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

# X tick labels
for ax in axs.flat:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")

plt.suptitle("Utility Metrics on CelebA (5000 images)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUT_PATH, dpi=200)
print(f"Saved plot to {OUT_PATH}")

