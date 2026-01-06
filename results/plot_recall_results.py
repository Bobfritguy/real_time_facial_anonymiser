import json
import glob
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

labels = []
recalls = []

for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    with open(path, "r") as f:
        data = json.load(f)

    detector = data.get("detector", "UnknownDetector")
    anonymiser = data.get("anonymiser", "None")
    recall = data.get("recall", 0.0)

    label = f"{detector}\n{anonymiser}"
    labels.append(label)
    recalls.append(recall)

plt.figure()
plt.bar(range(len(recalls)), recalls)
plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
plt.ylabel("Detection Recall")
plt.ylim(0, 1.0)
plt.title("Face Detection Recall on CelebA (5000 images)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("results/recall_comparison.png", dpi=200)
print("Saved plot to results/recall_comparison.png")
