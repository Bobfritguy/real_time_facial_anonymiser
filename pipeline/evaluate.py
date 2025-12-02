import os, cv2, time, json
from detectors.yolo_detector import YOLOFaceDetector
from metrics.detection import DetectionRecallMetric
from datasets.celebA_loader import load_celeba_bboxes

def evaluate_video(detector, anonymiser, metrics, video_path):
    cap = cv2.VideoCapture(video_path)
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces_clear = detector.detect(frame)
        anon_frame = anonymiser.apply(frame, faces_clear)
        faces_anon = detector.detect(anon_frame)

        # Compute any metrics you’ve plugged in:
        metrics.update(frame, faces_clear, anon_frame, faces_anon)

    cap.release()
    return metrics.finalise()



def evaluate_celeba(detector, celebA_root, limit=None, anonymiser=None, save=True):
    bbox_dict = load_celeba_bboxes(celebA_root)
    img_dir = os.path.join(celebA_root, "Img/img_celeba")

    metric = DetectionRecallMetric()

    images = sorted(bbox_dict.keys())
    if limit:
        images = images[:limit]

    for img_name in images:
        print("Processing", img_name)
        path = os.path.join(img_dir, img_name)
        frame = cv2.imread(path)
        if frame is None:
            continue

        # 1. Ground truth
        gt = [bbox_dict[img_name]]

        # 2. If anonymiser is None → evaluate clear image
        if anonymiser is None:
            eval_frame = frame
        else:
            # we need detections first to know what to anonymise
            faces_clear = detector.detect(frame)
            eval_frame = anonymiser.apply(frame, faces_clear)

        # 3. Detect on either clear or anonymised frame
        faces_pred = detector.detect(eval_frame)

        pred_boxes = [f["bbox"] for f in faces_pred]

        # 4. Update recall metric
        metric.update(gt, pred_boxes)

    # Final result
    recall = metric.finalise()
    print(f"Detection Recall: {recall:.4f}")

    # Save JSON summary
    if save:
        os.makedirs("results", exist_ok=True)

        out = {
            "detector": detector.__class__.__name__,
            "anonymiser": anonymiser.__class__.__name__ if anonymiser else "None",
            "dataset": "CelebA",
            "images_evaluated": len(images),
            "recall": recall,
            "timestamp": time.time(),
        }

        fname = f"results/celeba_recall_{out['detector']}_{out['anonymiser']}.json"
        with open(fname, "w") as f:
            json.dump(out, f, indent=4)

        print(f"[saved] {fname}")

    return recall

