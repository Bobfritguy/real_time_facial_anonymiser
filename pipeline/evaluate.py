import os
import cv2
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


def evaluate_celeba(detector, celebA_root, limit=None):
    # Load GT annotations
    bbox_dict = load_celeba_bboxes(celebA_root)

    # Folder with images:
    img_dir = os.path.join(celebA_root, "Img/img_celeba")

    metric = DetectionRecallMetric()

    images = sorted(bbox_dict.keys())
    if limit:
        images = images[:limit]

    for img_name in images:
        path = os.path.join(img_dir, img_name)
        frame = cv2.imread(path)
        if frame is None:
            continue

        gt = [bbox_dict[img_name]]  # CelebA always single face

        # Run the detector
        faces = detector.detect(frame)

        pred_boxes = [f["bbox"] for f in faces]

        # Update metric
        metric.update(gt, pred_boxes)

    recall = metric.finalise()
    print(f"Detection Recall on CelebA: {recall:.3f}")

    return recall
