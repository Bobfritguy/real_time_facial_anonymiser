from detectors.yolo_detector import YOLOFaceDetector
from pipeline.evaluate import evaluate_celeba

if __name__ == "__main__":
    detector = YOLOFaceDetector("weights/pretrained_model.pt")
    celebA_root = "datasets/celebA"

    evaluate_celeba(
        detector,
        celebA_root,
        limit=5000,
        save=True   
        )
