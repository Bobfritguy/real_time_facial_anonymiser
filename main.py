from detectors.yolo_detector import YOLOFaceDetector
from anonymisers.blur import BlurAnonymiser
from anonymisers.cartoon import CartoonAnonymiser
from pipeline.evaluate import evaluate_celeba

if __name__ == "__main__":
    detector = YOLOFaceDetector("weights/pretrained_model.pt")
    celebA_root = "datasets/celebA"

    print("### CLEAR recall:")
    evaluate_celeba(detector, celebA_root, limit=5000)

    print("\n### BLUR recall:")
    evaluate_celeba(detector, celebA_root, limit=5000, anonymiser=BlurAnonymiser())

    detector = MediaPipeMeshDetector()
    print("\n### CARTOON recall:")
    evaluate_celeba(detector, celebA_root, limit=5000, anonymiser=CartoonAnonymiser())

