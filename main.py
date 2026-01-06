from detectors.yolo_detector import YOLOFaceDetector
from anonymisers.blur import BlurAnonymiser
from anonymisers.cartoon import CartoonAnonymiser
from pipeline.evaluate import evaluate_celeba
from detectors.mp_mesh_detector import MediaPipeMeshDetector

if __name__ == "__main__":
    yolo = YOLOFaceDetector("weights/pretrained_model.pt")
    landmarker = MediaPipeMeshDetector(model_path="weights/face_landmarker.task")
    celebA_root = "datasets/celebA"

    print("### CLEAR metrics:")
    evaluate_celeba(yolo, celebA_root, limit=5000, anonymiser=None, landmarker=landmarker)

    print("\n### BLUR metrics:")
    evaluate_celeba(yolo, celebA_root, limit=5000, anonymiser=BlurAnonymiser(), landmarker=landmarker)

    print("\n### CARTOON metrics:")
    evaluate_celeba(yolo, celebA_root, limit=5000, anonymiser=CartoonAnonymiser(), landmarker=landmarker)
