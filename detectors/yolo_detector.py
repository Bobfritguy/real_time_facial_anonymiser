from ultralytics import YOLO
from .base_detector import BaseDetector

class YOLOFaceDetector(BaseDetector):
    def __init__(self, model_path="pretrained_model.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, stream=False)[0]
        faces = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            faces.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "landmarks": None   # YOLO baseline has no landmarks
            })
        return faces

