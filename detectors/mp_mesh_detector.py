import mediapipe as mp
import numpy as np
from .base_detector import BaseDetector

class MediaPipeMeshDetector(BaseDetector):
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            static_image_mode=False
        )


    def detect(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        h, w = frame.shape[:2]
        faces = []

        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks:

                # Compute bounding box from MediaPipe landmark coordinates
                xs = [lm.x * w for lm in fl.landmark]
                ys = [lm.y * h for lm in fl.landmark]

                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))

                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "landmarks": fl      # IMPORTANT: return real MediaPipe object
                })

        return faces


