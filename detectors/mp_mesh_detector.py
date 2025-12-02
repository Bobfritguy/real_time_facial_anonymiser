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
                pts = np.array([
                    (lm.x * w, lm.y * h) for lm in fl.landmark
                ], dtype=np.float32)

                x1, y1 = pts[:,0].min(), pts[:,1].min()
                x2, y2 = pts[:,0].max(), pts[:,1].max()

                faces.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "landmarks": pts
                })

        return faces

