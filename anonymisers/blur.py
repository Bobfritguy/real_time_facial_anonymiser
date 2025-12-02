import cv2
from .base_anon import BaseAnonymiser

class BlurAnonymiser(BaseAnonymiser):
    def apply(self, frame, faces):
        out = frame.copy()
        for f in faces:
            x1,y1,x2,y2 = f["bbox"]
            roi = out[y1:y2, x1:x2]
            if roi.size > 0:
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (59,59), 30)
        return out
