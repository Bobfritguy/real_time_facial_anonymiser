import numpy as np
import cv2
import mediapipe as mp

class MediaPipeMeshDetector:
    def __init__(self, model_path="weights/face_landmarker.task", max_faces=2):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        self._FaceLandmarker = FaceLandmarker
        self._options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = FaceLandmarker.create_from_options(self._options)

    def detect(self, frame):
        """
        Returns:
          faces: list of dicts with:
            {
              "bbox": [x1,y1,x2,y2],
              "landmarks": <list of NormalizedLandmark>   # Tasks API result
            }
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        faces = []
        if not result.face_landmarks:
            return faces

        for lms in result.face_landmarks:
            # lms is a list of NormalizedLandmark (x,y in [0,1])
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            faces.append({
                "bbox": [x1, y1, x2, y2],
                "landmarks": lms,   # keep Tasks landmarks (not numpy)
            })

        return faces



class MediaPipeTasksFaceLandmarker:
    def __init__(self, model_path="weights/face_landmarker.task", max_faces=2):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        self._landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_faces=max_faces,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        faces = []
        for lms in (result.face_landmarks or []):
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            faces.append({
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "landmarks": lms,   # list of NormalizedLandmark (.x/.y)
            })
        return faces

