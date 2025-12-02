class BaseDetector:
    """
    All detectors must output:
    - faces: list of dicts with:
        {
          "bbox": [x1, y1, x2, y2],
          "landmarks": Nx2 array or None
        }
    """
    def detect(self, frame):
        raise NotImplementedError
