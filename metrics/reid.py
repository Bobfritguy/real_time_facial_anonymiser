from __future__ import annotations

import math
import numpy as np
import cv2


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / (na * nb))


def _clamp_bbox(b, w, h):
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def crop_face(frame_bgr, bbox, pad: float = 0.15):
    """
    Crop bbox with padding. Returns cropped BGR image or None.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    px = int(bw * pad)
    py = int(bh * pad)
    x1p, y1p, x2p, y2p = _clamp_bbox([x1 - px, y1 - py, x2 + px, y2 + py], w, h)
    crop = frame_bgr[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None
    return crop


class InsightFaceEmbedder:
    """
    Uses InsightFace (ArcFace) pretrained model to get 512-d embeddings.
    Requires: pip install insightface on your environment.
    """
    def __init__(self, device: str = "cpu"):
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise ImportError(
                "InsightFace not available. Install with: pip install insightface\n"
                f"Import error: {e}"
            )

        # providers choice; keep it simple for CPU
        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda":
            # Only works if onnxruntime-gpu is installed; otherwise it will error.
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        # det_size affects internal detection; we are not using its detection heavily,
        # but it may still run. keep moderate.
        self.app.prepare(ctx_id=0 if device.lower() == "cuda" else -1, det_size=(320, 320))

    def embed_bgr(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """
        Returns 512-d embedding or None if embedding failed.
        """
        # InsightFace expects RGB internally, but FaceAnalysis accepts BGR numpy too.
        faces = self.app.get(face_bgr)
        if not faces:
            return None
        # Pick the highest score face within the crop
        faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
        emb = getattr(faces[0], "embedding", None)
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)


def summarise_floats(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "min": None, "max": None}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p):
        if n == 1:
            return xs_sorted[0]
        k = int(round((p / 100.0) * (n - 1)))
        k = max(0, min(n - 1, k))
        return xs_sorted[k]

    return {
        "n": n,
        "mean": float(sum(xs_sorted) / n),
        "p50": float(pct(50)),
        "p90": float(pct(90)),
        "p95": float(pct(95)),
        "min": float(xs_sorted[0]),
        "max": float(xs_sorted[-1]),
    }

