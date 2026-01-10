from __future__ import annotations
import numpy as np

def summarise_seconds(samples):
    """
    Summarise a list of durations in seconds.
    Returns ms stats: mean, p50, p90, p95, max.
    """
    arr = np.array(samples, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }

    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean() * 1000.0),
        "p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "p90_ms": float(np.percentile(arr, 90) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "max_ms": float(arr.max() * 1000.0),
    }

