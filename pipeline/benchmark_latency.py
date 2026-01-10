# pipeline/benchmark_latency.py
from __future__ import annotations
import time
import cv2
import numpy as np
import json
import os

from metrics.perf import summarise_seconds


def benchmark_video_latency(
    detector,
    anonymiser,
    video_path,
    landmarker=None,
    num_frames=300,
    warmup=30,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    # ---- Warmup ----
    for _ in range(warmup):
        ret, frame = cap.read()
        if not ret:
            break
        _ = detector.detect(frame)

    detect_times = []
    anon_times = []
    total_times = []

    frames = 0
    while frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        faces = detector.detect(frame)
        t1 = time.perf_counter()

        if anonymiser is not None:
            if landmarker is not None and anonymiser.__class__.__name__ == "CartoonAnonymiser":
                faces_for_anon = landmarker.detect(frame)
            else:
                faces_for_anon = faces
            _ = anonymiser.apply(frame, faces_for_anon)
        t2 = time.perf_counter()

        detect_times.append(t1 - t0)
        anon_times.append(t2 - t1)
        total_times.append(t2 - t0)

        frames += 1

    cap.release()

    return {
        "frames": frames,
        "detect": summarise_seconds(detect_times),
        "anonymise": summarise_seconds(anon_times),
        "total_process": summarise_seconds(total_times),
    }

