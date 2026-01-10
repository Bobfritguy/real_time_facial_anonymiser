#  GDPR Facial Anonymiser

This project aims to implemented an embedded solution that anonymises faces in real time, before leaving the embedded device. Making it GDPR compliant.

The framework is modular:

- `detectors/` – YOLO, MediaPipe FaceMesh, etc.
- `anonymisers/` – Blur, Cartoon FaceMesh, etc.
- `metrics/` – Detection recall, IoU shift, pose error, etc.
- `pipeline/` – Evaluation + real-time runner
- `datasets/` – Datasets and their loaders. 

---

## Project Timeline & Progress 

### ✅ WP1 — Background Research & Foundation Techniques (Oct–Nov)
- [x] **T1:** Literature review (face de-identification, obfuscation methods)
- [x] **T2:** GDPR & legal review (define anonymisation goals/permitted scope)
- [x] **T3:** Platform, tools, and hardware selection

---

### ✅ WP2 — Feasibility Testing / Proof-of-Concept (Nov–Dec)
- [x] **T1:** Integrate baseline obfuscation (blur/pixelate) into detector  
- [x] **T2:** Baseline performance benchmarking (FPS, latency, detection, re-ID)
- [x] **T3:** Prototype advanced anonymisation approaches (workstation PoC)

---

### WP3 — Core Project Implementation (Dec–Feb)
- [ ] **T1:** Implement full pipeline (capture → detect → track → anonymise → output)
- [ ] **T2:** Temporal consistency tests (mouth, gaze, pose stability)
- [ ] **T3:** Real-time optimisation strategies (embedded constraints)

---

### WP4 — Evaluation & Refinement (Jan–Mar)
- [ ] **T1:** Define full evaluation metrics (privacy vs utility)
- [ ] **T2:** GDPR compliance & de-anonymisation risk assessment
- [ ] **T3:** Iterate on chosen anonymisation method (refinement loop)
- [ ] **T4:** Prepare demo, poster, and presentation materials

---

### WP5 — Enhancements & Extra Features (Mar–Apr)
- [ ] **T1:** Zone-based policies & paparazzi mode
- [ ] **T2:** Build evaluation/demo dataset
- [ ] **T3:** Package cross-platform demonstrator (Docker/app + guide)

## Project Structure

```
live_face_detector/
├── anonymisers/
├── detectors/
├── metrics/
├── pipeline/
├── datasets/
│   ├── celebA/
│   │   ├── Anno/
│   │   ├── Img/
│   │   ├── Eval/
│   │   └── README.txt
│   └── celebV/
├── weights/
├── requirements.txt
└── README.md
```

## Metrics
See **[metrics readme](metrics/METRICS.md)** for detailed descriptions of all evaluation metrics.
