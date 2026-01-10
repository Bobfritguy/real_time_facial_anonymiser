import time
from datetime import timedelta

from detectors.yolo_detector import YOLOFaceDetector
from anonymisers.blur import BlurAnonymiser
from anonymisers.cartoon import CartoonAnonymiser
from pipeline.evaluate import evaluate_celeba, benchmark_celeba_latency
from detectors.mp_mesh_detector import MediaPipeMeshDetector

from pipeline.evaluate import evaluate_celeba_reid_similarity_drop

if __name__ == "__main__":
    t_start = time.perf_counter()
    yolo = YOLOFaceDetector("weights/pretrained_model.pt")
    landmarker = MediaPipeMeshDetector(model_path="weights/face_landmarker.task")
    celebA_root = "datasets/celebA"

#    # ---- Latency benchmarks (camera-independent) ----
#    print("### CLEAR latency:")
#    benchmark_celeba_latency(yolo, celebA_root, limit=500, anonymiser=None, landmarker=landmarker)
#
#    print("\n### BLUR latency:")
#    benchmark_celeba_latency(yolo, celebA_root, limit=500, anonymiser=BlurAnonymiser(), landmarker=landmarker)
#
#    print("\n### CARTOON latency:")
#    benchmark_celeba_latency(yolo, celebA_root, limit=500, anonymiser=CartoonAnonymiser(), landmarker=landmarker)
#
#    # ---- Existing metric evaluations ----
#    print("\n### CLEAR metrics:")
#    evaluate_celeba(yolo, celebA_root, limit=5000, anonymiser=None, landmarker=landmarker)
#
#    print("\n### BLUR metrics:")
#    evaluate_celeba(yolo, celebA_root, limit=5000, anonymiser=BlurAnonymiser(), landmarker=landmarker)
#
#    print("\n### CARTOON metrics:")
#    evaluate_celeba(yolo, celebA_root, limit=5000, anonymiser=CartoonAnonymiser(), landmarker=landmarker)
#
    
    print("\n### Re-ID similarity drop (NONE):")
    evaluate_celeba_reid_similarity_drop(
        yolo, celebA_root, anonymiser=None, landmarker=landmarker, limit_identities=500
    )

    print("\n### Re-ID similarity drop (BLUR):")
    evaluate_celeba_reid_similarity_drop(
        yolo, celebA_root, anonymiser=BlurAnonymiser(), landmarker=landmarker, limit_identities=500
    )

    print("\n### Re-ID similarity drop (CARTOON):")
    evaluate_celeba_reid_similarity_drop(
        yolo, celebA_root, anonymiser=CartoonAnonymiser(), landmarker=landmarker, limit_identities=500
    )


    elapsed = time.perf_counter() - t_start
    print(f"TOTAL EXPERIMENT RUNTIME: {timedelta(seconds=elapsed)}")
