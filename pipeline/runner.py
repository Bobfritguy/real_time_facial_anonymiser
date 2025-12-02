def run_realtime(detector, anonymiser):
    import cv2
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        anon = anonymiser.apply(frame, faces)

        cv2.imshow("Anonymised", anon)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
