from ultralytics import YOLO
import cv2

model = YOLO("runs/classify/yolov8n-cls/weights/best.pt")

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    results = model(frame)

    probs = results[0].probs
    class_id = probs.top1
    confidence = probs.top1conf

    class_name = model.names[class_id]

    cv2.putText(frame, f"{class_name} {confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Test YOLO", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
