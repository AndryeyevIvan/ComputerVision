import os
import cv2
import time
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, 'dogocat.mp4')
    cap = cv2.VideoCapture(VIDEO_PATH)


model = YOLO('yolov8n.pt')

CONF_THRESHOLD = 0.4

RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(scale * w)
        new_h = int(scale * h)
        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf=CONF_THRESHOLD, verbose=False)

    cat_count = 0
    dog_count = 0
    total_animals = 0

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID:
                cat_count += 1
                total_animals += 1
                color = (255, 0, 0)
                label = f'Cat {conf:.2f}'

            elif cls == DOG_CLASS_ID:
                dog_count += 1
                total_animals += 1
                color = (0, 255, 0)
                label = f'Dog {conf:.2f}'

            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 1.0 / dt

    cv2.putText(frame, f'Cats: {cat_count}', (20, 40),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv2.putText(frame, f'Dogs: {dog_count}', (20, 80),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Total animals: {total_animals}', (20, 120),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 160),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
