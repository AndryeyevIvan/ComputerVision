import cv2
import os
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(PROJECT_DIR, 'video')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_VIDEO_PATH = os.path.join(INPUT_DIR, 'Cars.mp4')
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'output_video.mp4')

USE_WEBCAM = False
source = 0 if USE_WEBCAM else INPUT_VIDEO_PATH

MODEL_PATH = "yolov8m.pt"
CONF_THRESH = 0.4
TRACKER = "bytetrack.yaml"
SAVE_VIDEO = True

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(source)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train'}

seen_id_total = set()
seen_id_class = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=False)
    r = result[0]

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imshow('frame', frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    track_id = boxes.id.cpu().numpy() if boxes.id is not None else None

    frame_count_class = {}

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        class_id = int(cls[i])
        class_name = model.names[class_id]
        score = conf[i]
        tid = int(track_id[i]) if track_id is not None else -1

        if class_name not in VEHICLE_CLASSES:
            continue

        if tid != -1:
            seen_id_total.add(tid)
            if class_name not in seen_id_class:
                seen_id_class[class_name] = set()
            seen_id_class[class_name].add(tid)

        if class_name not in frame_count_class:
            frame_count_class[class_name] = 0
        frame_count_class[class_name] += 1

        label = f"{class_name} {score:.2f}"
        if tid != -1:
            label += f" Id{tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    y_offset = 30
    for class_name, count in frame_count_class.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    cv2.putText(frame, f"Total unique objects: {len(seen_id_total)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('frame', frame)
    if writer:
        writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()