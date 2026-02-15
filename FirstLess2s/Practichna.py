import cv2
import numpy as np
import yt_dlp
import time
import os
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4

DISTANCE_METERS = 150

LINE1 = (800, 350, 1500, 600)
LINE2 = (600, 550, 1300, 800)

OUTPUT_DIR = "car_speeds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ydl_opts = {
    "format": "best[ext=mp4]",
    "quiet": True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    stream_url = info["url"]

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    exit()

model = YOLO(MODEL_PATH)

entry_times = {}
speeds = {}
start_line = {}
last_side_line1 = {}
last_side_line2 = {}

total_cars = 0

start_time = time.time()

def side_of_line(x, y, line):
    x1, y1, x2, y2 = line
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time() - start_time

    results = model.track(frame, conf=CONF_THRESH, tracker="bytetrack.yaml", persist=True, verbose=False)[0]

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        ids = results.boxes.id

        if ids is not None:
            ids = ids.cpu().numpy()

            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = model.names[class_id]

                if class_name not in ["car", "truck", "bus"]:
                    continue

                x1, y1, x2, y2 = boxes[i].astype(int)
                tid = int(ids[i])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                side1 = side_of_line(cx, cy, LINE1)
                side2 = side_of_line(cx, cy, LINE2)

                if tid not in last_side_line1:
                    last_side_line1[tid] = side1
                    last_side_line2[tid] = side2

                crossed1 = last_side_line1[tid] * side1 < 0
                crossed2 = last_side_line2[tid] * side2 < 0

                if tid not in start_line:
                    if crossed1:
                        start_line[tid] = 1
                        entry_times[tid] = current_time
                    elif crossed2:
                        start_line[tid] = 2
                        entry_times[tid] = current_time

                elif tid not in speeds:
                    dt = None

                    if start_line[tid] == 1 and crossed2:
                        dt = current_time - entry_times[tid]
                        direction = "1->2"
                    elif start_line[tid] == 2 and crossed1:
                        dt = current_time - entry_times[tid]
                        direction = "2->1"

                    if dt is not None and 0.3 < dt < 10:
                        speed_m_s = DISTANCE_METERS / dt
                        speed_kmh = speed_m_s * 3.6

                        speeds[tid] = speed_kmh
                        total_cars += 1

                        filename = os.path.join(OUTPUT_DIR, f"car_{tid}.txt")
                        with open(filename, "w") as f:
                            f.write(f"Car ID: {tid}\n")
                            f.write(f"Direction: {direction}\n")
                            f.write(f"Speed: {speed_kmh:.2f} km/h\n")
                            f.write(f"Time between lines: {dt:.2f} sec\n")

                        with open(os.path.join(OUTPUT_DIR, "all_cars.txt"), "a") as f:
                            f.write(f"Car {tid}: {direction} {speed_kmh:.2f} km/h | dt={dt:.2f} sec\n")

                last_side_line1[tid] = side1
                last_side_line2[tid] = side2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"ID {tid}"
                if tid in speeds:
                    label += f" {speeds[tid]:.1f} km/h"

                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.line(frame, (LINE1[0], LINE1[1]), (LINE1[2], LINE1[3]), (0, 0, 255), 3)
    cv2.line(frame, (LINE2[0], LINE2[1]), (LINE2[2], LINE2[3]), (255, 0, 0), 3)

    cv2.putText(frame, f"Total cars: {total_cars}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Traffic Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
