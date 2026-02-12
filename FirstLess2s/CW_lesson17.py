import cv2
import numpy as np
import yt_dlp
import time
from ultralytics import YOLO

# =============================
# НАСТРОЙКИ
# =============================

YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4

DISTANCE_METERS = 10  # расстояние между линиями

# координаты линий (ПОТОМ ПОДКОРРЕКТИРУЕШЬ)
# формат: (x1, y1, x2, y2)
LINE1 = (300, 200, 800, 600)
LINE2 = (500, 100, 1000, 500)

# =============================
# ПОЛУЧАЕМ ПРЯМОЙ STREAM URL
# =============================

ydl_opts = {
    'format': 'best[ext=mp4]',
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    stream_url = info['url']

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Ошибка открытия потока")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

model = YOLO(MODEL_PATH)

# =============================
# ХРАНЕНИЕ ДАННЫХ
# =============================

entry_times = {}
speeds = {}
counted_ids = set()

total_cars = 0
frame_index = 0

# =============================
# ФУНКЦИЯ ПРОВЕРКИ ПЕРЕСЕЧЕНИЯ
# =============================

def side_of_line(x, y, line):
    x1, y1, x2, y2 = line
    return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

# =============================
# ОСНОВНОЙ ЦИКЛ
# =============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    current_time = frame_index / fps

    results = model.track(frame, conf=CONF_THRESH,
                          tracker="bytetrack.yaml",
                          persist=True, verbose=False)[0]

    if results.boxes is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        ids = results.boxes.id

        if ids is not None:
            ids = ids.cpu().numpy()

            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = model.names[class_id]

                # Фильтр только машин
                if class_name not in ["car", "truck", "bus"]:
                    continue

                x1, y1, x2, y2 = boxes[i].astype(int)
                tid = int(ids[i])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Проверяем пересечение первой линии
                if tid not in entry_times:
                    if side_of_line(cx, cy, LINE1) < 0:
                        entry_times[tid] = current_time

                # Проверяем пересечение второй линии
                if tid in entry_times and tid not in speeds:
                    if side_of_line(cx, cy, LINE2) < 0:
                        dt = current_time - entry_times[tid]
                        if dt > 0:
                            speed_m_s = DISTANCE_METERS / dt
                            speed_kmh = speed_m_s * 3.6
                            speeds[tid] = speed_kmh
                            total_cars += 1
                            counted_ids.add(tid)

                # Рисуем рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                label = f"ID {tid}"

                if tid in speeds:
                    label += f" {speeds[tid]:.1f} km/h"

                cv2.putText(frame, label,
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,255), 2)

    # Рисуем линии
    cv2.line(frame, (LINE1[0], LINE1[1]),
             (LINE1[2], LINE1[3]), (0,0,255), 3)

    cv2.line(frame, (LINE2[0], LINE2[1]),
             (LINE2[2], LINE2[3]), (255,0,0), 3)

    # Общий счётчик
    cv2.putText(frame, f"Total cars: {total_cars}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,255,255), 2)

    cv2.imshow("Traffic Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
