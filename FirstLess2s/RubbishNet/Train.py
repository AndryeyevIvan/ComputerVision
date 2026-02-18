from ultralytics import YOLO
import shutil
import os
import random

DATASET_PATH = "dataset"
TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/val"

# ---------- Разделение на train/val ----------
if not os.path.exists(TRAIN_PATH):
    for cls in os.listdir(DATASET_PATH):
        if cls in ["train", "val"]:
            continue

        images = os.listdir(os.path.join(DATASET_PATH, cls))
        random.shuffle(images)

        split = int(len(images) * 0.8)

        train_imgs = images[:split]
        val_imgs = images[split:]

        os.makedirs(os.path.join(TRAIN_PATH, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_PATH, cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(
                os.path.join(DATASET_PATH, cls, img),
                os.path.join(TRAIN_PATH, cls, img)
            )

        for img in val_imgs:
            shutil.copy(
                os.path.join(DATASET_PATH, cls, img),
                os.path.join(VAL_PATH, cls, img)
            )

print("Датасет подготовлен.")

# ---------- Обучение ----------
model = YOLO("yolov8n-cls.pt")  # классификационная версия

model.train(
    data="dataset",
    epochs=30,
    imgsz=224,
    batch=16
)

# ---------- Сохранение ----------
model.export(format="onnx")  # если нужно для Unity
model.save("plastic_metal_model.pt")

print("Модель сохранена!")
