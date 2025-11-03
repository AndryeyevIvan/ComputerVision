import cv2
import numpy as np
import os

print("Натискайте 0, щоб перейти до наступного зображення")

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", "data/MobileNet/mobilenet.caffemodel")

classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

answers = []

folder = "images/MobileNet"

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(folder, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5,(224, 224),(127.5, 127.5, 127.5))
        net.setInput(blob)
        preds = net.forward()
        index = preds[0].argmax()
        label = classes[index] if index < len(classes) else "Unknown"
        conf = preds[0][index][0][0] * 100
        cv2.putText(image, f"{label}: {int(conf)}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        if not any(a[0] == label for a in answers):
            answers.append([label, f"{int(conf)}%", 1])
        else:
            for a in answers:
                if a[0] == label:
                    a[2] += 1
                    break

for i in answers:
    print(i)
