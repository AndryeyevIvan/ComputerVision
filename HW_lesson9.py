import cv2
import numpy as np
net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", "data/MobileNet/mobilenet.caffemodel")

classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        linne = line.strip()
        if not line: continue
        parts = linne.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

names = ["Chih", "Meduzka", "Papuga", "Pinguinos"]
answers = []

for name in names:
    image = cv2.imread(f"images/MobileNet/{name}.jpg")
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5,(224, 224),(127.5, 127.5, 127.5))
    net.setInput(blob)
    preds = net.forward()
    index = preds[0].argmax()
    label = classes[index] if index < len(classes) else "Unknown"
    conf = preds[0][index][0][0] * 100
    answers.append([label, f"{int(conf)}%"])


image = np.full((500, 500, 3), 255, dtype=np.uint8)
cv2.rectangle(image, (30, 30), (470, 470), (0, 0, 0), 2)
cv2.line(image, (30, 140), (470, 140), (0, 0, 0), 2)
cv2.line(image, (30, 250), (470, 250), (0, 0, 0), 2)
cv2.line(image, (30, 360), (470, 360), (0, 0, 0), 2)
cv2.line(image, (140, 30), (140, 470), (0, 0, 0), 2)
cv2.line(image, (360, 30), (360, 470), (0, 0, 0), 2)
image[40:40+90, 40:40+90] = cv2.resize(cv2.imread("images/MobileNet/Chih.jpg"), (90, 90))
image[150:150+90, 40:40+90] = cv2.resize(cv2.imread("images/MobileNet/Meduzka.jpg"), (90, 90))
image[260:260+90, 40:40+90] = cv2.resize(cv2.imread("images/MobileNet/Papuga.jpg"), (90, 90))
image[370:370+90, 40:40+90] = cv2.resize(cv2.imread("images/MobileNet/Pinguinos.jpg"), (90, 90))
cv2.putText(image, answers[0][0], (205, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(image, answers[0][1], (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(image, answers[1][0], (215, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(image, answers[1][1], (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(image, answers[2][0], (165, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
cv2.putText(image, answers[2][1], (400, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(image, answers[3][0], (145, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.346, (0, 0, 0), 2)
cv2.putText(image, answers[3][1], (400, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


cv2.imshow("white_image.png", image)
cv2.waitKey(0)
cv2.destroyAllWindow()