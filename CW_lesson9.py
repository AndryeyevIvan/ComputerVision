import cv2

# Завантажуємо моделі
net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", "data/MobileNet/mobilenet.caffemodel")

# 2 Завантажуємо список класів

classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        linne = line.strip()
        if not line: continue
        parts = linne.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

# 3 Підвантажуємо зображення

image = cv2.imread("images/dolgopyat.jpg")

# 4 Адаптуємо зображення під нейронку

blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5,(224, 224),(127.5, 127.5, 127.5))

# 5 Кладемо зображення в мережу і запускаємо

net.setInput(blob)
preds = net.forward()

# 6 Знаходимо індекс класа з найбільшою ймовірністю

index = preds[0].argmax()

# 7 Дістаємо назву класа і впевненість

label = classes[index] if index < len(classes) else "Unknown"
conf = float(preds[0][index]) * 100

print("Class:", label)
print("Confidence:", conf)

text = f"{label}: {int(conf)}%"

cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()