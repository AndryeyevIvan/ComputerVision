import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Створюємо функцію для генерації простих фігур

def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

# 2 Формуємо набори даних

# Список ознак
X = []
# Список міток
Y = []

colors = {
    "red" : (0, 0, 255),
    "green" : (0, 255, 0),
    "blue" : (255, 0, 0)
}

shapers = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapers:
        for _ in range(10):
            img = generate_image(bgr, shape)
            # Повертає значення середніх кольорів
            mean_color = cv2.mean(img)[:3] # (b, g, r, alpha)
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features)
            Y.append(f"{color_name}_{shape}")

# 3 Розділяємо дані

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)

# 4 Навчаємо модель

model = KNeighborsClassifier(n_neighbors=3) # Бажано непарні числа
model.fit(X_train, Y_train)


# 5 Перевіряємо точність

accurancy = model.score(X_test, Y_test)
print(f"Accuracy: {round(accurancy * 100, 2)}%")

test_image = generate_image((255, 10, 255), "triangle")
mean_color = cv2.mean(test_image)[:3]
prediction = model.predict([mean_color])
print(f"Prediction: {prediction}")

cv2.imshow("Image", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()