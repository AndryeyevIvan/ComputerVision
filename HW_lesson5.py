import cv2
import numpy as np

img = cv2.imread("images/hw5(2).jpg")
img = cv2.resize(img, (img.shape[1], img.shape[0]))
img_copy = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([51, 12, 0])
upper = np.array([125, 255, 255])
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = round(w / h, 2)
            compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)

            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Quadratic"
            elif len(approx) == 10:
                shape = "Star"
            else:
                shape = "Oval"
            cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
            cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0))
            cv2.putText(img_copy, f'shape: {shape}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(img_copy, f'A: {int(area)}, P:{int(perimeter)}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
cv2.imwrite("result.jpg", img_copy)
cv2.imshow("Figurki", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()