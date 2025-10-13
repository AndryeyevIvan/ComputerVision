from operator import indexOf

import cv2
import numpy as np

img = cv2.imread('images/img.jpg')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
img_copy = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 51, 0])
upper_red = np.array([12, 255, 255])

lower_green = np.array([102, 40, 0])
upper_green = np.array([139, 255, 196])

lower_blue = np.array([41, 40, 0])
upper_blue = np.array([93, 255, 255])

lower_yellow = np.array([22, 144, 0])
upper_yellow = np.array([29, 255, 255])

mask_red = cv2.inRange(img, lower_red, upper_red)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_green = cv2.inRange(img, lower_green, upper_green)
mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

            # contours1, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours2, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours3, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours4, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # if contours1 == cnt:
            #     color = "red"
            # elif contours2 == cnt:
            #     color = "green"
            # elif contours3 == cnt:
            #     color = "blue"
            # elif contours1 == cnt:
            #     color = "yellow"

            approx = cv2.approxPolyDP(cnt, 0.048 * perimeter, True)
            if len(approx) == 4:
                shape = "Quadratic"
            elif len(approx) >= 5:
                shape = "Oval"
            else:
                shape = "Inshe"

            cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
            cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0))
            cv2.putText(img_copy, f'shape: {shape}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(img_copy, f'A: {int(area)}, P:{int(perimeter)}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255))
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255))
            cv2.putText(img_copy, f'color:', (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255))

cv2.imwrite("result.jpg", img_copy)
cv2.imshow("Image", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()