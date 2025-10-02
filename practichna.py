import cv2
import numpy as np

img = np.zeros((400,600,3), np.uint8)
img[:] = 227, 227, 227
cv2.rectangle(img, (10, 10), (590, 390), (135, 135, 135), 3)
photo = cv2.imread("images/VlasnePhoto.png")
photo = cv2.resize(photo, (photo.shape[1] // 2 + 20, photo.shape[0] // 2 + 20))
img[30:30 + photo.shape[0], 30:30 + photo.shape[1]] = photo
photo2 = cv2.imread("images/qr-code.png")
print(photo.shape)
photo2 = cv2.resize(photo2, (photo2.shape[1] // 4, photo2.shape[1] // 4))
img[245:245 + photo2.shape[0], 490:490 + photo2.shape[1]] = photo2
cv2.putText(img, "Ivan Andryeyev", (190, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
cv2.putText(img, "Computer Vision Student", (190, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (82, 82, 82), 2)
cv2.putText(img, "Email: ivanandryeyev@gmail.com", (190, 220), cv2.FONT_HERSHEY_DUPLEX, 0.7, (44, 61, 115), 1)
cv2.putText(img, "Phone: +380950280109", (190, 260), cv2.FONT_HERSHEY_DUPLEX, 0.7, (44, 61, 115), 1)
cv2.putText(img, "09/02/2010", (190, 300), cv2.FONT_HERSHEY_DUPLEX, 0.7, (44, 61, 115), 1)
cv2.putText(img, "OpenCV Business Card", (113, 360), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
cv2.imwrite("business_card.png", img)
cv2.imshow("Vizitka", img)

cv2.waitKey(0)