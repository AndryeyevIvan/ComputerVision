import cv2
import numpy as np

image = cv2.imread("images/VlasnePhoto.png")
image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 80, 80)
kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)
image = cv2.erode(image, kernel, iterations=1)
cv2.imshow("Ivan", image)

image2 = cv2.imread("images/Email.png")
image2 = cv2.resize(image2, (image.shape[1] * 2, image.shape[0] * 2))
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = cv2.Canny(image2, 250, 250)
kernel2 = np.ones((5, 5), np.uint8)
image2 = cv2.dilate(image2, kernel2, iterations=1)
image2 = cv2.erode(image2, kernel2, iterations=1)
cv2.imshow("Email", image2)

cv2.waitKey(0)
cv2.destroyWindow()