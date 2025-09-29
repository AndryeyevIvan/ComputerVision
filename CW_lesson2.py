import cv2
import numpy as np

# image = cv2.imread('images/bird.jpg')
#
# #image = cv2.resize(image, (500, 250))
# image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# print(image.shape)
# #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# #image = cv2.flip(image, 1)
# #image = cv2.GaussianBlur(image, (5, 5), 7) #Тільке непарні значення в кортежі
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 200, 200)
# # image = cv2.dilate(image, None, iterations=1)
# kernel = np.ones((5, 5), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# image = cv2.erode(image, kernel, iterations=1)
#
#
# cv2.imshow('forest', image)
# # cv2.imshow('forest', image[0:150, 0:100])

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# video = cv2.VideoCapture('videos/vid.mp4')
video = cv2.VideoCapture(0)
while True:
    success, frame = video.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break