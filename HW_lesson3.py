import cv2
import numpy as np

img = cv2.imread('images/VlasnePhoto.png')
img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
cv2.rectangle(img,(120,30),(336,350),(235, 73, 52), 1)
cv2.putText(img, "Andryeyev Ivan", (img.shape[1] // 2 - 84, 400), cv2.FONT_HERSHEY_PLAIN, 1.2, (235, 73, 52), 2)
cv2.imshow('VlasnePhoto', img)
cv2.waitKey(0)
