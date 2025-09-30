import cv2
import numpy as np

img = np.zeros((512,400,3), np.uint8)
# rgb = bgr

# img[:] = 52, 235, 161 # Залити все

# img[100:150, 200:280] = 52, 235, 161 # Залити не все

cv2.rectangle(img,(100,100),(200,200),(52, 235, 161), 3) # (left upper), (right lower), (color), fet (thickness) cv2.FILLED
cv2.line(img,(100,100),(200,200),(255, 255, 255), 3)
print(img.shape)
cv2.line(img,(0, img.shape[0] // 2),(img.shape[1], img.shape[0] // 2),(255, 255, 255), 3)
cv2.line(img,(img.shape[1] // 2, 0),(img.shape[1] // 2, img.shape[0]),(255, 255, 255), 3)
cv2.circle(img,(150,150),50,(255, 255, 255), 3)
cv2.putText(img, "Hello", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 235, 161), 2)
cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()