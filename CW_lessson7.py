import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0,100,100])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([160,100,100])
upper_red2 = np.array([180,255,255])
points = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hcv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hcv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hcv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (255, 0, 0), 2)
                points.append((cx, cy))
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
                cv2.putText(frame, "Found!", (frame.shape[1] - 200, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
            cv2.putText(frame, "Not found!", (frame.shape[1] - 200, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255))

    # for i in range(1, len(points)):
    #     cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)

    cv2.imshow('video', frame)
    cv2.imshow('video2', mask)

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()