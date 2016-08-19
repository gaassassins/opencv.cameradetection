import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(3,800)
cap.set(4,490)

cat = cv2.imread('test.jpg')


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = 255-cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 0)
    im2, contours, hierarhy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for index, contour in enumerate(contours):

        area = cv2.contourArea(contour)
        if area < 2500:
            continue

        contourLen = cv2.arcLength(contour, True)
        approxed = cv2.approxPolyDP(contour, contourLen / 10, True)
        if len(approxed) != 4:
            continue

        parent = hierarhy[0][index][3]
        if parent == -1:
            continue

        h,w = cat.shape[:2]
        catrect = np.float32([[0,0], [w,0], [w,h], [0,h]])

        cv2.drawContours(frame, [approxed], 0, (255, 0, 0), 3)
        framerect = np.float32([x[0] for x in approxed])
        markerrect = np.float32([[0,0], [255,0], [255,255], [0,255]])

        transform =cv2.getPerspectiveTransform(catrect, framerect)

        marker = cv2.warpPerspective(gray, transform, (256,256))
        frame = cv2.warpPerspective(cat, transform,frame.shape[:2][::-1], frame, borderMode=cv2.BORDER_TRANSPARENT)

    cv2.imshow('frame', frame)
    cv2.imshow('binary', binary)

    if 27 == cv2.waitKey(30):
        break