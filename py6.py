import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,490)

image = cv2.imread("test.jpg")
board_size = (7,10)

while True:

    _, frame = cap.read()
    ret, corners = cv2.findChessboardCorners(frame, board_size)
    if ret:
        chess_corners = np.float32([tuple(corners[0][0]),
                                    tuple(corners[board_size[0] - 1][0]),
                                    tuple(corners[len(corners) - board_size[0]][0]),
                                    tuple(corners[len(corners) - 1][0])])
        w, h = image.shape[:2]
        image_corners=np.float32(((0,0), (0, w), (h, 0), (h, w)))


        transform = cv2.getPerspectiveTransform(image_corners, chess_corners)
        frame = cv2.warpPerspective(image, transform, frame.shape[:2][::-1],
                                          frame, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.imshow('Lol', frame)
    cv2.waitKey(30)



