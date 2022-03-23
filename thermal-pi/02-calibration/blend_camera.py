#!/usr/bin/python3
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#
# blend_camera.py
# Blend thermal camera and visible camera
#
# Author : sosorry
# Date   : 2020/07/24
# Usage  : python3 blend_camera.py

import numpy as np
import cv2 
import time
import imutils
from pylepton import Lepton

def nothing(x):
    pass

cv2.namedWindow("blend_camera", cv2.WINDOW_NORMAL)
cv2.createTrackbar("alpha", "blend_camera", 0, 10, nothing)

cap = cv2.VideoCapture(0)

try:
    with Lepton() as l:
        while True:
            alpha = cv2.getTrackbarPos("alpha", "blend_camera")

            a, _ = l.capture()
            cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
            np.right_shift(a, 8, a)
            _a = np.asarray(a, np.uint8)
            _a_rgb = cv2.cvtColor(_a, cv2.COLOR_GRAY2RGB)
            img1 = cv2.resize(_a_rgb, (160, 120), interpolation = cv2.INTER_CUBIC)

            _, img2 = cap.read()
            img2 = imutils.resize(img2, 160)

            cv2.resizeWindow("blend_camera", 160, 120)
            visible_alpha = float(alpha)/10
            thermal_alpha = float(10-alpha)/10
            dst = cv2.addWeighted(img1, visible_alpha, img2, thermal_alpha, 0)
            cv2.imshow("blend_camera", dst)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
            time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()

