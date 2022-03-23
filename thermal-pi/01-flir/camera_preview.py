#!/usr/bin/python3
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#
# camera_preview.py
# Preview from camera
#
# Author : sosorry
# Date   : 2015/04/18
# Usage  : python3 camera_preview.py

import cv2
import time
import imutils

cap = cv2.VideoCapture(1)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

try:
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, 320)
        cv2.putText(frame, "press 'q' to quit ", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.imshow("preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()
