#!/usr/bin/python3
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#
# Author : sosorry
# Date   : 2020/07/24
# Usage  : python3 calibrated_dual_camera

import numpy as np
import cv2 
import time
import imutils
import configparser 
from pylepton import Lepton

config = configparser.ConfigParser()
config.read('../fusion.conf')
visible_win_w = int(config.get('visible', 'win_w'))
startX = int(config.get('stereo', 'startX'))
startY = int(config.get('stereo', 'startY'))
endX = int(config.get('stereo', 'endX'))
endY = int(config.get('stereo', 'endY'))

cap = cv2.VideoCapture(0)

try:
    with Lepton() as l:
        while True:
            a, _ = l.capture()
            cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
            np.right_shift(a, 8, a)
            _a = np.asarray(a, np.uint8)
            _a_rgb = cv2.cvtColor(_a, cv2.COLOR_GRAY2RGB)
            img1 = cv2.resize(_a_rgb, (160, 120), interpolation = cv2.INTER_CUBIC)

            _, img2 = cap.read()
            img2 = imutils.resize(img2, visible_win_w)
            crop_img2 = img2[startY:endY, startX:endX]
            crop_img2 = cv2.resize(crop_img2, (160, 120))

            horizontal = np.hstack((img1, crop_img2))
            cv2.imshow("dual_camera", horizontal)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
            time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()

