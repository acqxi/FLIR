#!/usr/bin/python3
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#
# pylepton_preview.py
# Preview from FLIR Lepton
#
# Author : sosorry
# Date   : 2020/07/24
# Usage  : python3 pylepton_preview.py

import numpy as np
import cv2 
import time
from pylepton import Lepton

with Lepton() as l:
    while True:
        a, _ = l.capture()
        cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(a, 8, a)

        r = cv2.resize(a, (160, 120), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("preview", np.uint8(r))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
        time.sleep(0.01)

cv2.destroyAllWindows()

