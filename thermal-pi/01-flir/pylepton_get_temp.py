#!/usr/bin/python3
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#
# pylepton_get_temp.py
# read temperature from flir lepton raw data
#
# Author : sosorry
# Date   : 2020/07/24

import numpy as np
import cv2
import time
from pylepton import Lepton

win_w = 320
win_h = 240


with Lepton() as l:
    while True:
        lepton_buf, nr = l.capture()

        lepton_temp = np.copy(lepton_buf)
        a = np.copy(lepton_buf)
        np.right_shift(a, 8, a) # fit data into 8 bits
        cv2.normalize(lepton_buf, lepton_buf, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(lepton_buf, 8, lepton_buf)

        _lepton = np.asarray(lepton_buf, np.uint8)
        _lepton_gray = cv2.cvtColor(_lepton, cv2.COLOR_GRAY2RGB)
        _lepton_gray = cv2.resize(_lepton_gray, (win_w, win_h))
        _lepton_gray = cv2.applyColorMap(_lepton_gray, cv2.COLORMAP_JET)
        #_lepton_gray = cv2.applyColorMap(_lepton_gray, cv2.COLORMAP_RAINBOW)

        t = lepton_temp[3][3]
        print(t, int((t - 27315)/100))

        cv2.rectangle(_lepton_gray, (2, 2), (4, 4), (0, 0, 255), -1)
        cv2.imshow('temperature', _lepton_gray)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.01)

cv2.destroyAllWindows()

