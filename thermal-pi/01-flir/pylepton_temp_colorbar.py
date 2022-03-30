#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Copyright (c) 2020, raspberrypi.com.tw
# All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# pylepton_temp_colorbar.py
#
# Author : sosorry
# Date   : 2020/03/19


import numpy as np
import cv2
from pylepton import Lepton
import time
import numpy as np
import sys


try:
    low  = int(sys.argv[1])
    high = int(sys.argv[2])
except:
    low  = 20
    high = 50


_high = int(high) * 100 + 27315
_low  = int(low)  * 100 + 27315


def setColorBar(lepton_buf, _low, _high):
    d = (_high - _low) / 60.0
    i = 0 

    for i in range(1, 60):
        _d = d * i
        lepton_buf[i][74] = _high - int(_d)
        lepton_buf[i][73] = _high - int(_d)


with Lepton() as l:
    while True:
        lepton_buf, nr = l.capture()
        
        lepton_temp = np.copy(lepton_buf)
        a = np.copy(lepton_buf)
        np.right_shift(a, 8, a) # fit data into 8 bits

        lepton_buf = np.clip(lepton_buf, _low, _high)
        setColorBar(lepton_buf, _low, _high)

        cv2.normalize(lepton_buf, lepton_buf, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(lepton_buf, 8, lepton_buf)

        #h, w, _ = image.shape
        win_h = 60 *4
        win_w = 80 *4
        _lepton = np.asarray(lepton_buf, np.uint8)
        _lepton_gray = cv2.cvtColor(_lepton, cv2.COLOR_GRAY2RGB)
        _lepton_gray = cv2.resize(_lepton_gray, (win_w, win_h))
        _lepton_gray = cv2.applyColorMap(_lepton_gray, cv2.COLORMAP_JET)
        #_lepton_gray = cv2.applyColorMap(_lepton_gray, cv2.COLORMAP_RAINBOW)

        res = np.copy(_lepton_gray)

        w = 1
        h = 1
        x3 = int(160 / 4 * 2 - w)
        y3 = int(120 / 4 * 2 - h)
        
        cv2.rectangle(res, (x3 * 4, y3 * 4), ((x3 + w) * 4, (y3 + h) * 4), (0, 0, 255), -1)
        lepton_temp3 = lepton_temp[y3:y3+h, x3:x3+w]

        try:
            tmp3 = (np.max(lepton_temp3.ravel()) - 27315) / 100.0
            cv2.putText(res, str(tmp3), (x3 * 4 - 40, y3 * 4 + 40), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
            print([np.max(lepton_temp3.ravel())], tmp3)

            j = 10
            d = (high - low) / 5.0
            for i in range(6):
                _d = d * i
                _m = "-" + str(high - int(_d))
                _n = j + (i * 42)
                cv2.putText(res, _m, (290,  _n), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,255,255)) # 320*240
        
            cv2.imshow('image', res)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except:
            pass

        time.sleep(0.01)
    

cv2.destroyAllWindows()

