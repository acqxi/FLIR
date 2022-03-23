#!/usr/bin/python3
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#
# Author : sosorry
# Date   : 2020/07/24

import numpy as np
import cv2 
import time
import imutils
import sys
import threading
from pylepton import Lepton
import configparser 

config = configparser.ConfigParser()
config.read('../fusion.conf')
visible_win_w = int(config.get('visible', 'win_w'))
visible_win_h = int(config.get('visible', 'win_h'))

cap = cv2.VideoCapture(0)

try:
    wait = int(sys.argv[1])
except:
    wait = 0


def take_photo(wait):
    while wait > 0:
        print(str(wait) + "...")
        wait -= 1
        time.sleep(1)

    t = int(time.time())
    cv2.imwrite("thermal/" + str(t) + ".jpg", img1)
    cv2.imwrite("visible/" + str(t) + ".jpg", img2)
    print("save " + str(t) + " OK")


try:
    with Lepton() as l:
        while True:
            try:
                a, _ = l.capture()
                cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
                np.right_shift(a, 8, a)
                _a = np.asarray(a, np.uint8)
                _a_rgb = cv2.cvtColor(_a, cv2.COLOR_GRAY2RGB)
                img1 = cv2.resize(_a_rgb, (160, 120), interpolation = cv2.INTER_CUBIC)
                cv2.imshow("thermal", img1)
            except Exception as e:
                print(e)
                pass

            _, img2 = cap.read()
            img2 = imutils.resize(img2, visible_win_w)
            cv2.imshow("visible", img2)

            if cv2.waitKey(1) & 0xFF == ord("c"):
                th = threading.Thread(target=take_photo, args=(wait,))
                th.start()

            elif cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
            time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()

