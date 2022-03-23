#!/usr/bin/python3
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Copyright (c) 2020, raspberrypi.com.tw
# All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# pylepton_fusion_pi.py
#
# Author : sosorry
# Date   : 2020/07/24
# Usage  : python3 pylepton_fusion_pi.py haarcascade_frontalface_default.xml

import numpy as np
import cv2 
import sys
import time
import imutils
import configparser 
from pylepton import Lepton

config = configparser.ConfigParser()
config.read('../fusion.conf')
visible_win_w = int(config.get('visible', 'win_w'))
mFactor = int(visible_win_w / 160)    # multiple factor
startX = int(config.get('stereo', 'startX'))
startY = int(config.get('stereo', 'startY'))
endX = int(config.get('stereo', 'endX'))
endY = int(config.get('stereo', 'endY'))

try:
    cascPath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(cascPath)
except:
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)

try:
    with Lepton() as l:
        while True:
            a, _ = l.capture()
            lepton_temp = np.copy(a)
            #print(lepton_temp)
            cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
            np.right_shift(a, 8, a)
            _a = np.asarray(a, np.uint8)
            _a_rgb = cv2.cvtColor(_a, cv2.COLOR_GRAY2RGB)
            img1 = cv2.resize(_a_rgb, (320, 240), interpolation = cv2.INTER_CUBIC)

            _, img2 = cap.read()
            img2 = imutils.resize(img2, visible_win_w)
            crop_img2 = img2[startY:endY, startX:endX]
            crop_img2 = cv2.resize(crop_img2, (320, 240))
            gray = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
    
            if len(faces):
                print("Found {0} faces!".format(len(faces)))
    
            for (x, y, w, h) in faces:
                cv2.rectangle(crop_img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #cv2.rectangle(crop_img2, (x, y-20), (x+int(w/2), y), (0, 255, 0), 2)
    
                temp_face = lepton_temp[int(x/mFactor):int((x+w)/mFactor), int(y/mFactor):int((y+h)/mFactor)]
                #temp_face = lepton_temp[int(x/4):int((x+w)/4), int(y/4):int((y+h)/4)]
    
                try:
                    print((np.max(temp_face.ravel()) - 27315) / 100.0)
                    temp_max = (np.max(temp_face.ravel()) - 27315) / 100.0
                    cv2.putText(crop_img2, str(temp_max), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                except Exception as error:
                    print(error)
                    pass

            cv2.imshow("fusion_pi", crop_img2)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
            time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()
