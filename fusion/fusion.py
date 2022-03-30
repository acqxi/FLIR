import numpy as np
import cv2 
import sys
import time
# import pandas as pd
import argparse
import imutils
import gspread
# import gspread_dataframe as gd
import RPi.GPIO as GPIO
import configparser 
from pylepton import Lepton
from imutils.video import VideoStream
from time import sleep

trig = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(trig,GPIO.OUT)

config = configparser.ConfigParser()
config.read('./fusion.conf')
visible_win_w = int(config.get('visible', 'win_w'))
visible_win_h = int(config.get('visible', 'win_h'))
mFactor = int(visible_win_w / 80)    # multiple factor
startX = int(config.get('stereo', 'startx'))
startY = int(config.get('stereo', 'starty'))
endX = int(config.get('stereo', 'endx'))
endY = int(config.get('stereo', 'endy'))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", type=str, default='deploy.prototxt.txt', help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--temperature", type=float, default=50, help="the number of fever setting")
ap.add_argument("-f", "--fusion", type=str, default='off', help="Open the fusion of image")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["prototxt"], args["model"])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

person_count = 0
max_product = 0
high = 0
normal = 0
nomask = 0
now_time = time.time()
# df = pd.DataFrame({'ID', 'Temperature', 'Sub_temp', 'Fever', 'No_mask'})

try:
    with Lepton() as l:
        while True:

            a, _ = l.capture()
            lepton_temp = a.copy()
            cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
            np.right_shift(a, 8, a)
            _a = np.asarray(a, np.uint8)
            ratio = 3
            img_thermal = cv2.cvtColor(_a, cv2.COLOR_GRAY2RGB)
            img_thermal = cv2.resize(img_thermal, (80*ratio, 60*ratio))
            frame = vs.read()
            frame = frame[startY:endY, startX:endX] 
            frame = cv2.resize(frame, (640, 480))
            
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
         
            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            
            max_tmp  = round(((np.max(lepton_temp.ravel()) - 27315) / 97.0),3)
            flag = 0
            time_flag = 0
            # loop over the detections
            for i in range(0, detections.shape[2]):
                mask_check = 'N'
                temp_check = 'N'
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < args["confidence"]: continue
                
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sX, sY, eX, eY) = box.astype("int")
                
                
                temp_face = lepton_temp[int(sY/mFactor):int(eY/mFactor), int(sX/mFactor):int(eX/mFactor)]
                facelong = abs(eY-sY)
                facewidth = abs(eX-sX)
                mid = lepton_temp[int((sY+int((1/5)*facelong))/mFactor):int((eY)/mFactor), int((sX+(1/4)*facewidth)/mFactor):int((eX-(1/4)*facewidth)/mFactor)]
                midh = lepton_temp[int((sY+int((1/5)*facelong))/mFactor):int((eY-int((1/2)*facelong))/mFactor), int((sX+(2/5)*facewidth)/mFactor):int((eX-(2/5)*facewidth)/mFactor)]
                midl = lepton_temp[int((sY+int((1/2)*facelong))/mFactor):int((eY-int((1/4)*facelong))/mFactor), int((sX+(2/5)*facewidth)/mFactor):int((eX-(2/5)*facewidth)/mFactor)]

                try:
                    
                    Max_pos = np.argmax(lepton_temp)
                    pos_x = Max_pos % 80
                    pos_y = int(Max_pos / 80)
                    temp_max = round(((np.max(temp_face.ravel()) - 27315) / 97.0),3)
                    mid_h = round(((np.max(midh.ravel()) - 27315) / 97.0),3)
                    mid_l = round(((np.min(midl.ravel()) - 27315) / 97.0),3)
                    
                    if(int(temp_max)<=40 and int(temp_max)>=20 and facelong >= 80 and facewidth >= 60):
                        GPIO.output(trig,GPIO.HIGH)
                        bias = 150/(eY-sY)
                        temp_tuned = round(temp_max+bias, 3)
                        print(f'temp: {temp_tuned}')
                        print(f'confidence: {confidence*100}')
                        print(f'[Normal] \tperson count: {normal}')
                        print(f'[No mask] \tperson count: {nomask}')
                        print(f'[High temp] \tperson count: {high}\n')
                        
                        
                        cv2.circle(img_thermal, (pos_x*ratio,pos_y*ratio), 1, (255, 0, 0), -1)    
                        cv2.rectangle(img_thermal, (int(sX*ratio/mFactor),int(sY*ratio/mFactor)), (int(eX*ratio/mFactor),int(eY*ratio/mFactor)), (0, 0, 255), 1)    
                        display_color = (0, 0, 0)
                        #flag = 0
                        #time_flag = 0
                        if float(mid_h - mid_l) < 4.0 and float(confidence) >= 0.89:
                            flag = 1
                            nomask += 1
                            time_flag = 1
                            mask_check = 'Y'
                            p1_time = time.time()
                        
                        if temp_tuned <= args["temperature"] and flag == 0:
                            display_color = (15, 185, 255)
                            normal += 1
                        
                        elif temp_tuned > args["temperature"]:
                            high += 1
                            display_color = (50, 0, 255)
                            time_flag = 1
                            temp_check = 'Y'
                            p2_time = time.time()
                        
                        
                        if time_flag == 1: GPIO.output(trig,GPIO.LOW)
                        else: GPIO.output(trig,GPIO.HIGH)
                        
                        person_count += 1
                        upload = [person_count, temp_tuned, temp_max - mid_l, temp_check, mask_check]
                        sheet.append_row(upload)

                        cv2.rectangle(frame, (sX, sY), (eX, eY), display_color, 2)
                        cv2.putText(frame, str(temp_tuned), (sX, sY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, display_color, 2)
                    
                except Exception as error:
                    pass

            if(args["fusion"] == 'on'): cv2.imshow("Thermal", img_thermal)
            cv2.imshow("Fusion", frame)
            key = cv2.waitKey(1) & 0xFF
         
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                print(df)
                break
    
            time.sleep(0.01)
    
finally:
    # cap.release()
    cv2.destroyAllWindows()
    vs.stop()