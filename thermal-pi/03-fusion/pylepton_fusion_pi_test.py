#python3 pylepton_fusion_pi3.py -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel
import numpy as np
import cv2 
import sys
import time
import argparse
import imutils
import configparser 
from pylepton import Lepton
from imutils.video import VideoStream

config = configparser.ConfigParser()
config.read('../fusion.conf')
visible_win_w = int(config.get('visible', 'win_w'))
mFactor = int(visible_win_w / 160)    # multiple factor
startX = int(config.get('stereo', 'startX'))
startY = int(config.get('stereo', 'startY'))
endX = int(config.get('stereo', 'endX'))
endY = int(config.get('stereo', 'endY'))

try:
    print('DNN')
    cascPath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(cascPath)
except:
    print('haar open')
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#net = cv2.dnn.readNetFromCaffe(deploy.prototxt.txt, res10_300x300_ssd_iter_140000.caffemodel)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# cap = cv2.VideoCapture(0)

'''
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated
'''

try:
    with Lepton() as l:
        while True:
            a, _ = l.capture()
            lepton_temp = np.copy(a)
            #print(lepton_temp)
            # cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
            # np.right_shift(a, 8, a)
            # _a = np.asarray(a, np.uint8)
            # _a_rgb = cv2.cvtColor(_a, cv2.COLOR_GRAY2RGB)
            # img1 = cv2.resize(_a_rgb, (1280, 960), interpolation = cv2.INTER_CUBIC)

            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            #frame = rotate(frame, -90)
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
         
            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            
            
            
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < args["confidence"]: continue

                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
         
                # draw the bounding box of the face along with the associated
                # probability
                # text = "{:.2f}%".format(confidence * 100)
                x = startX
                y = startY
                #y = startY - 10 if startY - 10 > 10 else startY + 10
                temp_face = lepton_temp[int(x/mFactor):int((endX)/mFactor), int(y/mFactor):int((endY)/mFactor)]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # print(temp_face.ravel())
                '''
                temp_face.reshape(1,(int(x/mFactor):int((endX)/mFactor))*(int(y/mFactor):int((endY)/mFactor)))
                tt = temp_face.sorted(reverse = True)
                temp_sum = 0
                for i in range(0,10,1):
                    temp_sum += tt[i]
                    
                temp_sum = temp_sum/50
                '''
                try:
                    print('Size: ',abs(int(x/mFactor)-int(endX/mFactor))*abs(int(y/mFactor-int(endY/mFactor))),'  Temp:  ',round(((np.mean(temp_face.ravel()) - 27315) / 100.0),3))
                    #print(round(((np.mean(temp_face.ravel()) - 27315) / 100.0),3))
                    #temp_max = ((temp_sum-27315)/100.0)
                    temp_max = round(((np.mean(temp_face.ravel()) - 27315) / 100.0),3)
                    cv2.putText(frame, str(temp_max), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                except Exception as error:
                    # print(error)
                    pass
            # frame_ = cv2.resize(frame, (960, 540)) # Resize image  
            
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
         
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    
            time.sleep(0.01)

finally:
    # cap.release()
    cv2.destroyAllWindows()
    vs.stop()
