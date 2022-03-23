import cv2
import imutils

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, 320)
    cv2.imshow("preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()