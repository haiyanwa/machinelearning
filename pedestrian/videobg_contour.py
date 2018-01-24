import numpy as np
import cv2
import imutils
from imutils.video import FPS
from threading import Thread
import sys
import os.path

RESIZEFRAME = 360
CONTOUR_THREASHOLD_MIN = 300
CONTOUR_THREASHOLD_MAX = 6000

#cap = cv2.VideoCapture('video/68Position_001.mp4')
cap = cv2.VideoCapture('video/Camera320_4_45_09262017_33.mp4')
fps = FPS().start()

avg = None
background = 'config/background.npy'
if os.path.exists(background):
    print("load avg")
    avg = np.load(background)

while(True):
    ret, frame_ori = cap.read()

    if ret == True:
            
        if(RESIZEFRAME > 0 and RESIZEFRAME <1):
            frame = imutils.resize(frame_ori,width=int(RESIZEFRAME*w))
        elif (RESIZEFRAME > 1):
            frame = imutils.resize(frame_ori,width=int(RESIZEFRAME))
        else:
            frame = frame_ori.copy()
         
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = np.float32(gray)

        cv2.accumulateWeighted(gray, avg, 0.01)
        framediff = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(framediff, 50, 255, cv2.THRESH_BINARY)[1]
        blurred = cv2.dilate(thresh, None, iterations=1)
        
        (_, cnts, _) = cv2.findContours(blurred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for ct in cnts:
            if cv2.contourArea(ct) < CONTOUR_THREASHOLD_MIN or cv2.contourArea(ct) > CONTOUR_THREASHOLD_MAX:
                continue
            (x, y, w, h) = cv2.boundingRect(ct)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 0, 159), 2)
            cv2.rectangle(blurred, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow('frame',frame)
        cv2.imshow('bg',blurred)
        fps.update()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
       break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#np.save(background, avg)
cap.release()
cv2.destroyAllWindows()


# In[ ]:



