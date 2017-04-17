import numpy as np
import os.path
import re
import cv2

dir = "testimg_datala/16-5566_09-06-2016_002_SB/0700/"
first_file_path = dir + "image12630.jpg"
prefix = "image"
ext = ".jpg"
step = 1

n = 0

frame0 = cv2.imread(first_file_path)
gray_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

for dirName, subdirList, fileList in os.walk(dir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        file = re.sub(prefix,"",fname)
        file = re.sub(ext,"",file)
        file_path = dir + fname
        
        frame = cv2.imread(file_path)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ##compare with the previous
        if gray_frame0 is not None:
            #framediff = cv2.absdiff(gray_frame0,gray_frame)
            framediff = cv2.absdiff(frame0,frame)
            savepath = "./masked_bgr/" + fname
            print(savepath)
            cv2.imwrite(savepath, framediff)
        else:
            gray_frame0 = gray_frame
        
        next = int(file) + step
        next_file = dir + prefix + str(next) + ext
        
        ##when there's next frame
        if os.path.isfile(next_file):
            gray_frame0 = gray_frame
        else:
            gray_frame0 = None
            continue
