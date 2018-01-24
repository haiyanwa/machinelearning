import cv2
import numpy as np
import argparse
import os.path
from imutils.object_detection import non_max_suppression
from imutils import paths
from PIL import Image

imagePath="/Users/apple/Documents/CSULA/machine_learning/dataset_tool/python/testimg_datala/16-5566_09-06-2016_002_SB/0700/image12645.jpg"
image = cv2.imread(imagePath)
path, filename = os.path.split(imagePath)
print(path, " ", filename)

hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
orig = image.copy()

hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
for i in [(0,0),(4,4),(8,8),(16,16),(32,32)]:
#for i in [(0,0),(4,4)]:
    for j in [(0,0),(4,4),(8,8),(16,16),(32,32),(48,48),(64,64)]:
    #for j in [(0,0),(4,4)]:
        for k in [1.01,1.02,1.04,1.05]:
            for m in [0.05,0.10,0.20,0.25,0.45,0.65]:
                hog_dict = {'winStride': i,'padding': j, 'scale': k}
                #print(hog_dict)
                
                rects, w = hog.detectMultiScale(image, **hog_dict)
                for (x, y, w, h) in rects:
                    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, probs=None, overlapThresh= m)
            
                for (xA, yA, xB, yB) in pick:
                    cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
                    
                print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
            
                img = Image.fromarray(image)
            
                savepath = "./result/" + "marked_" + str(i[0]) + "_" + str(j[0]) + "_" + str(k) + "_" + str(m) + "_" + str(filename)
                print(savepath)
                img.save(savepath)
                
                ##refresh the image array
                image = cv2.imread(imagePath)
