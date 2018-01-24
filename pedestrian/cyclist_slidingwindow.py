from PIL import Image
import cv2
import os
import imutils
import sys
import re

##slice images into desired size by sliding through with specific step and window size

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], int(stepSize*windowSize[0])):
        for x in range(0, image.shape[1], int(stepSize*windowSize[1])):
            
            ##when x go above the range
            if((x + windowSize[1]) > image.shape[1] and y + windowSize[0] <= image.shape[0]):
                yield (x, y, image[y:y + windowSize[1], image.shape[1] - windowSize[0]:image.shape[1]])
                
            ##when y go above the range
            elif((y + windowSize[0]) > image.shape[0] and x + windowSize[1] < image.shape[1]):
                yield (x, y, image[image.shape[0] - windowSize[1]:image.shape[0], x:x + windowSize[0]])
            
            ##when both x and y go above the range
            elif((y + windowSize[0]) > image.shape[0] and (x + windowSize[1]) > image.shape[1]):
                yield (x, y, image[image.shape[0] - windowSize[1]:image.shape[0], image.shape[1] - windowSize[0]:image.shape[1]])
                return
            else:
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

imagedir = "./cyclist/image_Non/"
savepath = "./cyclist/48x96_Non_small/"
if not os.path.exists(savepath):
    os.makedirs(savepath)
    
window_list = {'s1':(48,96)}
#window_list = {'s1':(16,32),'s2':(32,64),'s3':(64,128),'s4':(100,200),'s5':(200,300)}

image_num = 0
##read in image
for dirName, subdirList, fileList in os.walk(imagedir):
    for fname in fileList:
        filepath = dirName + "/" + fname
        
        if(not re.search(r".png",fname)):
            continue
        
        image = cv2.imread(filepath)
        w,h = image.shape[:2]
        print(w,h)
        resized = cv2.resize(image, (int(w/2),int(h/2)))
        print(filepath)
        print("image_num", image_num)
        
        for k in window_list.keys():
            subpath = savepath + "48x96"
            count =0
            for (x, y, window) in sliding_window(resized, 0.5, windowSize=window_list[k]):
                filenum ="%05d"% image_num
                winfile = subpath + 'win_' + filenum + '.jpg'
                count = count + 1
                image_num += 1
                img = Image.fromarray(window)
                img.save(winfile)
