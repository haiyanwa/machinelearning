
# coding: utf-8

# In[ ]:

def mylogger(name):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger = logging.getLogger(name)
        hdlr = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

    return logger


#function overlap calculate how much 2 rectangles overlap
def overlap_area(boxes):
    if(len(boxes) == 0):
        return 0
    
    xx1 = max(boxes[0,0], boxes[1,0])
    yy1 = max(boxes[0,1], boxes[1,1])
    xx2 = min(boxes[0,2], boxes[1,2]) 
    yy2 = min(boxes[0,3], boxes[1,3])
    
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    
    ##box1 area
    area1 = (boxes[0,2]-boxes[0,0]) * (boxes[0,3]-boxes[0,1])
    ##box2 area
    area2 = (boxes[1,2]-boxes[1,0]) * (boxes[1,3]-boxes[1,1])
    if area1 > area2:
        area = area2
    else:
        area = area1
        
    overlap = (w * h) / area
    
    return overlap

def get_costs(pos,points):
    distances = [math.floor(math.sqrt((x2-pos[0])**2+(y2-pos[1])**2)) for (x2,y2) in points]
    return distances

