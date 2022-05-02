import glob 
from PIL import Image

def iou(bboxA,bboxB):
    left = max(bboxA[0],bboxB[0])
    right = min(bboxA[2],bboxB[2])
    top = max(bboxA[1],bboxB[1])
    bottom = min(bboxA[3],bboxB[3])

    if right<left or bottom < top:
        return 0.0
    iouArea = (right-left)*(bottom-top)
    bb1_Area = (bboxA[0]-bboxA[2])*(bboxA[1]-bboxA[3])
    bb2_Area = (bboxB[0]-bboxB[2])*(bboxB[1]-bboxB[3])

    return iouArea / float(bb1_Area+bb2_Area-iouArea)