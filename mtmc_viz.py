'''
After gettting the matching, we use this to visualize the MTMC result.
This is simply used for progress report visualization purpose 

hsiangwei
'''
import cv2
from numpy import genfromtxt
import numpy as np
import glob 
import copy
from PIL import Image


result = np.load('/home/wei/Desktop/test/npy/global_id_dic.npy',allow_pickle=True).tolist()
cnt = 1

for i,g_id in enumerate(result):
    print('processing matching {}/{}'.format(i+1,len(result)))
    for car in result[g_id]:
        if len(result[g_id])>3:
            labels = genfromtxt('/home/wei/Desktop/test/SCT/c0{}.txt'.format(str(car[0]+41)), delimiter=',', dtype=None)
            for label in labels:
                if label[1] == car[1]:
                    im = cv2.imread('/home/wei/Desktop/test/S06/c0{}/img/'.format(car[0]+41)+"%04d"%label[0]+'.jpg')
                    im_out = copy.deepcopy(im)
                    cv2.rectangle(im_out,(int(label[2]),int(label[3])),(int(label[2])+int(label[4]),int(label[3])+int(label[5])),(0,0,255),2)
                    cv2.putText(im_out,str(g_id),(int(label[2]),int(label[3])),cv2.FONT_HERSHEY_PLAIN,max(1.0, im.shape[1]/1200),(0,255,255),thickness = 2)
                    cv2.imwrite('/home/wei/Desktop/test/mtmc_viz/{}.jpg'.format("%04d"%cnt),im_out)
                    cnt+=1
