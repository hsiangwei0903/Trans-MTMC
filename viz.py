'''
We use this file to get the visualization of zone and the number of cars passing the zone.
This is used simply for visualization, drawing zones and debugging.

hsiangwei
'''
import cv2
from numpy import genfromtxt
import numpy as np
import glob 
import copy
from PIL import Image
from iou import iou

seqs = ['c041','c042','c043','c044','c045','c046']
path_in = '/home/wei/Desktop/test/S06/'

# zones = [   
#             [[(890,158),(1078,266)],[(0,200),(100,300)],[(0,400),(340,790)],[(629,69),(784,152)]], # c041 
#             [[(780,160),(1000,200)],[(0,300),(50,400)],[(0,450),(300,900)],[(520,110),(670,140)]], # c042
#             [[(1000,200),(1280,250)],[(0,200),(50,340)],[(0,450),(200,860)],[(750,85),(910,145)]], # c043
#             [[(1056,288),(1221,494)],[(0,200),(341,268)],[(251,679),(1201,951)],[(457,152),(639,184)]], # c044
#             [[(1075,200),(1269,372)],[(241,120),(406,166)],[(0,550),(1155,711)],[(500,84),(656,126)]], # c045
#             [[(1059,157),(1266,263)],[(0,200),(50,300)],[(0,550),(87,715)],[(755,41),(934,129)]] # c046
#         ]
# 1--down right  2--top left  3--down left  4--top right

zones = [   
            [[(0,240),(280,790)],[(1000,330),(1270,950)],[(560,75),(1075,260)],[(120,60),(400,180)]], # c041 
            [[(0,300),(170,927)],[(690,335),(993,954)],[(569,78),(927,218)],[(45,160),(358,250)]], # c042
            [[(0,200),(130,960)],[(715,400),(1273,930)],[(636,63),(1216,300)],[(16,80),(406,155)]], # c043
            [[(750,385),(1221,954)],[(743,75),(1007,278)],[(55,172),(619,289)],[(0,440),(218,956)]], # c044
            [[(429,317),(1269,719)],[(851,60),(1175,220)],[(163,92),(626,166)],[(0,200),(120,719)]], # c045
            [[(0,260),(189,719)],[(668,417),(1276,700)],[(811,56),(1233,217)],[(101,135),(511,177)]] # c046
        ]

# 0--from left 1--from bottom 2--from right 3--from top

if __name__ == "__main__":
    for s_id,seq in enumerate(seqs):
        zone_cnt = {1:[],2:[],3:[],4:[]}
        print('processing sequence {}'.format(seq))
        file_path = path_in + seq
        labels_path = '/home/wei/Desktop/test/SCT/{}.txt'.format(seq)
        imgs = sorted(glob.glob(file_path+'/img/*'))
        labels = genfromtxt(labels_path, delimiter=',', dtype=None)
        i = 0
        for frame,img in enumerate(imgs):
            frame += 1
            if frame%100 == 0:
                print('processing frame: ',frame)
            im = cv2.imread(img)
            im_out = copy.deepcopy(im)
            cv2.rectangle(im_out,zones[s_id][0][0],zones[s_id][0][1],(0,0,255),2)
            cv2.rectangle(im_out,zones[s_id][1][0],zones[s_id][1][1],(0,0,255),2)
            cv2.rectangle(im_out,zones[s_id][2][0],zones[s_id][2][1],(0,0,255),2)
            cv2.rectangle(im_out,zones[s_id][3][0],zones[s_id][3][1],(0,0,255),2)
    
            while labels[i][0] == frame:
                cv2.rectangle(im_out,(int(labels[i][2]),int(labels[i][3])),(int(labels[i][2])+int(labels[i][4]),int(labels[i][3])+int(labels[i][5])),(255,0,0),2)
                cv2.putText(im_out,str(labels[i][1]),(int(labels[i][2]),int(labels[i][3])),cv2.FONT_HERSHEY_PLAIN, 
                                                                            max(1.0, im.shape[1]/1200),(0,255,255),thickness = 2)
                

                for j,bbox in enumerate(zones[s_id]):
                    if iou([labels[i][2],labels[i][3],labels[i][2]+labels[i][4],
                                        labels[i][3]+labels[i][5]],[bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]) > 0.0:
                        if labels[i][1] not in zone_cnt[j+1]:
                            zone_cnt[j+1].append(labels[i][1])
                        else:
                            continue     

                i += 1
                if i == len(labels):
                    break 

            cv2.putText(im_out,str(len(zone_cnt[1])),(zones[s_id][0][0]),cv2.FONT_HERSHEY_PLAIN, 
                                                                            max(1.0, im.shape[1]/1200),(0,0,255),thickness = 2)
            cv2.putText(im_out,str(len(zone_cnt[2])),(zones[s_id][1][0]),cv2.FONT_HERSHEY_PLAIN, 
                                                                            max(1.0, im.shape[1]/1200),(0,0,255),thickness = 2)
            cv2.putText(im_out,str(len(zone_cnt[3])),(zones[s_id][2][0]),cv2.FONT_HERSHEY_PLAIN, 
                                                                            max(1.0, im.shape[1]/1200),(0,0,255),thickness = 2)
            cv2.putText(im_out,str(len(zone_cnt[4])),(zones[s_id][3][0]),cv2.FONT_HERSHEY_PLAIN, 
                                                                            max(1.0, im.shape[1]/1200),(0,0,255),thickness = 2)


            cv2.imwrite('/home/wei/Desktop/test/viz/{}/{}.jpg'.format(seq,"%04d"%frame),im_out)
        


