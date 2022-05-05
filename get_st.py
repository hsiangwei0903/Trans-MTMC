'''
Get the moving direction, entry time and exit time of every car (spatio-temporal information)

hsiangwei
'''
from numpy import genfromtxt
import numpy as np
import glob 
import copy
from PIL import Image
from iou import iou
import os
from viz import zones, seqs, path_in


def update_zone(dic,id,zone_id):
    if id not in dic:
        dic[id] = [zone_id]
    elif zone_id not in dic[id]:
        dic[id].append(zone_id)
    # else:
    #     if zone_id not in dic[id]:
    #         dic[id].append(zone_id)
    #     else:
    #         if dic[id][-1] != zone_id:
    #             dic[id].remove(zone_id)
    #             dic[id].append(zone_id)
    return dic

def update_time(dic,id,frame):
    if id not in dic:
        dic[id] = [frame] #initialize
    else:
        if len(dic[id])==1:
            dic[id].append(frame)
        else:
            dic[id][1] = frame
    return dic

def filter_dic(zone_dic,time_dic):
    for car in list(time_dic.keys()):
        if car not in zone_dic:
            time_dic.pop(car) # filter out those car not passing any zone
        elif len(zone_dic[car])==1:
            zone_dic[car].append(zone_dic[car][0])
        elif len(zone_dic[car])==3:
            zone_dic[car] = [zone_dic[car][0],zone_dic[car][-1]]
        elif len(zone_dic[car])==4:
            zone_dic[car] = [zone_dic[car][0],zone_dic[car][-1]]
    return zone_dic,time_dic
        

zone_dics = []
time_dics = []

for s_id,seq in enumerate(seqs):
    print('processing sequence {}'.format(seq))
    time_dic = {}
    zone_dic = {}
    file_path = path_in + seq
    labels_path = '/home/wei/Desktop/test/SCT/{}.txt'.format(seq)
    imgs = sorted(glob.glob(file_path+'/img/*')) # start with one
    labels = genfromtxt(labels_path, delimiter=',', dtype=None)
    for n,label in enumerate(labels):
        if n%10000 == 0:
            print('processing label {}/{}'.format(n,len(labels)-1))
        
        time_dic = update_time(time_dic,label[1],label[0])  # update timeline

        for j,bbox in enumerate(zones[s_id]):
                if iou([label[2],label[3],label[2]+label[4],label[3]+label[5]],[bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]) > 0:
                    zone_dic = update_zone(zone_dic,labels[n][1],j)

        if n == len(labels):
            break

    zone_dic, time_dic = filter_dic(zone_dic,time_dic)

    assert len(zone_dic)==len(time_dic)

    zone_dics.append(zone_dic)
    time_dics.append(time_dic)



np.save('/home/wei/Desktop/test/npy/zone_dics.npy',zone_dics)
np.save('/home/wei/Desktop/test/npy/time_dics.npy',time_dics)



