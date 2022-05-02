'''
We use this file to get the matching result.

hsiangwei
'''
import numpy as np
from numpy import genfromtxt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
import cv2
from numpy import genfromtxt
import glob 
import copy
from PIL import Image
from iou import iou

def update_global(global_id_dic,global_emb_dic,seq1,id1,emb1,seq2,id2,emb2):
    new = True
    for g_id in global_id_dic:
        if (seq1,id1) in global_id_dic[g_id]:
            global_id_dic[g_id].append((seq2,id2))
            global_emb_dic[g_id] = ((np.array(global_emb_dic[g_id])+np.array(emb2))/2).tolist()
            new = False
            break
    if new:
        new_g_id = len(global_id_dic)+1
        global_id_dic[new_g_id] = [(seq1,id1),(seq2,id2)]
        global_emb_dic[new_g_id] = ((np.array(emb1)+np.array(emb2))/2).tolist()

def matcher(seq_in,seq_out,emb_thres,time_thres,global_id_dic,global_emb_dic,zone_dics,emb_dics,time_dics): # seq_in,out = seq_id
    assert seq_in != seq_out
    print('matching {} and {} with emb_thres = {} and time_thres = {}'.format(seq_in+41,seq_out+41,emb_thres,time_thres))
    if seq_in > seq_out: # going right
        car_in = []
        car_out = []
        for car in zone_dics[seq_in]:
            if zone_dics[seq_in][car][1] == 2:
                car_in.append(car)
        for car in zone_dics[seq_out]:
            if zone_dics[seq_out][car][0] == 0:
                car_out.append(car)

        cost = np.ones([len(car_in),len(car_out)])

        for n1,car1 in enumerate(car_in):
            for n2,car2 in enumerate(car_out):
                if distance.cosine(emb_dics[seq_in][car1],emb_dics[seq_out][car2])<0.5: # TODO Tune this
                    if time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] > time_thres[0] and time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] < time_thres[1]:
                        cost[n1][n2] = distance.cosine(emb_dics[seq_in][car1],emb_dics[seq_out][car2])     
                    else:
                        cost[n1][n2] = 1
                else:
                    cost[n1][n2] = 1

        row_ind,col_ind = linear_sum_assignment(cost)

        matches = 0

        for match in range(min(len(row_ind),len(col_ind))):
            if cost[row_ind[match]][col_ind[match]]<emb_thres:
                matches += 1
                t = time_dics[seq_out][car_out[col_ind[match]]][0] - time_dics[seq_in][car_in[row_ind[match]]][1]
                #print('matching c0{} {} and c0{} {} with time interval {}'.format(seq_in+41,car_in[row_ind[match]],seq_out+41,car_out[col_ind[match]],t))
                update_global(global_id_dic,global_emb_dic,seq_in,car_in[row_ind[match]],emb_dics[seq_in][car1],seq_out,car_out[col_ind[match]],emb_dics[seq_out][car2])
        
        print('total matches {}'.format(matches))

    elif seq_in < seq_out: # going left
        car_in = []
        car_out = []
        for car in zone_dics[seq_in]:
            if zone_dics[seq_in][car][1] == 0:
                car_in.append(car)
        for car in zone_dics[seq_out]:
            if zone_dics[seq_out][car][0] == 2:
                car_out.append(car)

        cost = np.ones([len(car_in),len(car_out)])

        for n1,car1 in enumerate(car_in):
            for n2,car2 in enumerate(car_out):
                if distance.cosine(emb_dics[seq_in][car1],emb_dics[seq_out][car2])<0.5: # TODO Tune this
                    if time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] > time_thres[0] and time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] < time_thres[1]:
                        cost[n1][n2] = distance.cosine(emb_dics[seq_in][car1],emb_dics[seq_out][car2])     
                    else:
                        cost[n1][n2] = 1
                else:
                    cost[n1][n2] = 1

        row_ind,col_ind = linear_sum_assignment(cost)

        matches = 0

        for match in range(min(len(row_ind),len(col_ind))):
            if cost[row_ind[match]][col_ind[match]]<emb_thres:
                matches += 1
                t = time_dics[seq_out][car_out[col_ind[match]]][0] - time_dics[seq_in][car_in[row_ind[match]]][1]
                #print('matching c0{} {} and c0{} {} with time interval {}'.format(seq_in+41,car_in[row_ind[match]],seq_out+41,car_out[col_ind[match]],t))
                update_global(global_id_dic,global_emb_dic,seq_in,car_in[row_ind[match]],emb_dics[seq_in][car1],seq_out,car_out[col_ind[match]],emb_dics[seq_out][car2])
        
        print('total matches {}'.format(matches))


'''

def check_overlap(car1,car2,global_id_dic,global_emb_dic,zone_dics,emb_dics,time_dics):
    if distance.cosine(global_emb_dic[car1],global_emb_dic[car2])<0.3:
        pass_camera1 = set()
        pass_camera2 = set()
        for pair1 in global_id_dic[car1]:
            pass_camera1.add(pair1[0])
        for pair2 in global_id_dic[car2]:
            pass_camera2.add(pair2[0])
        if len(pass_camera1.intersection(pass_camera2))==0:
            for pair in global_id_dic[car2]:
                global_id_dic[car1].append(pair)
                print('merging {} and {}'.format(car1,car2))
            global_id_dic.pop(car2)

'''




if __name__ == "__main__":
    zone_dics = np.load('/home/wei/Desktop/test/npy/zone_dics.npy',allow_pickle=True).tolist()
    emb_dics = np.load('/home/wei/Desktop/test/npy/emb_dics.npy',allow_pickle=True).tolist()
    time_dics = np.load('/home/wei/Desktop/test/npy/time_dics.npy',allow_pickle=True).tolist()
    
    emb_thresholds_r = [0.40,0.45,0.38,0.35,0.35] # [46-45,45-44,44-43,43-42,42-41]
    time_thresholds_r = [(540,750),(260,500),(350,633),(200,500),(400,800)] # [46-45,45-44,44-43,43-42,42-41]
    emb_thresholds_l = [0.32,0.45,0.38,0.35,0.35] # [46-45,45-44,44-43,43-42,42-41]
    time_thresholds_l = [(410,550),(210,400),(400,800),(160,300),(680,900)] # [46-45,45-44,44-43,43-42,42-41]
    
    global_id_dic = {}
    global_emb_dic = {}

    # going left association
    for s_id in range(0,5):
        matcher(s_id,s_id+1,emb_thresholds_l[4-s_id],time_thresholds_l[4-s_id],global_id_dic,global_emb_dic,zone_dics,emb_dics,time_dics)
    
    assert len(global_id_dic) == len(global_emb_dic)

    # going right association
    for s_id in range(5,0,-1):
        matcher(s_id,s_id-1,emb_thresholds_r[5-s_id],time_thresholds_r[5-s_id],global_id_dic,global_emb_dic,zone_dics,emb_dics,time_dics)
    
    assert len(global_id_dic) == len(global_emb_dic)

    '''
    for car1 in global_emb_dic:
            for car2 in global_emb_dic:
                if car1 != car2:
                    check_overlap(car1,car2,global_id_dic,global_emb_dic,zone_dics,emb_dics,time_dics)
    '''

    print('===========================================')
    print('total matching : ', len(global_id_dic))
    print('===========================================')
    print('finish matching')
    
    print(global_id_dic)

    np.save('/home/wei/Desktop/test/npy/global_id_dic.npy',global_id_dic)