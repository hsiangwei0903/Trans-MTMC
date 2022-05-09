'''
We use this file to get the matching result. 
(Using Euclidean distance, subtract mean feature of every camera and use top k(3) feature association.)

hsiangwei
'''
import numpy as np
from numpy import genfromtxt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from numpy import genfromtxt

def update_global(global_id_dic,seq1,id1,seq2,id2):
    new = True
    for g_id in global_id_dic:
        if (seq1,id1) in global_id_dic[g_id]:
            global_id_dic[g_id].append((seq2,id2))
            new = False
            break
    if new:
        new_g_id = len(global_id_dic)+1
        global_id_dic[new_g_id] = [(seq1,id1),(seq2,id2)]

def get_dist(emb_dics,seq_in,car1,seq_out,car2):
    dist = []
    for emb_in in emb_dics[seq_in][car1]:
        for emb_out in emb_dics[seq_out][car2]:
            dist.append([distance.euclidean(emb_in,emb_out)])
    
    k = 3 # TODO experiemnt on this

    dist = np.mean(np.array(sorted(dist)[:k]),axis = 0).tolist()[0]

    return dist

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

        cost = np.ones([len(car_in),len(car_out)])*100

        for n1,car1 in enumerate(car_in):
            for n2,car2 in enumerate(car_out):
                if len(time_dics[seq_out][car2])==1 or len(time_dics[seq_in][car1])==1 or (time_dics[seq_out][car2][1]-time_dics[seq_out][car2][0])<=5 or (time_dics[seq_in][car1][1]-time_dics[seq_in][car1][0])<=5: # blink detection
                        cost[n1][n2] == 100
                # if car1 only pass 1 zone and it is not in first nor last frame
                elif (zone_dics[seq_in][car1][0]==zone_dics[seq_in][car1][1]) and (time_dics[seq_in][car1][0]!=1 and time_dics[seq_in][car1][1]!=2000):
                    cost[n1][n2] == 100
                # if car2 only pass 1 zone and it is not in first nor last frame
                elif (zone_dics[seq_out][car2][0]==zone_dics[seq_out][car2][1]) and (time_dics[seq_out][car2][0]!=1 and time_dics[seq_out][car2][1]!=2000):
                    cost[n1][n2] == 100
                elif time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] > time_thres[0] and time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] < time_thres[1]:
                    dis = get_dist(emb_dics,seq_in,car1,seq_out,car2)
                    if dis<20:    
                        cost[n1][n2] = dis

        np.save('/home/wei/Desktop/test/cost_matrix/{}to{}.npy'.format(seq_in,seq_out),cost)

        row_ind,col_ind = linear_sum_assignment(cost)

        matches = 0

        for match in range(min(len(row_ind),len(col_ind))):
            if cost[row_ind[match]][col_ind[match]]<emb_thres:
                matches += 1
                t = time_dics[seq_out][car_out[col_ind[match]]][0] - time_dics[seq_in][car_in[row_ind[match]]][1]
                t2 = (time_dics[seq_in][car_in[row_ind[match]]][1],time_dics[seq_out][car_out[col_ind[match]]][0])
                print('matching c0{} {} and c0{} {} with time interval {} and cost {:.2f}'.format(seq_in+41,car_in[row_ind[match]],seq_out+41,car_out[col_ind[match]],t2,cost[row_ind[match]][col_ind[match]]))
                update_global(global_id_dic,seq_in,car_in[row_ind[match]],seq_out,car_out[col_ind[match]])
        
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

        cost = np.ones([len(car_in),len(car_out)])*100

        for n1,car1 in enumerate(car_in):
            for n2,car2 in enumerate(car_out):
                if len(time_dics[seq_out][car2])==1 or len(time_dics[seq_in][car1])==1 or (time_dics[seq_out][car2][1]-time_dics[seq_out][car2][0])<=5 or (time_dics[seq_in][car1][1]-time_dics[seq_in][car1][0])<=5: # blink detection
                        cost[n1][n2] == 100
                # if car1 only pass 1 zone and it is not in first nor last frame
                elif (zone_dics[seq_in][car1][0]==zone_dics[seq_in][car1][1]) and (time_dics[seq_in][car1][0]!=1 and time_dics[seq_in][car1][1]!=2000):
                    cost[n1][n2] == 100
                # if car2 only pass 1 zone and it is not in first nor last frame
                elif (zone_dics[seq_out][car2][0]==zone_dics[seq_out][car2][1]) and (time_dics[seq_out][car2][0]!=1 and time_dics[seq_out][car2][1]!=2000):
                    cost[n1][n2] == 100
                elif time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] > time_thres[0] and time_dics[seq_out][car2][0] - time_dics[seq_in][car1][1] < time_thres[1]:
                    dis = get_dist(emb_dics,seq_in,car1,seq_out,car2)
                    if dis<20:    
                        cost[n1][n2] = dis
                
        np.save('/home/wei/Desktop/test/cost_matrix/{}to{}.npy'.format(seq_in,seq_out),cost)

        row_ind,col_ind = linear_sum_assignment(cost)
        matches = 0

        for match in range((len(row_ind))):
            #print('matching c0{} {} and c0{} {}'.format(seq_in+41,car_in[row_ind[match]],seq_out+41,car_out[col_ind[match]]))
            if cost[row_ind[match]][col_ind[match]]<emb_thres:
                matches += 1
                t = time_dics[seq_out][car_out[col_ind[match]]][0] - time_dics[seq_in][car_in[row_ind[match]]][1]
                t2 = (time_dics[seq_in][car_in[row_ind[match]]][1],time_dics[seq_out][car_out[col_ind[match]]][0])
                print('matching c0{} {} and c0{} {} with time interval {} and cost {:.2f}'.format(seq_in+41,car_in[row_ind[match]],seq_out+41,car_out[col_ind[match]],t2,cost[row_ind[match]][col_ind[match]]))
                update_global(global_id_dic,seq_in,car_in[row_ind[match]],seq_out,car_out[col_ind[match]])

        print('total matches {}'.format(matches))


if __name__ == "__main__":
    zone_dics = np.load('/home/wei/Desktop/test/npy/zone_dics.npy',allow_pickle=True).tolist()
    emb_dics = np.load('/home/wei/Desktop/test/npy/emb_dics_all.npy',allow_pickle=True).tolist() # Read frame based embedding feature from the data
    time_dics = np.load('/home/wei/Desktop/test/npy/time_dics.npy',allow_pickle=True).tolist()
    
    emb_thresholds_r = [20,20,20,20,20] # [46-45,45-44,44-43,43-42,42-41]
    time_thresholds_r = [(320,550),(260,500),(350,633),(200,500),(400,800)] # [46-45,45-44,44-43,43-42,42-41]
    emb_thresholds_l = [20,20,20,20,20] # [46-45,45-44,44-43,43-42,42-41]
    time_thresholds_l = [(250,450),(210,400),(350,700),(160,400),(680,1115)] # [46-45,45-44,44-43,43-42,42-41]
    
    global_id_dic = {}
    global_emb_dic = {}

    # substract mean feature of every sequence
    for s_id,seq in enumerate(emb_dics):
        seq_feature = []
        for car in seq:
            for feature in seq[car]:
                seq_feature.append(feature)
        mean_feature = np.mean(np.array(seq_feature),axis=0)
        for car in seq:
            for f_id,feature in enumerate(seq[car]):
                emb_dics[s_id][car][f_id] = (feature-mean_feature).tolist()

    # going left association
    for s_id in range(0,5):
        matcher(s_id,s_id+1,emb_thresholds_l[4-s_id],time_thresholds_l[4-s_id],global_id_dic,global_emb_dic,zone_dics,emb_dics,time_dics)
    
    #assert len(global_id_dic) == len(global_emb_dic)

    # going right association
    for s_id in range(5,0,-1):
        matcher(s_id,s_id-1,emb_thresholds_r[5-s_id],time_thresholds_r[5-s_id],global_id_dic,global_emb_dic,zone_dics,emb_dics,time_dics)
    
    #assert len(global_id_dic) == len(global_emb_dic)


    print('===========================================')
    print('total matching : ', len(global_id_dic))
    print('===========================================')
    print('finish matching')
    
    #print(global_id_dic)

    np.save('/home/wei/Desktop/test/npy/global_id_dic_euclidean_topk.npy',global_id_dic)
    #np.save('/home/wei/Desktop/test/npy/global_emb_dic.npy',global_emb_dic)