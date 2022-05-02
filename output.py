'''
We use this file to generate the final tracking result.

hsiangwei
'''
import numpy as np
from numpy import genfromtxt

path_out = '/home/wei/Desktop/track1.txt'
result = np.load('/home/wei/Desktop/test/npy/global_id_dic.npy',allow_pickle=True).tolist()
file = open(path_out,"w")
for car in result:
    if car/len(result)==10:
        print('processing car {}'.format(car))
    for pair in result[car]:
        sct_file = ''
        sct = genfromtxt('/home/wei/Desktop/test/SCT/c0{}.txt'.format(str(41+pair[0])), delimiter=',', dtype=None)
        for label in sct:
            if label[1] == pair[1]:
                file.write(str(41+pair[0])+" ") # cam id
                file.write(str(car)+" ") # car id
                file.write(str(label[0])+" ") # frame id
                file.write(str(int(label[2]))+" "+str(int(label[3]))+" "+str(int(label[4]))+" "+str(int(label[5]))+" "+str(-1)+" "+str(-1)+"\n")