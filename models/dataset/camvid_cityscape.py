import numpy as np
def c2c(label,mode):
    trans=[255,255,18,255,2,13,255,255,5,4,255,255,255,17,255,255,11,0,9,1,7,10,14,255,6,16,8,15,255,8,255,3,255]
    if mode=='train':
        res = label
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                res[i][j]=trans[label[i][j]]
    elif mode=='val':
        res = np.zeros((label.shape[0],label.shape[1],label.shape[2]))
        for i in range(label.shape[1]):
            for j in range(label.shape[2]):
                res[0][i][j]=trans[label[0][i][j]]
    return res