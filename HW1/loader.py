import sys
import os

from array import array
from struct import *

import numpy as np

def load(batchsize, fp_img, fp_label):
    tmp_img = fp_img.read(batchsize * 784)
    tmp_label = fp_label.read(batchsize)

    Y = np.zeros((0,10))

    X = np.reshape(unpack(len(tmp_img)*'B',tmp_img),(batchsize,784))
    tmp_Y = np.reshape(unpack(len(tmp_label)*'B',tmp_label),(batchsize))

    for i in range(0,batchsize):
        lbl = np.zeros((10))
        lbl[tmp_Y[i]] = 1
        Y = np.append(Y,np.reshape(lbl,(1,10)),axis=0)
    
    return X.T, Y.T