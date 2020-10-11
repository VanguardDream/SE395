import numpy as np
import sys
import os

from array import array
from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import timeit

import activation
import loss

codePath = os.path.dirname( os.path.abspath("HW1.py"))

fp_trainImage = open(codePath+'\\data\\train\\train-images.idx3-ubyte','rb')
fp_trainLabel = open(codePath+'\\data\\train\\train-labels.idx1-ubyte','rb')

#Jump MNIST file header
tmp = fp_trainImage.read(16)
tmp = fp_trainLabel.read(8)

img_load = np.zeros((28,28),dtype=int)
label_load = np.zeros((1,10),dtype=int)

bin_img = np.reshape(np.array([]),(0,28*28))
bin_label = np.reshape(np.array([]),(0,1))

# #iterative image loader
# while True:
# # for i in range(10000):
#     tmp_img = fp_trainImage.read((784))
#     tmp_label = fp_trainLabel.read(1)

#     if tmp_img == '':
#         break
#     if tmp_label == '':
#         break

#     tmp_img = np.reshape(unpack(784*'B',tmp_img),(1,784))
#     tmp_label = np.reshape(int.from_bytes(tmp_label,byteorder='big',signed=False),(1,1))

#     bin_img = np.append(bin_img, np.array(tmp_img),axis=0)
#     bin_label = np.append(bin_label, tmp_label, axis=0)

# # Python built in types debugging 
# print(unpack(len(tmp)*'B',tmp))
# print(type(unpack(len(tmp)*'B',tmp)))
# print(img_load)
# print(label_load)

#----------------------------------------------------------------
bin_img = fp_trainImage.read((784))
bin_label = fp_trainLabel.read(1)

layer_img = np.reshape(unpack(len(bin_img)*'B',bin_img),(784,1))
label = np.array([0,0,0,0,0,0,0,0,0,0])

label[int.from_bytes(bin_label,byteorder='big',signed=False)] = 1

w_1 = np.zeros((784,784))
b_1 = 1
w_2 = np.zeros((784,784))
b_2 = 1
w_3 = np.zeros((784,784))
b_3 = 1

w_out = np.ones((784,10))


z_1 = np.dot(layer_img.T,w_1).T + b_1
a_1 = activation.relu(z_1)

z_2 = np.dot(a_1.T,w_2).T + b_2
a_2 = activation.relu(z_2)

z_3 = np.dot(a_2.T,w_3).T + b_3
a_3 = activation.relu(z_3)

z_out = np.dot(a_3.T,w_out).T
output = loss.softmax(z_out)

t = loss.crossEntropy(output, label)

dw_3 = np.dot(a_2,(a_3 - label).T)
print(dw_3)

# # show image debugging
# img_load = np.reshape(unpack(len(bin_img)*'B',bin_img),(28,28))
# label_load = int.from_bytes(bin_label,byteorder='big',signed=False)
# plt.imshow(img_load,cmap=cm.binary)
# plt.show()
# print(img_load)
# print(label_load)