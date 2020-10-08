import numpy as np
import sys
import os

from array import array
from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

codePath = os.path.dirname( os.path.abspath("HW1.py"))

fp_trainImage = open(codePath+'\\data\\train\\train-images.idx3-ubyte','rb')
fp_trainLabel = open(codePath+'\\data\\train\\train-labels.idx1-ubyte','rb')

#Jump MNIST file header
tmp = fp_trainImage.read(16)
tmp = fp_trainLabel.read(8)

img_load = np.zeros((28,28),dtype=int)
label_load = np.zeros((1,10),dtype=int)

# Python built in types debugging
# print(unpack(len(tmp)*'B',tmp))
# print(type(unpack(len(tmp)*'B',tmp)))
# print(img_load)
# print(label_load)

bin_img = fp_trainImage.read(28*28)
bin_label = fp_trainLabel.read(1)

layer_img =np.reshape(unpack(len(bin_img)*'B',bin_img),1)

print(layer_img)

img_load = np.reshape(unpack(len(bin_img)*'B',bin_img),(28,28))
label_load = int.from_bytes(bin_label,byteorder='big',signed=False)

plt.imshow(img_load,cmap=cm.binary)
plt.show()

print(img_load)
print(label_load)
