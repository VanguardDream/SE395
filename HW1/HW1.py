import numpy as np
import sys
import os

from array import array
from struct import *
import matplotlib

codePath = os.path.dirname( os.path.abspath("HW1.py"))

# fp_trainImage = open(codePath+'\\data\\train\\train-images.idx3-ubyte','rt',encoding='UTF8')
# fp_trainLabel = open(codePath+'\\data\\train\\train-labels.idx1-ubyte','rt',encoding='UTF8')

# img_loaded = np.zeros((28,28),dtype=int)
# label_loaded = [[],[],[],[],[],[],[],[],[],[]]
# d = 0 
# l = 0 
# idx = 0

# s = fp_trainImage.read(16)
# l = fp_trainLabel.read(8)

# _iter = 0

# while True:
#     s = fp_trainImage.read(28*28)
#     l = fp_trainLabel.read(1)

#     if not s:
#         break
#     if not l:
#         break

#     idx = int(l[0])

#     img_loaded = np.reshape(unpack(len(s)*'b',s),(28,28))
#     label_loaded[idx].append(img)

#     k=k+1

# matplotlib.pyplot.imshow(img_loaded,cmap=cm.binary)
# matplotlib.pyplot.imshow()

fp_image = open(codePath+'\\data\\train\\train-images.idx3-ubyte')
fp_label = open(codePath+'\\data\\train\\train-labels.idx1-ubyte')

img = np.zeros((28,28))                                # image resolution (28 x 28)

lbl = [ [],[],[],[],[],[],[],[],[],[] ]
d = 0
l = 0
index=0


s = fp_image.read(16)	#read first 16byte
l = fp_label.read(8)	#read first  8byte

#print(s)
#print("s_len:",len(s))
#print(l)
#print("l_len:",len(l))

"""
#single example - no loop
s = fp_image.read(784)
l = fp_label.read(1)
print("number:",int(l[0]))
img = np.reshape( unpack(len(s)*'B',s), (28,28))

#print(img)
plt.imshow(img,cmap = cm.binary)
plt.show()
"""


k=0
#read mnist and show character
while True:	
	s = fp_image.read(784)
	l = fp_label.read(1)

	if not s:
		break;
	if not l:
		break;

	index = int(l[0])
	#print(k,":",index)
	
	#no-loop
	img = np.reshape( unpack(len(s)*'B',s), (28,28))

	"""	
	#loop
	for i in range(0,28):
		for j in range(0,28):
			#print(i,j)
			d = s[(i*28)+j]		
			img[i][j] = d
			#print('%02x'%(d),end="")
	"""	

	lbl[index].append(img)
		
	k=k+1
#print(img)

plt.imshow(img,cmap = cm.binary)
plt.show()

print(np.shape(lbl))

print("read done")