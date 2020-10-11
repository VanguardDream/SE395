e = 2.71828182

import numpy

def sigmoid (x):
    return 1 / (1 + numpy.power(e,x))

def relu (x):
    return numpy.maximum(0,x)



