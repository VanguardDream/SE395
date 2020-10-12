e = 2.71828182

import numpy

def sigmoid (x):
    return 1 / (1 + numpy.exp(-x))

def relu (x):
    return numpy.maximum(0,x)

def lrelu (x, slope = 0.2):
    x[x >= 0] = x
    x[x < 0] = slope * x

    return x