import numpy

def softmax(x):
    exps = numpy.exp(x)
    return exps / numpy.sum(exps)

def crossEntropy(m, A, Y):
    cost = (-1/m) * numpy.sum(Y * numpy.log(A) + (1 - Y) * (numpy.log(1 - A)))
    return cost