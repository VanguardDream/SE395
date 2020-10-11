import numpy

def softmax(x):
    exps = numpy.exp(x)
    return exps / numpy.sum(exps)

def crossEntropy(prediction, groundTruth):
    p = softmax(prediction)

    log_likelihood = -numpy.log(p)
    loss = numpy.sum(log_likelihood)
    return loss