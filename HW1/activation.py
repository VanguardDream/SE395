e = 2.71828182

import numpy

def sigmoid (x):
    cache = x
    A = 1 / (1 + numpy.exp(-x))
    return A, cache

def relu (x):
    cache = x
    A = numpy.maximum(0,x)
    return A, cache

def lrelu (x, slope = 0.2):
    cache = x
    x = numpy.where(x > 0, x, slope * x)

    return x, cache

def sigmoid_back(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_back(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def lrelu_back(dA,cache,slope = 0.2):
    Z = cache

    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = slope

    assert (dZ.shape == Z.shape)

    return dZ