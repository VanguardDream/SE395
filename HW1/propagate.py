import numpy as np
import activation
import loss

def propagate(w,b,X,Y):
    """
    Input
    w : weight mat.
    b : bias scalar.
    X : input image mat.
    Y : label mat.

    Return
    cost
    dw
    db
    """
    m = X.shape[1]

    A = activation.lrelu(np.dot(w.T,X) + b)
    # cost = loss.crossEntropy(m,A,Y)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)

    cost = np.squeeze(cost)

    print(A)
    print(cost)
    print(dw)
    print(db)

    return cost, dw ,db

def linear_forward(A,W,b):
    """
    Implement the linear part of a layer's forward propagation.
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation(A_prev, W, b, acti_func):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" and "lrelu"
    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if acti_func == 'sigmoid':
        Z, l_cache = linear_forward(A_prev,W,b)
        A, a_cache = activation.sigmoid(Z)
    
    elif acti_func == 'relu':
        Z, l_cache = linear_forward(A_prev,W,b)
        A, a_cache = activation.relu(Z)

    elif acti_func == 'lrelu':
        Z, l_cache = linear_forward(A_prev,W,b)
        A, a_cache = activation.lrelu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (l_cache, a_cache)

    return A, cache

def linear_activation_backward(dA, cache, acti_func):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    l_cache, a_cache = cache

    dA_prev = dA
    dW = l_cache
    db = np.sum(dA)


    if activation == "relu":
        dZ = activation.relu_back(dA, a_cache)
        dA_prev, dW, db = linear_backward(dZ, l_cache)

    elif activation == "lrelu":
        dZ = activation.lrelu_back(dA, a_cache)
        dA_prev, dW, db = linear_backward(dZ, l_cache)
        
    elif activation == "sigmoid":
        dZ = activation.sigmoid_back(dA, a_cache)
        dA_prev, dW, db = linear_backward(dZ, l_cache)

    return dA_prev, dW, db