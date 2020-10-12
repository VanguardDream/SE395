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