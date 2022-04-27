#Different type of activation function and their derivatives to be used in the Artificial Neural Network

import numpy as np

##############################################################################

def sigmoid(x):
    """Definition of the sigmoid function
    
    Parameters
    ----------
    x : float
        Input of the sigmoid function.

    Returns
    -------
    TYPE : float
        Output of the sigmoid function.

    """
    x = np.where(x > -709, x, -709)    # Avoid problem of inf value in np.exp(x)
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    """Definition of the jacobian (trivial) of the sigmoid function
    
    Parameters
    ----------
    x : float or array-like
        Input of the derivative/jacobian of the sigmoid function.

    Returns
    -------
    jacobian : numpy array
        Jacobian matrix of the sigmoid function, evaluated for the input x; it is trivial because all the elements outside the principal diagonal are equal to zero.
        In the particular case of a float as input, the ouput will be a numpy mono-dimensional array with one element (the simple derivative).

    """
    x = np.array(x)
    lenght = np.size(x)
    jacobian = np.zeros((lenght, lenght))
    for i in range(lenght):
        jacobian[i][i] = sigmoid(x[i]) * (1 - sigmoid(x[i]))
    return jacobian

##############################################################################

def softmax(x):
    """Definition of the softmax activation function
    
    Parameters
    ----------
    x : array-like
        Input of the softmax function.

    Returns
    -------
    TYPE : numpy array
        Output of the softmax function, same dimension of the input.

    """
    num = np.exp(x - np.max(x))
    den = np.sum(num)
    return num / den


def deriv_softmax(x):
    """Definition of the softmax function jacobian

    Parameters
    ----------
    x : array-like
        Input of the softmax function jacobian.

    Returns
    -------
    jacobian : numpy array
        Jacobian matrix of the softmax function, evaluated for the input x.

    """
    lenght = np.size(x)
    a = softmax(x)
    jacobian = np.zeros((lenght, lenght))
    for i in range(lenght):
        for j in range(lenght):
            if i==j:
                jacobian[i][j] = a[i] * (1 - a[i])
            else:
                jacobian[i][j] = - a[i] * a[j]
    return jacobian

##############################################################################
