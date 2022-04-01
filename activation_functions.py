#Different type of activation function and their derivatives to be used in the Artificial Neural Network

import numpy as np

##############################################################################

def sigmoid(x):
    "Definition of the sigmoid function"
    x = np.where(x > -709, x, -709)    # Avoid problem of inf value in np.exp(x)
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    "Definition of the jacobian (trivial) of the sigmoid function"
    x = np.array(x)
    lenght = np.size(x)
    jacobian = np.zeros((lenght, lenght))
    for i in range(lenght):
        jacobian[i][i] = sigmoid(x[i]) * (1 - sigmoid(x[i]))
    return jacobian

##############################################################################

def softmax(x):
    "Definition of the softmax activation function"
    num = np.exp(x - np.max(x))
    den = np.sum(num)
    return num / den


def deriv_softmax(x):
    "Definition of the softmax function jacobian"
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
"""
def relu(x):
    "Definition of the Rectified Linear Unit function"
    return x * (x>0)


def deriv_relu(x):
    "Definition of the jacobian of the Rectified Linear Unit function"
    lenght = len(x)
    jacobian = np.zeros((lenght, lenght))
    for i in range(lenght):
        jacobian[i][i] = 1 * (x[i]>0)
    return jacobian
   
##############################################################################

def elu (x, alpha=1):
    "Definition of the ELU function"
    result = np.where(x>0, x, alpha*(np.exp(x) - 1))
    return result


def deriv_elu(x, alpha=100):
    "Definition of the jacobian of the ELU function"
    result = np.where(x>0, 1, alpha * np.exp(x))
    lenght = len(x)
    jacobian = np.zeros((lenght, lenght))
    for i in range(lenght):
        jacobian[i][i] = result[i]
    return jacobian

##############################################################################
"""