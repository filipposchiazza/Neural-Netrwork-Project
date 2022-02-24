"Different type of activation function and their derivatives to be used in the Artificial Neural Network"

import numpy as np

##############################################################################

def sigmoid(x):
    "Definition of the sigmoid function"
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    "Definition of the derivative of the sigmoid function"
    return sigmoid(x) * (1 - sigmoid(x))

##############################################################################

def relu(x):
    "Definition of the Rectified Linear Unit function"
    return x * (x>0)


def deriv_relu(x):
    "Definition of the derivative of the Rectified Linear Unit function"
    return 1 * (x>0)
   
##############################################################################
    
def tanh(x):
    "Definition of the hyperbolic function tanh"
    num = 1 - np.exp(-2 * x)
    den = 1 + np.exp(-2 * x)
    return num/den


def deriv_tanh(x):
    "Definition of the derivative of the hyperbolic function tanh"
    return 1 - tanh(x)**2

##############################################################################
