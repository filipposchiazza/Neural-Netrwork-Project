# Define different types of Loss functions
import numpy as np


###############################################################################

# MSE is generally used for regression tasks
def mse(prediction, target):
    "Mean Squared Error loss function"
    difference = prediction - target
    difference_squared = difference**2
    return np.sum(difference_squared) / len(difference) 

def mse_deriv(prediction, target):
    "Derivative of Mean Squared Error"
    difference = prediction - target
    return 2 / len(difference) * difference

###############################################################################

# Cross Entropy is used for classification tasks
def cross_entropy(prediction, target):
    "Cross Entropy loss function"
    log = np.log(prediction)
    ylog = target * log
    return -np.sum(ylog)

def cross_entropy_deriv(prediction, target):
    "Derivative of the Cross Entropy"
    return - target / prediction

###############################################################################