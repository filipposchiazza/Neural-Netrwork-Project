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

# Binary Cross Entropy is used for binary classification tasks
def binary_cross_entropy(prediction, target):
    "Binary Cross Entropy loss function"
    term_1 = target * np.log(prediction + 1e-15)
    term_2 = (1 - target) * np.log(1 - prediction + 1e-15)
    return -(term_1 + term_2)

def binary_cross_entropy_deriv(prediction, target):
    "Derivative of the Cross Entropy"
    #prediction = np.clip(prediction, 1e-15, 1e-15)
    term_1 = target / prediction 
    term_2 = (1 - target) / (1 - prediction)
    return term_2 - term_1

###############################################################################