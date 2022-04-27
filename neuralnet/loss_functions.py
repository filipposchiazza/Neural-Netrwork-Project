#Define different types of Loss functions
import numpy as np


clip_value = 1e-15
###############################################################################

# Binary Cross Entropy is used for binary classification tasks

def binary_cross_entropy(prediction, target):
    """Binary Cross Entropy loss function

    Parameters
    ----------
    prediction : float
        Neural network prediction or, in other words, the output of the feedforward phase.
    target : float
        Label linked with the neural network input.

    Returns
    -------
    float
        Evaluation of the binary cross entropy between prediction and target.
        
    """
    # Use np.clip in order to avoid nan problems when evaluating np.log()
    prediction = np.clip(prediction, a_min=clip_value, a_max=1-clip_value)
    term_1 = target * np.log(prediction)
    term_2 = (1 - target) * np.log(1 - prediction)
    return -(term_1 + term_2)


def binary_cross_entropy_deriv(prediction, target):
    """Derivative of the Binary Cross Entropy

    Parameters
    ----------
    prediction : float
        Neural network prediction or, in other words, the output of the feedforward phase.
    target : float
        Label linked with the neural network input.

    Returns
    -------
    float
        Evaluation of the binary cross entropy derivative (with respect to the prediction) evaluated for prediction and target.

    """
    # Use np.clip in order to avoid nan problems when evaluating the fraction target/prediction
    prediction = np.clip(prediction, a_min=clip_value, a_max=1-clip_value)
    term_1 = target / prediction 
    term_2 = (1 - target) / (1 - prediction)
    return term_2 - term_1

###############################################################################

# Cross entropy is used for multiple-class classification tasks

def cross_entropy(prediction, target):
    """Cross Entropy loss function
    
    Parameters
    ----------
    prediction : array-like
        Neural network prediction or, in other words, the output of the feedforward phase.
    target : array-like
        Label linked with the neural network input.

    Returns
    -------
    float
       Evaluation of the cross entropy between prediction and target.

    """
    # Use np.clip in order to avoid nan problems when evaluating np.log()
    prediction = np.clip(prediction, a_min=clip_value, a_max=1)
    log = np.log(prediction)
    ylog = target * log
    return -np.sum(ylog)


def cross_entropy_deriv(prediction, target):
    """Derivative of the Cross Entropy

    Parameters
    ----------
    prediction : array-like
        Neural network prediction or, in other words, the output of the feedforward phase.
    target : array-like
        Label linked with the neural network input.

    Returns
    -------
    numpy array
        Cross entropy derivative (with respect to the prediction) evaluated for prediction and target.

    """
    # Use np.clip in order to avoid nan problems when evaluating the fraction target/prediction
    prediction = np.clip(prediction, a_min=clip_value, a_max=1)
    target = np.array(target)
    return - target/prediction

###############################################################################
