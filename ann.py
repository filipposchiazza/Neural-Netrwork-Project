"Diamo inizio al gioco...vediamo se mi ricordo ancora come si fa"

import numpy as np

class Ann:
    
    def __init__(self, num_inputs, num_hidden, num_outputs):
        
        """Initialize the Artificial Neural Network
        
        Parameters
        ----------
        num_inputs : int
            The number of inputs to be feed to the Neural network (the dimensionality of the dataset)
        num_hidden : array_like
            The element i of the array specifies the number of neurons in the hidden layer i, so the total number of hidden layer is given
            by the dimension of the array
        num_output : int
            The number of outputs of the Neural Network
        
        Returns
        -------
        out : ann
            An ann object with the specified number of inputs, hidden layers, number of neurons in each hidden layer and number of output
        
        Example
        -------
        >>> ann.Ann(10, [5, 3], 2)
        
        """
        
        
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        self.layers = [self.num_inputs] + self.num_hidden + [self.num_outputs] 
        
        self.weights = []
        
        for i in range (len(self.layers) - 1):
            w = np.random.rand(self.layers[i] + 1, self.layers[i+1])
            self.weights.append(w)
        
    
    def _forward_prop(self):
        pass
    
    def _backward_prop(self):
        pass
    
    def _gradient_descendent(self):
        pass
    
    def train(self):
        pass
    
    def predict(self):
        pass
    
    
    
        