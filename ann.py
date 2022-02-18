"Diamo inizio al gioco...vediamo se mi ricordo ancora come si fa"

import numpy as np
import activation_functions as act

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
        self.num_hidden = np.array(num_hidden)
        self.num_outputs = num_outputs
        self.layers = np.concatenate(([self.num_inputs], self.num_hidden, [self.num_outputs]))
        
        self.weights = []
        
        for i in range (self.layers.size - 1):
            w = np.random.rand(int(self.layers[i]) + 1, int(self.layers[i+1]))
            self.weights.append(w)      
        # Attention: here weights is a list of bi-dimensional numpy arrays
        #self.weights = np.array(self.weights)
        
    def forward_prop(self, inputs, activation_func):
        
        """Perform the forward propagation
        
        Parameters
        ----------
        inputs : array_like
            Vector of inputs for the Neural Network
        activation_func : function
            Activation function used by each neuron to produce its output
        
        Returns
        -------
        out : array_like
            Array of outputs of the neural network
            
        Example
        -------
        >> ann.Ann(4, [3, 2], 2)
        >> result = Ann.forward_prop([10, 20, 30, 40], sigmoid)
        
        """
        
        self.activation_func = activation_func
        activations = np.array(inputs)
        activations = np.insert(activations, 0, 1)
        
        for i in range(self.layers.size - 1):
            # Calculate the linear combination between inputs of the previous layer and weights of the current one
            linear_part = np.dot(activations, self.weights[i])      
            
            # Apply the activation function to the linear part
            activations = self.activation_func(linear_part)
            
            # Add the first term equal to 1 in order to consider the bias
            activations = np.insert(activations, 0, 1)
        
        # Remove the addictional term at the beginning of the output array
        activations = np.delete(activations, 0)
        return activations
            
    
    def backward_prop(self):
        pass
    
    def gradient_descendent(self):
        pass
    
    def train(self):
        pass
    
    def predict(self):
        pass
    
    
if __name__ == '__main__':
    
    nn = Ann(1, [], 2)
    result = nn.forward_prop([0.5], act.deriv_sigmoid)
    print(result)
    pass
        