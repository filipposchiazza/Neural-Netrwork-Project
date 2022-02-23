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
        
        # Create weights and initialize them with random values
        self.weights = []       
        for i in range (len(self.layers) - 1):
            w = np.random.rand(int(self.layers[i]) , int(self.layers[i+1]))
            self.weights.append(w)      
        # Attention: here weights is a list of bi-dimensional numpy arrays
        
        # Create biases and initialize them with random values
        self.biases = []
        for i in range (len(self.layers) - 1):
            b = np.random.rand(int(self.layers[i+1]))
            self.biases.append(b)
        
        # Create a list of array that will store the values of the neuron's linear combinations
        self.linear_comb = []
        for i in range (len(self.layers) - 1):
            single_layer = np.zeros(int(self.layers[i+1]))
            self.linear_comb.append(single_layer)
              
        
        # Create a list of array that will store the values of the neuron's activations
        self.activations = []
        for i in range (len(self.layers)):
            single_layer = np.zeros(int(self.layers[i])) 
            self.activations.append(single_layer)
        
        # Create a list of array that will store the weights' derivatives
        self.weights_deriv = []
        for i in range(len(self.layers) - 1):
            d_w = np.zeros((int(self.layers[i]) , int(self.layers[i+1])))
            self.weights_deriv.append(d_w)
        
        # Create a list of array that will store the biases' derivatives
        self.biases_deriv = []
        for i in range (len(self.layers) - 1):
            d_b = np.random.rand(int(self.layers[i+1]))
            self.biases_deriv.append(d_b)
        
        
        
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
        self.activations[0] = activations
        
        for i in range(self.layers.size - 1):
            # Calculate the linear combination between inputs of the previous layer and weights of the current one
            z = np.dot(activations, self.weights[i]) + self.biases[i]
            self.linear_comb[i] = z
            
            # Apply the activation function to the linear part
            activations = self.activation_func(z)
            
            # Store the activations in object attribute self.activations
            self.activations[i+1] = activations

        return activations
            
    
    def backward_prop(self, error, activation_deriv, verbose = False):
        for i in reversed(range(len(self.weights_deriv))):
            z = self.linear_comb[i]    
            delta = error * activation_deriv(z)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T          
            current_activation = self.activations[i]
            current_activation_reshaped = current_activation.reshape(current_activation.shape[0], -1)
            
            self.weights_deriv[i] = np.dot(current_activation_reshaped, delta_reshaped)
            self.biases_deriv[i] = delta
            
            error = np.dot(delta, self.weights[i].T)
            
            if verbose == True:
                print ("Derivatives for W{}: {}".format(i, self.weights_deriv[i]))
                print ("Derivatives for B{}: {}". format(i, self.biases_deriv[i]))
            
        return error
            
            
    def gradient_descendent(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate*self.weights_deriv[i]
            self.biases[i] -= learning_rate*self.biases_deriv[i]
    
    
    def train(self):
        pass
    
    def predict(self):
        pass
    
    
if __name__ == '__main__':
    
    nn = Ann(2, [5], 1)
    #create dummy data
    data = [0.1, 0.3]
    target = [0.3]
    #forward propagation
    output = nn.forward_prop(data, act.deriv_sigmoid)
    #calculate the error
    error = target - output
    #backpropagation
    nn.backward_prop(error, act.deriv_sigmoid)
    #gradient descend
    nn.gradient_descendent(0.1)
        