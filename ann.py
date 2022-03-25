"Diamo inizio al gioco...vediamo se mi ricordo ancora come si fa"

import numpy as np
import activation_functions as act
import loss_functions as lf
import pickle
from random import random


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
        ann
            An ann object with the specified number of inputs, hidden layers, number of neurons in each hidden layer and number of outputs
        
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
        
            
    def _forward_prop(self, inputs, activation_func):
        
        """Perform the forward propagation
        
        Parameters
        ----------
        inputs : array_like
            Vector of inputs for the Neural Network
        activation_func : function
            Activation function used by each neuron to produce its output
        
        Returns
        -------
        array_like
            Array of outputs of the neural network
        
        """
        
        #self.activation_func = activation_func
        activations = np.array(inputs)
        self.activations[0] = activations
        
        for i in range(self.layers.size - 1):
            # Calculate the linear combination between inputs of the previous layer and weights of the current one
            z = np.dot(activations, self.weights[i]) + self.biases[i]
            self.linear_comb[i] = z
            
            # Apply the activation function to the linear part
            activations = activation_func(z)
            
            # Store the activations in object attribute self.activations
            self.activations[i+1] = activations
        return activations
    
    
    def _backward_prop(self, error, activation_deriv, verbose = False):
        """ Perform the backpropagation algorithm
        
        Parameters
        ----------
        error : array_like
            Derivative of the error function evaluated in each neuron of the output layer.
        activation_deriv : function
            Derivative of the activation function given, as an argument, to the forward propagation function.
        verbose : Boolean, optional
            If it is set to True, print all the derivatives of the weights and biases. The default is False.

        Returns
        -------
        array_like
            Error backpropagated to the input layer.
        
        """
        
        
        for i in reversed(range(len(self.weights_deriv))):
            z = self.linear_comb[i]    
            #delta = error * activation_deriv(z)
            delta = np.dot(error, activation_deriv(z))
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
            
            
    def _gradient_descendent(self, learning_rate):
        """ Implementation of the stochastic gradient descendent to update weights and biases
        
        Parameters
        ----------
        learning_rate : float
            Learning rate used to update weights and biases.

        Returns
        -------
        array_like
            List of the weights updated for each layer.
        array_like
            List of biases updated for each layer.

        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate*self.weights_deriv[i]
            self.biases[i] -= learning_rate*self.biases_deriv[i]
        return self.weights, self.biases
    
    
    def train(self, inputs, targets, epochs, learning_rate, activation_function, activation_derivative, loss_func, deriv_loss_fun):
        """ Train method: the neural network update weights and biases, according to the inputs and the targets in order to minimize the loss function
        
        Parameters
        ----------
        inputs : array_like
            Array of input data used to train the network, together with the array of targets.
        targets : array_like
            Array of labels used to train the network, together with the array of input data.
        epochs : int
            Number of times the entire set of input data is given to the neural network for the process of training.
        learning_rate : float
            Learning rate used to update weights and biases with the gradient descendent method.
        activation_function : function
            Activation function used by each neuron to produce its output.
        activation_derivative : function
            Derivative of the activation function.
        loss_func : function
            Function used to evaluete the difference between the output produced by the network and the target one.
            The goal is to minimize this function with respect the weights and the biases.
        deriv_loss_fun : function
            Derivative of the loss function used during the backpropagation phase.

        Returns
        -------
        float
            Mean error evaluated with the loss function in the last epoch.
            
        Example
        -------
        >>> import import activation_functions as act
        >>> import loss_functions as lf
        >>> ann.Ann(10, [5, 3], 2)
        >>> Ann.train(data, labels, 1000, 0.1, act.sigmoid, act.deriv_sigmoid, lf.mse, lf.mse_deriv)

        """
        self.loss_func = loss_func
        self.activation_func = activation_function
        n = len(inputs)
        for i in range(epochs):
            
            sum_error = 0
            
            for single_input, target in zip(inputs, targets):
                
                # forward propagation
                output = self._forward_prop(single_input, activation_function)
                
                # calculate the error
                error = deriv_loss_fun(output, target)
                
                # backpropagation
                self._backward_prop(error, activation_derivative)
                
                # apply gradient descendent
                self._gradient_descendent(learning_rate)
                
                # evaluate the error for each input
                sum_error += loss_func(output, target)
            
            print("Epoch {}/{}-Error: {}".format(i+1, epochs, sum_error / n))
        
        return sum_error / n
                
                           
    def predict(self, inputs):
        """Once the neural network is trained, this method predicts the output of a given input
        
        Parameters
        ----------
        inputs : array_like
            Input data of which you are interested to predict the output of the neural network.

        Returns
        -------
        prediction : array_like
            Result of the forward propagation of the input data with the trained neural network (weights and biases updated).

        Example
        -------
        >>> Ann.predict(data)
        
        """
        prediction = self._forward_prop(inputs, self.activation_func)
        return prediction
    
    
    def evaluate(self, inputs, targets):
        """ Evaluation of the mean error for a test dataset (never seen before by the neural network)
        
        Parameters
        ----------
        inputs : array_like
            Data given to the neural network in order to have prediction.
        targets : array_like
            True labels of the data.
        
        Returns
        -------
        float
            Mean error between the targets and the outputs of the neural network evaluated with the loss function.
            
        Example
        -------
        >>> Ann.evaluate(data, labels)
        
        """
        sum_error = 0
        n = len(inputs)
        for i in range(len(inputs)):
            prediction = self.predict(inputs[i])
            target = targets[i]
            sum_error += self.loss_func(prediction, target)
        return float(sum_error / n)
    
    ###########################################################################################    
    # get methods
    
    def get_weights(self):
        """Return a list whose elements are the weight matrices for each couple of layers"""
        return self.weights
    
    
    def get_biases(self):
        """Return a list whose elements are the bias vectors for each couple of layers"""
        return self.biases
    
    
    def get_activation_function(self):
        """Return the activation function used in the neural network process of training"""
        return self.activation_func
    
    def get_loss_function(self):
        """Return the loss function used in the neural network process of training"""
        return self.loss_func
    
    ############################################################################################
    
    def set_parameters(self, saved_weights, saved_biases):
        self.weights = saved_weights
        self.biases = saved_biases
        
    
    def save_parameters(self, file_name, path='./'):
        """Save the structure of the network (neurons for each layer), weights and biases of the neural network in a .pkl file.
        
        Parameters
        ----------
        file_name : string
            Name of the file where weights and biases will be stored. The file will be a .pkl and it is possible to add or not the extension.
        path : string, optional
            Directory where the file will be saved. If the file already exists, it will be overwritten. The default is './'.

        Returns
        -------
        string:
            Message of save confirmation.

        """
        
        if file_name[-4:] != '.pkl':
            file_name += '.pkl'
        total_name = path + file_name
        parameters = [self.num_inputs, self.num_hidden, self.num_outputs, self.biases, self.weights]
        pickle.dump(parameters, open(total_name, 'wb'))
        return 'Save completed successfully'
    
    
    @classmethod
    def load_parameters(cls, file_name):
        """ Load from a file the weights, the biases and the number of neurons for each layer.
        
        Parameters
        ----------
        file_name : string
            File source where weights, biases and number of neurons for each layer are stored (look at 'save_parameters' method).

        Returns
        -------
        biases : list
            Biases of the neural network saved previously.
        weights : list
            Weights of the neural network saved previously.
        num_inputs : int
            Number of neurons in the input layer of the neural network saved previously.
        num_hidden : list
            Number of neurons in the hidden layers of the neural network saved previously.
        num_outputs : int
            Number of neurons in the output layer of the neural network saved previously.

        """
        if file_name[-4:] != '.pkl':
            file_name += '.pkl'
        parameters = pickle.load(open(file_name, 'rb'))
        num_inputs = parameters[0]
        num_hidd = parameters[1]
        num_outputs = parameters[2]
        biases = parameters[3]
        weights = parameters[4]
        return biases, weights, num_inputs, num_hidd, num_outputs
    
    
    @classmethod
    def load_and_set_network(cls, file_name):
        """Create a neural network with weights, biases and number of neuron for each layer stored in "file_name"
        
        Parameters
        ----------
        file_name : string
            File source where weights, biases and number of neurons for each layer are stored (look at 'save_parameters' method).

        Returns
        -------
        network_loaded : Ann
            Neural network with weights, biases and number of neuron for each layer stored in the file "file_name"

        """
        
        if file_name[-4:] != '.pkl':
            file_name += '.pkl'
        parameters = pickle.load(open(file_name, 'rb'))
        num_inputs = parameters[0]
        num_hidd = parameters[1]
        num_outputs = parameters[2]
        biases = parameters[3]
        weights = parameters[4]
        network_loaded = cls(num_inputs, num_hidd, num_outputs)
        network_loaded.biases = biases
        network_loaded.weights = weights
        return network_loaded
        
        
    
    

    
        
        
        



    
        
    



    
if __name__ == '__main__':
    
    print ("Hello")
    
        