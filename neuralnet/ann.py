"Diamo inizio al gioco...vediamo se mi ricordo ancora come si fa"

import numpy as np
import pickle
import activation_functions as act
import loss_functions as lf

class Ann:
    
    def __init__(self, num_inputs, num_hidden, num_outputs, activation_function, loss_function, seed=None):
        
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
        activation_function : function
            Activation function used by each neuron to produce its output.
        loss_function : function
            Function used to evaluete the difference between the output produced by the network and the target one.
            The goal is to minimize this function with respect the weights and the biases.
        seed : int
            Default value = None. Value used to set the seed for the initial random generation of biases and weights
        
        Returns
        -------
        ann
            An ann object with the specified number of inputs, hidden layers, number of neurons in each hidden layer and number of outputs
        
        Example
        -------
        >>> ann.Ann(num_inputs=10, num_hidden=[5, 3], num_outputs=2)
        
        """
        
        
        self.num_inputs = num_inputs
        self.num_hidden = np.array(num_hidden)
        self.num_outputs = num_outputs
        self.layers = np.concatenate(([self.num_inputs], self.num_hidden, [self.num_outputs]))
        self._set_activation_function(activation_function)
        self._set_loss_function(loss_function)
        
        
        # Set the seed for the random generation
        np.random.seed(seed)
        
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
        
            
    def _forward_prop(self, inputs):
        
        """Perform the forward propagation
        
        Parameters
        ----------
        inputs : array_like
            Vector of inputs for the Neural Network
        
        Returns
        -------
        array_like
            Array of outputs of the neural network
        
        """
        
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
    
    
    def _backward_prop(self, error,verbose = False):
        """ Perform the backpropagation algorithm
        
        Parameters
        ----------
        error : array_like
            Derivative of the error function evaluated in each neuron of the output layer.
        verbose : Boolean, optional
            If it is set to True, print all the derivatives of the weights and biases. The default is False.

        Returns
        -------
        array_like
            Error backpropagated to the input layer.
        
        """
        
        
        for i in reversed(range(len(self.weights_deriv))):
            z = self.linear_comb[i]    
            delta = np.dot(error, self.act_func_deriv(z))
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
            # Update weights and biases for each level of the neural network
            self.weights[i] -= learning_rate*self.weights_deriv[i]
            self.biases[i] -= learning_rate*self.biases_deriv[i]
        return self.weights, self.biases
    
    
    def train(self, inputs, targets, epochs, learning_rate, verbose=True):
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
        verbose : bool
            Default value: True. If it is equal to True, the error is printed for each epoch. 

        Returns
        -------
        float
            Mean error evaluated with the loss function in the last epoch.
            
        Example
        -------
        >>> import import activation_functions as act
        >>> import loss_functions as lf
        >>> ann.Ann(num_inputs=10, num_hidden=[5, 3], num_outputs=2, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
        >>> Ann.train(inputs=data, targets=labels, epochs=1000, learning_rate=0.1)

        """     
        n = len(inputs)
        
        for i in range(epochs):
            
            sum_error = 0
            
            for single_input, target in zip(inputs, targets):
                
                # forward propagation
                output = self._forward_prop(inputs=single_input)
                
                # calculate the error
                error = self.loss_func_deriv(prediction=output, target=target)
                
                # backpropagation
                self._backward_prop(error=error)
                
                # apply gradient descendent
                self._gradient_descendent(learning_rate=learning_rate)
                
                # evaluate the error for each input
                sum_error += self.loss_func(prediction=output, target=target)
            if verbose == True:
                print("Epoch {}/{} - Error: {}".format(i+1, epochs, float(sum_error / n)))
        
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
        >>> Ann.predict(inputs=data)
        
        """
        prediction = self._forward_prop(inputs)
        return prediction
        
    
    def evaluate_classification(self, inputs, targets):
        """Evaluate the percentage of correct classification on the test dataset.
        
        Parameters
        ----------
        inputs : array_like
            Test dataset.
        targets : array_like
            True labels of the test dataset.

        Returns
        -------
        predictions : array_like
            Output of the neural network for the test dataset.
        
        Example
        -------
        >>Ann.evaluate_classification(inputs=dataset_test, targets=target_test)

        """
        
        if np.size(targets[0]) == 1:
            predictions = self.predict(inputs)
            predictions = np.reshape(predictions, (len(predictions), ))
            targets = np.reshape(targets, (len(targets),))
            predictions = np.where(predictions > 0.5, 1, 0)
            correct_predictions = predictions == targets 
            
        else:
            predictions = np.zeros((len(inputs), np.size(targets[0])))
            for i in range(len(predictions)):
                predictions[i] = self.predict(inputs[i])
        
            for i in range(len(predictions)):
                support = np.max(predictions[i])
                predictions[i] = np.where(predictions[i] == support, 1, 0)
        
            correct_predictions = np.all(predictions==targets, axis=1)
            
            
        num_correct_prediction = np.sum(correct_predictions)
        percentage = num_correct_prediction / len(predictions) * 100
        
        print("Correct classification on the test dataset: {}/{}".format(num_correct_prediction, len(predictions)))
        print ("Percentage of correct classification on the test dataset: {:.2f}%".format(percentage))
        
        return predictions, num_correct_prediction, percentage
    
    
    
###############################################################################################################################    
    
    #Get methods
    
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
    
    #Set methods
    
    def _set_parameters(self, saved_weights, saved_biases):
        """Set the parameters(weights and biases) of the network"""
        self.weights = saved_weights
        self.biases = saved_biases
     
        
    def _set_activation_function(self, act_func):
        """Set the activation function of the network"""    
        self.activation_func = act_func
        
        if act_func == act.sigmoid:
            self.act_func_deriv = act.deriv_sigmoid
        elif act_func == act.softmax:
            self.act_func_deriv = act.deriv_softmax
        
    
    def _set_loss_function(self, loss_func):
        """"Set the loss function of the network"""
        self.loss_func = loss_func
        
        if loss_func == lf.binary_cross_entropy:
            self.loss_func_deriv = lf.binary_cross_entropy_deriv
        elif loss_func == lf.cross_entropy:
            self.loss_func_deriv = lf.cross_entropy_deriv 
        
    ##########################################################################################
    
    #Saving method      
    
    def save_parameters(self, file_name, path='./'):
        """Save the structure of the network (neurons for each layer), weights and biases, activation function
        and loss function of the neural network in a .pkl file.
        
        Parameters
        ----------
        file_name : string
            Name of the file where the neural network parameters will be stored. 
            The file will be a .pkl and it is possible to add or not the extension.
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
        parameters = [self.num_inputs, self.num_hidden, self.num_outputs, self.biases, self.weights, self.activation_func, self.loss_func]
        pickle.dump(parameters, open(total_name, 'wb'))
        return 'Save completed successfully'
    
    ###################################################################################################################################
    
    #Loading methods
    
    @classmethod
    def load_parameters(cls, file_name):
        """ Load from a file the weights, the biases, the number of neurons for each layer, the activation and the loss function.
        
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
        activation_function : func
            Activation function of the neural network
        loss_func : func
            Loss function of the neural network
            
        """
        
        if file_name[-4:] != '.pkl':
            file_name += '.pkl'
        parameters = pickle.load(open(file_name, 'rb'))
        num_inputs = parameters[0]
        num_hidd = parameters[1]
        num_outputs = parameters[2]
        biases = parameters[3]
        weights = parameters[4]
        activation_function = parameters[5]
        loss_function = parameters[6]
        return biases, weights, num_inputs, num_hidd, num_outputs, activation_function, loss_function
    
    
    @classmethod
    def load_and_set_network(cls, file_name):
        """Create a neural network with weights, biases, number of neuron for each layer, activation and loss functions stored in "file_name"
        
        Parameters
        ----------
        file_name : string
            File source where weights, biases, number of neurons for each layer,activation and loss function
            are stored (look at 'save_parameters' method).

        Returns
        -------
        network_loaded : Ann
            Neural network with weights, biases, number of neuron for each layer, activation and loss functions stored in the file "file_name"

        """
        biases, weights, num_inputs, num_hidd, num_outputs, activation_function, loss_function = cls.load_parameters(file_name)
        network_loaded = cls(num_inputs=num_inputs, num_hidden=num_hidd, 
                             num_outputs=num_outputs, activation_function=activation_function, 
                             loss_function=loss_function)
        cls._set_parameters(saved_weights=weights, saved_biases=biases)
        return network_loaded
        
  