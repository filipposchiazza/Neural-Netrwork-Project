import numpy as np
import os
import json
from neuralnet import activation_functions as act
from neuralnet import loss_functions as lf

class Ann:
    
    activation_list = {act.sigmoid.__name__ : act.sigmoid,
                           act.softmax.__name__ : act.softmax}
    
    loss_list = {lf.binary_cross_entropy.__name__ : lf.binary_cross_entropy,
                     lf.cross_entropy.__name__ : lf.cross_entropy}
    
    
    
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
            d_w = np.random.rand(int(self.layers[i]) , int(self.layers[i+1]))
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
        
        if self.num_outputs == 1:
            predictions = self._forward_prop(inputs)
            predictions = np.reshape(predictions, (len(predictions), 1))
        else:
            inp = np.reshape(inputs, (-1, self.num_inputs))
            predictions = np.zeros((len(inp), self.num_outputs))
            for i in range(len(predictions)):
                predictions[i] = self._forward_prop(inp[i])
                
        return predictions
    
    
    
    def _discretize_predictions(self, predictions):
        """Discretize the values of predictions: values 0 or 1.
        

        Parameters
        ----------
        predictions : array-like
            Array whose values are in the continuous range [0, 1].

        Returns
        -------
        predictions_discr : array-like
            Predictions with elements equal to 0 or 1.

        """
        if self.num_outputs == 1:
            predictions_discr = np.where(predictions > 0.5, 1, 0)
        else:
            support = np.max(predictions, axis=1)
            support = np.reshape(support, (-1,1))
            predictions_discr = np.where(predictions == support, 1, 0)
        
        return predictions_discr
    
    
    
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
        predictions_discr : array-like
            Predictions with elements equal to 0 or 1.
        num_correct_prediction : int
            Number of correct predictions.
        percentage : float
            Percentage of correct predictions.

        """
        predictions = self.predict(inputs)
        predictions_discr = self._discretize_predictions(predictions)
        targets_reshaped = np.reshape(targets, (-1, self.num_outputs))
        correct_predictions = np.all(predictions_discr == targets_reshaped, axis=1)
        num_correct_prediction = np.sum(correct_predictions)
        percentage = num_correct_prediction / len(predictions) * 100
        print("Correct classification on the test dataset: {}/{}".format(num_correct_prediction, len(predictions)))
        print ("Percentage of correct classification on the test dataset: {:.2f}%".format(percentage))
        
        return predictions_discr, num_correct_prediction, percentage
        
 
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
    
#################################################################################################
    
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
    
    def _save_building_parameters(self, location):
        """Save number of inputs, hidden layers(number and content) and outputs in the file "building_parameters.json",
        stored in the location given as argument to the function.
        

        Parameters
        ----------
        location : string
            Directory where the file "building_parameters.json" is stored.

        Returns
        -------
        None.

        """
        
        total_file_name = location + 'building_parameters.json'
        data = [self.num_inputs, self.num_hidden.tolist(), self.num_outputs]
        # save
        with open(total_file_name, 'w') as f:
            json.dump(data, f)
        f.close()

    
    def _save_biases(self, location):
        """Save biases in the file "biases.json", stored in the location given as argument to the function.
        

        Parameters
        ----------
        location : string
            Directory where the file "biases.json" is stored.

        Returns
        -------
        None.

        """
        
        total_file_name = location + 'biases.json'
        data = self.biases
        # transform numpy arrays into lists
        for i in range(len(data)):
            data[i] = data[i].tolist()
        # save
        with open(total_file_name, 'w') as f:
            json.dump(data, f)
        f.close()
        # restore biases
        for i in range(len(self.biases)):
            self.biases[i] = np.asarray(self.biases[i])
            
    
    def _save_weights(self, location):
        """Save weights in the file "weights.json", stored in the location given as argument to the function.
        

        Parameters
        ----------
        location : string
            Directory where the file "weights.json" is stored.

        Returns
        -------
        None.

        """
        
        total_file_name = location + 'weights.json'
        data = self.weights
        # transform numpy arrays into lists
        for i in range(len(data)):
            data[i] = data[i].tolist()
        # save
        with open(total_file_name, 'w') as f:
            json.dump(data, f)
        f.close()
        # restore weights
        for i in range(len(self.weights)):
            self.weights[i] = np.asarray(self.weights[i])
        
    
    def _save_activation_and_loss_functions(self, location):
        """Save the activation and loss functions in the file "activation_loss_functions.json",
        stored in the location given as argument to the function.
        

        Parameters
        ----------
        location : string
            Directory where the file "activation_loss_functions.json" is stored.

        Returns
        -------
        None.

        """
        
        total_file_name = location + 'activation_loss_functions.json'
        data = {'Activation function' : self.activation_func.__name__,
                'Loss function' : self.loss_func.__name__}
        # save
        with open(total_file_name, 'w') as f:
            json.dump(data, f)
        f.close()
        
        
    def save(self, directory_name = 'network_parameters/', path = './'):
        """Save the structure of the network (neurons for each layer), weights and biases, activation function
        and loss function of the neural network in json format.
        
        Parameters
        ----------
        directory_name : string
            Name of the directory where the neural neworks parameters will be saved. 
            Be sure that the directory does not already exist.
            
        path : string, optional
            Location where the directory will be created. The default is './'.

        Returns
        -------
        None.

        """
    
        total_directory_name = path + directory_name
        os.mkdir(total_directory_name)
        self._save_building_parameters(total_directory_name)
        self._save_biases(total_directory_name)
        self._save_weights(total_directory_name)
        self._save_activation_and_loss_functions(total_directory_name)

##########################################################################################################################
    
    #Loading methods
    @classmethod
    def _load_parameters(cls, location):
        """Load the number of neural network inputs, hidden layers and outputs from the file "building_parameters.json",
        placed in the location given as input to the function. 
        

        Parameters
        ----------
        location : string
            Location where "building_parameters.json" is stored.

        Returns
        -------
        num_inp : int
            Number of inputs of the neural network.
        num_hidd : list
            The element i of the array specifies the number of neurons in the hidden layer i, so the total number of hidden layer is given
            by the dimension of the array.
        num_out : int
            Number of outputs of the neural network.

        """
        
        with open(location + 'building_parameters.json', 'r') as f:
            parameters = json.load(f)
        f.close()
        num_inp = parameters[0]
        num_hidd = parameters[1]
        num_out = parameters[2]
        return num_inp, num_hidd, num_out
    
    
    @classmethod
    def _load_activation_and_loss(cls, location):
        """Load the activation and loss functions from the file "activation_loss_functions.json",
        placed in the location given as input to the function. 
        

        Parameters
        ----------
        cls : Ann
        location : string
            Location where "activation_loss_functions.json" is stored.

        Returns
        -------
        activation_function : func
            Activation function of the neural network.
        loss_function : func
            Loss function of the neural network.

        """
        
        with open(location + 'activation_loss_functions.json', 'r') as f:
            data = json.load(f)
        f.close()
        activation_function = cls.activation_list[data['Activation function']]
        loss_function = cls.loss_list[data['Loss function']]
        return activation_function, loss_function
    
    
    
    @classmethod
    def _load_biases(cls, location):
        """Load the neural network's biases from the file "biases.json",
        placed in the location given as input to the function. 
        

        Parameters
        ----------
        cls : Ann
        location : string
            Location where "biases.json" is stored.

        Returns
        -------
        biases : list
            Neural network's biases.

        """
        
        with open(location + 'biases.json', 'r') as f:
            biases = json.load(f)
        f.close()
        
        for i in range(len(biases)):
            biases[i] = np.asarray(biases[i])
        
        return biases
    
    @classmethod
    def _load_weights(cls, location):
        """Load the neural network's weights from the file "weights.json",
        placed in the location given as input to the function. 
        

        Parameters
        ----------
        cls : Ann
        location : string
            Location where "biases.json" is stored.

        Returns
        -------
        weights : list
            Neural network's weights.

        """
        
        with open(location + 'weights.json', 'r') as f:
            weights = json.load(f)
        f.close()
        
        for i in range(len(weights)):
            weights[i] = np.asarray(weights[i])
        
        return weights



    @classmethod
    def _load_all(cls, directory_name):
        """Clip all the load function together.
        

        Parameters
        ----------
        cls : Ann
        directory_name : string
            Directory where the json files needed to setup the neural network are stored.

        Returns
        -------
        num_inp : int
            Number of inputs of the neural network.
        num_hidd : list
            The element i of the array specifies the number of neurons in the hidden layer i, so the total number of hidden layer is given
            by the dimension of the array.
        num_out : int
            Number of outputs of the neural network.
        activation_function : func
            Activation function of the neural network.
        loss_function : func
            Loss function of the neural network.
        biases : list
            Neural network's biases.
        weights : list
            Neural network's weights.

        """
        
        num_inp, num_hidd, num_out = cls._load_parameters(directory_name)
        activation_function, loss_function = cls._load_activation_and_loss(directory_name)
        biases = cls._load_biases(directory_name)
        weights = cls._load_weights(directory_name)
        
        return num_inp, num_hidd, num_out, activation_function, loss_function, biases, weights
    

    @classmethod
    def load_neural_network(cls, directory_name):
        """Return an Ann object created with the parameters stored in the directory given as argument to the function.
        

        Parameters
        ----------
        cls : Ann
        directory_name : string
            Directory where the json files needed to setup the neural network are stored.

        Returns
        -------
        neural_network : Ann
            Ann object created with the parameters stored in the directory given as argument to the function.

        """
        
        
        num_inp, num_hidd, num_out, activation_function, loss_function, biases, weights = cls._load_all(directory_name)
        
        neural_network = cls(num_inputs = num_inp,
                             num_hidden = num_hidd, 
                             num_outputs = num_out,
                             activation_function = activation_function,
                             loss_function = loss_function)
        
        neural_network._set_parameters(weights, biases)
        
        return neural_network 

  
