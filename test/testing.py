"Testing"
import sys
sys.path.insert(0, '/home/filippo/Documenti/Università/Magistrale/NeuralNetworkFromScratch/neuralnet')


import numpy as np
from hypothesis import given
from hypothesis.strategies import data
import hypothesis.strategies as st
import pickle
import os

import ann
import activation_functions as act
import loss_functions as lf


max_num_neurons = 20

##################################################################################################################################
"""
#Test the costruction of the neural network (the function __init__)

@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_num_layers(inp, hidd, out):
    "Test if the number of layers is the correct one, according to the inputs given to build the ann"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    assert network.layers.size == len([network.num_inputs]) + network.num_hidden.size + len([network.num_outputs])
 

@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50)) 
def test_num_weights(inp, hidd, out):
    "Test if the total number of weights of the neural network is the correct one, according to the inputs given to build it"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    counter_first = 0
    for i in range(len(network.layers)-1):
        counter_first += (network.layers[i]) * (network.layers[i+1])
    counter_second = 0
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            counter_second += len(network.weights[i][j])
    assert counter_first == counter_second


@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50)) 
def test_num_biases(inp, hidd, out):
    "Test if the total number of biases is equal to (total number of neurons - number of neurons in the input layer)"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    counter_first = 0
    for b in network.biases:
        counter_first += b.size
    counter_second = 0
    for i in range(1, len(network.layers)):
        counter_second += network.layers[i]
    assert counter_first == counter_second


@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_num_linear_comb(inp, hidd, out):
    "Test if the total number of linear combinations is equal to (total number of neurons - number of neurons in the input layer)"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    counter_first = 0
    for lin in network.linear_comb:
        counter_first += lin.size
    counter_second = 0
    for i in range(1, len(network.layers)):
        counter_second += network.layers[i]
    assert counter_first == counter_second
    
    
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_num_activations(inp, hidd, out):
    "Test if the numbers of activations stored for each layer is the correct one (corresponding to the number of neurons)"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    for i in range(len(network.layers)):
        assert len(network.activations[i]) == network.layers[i] 

        
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_num_weights_derivatives(inp, hidd, out):
    "Test if the number of weights'derivatives stored is the same of the number of weights (as it should be)"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    for i in range(len(network.weights)):
        assert network.weights[i].size == network.weights_deriv[i].size


@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50)) 
def test_num_biases_derivatives(inp, hidd, out):
    "Test if the number of biases'derivatives stored is the same of the number of weights (as it should be)"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    for i in range(len(network.biases)):
        assert network.biases[i].size == network.biases_deriv[i].size
"""
##########################################################################################################################

# Test the forward propagation method

def test_specific_case_forward_propagation():
    "Test the output of the forward propation for a specific combination of inputs, weights and biases"
    w1 = np.array([[0.5, 0.4], [0.3, 0.7]])
    b1 = np.array([0.3, 0.6])
    weights = [w1]
    biases = [b1]
    neural_network = ann.Ann(num_inputs = 2, num_hidden = [], num_outputs = 2,
                             activation_function = act.sigmoid, 
                             loss_function = lf.binary_cross_entropy)
    neural_network._set_parameters(weights, biases)
    inputs = [1., 1.]
    result = neural_network._forward_prop(inputs)
    expected_result = np.array([0.75026011, 0.84553473])
    assert np.all(np.isclose(result, expected_result, rtol=0.1, atol=1e-5))
    
    
def test_limit_case_forward_propagation():
    "Test the output of the forward propation for the limit case when all the weights and biases are equal to 0"
    w1 = np.array([[0., 0.], [0., 0.]])
    b1 = np.array([0.0, 0.0])
    w2 = np.array([[0.], [0.]])
    b2 = np.array([0.])
    weights = [w1, w2]
    biases = [b1, b2]
    neural_network = ann.Ann(num_inputs = 2, num_hidden = [2], num_outputs = 1,
                             activation_function = act.sigmoid, 
                             loss_function = lf.binary_cross_entropy)
    neural_network._set_parameters(weights, biases)
    inputs = [1., 1.]
    result = neural_network._forward_prop(inputs)
    expected_result = np.array([0.5])
    assert np.all(np.isclose(result, expected_result, rtol=0.1, atol=1e-5))
    
    
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_forward_prop(inp, hidd, out):
    "Test that the dimension of the forward propagation result is the same as the number of output layers of the neural network"
    np.random.seed(1)
    dataset = np.random.randn(inp)
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    assert network.num_outputs == network._forward_prop(inputs=dataset).size
    
########################################################################################################################## 
"""
# Test the backward propagation method

@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_weights_deriv_after_backprop(inp, hidd, out):
    "Test that, after backpropagation, the dimensionalities of the weights'derivatives is still the same as the ones of weights"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    np.random.seed(1)
    error = np.random.uniform(-100, 100, out)
    network._backward_prop(error=error)
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            assert network.weights_deriv[i][j].size == network.weights[i][j].size
   
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.integers(min_value=1, max_value=max_num_neurons), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_biases_deriv_after_backprop(inp, hidd, out):
    "Test that, after backpropagation, the dimensionalities of the biases'derivatives is still the same as the ones of biases"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    np.random.seed(1)
    error = np.random.uniform(-100, 100, out)
    network._backward_prop(error=error)
    for i in range(len(network.biases)):
        assert network.biases_deriv[i].size == network.biases[i].size    


##########################################################################################################################

# Test the gradient descendent method

@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50),
       learning_rate = st.floats(min_value=0.0001, max_value=10, allow_nan=False))
def test_weights_structure_after_gradient (inp, hidd, out, learning_rate):
    "Test that the dimensionalities of the weights is still the same after the gradient descendent"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    np.random.seed(1)
    error = np.random.uniform(-100, 100, out)
    network._backward_prop(error=error)
    
    previous_weights = []
    for i in range(len(network.layers) - 1):
        previous_weights.append(np.copy(network.weights[i]))
    
    network._gradient_descendent(learning_rate=learning_rate)
    
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            assert network.weights[i][j].size == previous_weights[i][j].size
            
            
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50),
       learning_rate = st.floats(min_value=0.0001, max_value=10, allow_nan=False))
def test_biases_structure_after_gradient (inp, hidd, out, learning_rate):
    "Test if the dimensionalities of the biases is still the same after the gradient descendent"
    network = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    np.random.seed(1)
    error = np.random.uniform(-100, 100, out)
    network._backward_prop(error=error)
    
    previous_biases = []
    for i in range(len(network.layers) - 1):
        previous_biases.append(np.copy(network.biases[i]))
    
    network._gradient_descendent(learning_rate=learning_rate)
    
    for i in range(len(network.biases)):
        assert network.biases[i].size == previous_biases[i].size

##########################################################################################################################

# Test the train method  -> not necessary because it is the collection of methods already tested.

##########################################################################################################################

# Test the predict method  -> not necessary because it simply apply the forward propagation to an input vector and the
# forward propagation is already tested.

########################################################################################################################## 

# Test the evaluate classification method

target_values = [0, 1]

@given(data = data(),
       inp = st.integers(min_value=1, max_value=100),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=1, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_number_correct_prediction_is_in_meaningful_range(data, inp, hidd, out):
    "Test that the number of correct predictions in in the range between 0 and the total number of predictions"
    if out == 1:
        ann_test = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    else:
        ann_test = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.softmax, loss_function=lf.cross_entropy)
    
    size_dataset = data.draw(st.integers(min_value=1, max_value=1e2))
    targets = np.zeros((size_dataset, out))
    for i in range(len(targets)):
        targets[i] = data.draw(st.lists(st.sampled_from(target_values), min_size=out, max_size=out))
    np.random.seed(1)
    inputs = np.random.standard_normal((size_dataset, inp))
    inputs = np.array(inputs)
    targets = np.array(targets)
    a, num_correct_classification, c = ann_test.evaluate_classification(inputs=inputs, targets=targets) 
    assert num_correct_classification >= 0 and num_correct_classification <= len(targets)


##########################################################################################################################

# Test save parameters method
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=1, max_size=10),
       out = st.integers(min_value=1, max_value=50),
       file_name = st.text(alphabet = st.characters(whitelist_categories = ('L', 'N'))))
def test_save_correctly(inp, hidd, out, file_name):
    "Test that all the parameters of the neural network are saved correctly"
    ann_test = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    ann_test.save_parameters(file_name=file_name)
    parameters = pickle.load(open("./" + file_name + ".pkl", 'rb'))
    os.remove("./" + file_name + ".pkl")
    assert parameters[0] == ann_test.num_inputs
    assert np.all(parameters[1] == ann_test.num_hidden)
    assert parameters[2] == ann_test.num_outputs
    for i in range(len(parameters[3])):
        assert np.all(parameters[3][i] == ann_test.get_biases()[i])
    for i in range(len(parameters[4])):
        assert np.all(parameters[4][i] == ann_test.get_weights()[i])
    assert parameters[5] == ann_test.get_activation_function()
    assert parameters[6] == ann_test.get_loss_function()

##########################################################################################################################

# Test load parameters method
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=1, max_size=10),
       out = st.integers(min_value=1, max_value=50),
       file_name = st.text(alphabet = st.characters(whitelist_categories = ('L', 'N'))))
def test_loading_parameters(inp, hidd, out, file_name):
    "Test that all the parameters of the neural network are loaded correctly"
    ann_test = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    ann_test.save_parameters(file_name=file_name)
    biases, weights, num_inputs, num_hidd, num_outputs, activation_function, loss_function = ann.Ann.load_parameters(file_name="./" + file_name + ".pkl")
    os.remove("./" + file_name + ".pkl")
    assert num_inputs == ann_test.num_inputs
    assert np.all(num_hidd == ann_test.num_hidden)
    assert num_outputs == ann_test.num_outputs
    for i in range(len(biases)):
        assert np.all(biases[i] == ann_test.get_biases()[i])
    for i in range(len(weights)):
        assert np.all(weights[i] == ann_test.get_weights()[i])
    assert activation_function == ann_test.get_activation_function()
    assert loss_function == ann_test.get_loss_function()
    
    

##########################################################################################################################

#Test load and set the network method

@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=1, max_size=10),
       out = st.integers(min_value=1, max_value=50),
       file_name = st.text(alphabet = st.characters(whitelist_categories = ('L', 'N'))))
def test_bilding_network_from_saving_parameters(inp, hidd, out, file_name):
    "Test that the neural network is loaded correctly from the saving file"
    ann_test = ann.Ann(num_inputs=inp, num_hidden=hidd, num_outputs=out, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)
    ann_test.save_parameters(file_name=file_name)
    ann_loaded = ann.Ann.load_and_set_network(file_name="./" + file_name + ".pkl")
    os.remove("./" + file_name + ".pkl")
    assert ann_loaded.num_inputs == ann_test.num_inputs
    assert np.all(ann_loaded.num_hidden == ann_test.num_hidden)
    assert ann_loaded.num_outputs == ann_test.num_outputs
    for i in range(len(ann_test.get_biases())):
        assert np.all(ann_loaded.get_biases()[i] == ann_test.get_biases()[i])
    for i in range(len(ann_test.get_weights())):
        assert np.all(ann_loaded.get_weights()[i] == ann_test.get_weights()[i])
    assert ann_loaded.get_activation_function() == ann_test.get_activation_function()
    assert ann_loaded.get_loss_function() == ann_test.get_loss_function()

############################################################################################################################

#Test the activation functions

    # sigmoid function

def test_notable_sigmoid_values():
    "Test some notable values of the sigmoid function"
    assert np.abs(act.sigmoid(0) - 0.5 < 1e-10)
    assert np.abs(act.sigmoid(1) - 0.73 < 0.2)
    assert np.abs(act.sigmoid(-1) - 0.27 < 0.2)
    assert np.abs(act.sigmoid(100) - 1. < 1e-10)
    assert np.abs(act.sigmoid(-100) < 1e-10)

    
def test_limit_case_sigmoid():
    "Test the limit case when normally the exponential produce an inf value, avoided in the sigmoid implementation"
    assert act.sigmoid(-800) < 1e-10
    

@given(inputs = st.lists(st.floats(allow_nan=False, allow_infinity=True), min_size=1, max_size=100))
def test_range_sigmoid(inputs):
    "Test that the output of the sigmoid function is in the range [0, 1]"
    result = act.sigmoid(np.asarray(inputs))
    assert np.all(result >= 0)  and np.all(result <= 1)


    # derivative sigmoid function 
    
    
def test_notable_sigmoid_derivative_values_scalar_case():
    "Test some notable values of the sigmoid derivative function in the scalar case"
    assert np.abs(act.deriv_sigmoid([0]) - 0.25 < 1e-10)
    assert np.abs(act.deriv_sigmoid([1]) - 0.20 < 1e-10)
    assert np.abs(act.deriv_sigmoid([-1]) - 0.20 < 1e-10)
    assert np.abs(act.deriv_sigmoid([100]) < 1e-10)
    assert np.abs(act.deriv_sigmoid([-100]) < 1e-10)
    
    
def test_notable_sigmoid_derivative_values_vectorial_case():
    "Test some notable values of the sigmoid derivative function in the vectorial case"
    x = [0, 1, 50]
    expected_result = np.array([[0.25, 0., 0.], [0., 0.2, 0.], [0., 0., 0.]])
    assert np.all(np.isclose(act.deriv_sigmoid(x), expected_result, rtol=0.1, atol=1e-5))
    
    
@given(inputs = st.lists(st.floats(allow_nan=False, allow_infinity=True), min_size=1, max_size=100))
def test_range_derivative_sigmoid(inputs):   
    "Test that all the elements of the sigmoid jacobian are in the range [0, 0.25]"
    jacobian = act.deriv_sigmoid(np.asarray(inputs))
    assert np.all(jacobian >= 0)  and np.all(jacobian <= 0.25)
    
    
@given(x = st.floats(allow_nan=False, allow_infinity=True))
def test_symmetry_derivative_sigmoid(x):
    "Test that the derivative of the sigmoid function is symmetric with respect the origin"
    assert np.abs(act.deriv_sigmoid([x]) - act.deriv_sigmoid([-x])) < 1e-5
    
    
    # softmax function
    
    
def test_softmax_output_value():
    "Test specific output value of the softmax function"
    inputs = [8.0, 5.0, 0.0]
    result = act.softmax(inputs)
    expected_result = [0.9523, 0.0474, 0.0003]
    evaluation = np.isclose(result, expected_result, rtol=0.1, atol=1e-5)
    assert np.all(evaluation)
      
    
@given(inputs = st.lists(st.floats(max_value=1e100, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_range_softmax_function(inputs):
    "Test the range of the softmax function"
    inputs = np.array(inputs)
    result = act.softmax(inputs)
    assert np.all(result >= 0) and np.all(result <= 1)


@given(inputs = st.lists(st.floats(max_value=1e100, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_normalization_softmax_function(inputs):
    "Test that the sum of the elements of the softmax output is equal to one (condition to represent a probability)"
    inputs = np.array(inputs)
    result = act.softmax(inputs)
    summation = np.sum(result)
    assert np.abs(summation - 1) < 1e-15
    
    
    # derivative softmax function

    
def test_trivial_values_derivative_softmax_function():
    "Test specific output value of the softmax function derivative"
    inputs = [0., 1.]
    jacobian = act.deriv_softmax(inputs)
    expected_jacobian = np.asarray([[0.19661193324148185, -0.19661193324148185], 
                         [-0.19661193324148185, 0.19661193324148185]])
    evaluation = np.isclose(jacobian, expected_jacobian, rtol=0.1, atol=1e-5)
    assert np.all(evaluation)
    

@given(inputs = st.lists(st.floats(max_value=1e100, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_range_softmax_function_derivative(inputs): 
    "Test the range of the elements of the softmax function jacobian ([-1, 1])"
    inputs = np.array(inputs)
    jacobian = act.deriv_softmax(inputs)
    assert np.all(jacobian >= -1) and np.all(jacobian <= 1)
    
 
##################################################################################################################################

#Test the loss functions
    
    # binary cross entropy


def test_specifi_value_binary_cross_entropy():
    "Test specific output value of the binary cross entropy, when target = 1"
    target = 1.
    prediction = 0.7
    binary_ce = lf.binary_cross_entropy(prediction, target)
    expected_output = -np.log(0.7)
    assert np.abs(binary_ce - expected_output) < 1e-10
    

def test_specifi_value_binary_cross_entropy_bis():
    "Test specific output value of the binary cross entropy, when target = 0"
    target = 0.
    prediction = 0.2
    binary_ce = lf.binary_cross_entropy(prediction, target)
    expected_output = -np.log(1 - 0.2)
    assert np.abs(binary_ce - expected_output) < 1e-10
    

def test_limit_cases_binary_cross_entropy():
    "Test limit cases when the prediction is equal to 0 or 1"
    assert np.abs(lf.binary_cross_entropy(prediction=1., target=1.)) < 1e-10
    assert np.abs(lf.binary_cross_entropy(prediction=1., target=0.) + np.log(lf.clip_value)) < 1e-3
    assert np.abs(lf.binary_cross_entropy(prediction=0., target=1.) + np.log(lf.clip_value)) < 1e-3
    assert np.abs(lf.binary_cross_entropy(prediction=0., target=0.)) < 1e-10


@given(target = st.integers(min_value=0, max_value=1),
       prediction = st.floats(min_value=0., max_value=1.))
def test_symmetric_property_binary_cross_entropy(prediction, target):
    "Test the symmetry property of the binary cross entropy: same result for (prediction, target) and (1-prediction, 1-target)"
    result = lf.binary_cross_entropy(prediction, target)
    reverse_result = lf.binary_cross_entropy(1-prediction,1- target)
    assert np.abs(result - reverse_result) < 1e-3


@given(target = st.integers(min_value=0, max_value=1),
       prediction = st.floats(min_value=0, max_value=1))
def test_range_binary_cross_entropy(target, prediction):
    "Test that the result of the binary cross entropy is in the expected interval [0, -*log(clip_value)]"
    minimum = 0
    maximum = - np.log(lf.clip_value)
    binary_cross_entropy = lf.binary_cross_entropy(prediction, target)
    assert binary_cross_entropy >= minimum and binary_cross_entropy <= maximum


    # binary cross entropy deriv
    
    
def test_specifi_value_binary_cross_entropy_deriv():
    "Test specific output value of the binary cross entropy derivative, when target = 1"
    target = 1.
    prediction = 0.7
    binary_ced = lf.binary_cross_entropy_deriv(prediction, target)
    expected_output = -1 / 0.7
    assert np.abs(binary_ced - expected_output) < 1e-10
    

def test_specifi_value_binary_cross_entropy_deriv_bis():
    "Test specific output value of the binary cross entropy, when target = 0"
    target = 0.
    prediction = 0.3
    binary_ced = lf.binary_cross_entropy_deriv(prediction, target)
    expected_output = 1 / 0.7
    assert np.abs(binary_ced - expected_output) < 1e-10
    
    
def test_limit_cases_binary_cross_entropy_deriv():
    "Test limit cases when the prediction is equal to 0 or 1"
    assert np.abs(lf.binary_cross_entropy_deriv(prediction=1., target=1.) + 1) < 1e-10
    assert np.abs(lf.binary_cross_entropy_deriv(prediction=1., target=0.) - 1/(1 - (1-lf.clip_value))) < 1e-3
    assert np.abs(lf.binary_cross_entropy_deriv(prediction=0., target=1.) + 1/lf.clip_value) < 1e-3
    assert np.abs(lf.binary_cross_entropy_deriv(prediction=0., target=0.) - 1) < 1e-10
    

@given(target = st.integers(min_value=0, max_value=1),
       prediction = st.floats(min_value=0, max_value=1))
def test_range_binary_cross_entropy_deriv(target, prediction):
    "Test that the result of the binary cross entropy derivative is in the expected interval [-1/clip_value, 1/clip_value]"
    minimum = -1/lf.clip_value
    maximum = 1/lf.clip_value
    binary_cross_entropy_deriv = lf.binary_cross_entropy_deriv(prediction, target)
    assert binary_cross_entropy_deriv >= minimum and binary_cross_entropy_deriv <= maximum
    
  
@given(target = st.integers(min_value=0, max_value=1),
       prediction = st.floats(min_value=1e-5, max_value=1-1e-5))
def test_antisymmetric_property_binary_cross_entropy_deriv(prediction, target):
    "Test that the result of the binary cross entropy derivative is antisymmetric for the change (prediction, target) -> (1-prediction, 1-target)"
    result = lf.binary_cross_entropy_deriv(prediction, target)
    reverse_result = lf.binary_cross_entropy_deriv(1 - prediction, 1 - target)
    assert np.abs(result + reverse_result) < 1e-5
    
    
    # cross entropy
    

def test_specifi_value_cross_entropy():
    "Test specific output value of the cross entropy"
    target = [1., 0., 0.]
    prediction = [0.7, 0.2, 0.1]
    ce = lf.cross_entropy(prediction, target)
    expected_output = -np.log(0.7)
    assert np.abs(ce - expected_output) < 1e-10
    

def test_limit_case_cross_entropy():
    "Test limit case of cross entropy when target and prediction are equal"
    target = [1., 0., 0.]
    prediction = [1., 0., 0.]
    ce = lf.cross_entropy(prediction, target)
    assert np.abs(ce) < 1e-10
    
    
def test_limit_case_cross_entropy_bis():
    "Test limit case of cross entropy when target and prediction are different end prediction elements are eighter 0. or 1."
    target = [1., 0., 0.]
    prediction = [0., 1., 0.]
    ce = lf.cross_entropy(prediction, target)
    expected_ce = np.log(lf.clip_value)
    assert np.abs(ce + expected_ce) < 1e-10


# Here use data() rather than st.list(st.float, min, max) because it is required to have two lists of the same dimension
@given(data()) 
def test_range_cross_entropy(data):
    "Test that the result of the cross entropy is in the expected interval [0, -dim*log(clip_value)]"
    dim = data.draw(st.integers(min_value=2, max_value=20))
    target = data.draw(st.lists(st.sampled_from(target_values), min_size=dim, max_size=dim))
    prediction = data.draw(st.lists(st.sampled_from(prediction_values), min_size=dim, max_size=dim))
    minimum = 0
    maximum = - dim * np.log(lf.clip_value)
    cross_entropy = lf.cross_entropy(prediction, target)
    assert cross_entropy >= minimum and cross_entropy <= maximum
    
    
    # cross entropy derivative
    
    
def test_specifi_value_cross_entropy_deriv():
    "Test specific output value of the cross entropy derivative"
    target = [1., 0., 0., 0.]
    prediction = [0.65, 0.25, 0.005, 0.005]
    ced = lf.cross_entropy_deriv(prediction, target)
    expected_output = np.asarray([-1 / 0.65, 0., 0., 0.])
    assert np.all(np.isclose(ced, expected_output, rtol=0.1, atol=1e-5))
    

def test_limit_case_cross_entropy_deriv():
    "Test limit case of cross entropy derivative when target and prediction are equal"
    target = [1., 0., 0., 0.]
    prediction = [1., 0., 0., 0.]
    ced = lf.cross_entropy_deriv(prediction, target)
    expected_output = np.asarray([-1, 0, 0, 0])
    assert np.all(np.isclose(ced, expected_output, rtol=0.1, atol=1e-5))
    
    
def test_limit_case_cross_entropy_deriv_bis():
    "Test limit case of cross entropy when target and prediction are different end prediction elements are eighter 0. or 1."
    target = [0., 1., 0.]
    prediction = [0., 0., 1.]
    ced = lf.cross_entropy_deriv(prediction, target)
    expected_output = np.asarray([-0/lf.clip_value, -1/lf.clip_value, 0.])
    assert np.all(np.isclose(ced, expected_output, rtol=0.1, atol=1e-5))
    

@given(data())
def test_nan_values_cross_entropy_deriv(data):
    "Test that the derivative of the cross entropy does not produce nan values"
    dim = data.draw(st.integers(min_value=2, max_value=20))
    target = data.draw(st.lists(st.sampled_from(target_values), min_size=dim, max_size=dim))
    prediction = data.draw(st.lists(st.sampled_from(prediction_values), min_size=dim, max_size=dim))
    prediction = np.array(prediction)
    target = np.array(target)
    result = lf.cross_entropy_deriv(prediction, target)
    assert not np.any(np.isnan(result))


@given(data())
def test_inf_values_cross_entropy_deriv(data):
    "Test that the derivative of the cross entropy does not produce inf values"
    dim = data.draw(st.integers(min_value=2, max_value=20))
    target = data.draw(st.lists(st.sampled_from(target_values), min_size=dim, max_size=dim))
    prediction = data.draw(st.lists(st.sampled_from(prediction_values), min_size=dim, max_size=dim))
    prediction = np.array(prediction)
    target = np.array(target)
    result = lf.cross_entropy_deriv(prediction, target)
    assert np.all(result > -np.inf) and np.all(result < np.inf)

"""