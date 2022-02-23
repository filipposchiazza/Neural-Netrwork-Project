"Testing"

import numpy as np
from hypothesis import given
import hypothesis.strategies as st

import ann
import activation_functions as act


max_num_neurons = 10
num_neuron = [i for i in range(max_num_neurons + 1)]

##################################################################################################################################

# Test the costruction of the neural network (the function __init__)

@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_num_layers(inp, hidd, out):
    #Test if the number of layers is the correct one, according to the inputs given to build the ann
    network = ann.Ann(inp, hidd, out)
    assert network.layers.size == len([network.num_inputs]) + network.num_hidden.size + len([network.num_outputs])
 
    
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))   
def test_num_weights(inp, hidd, out):
    # Test if the total number of weights of the neural network is the correct one, according to the inputs given to build it
    network = ann.Ann(inp, hidd, out)
    counter_first = 0
    for i in range(len(network.layers)-1):
        counter_first += (network.layers[i]) * (network.layers[i+1])
    counter_second = 0
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            counter_second += len(network.weights[i][j])
    assert counter_first == counter_second


@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50)) 
def test_num_biases(inp, hidd, out):
    # Test if the total number of biases is equal to (total number of neurons - number of neurons in the input layer)
    network = ann.Ann(inp, hidd, out)
    counter_first = 0
    for b in network.biases:
        counter_first += b.size
    counter_second = 0
    for i in range(1, len(network.layers)):
        counter_second += network.layers[i]
    assert counter_first == counter_second


@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_num_linear_comb(inp, hidd, out):
    # Test if the total number of linear combinations is equal to (total number of neurons - number of neurons in the input layer)
    network = ann.Ann(inp, hidd, out)
    counter_first = 0
    for lin in network.linear_comb:
        counter_first += lin.size
    counter_second = 0
    for i in range(1, len(network.layers)):
        counter_second += network.layers[i]
    assert counter_first == counter_second
    
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))   
def test_num_activations(inp, hidd, out):
    # Test if the numbers of activations stored for each layer is the correct one (corresponding to the number of neurons)
    network = ann.Ann(inp, hidd, out)
    for i in range(len(network.layers)):
        assert len(network.activations[i]) == network.layers[i] 

        
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50)) 
def test_num_weights_derivatives(inp, hidd, out):
    # Test if the number of weights'derivatives stored is the same of the number of weights (as it should be)
    network = ann.Ann(inp, hidd, out)
    for i in range(len(network.weights)):
        assert network.weights[i].size == network.weights_deriv[i].size


@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50)) 
def test_num_biases_derivatives(inp, hidd, out):
    # Test if the number of biases'derivatives stored is the same of the number of weights (as it should be)
    network = ann.Ann(inp, hidd, out)
    for i in range(len(network.biases)):
        assert network.biases[i].size == network.biases_deriv[i].size

##########################################################################################################################
 
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_forward_prop(inp, hidd, out):
    data = np.random.randn(inp)
    network = ann.Ann(inp, hidd, out)
    assert network.num_outputs == network.forward_prop(data, act.sigmoid).size
    
##########################################################################################################################    

"""

@given(x = st.floats(allow_nan=False))
def test_range_sigmoid(x):
    # Test that the result of the Sigmoid function is in the interval [0,1]
    assert act.sigmoid(x) >= 0 and act.sigmoid(x) <= 1
   
"""
    



