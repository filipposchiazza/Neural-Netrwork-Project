"Testing"

import numpy as np
from hypothesis import given
import hypothesis.strategies as st

import ann
import activation_functions as act


max_num_neurons = 10
num_neuron = [i for i in range(max_num_neurons + 1)]


@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_num_layers(inp, hidd, out):
    #Test if the number of layers is the correct one, according to the inputs given to build the ann'
    network = ann.Ann(inp, hidd, out)
    assert network.layers.size == len([network.num_inputs]) + network.num_hidden.size + len([network.num_outputs])
 
    
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))   
def test_num_weights(inp, hidd, out):
    #Test if the total number of weights of the neural network is the correct one, according to the inputs given to build it'
    network = ann.Ann(inp, hidd, out)
    
    counter_first = 0
    for i in range(network.layers.size-1):
        counter_first += (network.layers[i] + 1) * (network.layers[i+1])
    
    counter_second = 0
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            counter_second += len(network.weights[i][j])
          
    assert counter_first == counter_second


 
  
@given(inp = st.integers(min_value=1, max_value=1e5),
       hidd = st.lists(st.sampled_from(num_neuron), min_size=0, max_size=10),
       out = st.integers(min_value=1, max_value=50))
def test_forward_prop(inp, hidd, out):
    data = np.random.randn(inp)
    network = ann.Ann(inp, hidd, out)
    assert network.num_outputs == network.forward_prop(data, act.sigmoid).size
"""

@given(x = st.floats(allow_nan=False))
def test_range_sigmoid(x):
    # Test that the result of the Sigmoid function is in the interval [0,1]
    assert act.sigmoid(x) >= 0 and act.sigmoid(x) <= 1
   
"""
    



