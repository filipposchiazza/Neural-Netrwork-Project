"Testing"

import numpy as np
import ann


network = ann.Ann(5, [8,3], 2)

def test_num_layers():
    'Test if the number of layers is the correct one, according to the inputs given to build the ann'
    network = ann.Ann(5, [8,3], 2)
    assert len(network.layers) == len([network.num_inputs]) + len(network.num_hidden) + len([network.num_outputs])
    
def test_num_weights():
    'Test if the total number of weights of the neural network is the correct one, according to the inputs given to build it'
    network = ann.Ann(5, [4,3], 2)
    
    counter_first = 0
    for i in range(len(network.layers)-1):
        counter_first += (network.layers[i] + 1) * (network.layers[i+1])
    
    counter_second = 0
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            counter_second += len(network.weights[i][j])
            
    assert counter_first == counter_second
    
    

    



