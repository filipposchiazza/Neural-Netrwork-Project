
import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from neuralnet import activation_functions as act


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


@given(inputs = st.lists(st.floats(max_value=1e100, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_sorting_invariance(inputs):
    "Test that sorting the inputs and than appling the softmax function is equivalent to sort the output of the softmax function"
    inputs = np.asarray(inputs)
    softmax = act.softmax(inputs)
    softmax.sort()
    inputs.sort()
    softmax_after_sorting = act.softmax(inputs)
    assert np.all(np.isclose(softmax, softmax_after_sorting))
    
    
@given(inputs = st.lists(st.floats(max_value=1e100, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_shuffle_invariance(inputs):
    "Test that shuffling the inputs does not change the output of the softmax function"
    inputs = np.asarray(inputs)
    softmax = act.softmax(inputs)
    softmax.sort()
    np.random.shuffle(inputs)
    softmax_after_shuffle = act.softmax(inputs)
    softmax_after_shuffle.sort()
    assert np.all(np.isclose(softmax, softmax_after_shuffle))
    
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
    

