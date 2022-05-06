
import numpy as np
from hypothesis import given
from hypothesis.strategies import data
import hypothesis.strategies as st

from neuralnet import loss_functions as lf


#Test the loss functions

target_values = [0., 1.]
    
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
    prediction = data.draw(st.lists(st.floats(min_value=0., max_value=1.), min_size=dim, max_size=dim))
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
    prediction = data.draw(st.lists(st.floats(min_value=0., max_value=1.), min_size=dim, max_size=dim))
    prediction = np.array(prediction)
    target = np.array(target)
    result = lf.cross_entropy_deriv(prediction, target)
    assert not np.any(np.isnan(result))


@given(data())
def test_inf_values_cross_entropy_deriv(data):
    "Test that the derivative of the cross entropy does not produce inf values"
    dim = data.draw(st.integers(min_value=2, max_value=20))
    target = data.draw(st.lists(st.sampled_from(target_values), min_size=dim, max_size=dim))
    prediction = data.draw(st.lists(st.floats(min_value=0., max_value=1.), min_size=dim, max_size=dim))
    prediction = np.array(prediction)
    target = np.array(target)
    result = lf.cross_entropy_deriv(prediction, target)
    assert np.all(result > -np.inf) and np.all(result < np.inf)


