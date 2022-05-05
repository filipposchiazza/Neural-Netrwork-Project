import sys
sys.path.insert(0, '/home/filippo/Documenti/Università/Magistrale/NeuralNetworkFromScratch/neuralnet')

import ann 
import activation_functions as act 
import loss_functions as lf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#Load the dataset on breast cancer
breast_cancer = load_breast_cancer()
data = breast_cancer.data
targets = breast_cancer.target

#Perform normalization
scaling = MinMaxScaler()
data = scaling.fit_transform(data)

#Split the dataset in 75% training and 25% test
data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.25)

#Create the neural network
neural_network = ann.Ann(num_inputs=len(data[0]), num_hidden=[5], num_outputs=1, activation_function=act.sigmoid, loss_function=lf.binary_cross_entropy)

#Train the neural network
neural_network.train(inputs=data_train, targets=targets_train, epochs=20, learning_rate=0.2)

#Evaluate the performances of the neural network on the test dataset
neural_network.evaluate_classification(inputs=data_test, targets=targets_test)