import sys
sys.path.insert(0, '/home/filippo/Documenti/Università/Magistrale/NeuralNetworkFromScratch/neuralnet')

import numpy as np
import ann
import activation_functions as act
import loss_functions as lf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# IRIS DATASET

#Load the Iris dataset
iris = load_iris()
data = iris.data
targets = iris.target

#Data pre-processing 
targets_rev = np.zeros((len(targets),3))

for i in range(len(targets)):
    if targets[i] == 0:
        targets_rev[i] = np.array([1, 0, 0])
    elif targets[i] == 1:
        targets_rev[i] = np.array([0, 1, 0])
    else:
        targets_rev[i] = np.array([0, 0, 1])
        
#Join data and targets_rev
joint = np.concatenate((data, targets_rev), axis=1)
#Shuffle the dataset
np.random.shuffle(joint)
joint = np.split(joint, [4], axis = 1)
#Separate again data and targets_rev
data = joint[0]
targets_rev = joint[1]

#Perform normalization
scaling = MinMaxScaler()
data = scaling.fit_transform(data)

#Split the dataset in 70% training and 30% test
data_train, data_test, targets_train, targets_test = train_test_split(data, targets_rev, test_size=0.3)

#Create the neural network       
neural_network = ann.Ann(num_inputs=4, num_hidden=[5], num_outputs=3, activation_function=act.softmax, loss_function=lf.cross_entropy)
#Train the neural network
neural_network.train(inputs=data_train, targets=targets_train, epochs=20, learning_rate=0.1)

#Evaluate the performances of the neural network on the test dataset
neural_network.evaluate_classification(inputs=data_test, targets=targets_test)