import sys
sys.path.insert(0, '/home/filippo/Scrivania/Università/Magistrale/project_ANN/neuralnet')

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
my_nn = ann.Ann(len(data[0]), [5], 1)

#Train the neural network
my_nn.train(data_train, targets_train, 20, 0.2, act.sigmoid, lf.binary_cross_entropy)

#Evaluate the performances of the neural network on the test dataset
my_nn.evaluate_classification(data_test, targets_test)