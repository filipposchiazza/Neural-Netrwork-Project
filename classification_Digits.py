import numpy as np
import ann
import activation_functions as act
import loss_functions as lf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#HANDWRITTEN DIGITS DATASET

#Load the handwritten digits dataset
digits = load_digits()
data = digits.data
targets = digits.target

#Data pre-processing 
targets_rev = np.zeros((len(targets),10))

for i in range(len(targets)):
    if targets[i] == 0:
        targets_rev[i][0] = 1
    elif targets[i] == 1:
        targets_rev[i][1] = 1
    elif targets[i] == 2:
        targets_rev[i][2] = 1
    elif targets[i] == 3:
        targets_rev[i][3] = 1
    elif targets[i] == 4:
        targets_rev[i][4] = 1
    elif targets[i] == 5:
        targets_rev[i][5] = 1
    elif targets[i] == 6:
        targets_rev[i][6] = 1
    elif targets[i] == 7:
        targets_rev[i][7] = 1
    elif targets[i] == 8:
        targets_rev[i][8] = 1
    elif targets[i] == 9:
        targets_rev[i][9] = 1   

#Perform normalization
scaling = MinMaxScaler()
data = scaling.fit_transform(data)

#Split the dataset in 70% training and 30% test
data_train, data_test, targets_train, targets_test = train_test_split(data, targets_rev, test_size=0.3)

#Create the neural network      
my_ann = ann.Ann(64, [15], 10)

#Train the neural network
my_ann.train(data_train, targets_train, epochs=30, learning_rate=0.1, activation_function=act.softmax, loss_func=lf.cross_entropy)

#Evaluate the performances of the neural network on the test dataset
my_ann.evaluate_classification(data_test, targets_test)



