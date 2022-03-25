import numpy as np
import ann
import activation_functions as act
import loss_functions as lf
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#HANDWRITTEN DIGITS DATASET

# load the dataset
digits = load_digits()
data = digits.data
targets = digits.target

# data pre-processing 
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

# perform normalization
scaling = MinMaxScaler()
data = scaling.fit_transform(data)

# split the dataset in 70% training and 30% test
data_train, data_test, targets_train, targets_test = train_test_split(data, targets_rev, test_size=0.3)

# Neural network training       
my_ann = ann.Ann(64, [15], 10)

my_ann.train(data_train, targets_train, 50, 0.1, act.softmax, lf.cross_entropy)

predictions = np.zeros((len(data_test), len(targets_test[0])))
for i in range(len(data_test)):
    predictions[i] = my_ann.predict(data_test[i])

for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if predictions[i][j] < 0.5:
            predictions[i][j] = 0
        else: 
            predictions[i][j] = 1
    print("Prediction: {}    true value: {}".format(predictions[i], targets_test[i]))

correct_predictions = np.all(predictions == targets_test, axis=1)        
num_correct_prediction = np.sum(correct_predictions)
percentage = num_correct_prediction / len(predictions) * 100

print("Correct classification on the test dataser: {}/{}".format(num_correct_prediction, len(predictions)))
print ("Percentage of correct classification on the test dataset: {}%".format(percentage))