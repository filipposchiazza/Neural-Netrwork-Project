import numpy as np
import ann 
import activation_functions as act 
import loss_functions as lf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
breast_cancer = load_breast_cancer()
data = breast_cancer.data
targets = breast_cancer.target

# perform normalization
scaling = MinMaxScaler()
data = scaling.fit_transform(data)

# split the dataset in 75% training and 25% test
data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.25)

# Create the neural network
my_nn = ann.Ann(len(data[0]), [5], 1)

# Train the neural network
my_nn.train(data_train, targets_train, 50, 0.2, act.sigmoid, act.deriv_sigmoid, lf.binary_cross_entropy, lf.binary_cross_entropy_deriv)

# Predictions on the test dataset
predictions = my_nn.predict(data_test)
predictions = np.reshape(predictions, (len(predictions), ))

for i in range(len(predictions)):
    if predictions[i] < 0.5:
        predictions[i] = 0
    else: 
        predictions[i] = 1

correct_predictions = predictions == targets_test        
num_correct_prediction = np.sum(correct_predictions)
percentage = num_correct_prediction / len(predictions) * 100

print("Correct classification on the test dataser: {}/{}".format(num_correct_prediction, len(predictions)))
print ("Percentage of correct classification on the test dataset: {}%".format(percentage))