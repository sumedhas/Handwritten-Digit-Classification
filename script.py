import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import time
import pickle
from math import sqrt
from sklearn.svm import SVC
from pylab import *

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all(2).mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    
    w = initialWeights.reshape((n_features+1,1))
    bias = np.ones((n_data, 1))
    train_data = np.hstack((train_data, bias))
    
    
    
    #Computing theta_n
    theta_n = sigmoid(np.dot(train_data, w))
    #Computing error
    #yn*ln(θn)
    t1 = (labeli*np.log(theta_n))
    
    
    #(1 − yn)ln(1 − θn)
    t2 = ((1.0 - labeli)*np.log(1.0 - theta_n))
    
    error = -np.sum(t1+t2)/theta_n.shape[0]
    
    #Computing error_grad
    error_grad = (theta_n - labeli) * train_data
    error_grad = np.sum(error_grad, axis=0)/theta_n.shape[0]
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    bias = np.ones((data.shape[0], 1))
    data = np.hstack((data, bias))
    
    res = sigmoid(np.dot(data, W))
    label = np.argmax(res, axis=1)
    
    label = label.reshape((data.shape[0],1))

    return label
    


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # w = (D + 1) * K where K is the number of classifier, in this case n_class = 10
    w = params.reshape(n_feature + 1, n_class)

    # adding bias X size of N x D
    bias = np.ones((n_data, 1))
    x = np.hstack((bias, train_data))

    # theta of nk, refer to (5) using 1-of-K
    theta = (np.exp(np.dot(x, w)) / np.sum(np.exp(np.dot(x, w)), axis=1).reshape(n_data, 1))

    # calculating error, refer to (7)
    error = -1 * np.sum(np.sum(labeli * np.log(theta)))

    # calculating error_grad
    error_grad = (np.dot(x.T, theta - labeli))

    # get optimal parameter vector w iteratively, refer (9)
    error = error / n_data
    error_grad = error_grad / n_data

    return error, error_grad.flatten()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # adding bias
    bias = np.ones((data.shape[0], 1))
    x = np.hstack((bias, data))

    # refer to (5)
    temp = np.exp(np.dot(x, W)) / np.sum(np.exp(np.dot(x, W)))
    for i in range(temp.shape[0]):
        label[i] = np.argmax(temp[i])
    label = label.reshape(label.shape[0], 1)

    return label


"""
Script for Logistic Regression
"""
timer = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

getTime = time.time() - timer
print('\n It took: ' + str(getTime) + ' seconds to complete')

f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()


"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

###################### Linear kernel (all other parameters are kept default) ######################
print('SVM with linear kernel')
timer = time.time()
clf = SVC(kernel='linear')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')
getTime = time.time() - timer
print('\n It took ' + str(getTime) + ' seconds to complete')


###################### Kernel as radical basis function and Gamma = 1.0 (all other parameters are kept default) ######################
print('\n\n SVM with radial basis function, gamma = 1')
timer = time.time()
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')
getTime = time.time() - timer
print('\n It took ' + str(getTime) + ' seconds to complete')


###################### Kernel as radical basis function and all other parameters are kept as default ######################
print('\n\n SVM with radial basis function, gamma = 0')
timer = time.time()
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')
getTime = time.time() - timer
print('\n It took ' + str(getTime) + ' seconds to complete')


###################### All parameters including gamma are kept as default ######################
print('\n\n SVM with radial basis function, different values of C')
timer = time.time()
train_accuracy = np.zeros(11)
valid_accuracy = np.zeros(11)
test_accuracy = np.zeros(11)
c_values = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
#c_values[0] = 1.0   # first value is 1
#c_values[1:] = [x for x in np.arange(10.0, 101.0, 10.0)]  

### for every C, train and compute accuracy ###
for i in range(11):
    print('\n For C value: ' + str(c_values[i])
          
    clf = SVC(C=c_values[i],kernel='rbf')
    clf.fit(train_data, train_label.flatten())
    
    train_accuracy[i] = 100*clf.score(train_data, train_label)
    print('\n Training set Accuracy:' + str(train_accuracy[i]) + '%')
    
    valid_accuracy[i] = 100*clf.score(validation_data, validation_label)
    print('\n Validation set Accuracy:' + str(valid_accuracy[i]) + '%')
    
    test_accuracy[i] = 100*clf.score(test_data, test_label)
    print('\n Testing set Accuracy:' + str(test_accuracy[i]) + '%')
    
getTime = time.time() - timer
print('\n It took ' + str(getTime) + ' seconds to complete')

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
timer = time.time()
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

getTime = time.time() - timer
print('\n It took: ' + str(getTime) + ' seconds to complete')

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
