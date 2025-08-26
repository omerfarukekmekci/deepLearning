import numpy as np
import h5py as h5py
import matplotlib.pyplot as plt


# helper functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def tanh(Z):
    return np.tanh(Z)


# loading the dataset
train_dataset = h5py.File("cats_vs_dogs_96.h5", "r")
X_train = np.array(train_dataset["X_train"][:])
Y_train = np.array(train_dataset["Y_train"][:])
X_test = np.array(train_dataset["X_test"][:])
Y_test = np.array(train_dataset["Y_test"][:])


# reshaping the matrices
Y_train = Y_train.reshape(1, -1)
Y_test = Y_test.reshape(1, -1)
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
m = X_train.shape[1]


# checking the dimensions
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# forward propagation
layer_dims = [X_train.shape[1], 10, 3, 1]
A0 = X_train
caches = []

W1 = np.random.randn(layer_dims[1], A0.shape[0]) * 0.01
b1 = np.zeros((layer_dims[1], 1))
Z1 = np.dot(W1, A0) + b1
A1 = tanh(Z1)
linear_cache_1 = (A0, W1, b1)
activation_cache_1 = Z1
cache_1 = (linear_cache_1, activation_cache_1)
caches.append(cache_1)

W2 = np.random.randn(layer_dims[2], A1.shape[0]) * 0.01
b2 = np.zeros((layer_dims[2], 1))
Z2 = np.dot(W2, A1) + b2
A2 = tanh(Z2)
linear_cache_2 = (A1, W2, b2)
activation_cache_2 = Z2
cache_2 = (linear_cache_2, activation_cache_2)
caches.append(cache_2)

W3 = np.random.randn(layer_dims[3], A2.shape[0]) * 0.01
b3 = np.zeros((layer_dims[3], 1))
Z3 = np.dot(W3, A2) + b3
A3 = sigmoid(Z3)
linear_cache_3 = (A2, W3, b3)
activation_cache_3 = Z3
cache_3 = (linear_cache_3, activation_cache_3)
caches.append(cache_3)


# calculate the cost
cost = -np.sum(Y_train * np.log(A3) + (1 - Y_train) * np.log(1 - A3)) / m
cost = np.squeeze(cost)  # this makes the value a scalar instead of a 1*1 matrix


# calculate the derivatives for the backwards propagation
dZ3 = A3 - Y_train
dW3 = np.dot(dZ3, A2.T) / m
db3 = np.sum(dZ3, axis=1, keepdims=True) / m
dA2 = np.dot(W3.T, dZ3)

dZ2 = dA2 * (1 - A2**2)
dW2 = np.dot(dZ2, A1.T) / m
db2 = np.sum(dZ2, axis=1, keepdims=True) / m
dA1 = np.dot(W2.T, dZ2)

dZ1 = dA1 * (1 - A1**2)
dW1 = np.dot(dZ1, A0.T) / m
db1 = np.sum(dZ1, axis=1, keepdims=True) / m


# update parameters
learning_rate = 0.01

for i in range(0, 100):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
