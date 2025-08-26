import numpy as np
import h5py as h5py
import matplotlib.pyplot as plt


# helper functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def tanh(Z):
    return np.tanh(Z)


def initialiseParameters(layer_dims):
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        parameters["W" + str(l)] = (
            np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        )
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linearForward(parameters, A0):
    A = A0
    caches = []
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters["W" + str(l)], A_prev) + parameters["b" + str(l)]
        linear_cache = (A_prev, parameters["W" + str(l)], parameters["b" + str(l)])

        A = tanh(Z)
        activation_cache = Z

        cache = (linear_cache, activation_cache)
        caches.append(cache)

    A_prev = A
    Z = np.dot(parameters["W" + str(L)], A_prev) + parameters["b" + str(L)]
    linear_cache = (A_prev, parameters["W" + str(L)], parameters["b" + str(L)])

    A = sigmoid(Z)
    activation_cache = Z

    cache = (linear_cache, activation_cache)
    caches.append(cache)

    return A, caches


def compute_cost(A, Y_train):
    m = Y_train.shape[1]
    cost = -np.sum(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A)) / m
    cost = np.squeeze(cost)  # this makes the value a scalar instead of a 1*1 matrix

    return cost


def calculateDerivatives(A, Y_train, caches, parameters):
    grads = {}
    L = len(caches)
    m = Y_train.shape[1]

    linear_cache, activation_cache = caches[L]
    A_prev, W, b = linear_cache
    Z = activation_cache

    dZ = A - Y_train
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    for l in reversed(range(L - 1)):
        linear_cache, activation_cache = caches[l]
        A_prev, W, b = linear_cache
        Z = activation_cache

        dZ = dA_prev * (1 - np.tanh(Z) ** 2)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def updateParameters(parameters, grads, learning_rate=0.01):
    L = len(parameters) // 2

    for l in range(1, L):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


def train(layer_dims, numberOfGenerations=10):
    parameters = initialiseParameters()

    for i in range(1, numberOfGenerations):
        A, caches = linearForward(parameters, A0)
        grads = calculateDerivatives(A, Y_train, caches, parameters)
        parameters = updateParameters(parameters, grads, learning_rate=0.01)

    return XXX
