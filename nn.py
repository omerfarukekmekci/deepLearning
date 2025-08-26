import numpy as np
import h5py as h5py
import matplotlib.pyplot as plt


# helper functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def tanh(Z):
    return np.tanh(Z)


# forward propagation
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


# backward propagation
def calculateDerivatives(A, Y_train, caches):
    grads = {}
    L = len(caches)
    m = Y_train.shape[1]

    linear_cache, activation_cache = caches[L - 1]
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


def updateParameters(parameters, grads, learning_rate=0.1):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


def train(A0, Y_train, layer_dims, numberOfGenerations, ifPrint):
    parameters = initialiseParameters(layer_dims)

    for i in range(numberOfGenerations):
        A, caches = linearForward(parameters, A0)
        grads = calculateDerivatives(A, Y_train, caches)
        parameters = updateParameters(parameters, grads, learning_rate=2)

        if ifPrint and i % 10 == 0:
            print("Training cost: " + str(compute_cost(A, Y_train)))

    return parameters


def predict(ifPrint=True):
    # loading the dataset
    train_dataset = h5py.File("cats_vs_dogs_64.h5", "r")
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

    n_x = X_train.shape[0]
    layer_dims = [n_x, 10, 3, 1]

    finalParameters = train(
        X_train, Y_train, layer_dims, numberOfGenerations=500, ifPrint=True
    )

    final_values, _ = linearForward(finalParameters, X_test)
    predictions = (final_values > 0.5).astype(int)
    test_accuracy = np.mean(predictions == Y_test)
    print("Test cost: " + str(compute_cost(final_values, Y_test)))
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


predict()
