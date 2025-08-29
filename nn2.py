import numpy as np
import h5py as h5py
import matplotlib.pyplot as plt


# helper functions
def sigmoid(Z):
    return np.where(Z >= 0, 1 / (1 + np.exp(-Z)), np.exp(Z) / (1 + np.exp(Z)))


def ReLU(Z):
    return np.maximum(0, Z)


# forward propagation
def initialiseParameters(layer_dims):
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        # He initialization for ReLU
        parameters["W" + str(l)] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(2.0 / layer_dims[l - 1])
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

        A = ReLU(Z)
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
    # Add epsilon to prevent log(0) - IMPROVEMENT 1
    epsilon = 1e-8
    A = np.clip(A, epsilon, 1 - epsilon)
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

        dZ = dA_prev * (Z > 0).astype(float)
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


# IMPROVEMENT 2: Accept learning_rate as parameter
def train(
    A0, Y_train, layer_dims, numberOfGenerations, learning_rate=0.005, ifPrint=True
):
    parameters = initialiseParameters(layer_dims)

    for i in range(numberOfGenerations):
        A, caches = linearForward(parameters, A0)
        grads = calculateDerivatives(A, Y_train, caches)
        parameters = updateParameters(parameters, grads, learning_rate)

        if ifPrint and i % 100 == 0:
            print(f"Generation: {i},    Training cost: {compute_cost(A, Y_train):.6f}")

    return parameters


def predict(ifPrint=True):
    # loading the dataset
    train_dataset = h5py.File("large_set_64.h5", "r")
    X_train = np.array(train_dataset["X_train"][:])
    Y_train = np.array(train_dataset["Y_train"][:])
    X_test = np.array(train_dataset["X_test"][:])
    Y_test = np.array(train_dataset["Y_test"][:])
    train_dataset.close()

    # reshaping the matrices
    Y_train = Y_train.reshape(1, -1)
    Y_test = Y_test.reshape(1, -1)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    # data normalization check
    if X_train.max() > 1:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        print("Data normalized from [0,255] to [0,1] range")

    m = X_train.shape[1]

    # checking the dimensions
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    n_x = X_train.shape[0]
    layer_dims = [n_x, 240, 120, 60, 30, 1]

    # test multiple learning rates
    learning_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
    best_accuracy = 0
    best_params = None
    best_lr = None

    print("Testing multiple learning rates...")
    print("=" * 50)

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        print("-" * 30)

        # Train with current learning rate
        finalParameters = train(
            X_train,
            Y_train,
            layer_dims,
            numberOfGenerations=1501,
            learning_rate=lr,
            ifPrint=True,
        )

        # Training set evaluation
        train_values, _ = linearForward(finalParameters, X_train)
        train_predictions = (train_values > 0.5).astype(int)
        train_accuracy = np.mean(train_predictions == Y_train)

        # Test set evaluation
        test_values, _ = linearForward(finalParameters, X_test)
        test_predictions = (test_values > 0.5).astype(int)
        test_accuracy = np.mean(test_predictions == Y_test)

        # Print results for this learning rate
        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"Training cost: {compute_cost(train_values, Y_train):.6f}")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test cost: {compute_cost(test_values, Y_test):.6f}")

        # overfitting check
        if train_accuracy - test_accuracy > 0.03:  # 3% difference
            print("WARNING: Possible overfitting detected")
            print(
                f"Training accuracy is {(train_accuracy - test_accuracy)*100:.1f}% higher than test accuracy"
            )

        # Keep track of best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_params = finalParameters
            best_lr = lr

    # Print final results
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Best learning rate: {best_lr}")
    print(f"Best test accuracy: {best_accuracy * 100:.2f}%")
    print("=" * 50)

    return best_params, best_accuracy


predict()
