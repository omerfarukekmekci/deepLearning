import cupy as cp
import h5py as h5py
import matplotlib.pyplot as plt

# print(cp.__version__) = 13.6.0
x = cp.arange(10)
print(x.device)


# helper functions
def sigmoid(Z):
    return cp.where(Z >= 0, 1 / (1 + cp.exp(-Z)), cp.exp(Z) / (1 + cp.exp(Z)))


def ReLU(Z):
    return cp.maximum(0, Z)


# forward propagation
def initialiseParameters(layer_dims):
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        # He initialization for ReLU
        parameters["W" + str(l)] = cp.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * cp.sqrt(2.0 / layer_dims[l - 1])
        parameters["b" + str(l)] = cp.zeros((layer_dims[l], 1))

    return parameters


def linearForward(parameters, A0, dropout_rates=None, training=True):
    A = A0
    caches = []
    dropout_caches = []
    L = len(parameters) // 2

    # Set default dropout rates if not provided
    if dropout_rates is None:
        dropout_rates = [0.4, 0.3, 0.3, 0.2]  # Different rates for each hidden layer

    for l in range(1, L):
        A_prev = A
        Z = cp.dot(parameters["W" + str(l)], A_prev) + parameters["b" + str(l)]
        linear_cache = (A_prev, parameters["W" + str(l)], parameters["b" + str(l)])

        A = ReLU(Z)

        # Apply dropout to hidden layers during training
        if training and l <= len(dropout_rates):
            dropout_rate = dropout_rates[l - 1]
            dropout_mask = (cp.random.rand(*A.shape) > dropout_rate).astype(float)
            A = A * dropout_mask / (1 - dropout_rate)  # Inverted dropout
            dropout_caches.append(dropout_mask)
        else:
            dropout_caches.append(None)

        activation_cache = Z
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    # Final layer (no dropout on output)
    A_prev = A
    Z = cp.dot(parameters["W" + str(L)], A_prev) + parameters["b" + str(L)]
    linear_cache = (A_prev, parameters["W" + str(L)], parameters["b" + str(L)])

    A = sigmoid(Z)
    activation_cache = Z
    dropout_caches.append(None)  # No dropout on output layer

    cache = (linear_cache, activation_cache)
    caches.append(cache)

    return A, caches, dropout_caches


def compute_cost(A, Y_train):
    m = Y_train.shape[1]
    # Add epsilon to prevent log(0) - IMPROVEMENT 1
    epsilon = 1e-8
    A = cp.clip(A, epsilon, 1 - epsilon)
    cost = -cp.sum(Y_train * cp.log(A) + (1 - Y_train) * cp.log(1 - A)) / m
    cost = cp.squeeze(cost)  # this makes the value a scalar instead of a 1*1 matrix

    return cost


# backward propagation
def calculateDerivatives(
    A, Y_train, caches, dropout_caches, dropout_rates=None, training=True
):
    grads = {}
    L = len(caches)
    m = Y_train.shape[1]

    # Set default dropout rates if not provided
    if dropout_rates is None:
        dropout_rates = [0.4, 0.3, 0.3, 0.2]

    # Final layer (sigmoid)
    linear_cache, activation_cache = caches[L - 1]
    A_prev, W, b = linear_cache
    Z = activation_cache

    dZ = A - Y_train
    dW = cp.dot(dZ, A_prev.T) / m
    db = cp.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = cp.dot(W.T, dZ)

    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # Hidden layers (ReLU with dropout)
    for l in reversed(range(L - 1)):
        linear_cache, activation_cache = caches[l]
        A_prev, W, b = linear_cache
        Z = activation_cache

        # Apply dropout mask to gradients during training
        if training and l < len(dropout_rates) and dropout_caches[l] is not None:
            dropout_rate = dropout_rates[l]
            dA_prev = dA_prev * dropout_caches[l] / (1 - dropout_rate)

        dZ = dA_prev * (Z > 0).astype(float)
        dW = cp.dot(dZ, A_prev.T) / m
        db = cp.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = cp.dot(W.T, dZ)

        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def updateParameters(parameters, grads, learning_rate=0.1):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


# IMPROVEMENT 2: Accept learning_rate and dropout_rates as parameters
def train(
    A0,
    Y_train,
    layer_dims,
    numberOfGenerations,
    learning_rate=0.005,
    dropout_rates=None,
    ifPrint=True,
):
    parameters = initialiseParameters(layer_dims)

    # Set default dropout rates if not provided
    if dropout_rates is None:
        dropout_rates = [0.4, 0.3, 0.3, 0.2]  # For 4 hidden layers

    for i in range(numberOfGenerations + 1):
        A, caches, dropout_caches = linearForward(
            parameters, A0, dropout_rates, training=True
        )
        grads = calculateDerivatives(
            A, Y_train, caches, dropout_caches, dropout_rates, training=True
        )
        parameters = updateParameters(parameters, grads, learning_rate)

        if ifPrint and i % 100 == 0:
            print(f"Generation: {i},    Training cost: {compute_cost(A, Y_train):.6f}")

    return parameters


def predict(ifPrint=True):
    # loading the dataset
    train_dataset = h5py.File("large_set_64.h5", "r")
    X_train = cp.array(train_dataset["X_train"][:], dtype=cp.float32)
    Y_train = cp.array(train_dataset["Y_train"][:], dtype=cp.float32)
    X_test = cp.array(train_dataset["X_test"][:], dtype=cp.float32)
    Y_test = cp.array(train_dataset["Y_test"][:], dtype=cp.float32)
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
    # Updated smaller architecture
    layer_dims = [n_x, 512, 256, 128, 64, 1]

    # Dropout rates for each hidden layer (4 hidden layers)
    dropout_rates = [0.3, 0.25, 0.2, 0.15]

    # test multiple learning rates
    learning_rates = [0.03, 0.01]
    best_accuracy = 0
    best_params = None
    best_lr = None

    print("Testing multiple learning rates with dropout...")
    print("=" * 50)

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        print("-" * 30)

        # Train with current learning rate and dropout
        finalParameters = train(
            X_train,
            Y_train,
            layer_dims,
            numberOfGenerations=2000,
            learning_rate=lr,
            dropout_rates=dropout_rates,
            ifPrint=True,
        )

        # Training set evaluation (no dropout during evaluation)
        train_values, _, _ = linearForward(finalParameters, X_train, training=False)
        train_predictions = (train_values > 0.5).astype(int)
        train_accuracy = cp.mean(train_predictions == Y_train)

        # Test set evaluation (no dropout during evaluation)
        test_values, _, _ = linearForward(finalParameters, X_test, training=False)
        test_predictions = (test_values > 0.5).astype(int)
        test_accuracy = cp.mean(test_predictions == Y_test)

        # Print results for this learning rate
        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"Training cost: {compute_cost(train_values, Y_train):.6f}")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test cost: {compute_cost(test_values, Y_test):.6f}")

        # overfitting check
        accuracy_gap = train_accuracy - test_accuracy
        if accuracy_gap > 0.03:  # 3% difference
            print("WARNING: Possible overfitting detected!")
            print(
                f"Training accuracy is {accuracy_gap*100:.1f}% higher than test accuracy"
            )
        else:
            print("✓ Good generalization - no significant overfitting detected")

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


# Run the training
predict()
