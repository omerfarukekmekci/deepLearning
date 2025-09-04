import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # hide TensorFlow startup INFO

import numpy as np
import tensorflow as tf

# Define a simple Sequential model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(1,)),  # input shape must be in parentheses
        tf.keras.layers.Dense(units=1),  # single neuron, linear output
    ]
)

# Compile the model (choose optimizer + loss function)
model.compile(optimizer="sgd", loss="mean_squared_error")

# Training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 20.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 39.0], dtype=float)

# Train the model for 500 epochs (iterations over the dataset)
model.fit(xs, ys, epochs=100)

# Make a prediction
print(model.predict(np.array([10.0])))
