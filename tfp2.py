import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # hide TensorFlow startup INFO

import numpy as np
import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["loss"] < 0.2:
            print("Loss is low so cancelling training!")
            self.model.stop_training = True


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28)),  # input shape must be in parentheses
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Compile the model (choose optimizer + loss function)
model.compile(optimizer=tf.optimizers.Adam(), loss="sparse_categorical_crossentropy")

model.fit(train_images, train_labels, epochs=5, callbacks=[myCallback()])


# Make a prediction
# print(model.predict(test_images))
