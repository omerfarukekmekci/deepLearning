import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # to hide tensorflow startup info

import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") and logs["accuracy"] > 0.95:
            print("\nAccuracy is high enough, cancelling training.")
            self.model.stop_training = True


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


def create_fashion_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (2, 2), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),  # 10 categories
        ]
    )
    return model


fashion_model = create_fashion_model()
fashion_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fashion_model.fit(train_images, train_labels, epochs=10, callbacks=[myCallback()])

# evaluation
test_loss, test_acc = fashion_model.evaluate(test_images, test_labels)
print(f"Fashion-MNIST Test accuracy: {test_acc:.4f}")
