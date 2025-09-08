import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # to hide tensorflow startup info

import tensorflow as tf

train_dir = r"C:\Users\omerf\Desktop\Deep Learning\PetImages"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(150, 150), batch_size=20, label_mode="binary"
)

SHUFFLE_BUFFER_SIZE = 1000
train_dataset_final = (
    train_dataset.cache()
    .shuffle(SHUFFLE_BUFFER_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# data augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.2, fill_mode="nearest"),
    ]
)


def create_dog_cat_model():
    model = tf.keras.Sequential(
        [
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),  # binary classification
        ]
    )
    return model


dog_cat_model = create_dog_cat_model()
dog_cat_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

dog_cat_model.fit(train_dataset_final, epochs=10)
