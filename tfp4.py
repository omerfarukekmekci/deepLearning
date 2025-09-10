import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # to hide tensorflow startup info

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Variables to test for best accuracy:
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
NUM_LAYERS = 2
NUM_CONVOS = 3
USE_AUG = 0
USE_REG = 0
USE_NORM = 0
USE_CALLB = 0


def print_parameters(
    IMAGE_SIZE,
    BATCH_SIZE,
    LR,
    EPOCHS,
    NUM_LAYERS,
    NUM_CONVOS,
    USE_AUG,
    USE_REG,
    USE_NORM,
    USE_CALLB,
):
    print("")


TRAIN_DIR = "PetImages\\train"
VAL_DIR = "PetImages\\validation"

import os
from PIL import Image


def create_no_aug_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            # tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            # tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def create_augmentation():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.2, fill_mode="nearest"),
        ]
    )

    return model


no_aug_model, augmentation = create_no_aug_model(), create_augmentation()

model = tf.keras.models.Sequential([no_aug_model])
# model = tf.keras.models.Sequential([augmentation, no_aug_model])


train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(LR),
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1,
    validation_data=validation_dataset,
    # callbacks=[EarlyStoppingCallback()],
)

print_parameters()
