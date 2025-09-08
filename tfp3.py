import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # to hide tensorflow startup info

import tensorflow as tf

TRAIN_DIR = "horse-or-human/train"
VAL_DIR = "horse-or-human/validation"

# Directory with training horse pictures
train_horse_dir = os.path.join(TRAIN_DIR, "horses")

# Directory with training human pictures
train_human_dir = os.path.join(TRAIN_DIR, "humans")

# Directory with validation horse pictures
validation_horse_dir = os.path.join(VAL_DIR, "horses")

# Directory with validation human pictures
validation_human_dir = os.path.join(VAL_DIR, "humans")


def create_model():
    """Builds a CNN for image binary classification"""

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(300, 300, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


BATCH_SIZE = 32
IMAGE_SIZE = (300, 300)
LABEL_MODE = "binary"

# Instantiate the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode=LABEL_MODE
)

# Instantiate the validation set
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode=LABEL_MODE
)

# Optimize the datasets for training
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = (
    train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
)

validation_dataset_final = validation_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE)

FILL_MODE = "nearest"

# Create the augmentation model.
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(300, 300, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2, fill_mode=FILL_MODE),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode=FILL_MODE),
        tf.keras.layers.RandomZoom(0.2, fill_mode=FILL_MODE),
    ]
)

# Instantiate the base model
model_without_aug = create_model()

# Prepend the data augmentation layers to the base model
model_with_aug = tf.keras.models.Sequential([data_augmentation, model_without_aug])

# Compile the model
model_with_aug.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=["accuracy"],
)


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= 0.8 and logs.get("val_accuracy") >= 0.8:
            self.model.stop_training = True
            print(
                "\nReached 80% train accuracy and 80% validation accuracy, so cancelling training!"
            )


# Constant for epochs

# Train the model
history = model_with_aug.fit(
    train_dataset_final,
    epochs=20,
    verbose=2,
    validation_data=validation_dataset_final,
    callbacks=[EarlyStoppingCallback()],
)
