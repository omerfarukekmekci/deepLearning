import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # to hide tensorflow startup info

import tensorflow as tf

# Variables optimized for best accuracy:
IMAGE_SIZE = (250, 250)
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
NUM_LAYERS = 3
NUM_CONVOS = 5
USE_AUG = 1
USE_DRPT = 0


def print_parameters(
    IMAGE_SIZE,
    BATCH_SIZE,
    LR,
    EPOCHS,
    NUM_LAYERS,
    NUM_CONVOS,
    USE_AUG,
    USE_DRPT,
    convos,
    history=None,
    log_file="CNN_results.txt",
):
    with open(log_file, "a") as f:
        f.write("=" * 45 + "\n")
        f.write("MODEL TRAINING PARAMETERS\n")
        f.write("=" * 45 + "\n")

        f.write(f"Image Size        : {IMAGE_SIZE}\n")
        f.write(f"Batch Size        : {BATCH_SIZE}\n")
        f.write(f"Learning Rate     : {LR}\n")
        f.write(f"Epochs            : {EPOCHS}\n")
        f.write(f"Number of Layers  : {NUM_LAYERS}\n")
        f.write(f"Number of Convos  : {NUM_CONVOS} {convos}\n")
        f.write(f"Use Augmentation  : {bool(USE_AUG)}\n")
        f.write(f"Use Dropout       : {bool(USE_DRPT)}\n")

        if history is not None:
            last_epoch = -1  # last epoch index
            train_loss = history.history["loss"][last_epoch]
            train_acc = history.history["accuracy"][last_epoch]
            val_loss = history.history["val_loss"][last_epoch]
            val_acc = history.history["val_accuracy"][last_epoch]

            f.write("\nLAST EPOCH PERFORMANCE\n")
            f.write("-" * 45 + "\n")
            f.write(f"Training Loss     : {train_loss:.4f}\n")
            f.write(f"Training Accuracy : {train_acc:.4f}\n")
            f.write(f"Validation Loss   : {val_loss:.4f}\n")
            f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write("=" * 45 + "\n\n\n\n")


TRAIN_DIR = "PetImages\\train"
VAL_DIR = "PetImages\\validation"
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE


def create_no_aug_model(USE_DRPT, NUM_CONVOS, NUM_LAYERS):
    model = tf.keras.Sequential()

    # Input + rescaling
    model.add(tf.keras.Input(shape=(*IMAGE_SIZE, 3)))
    model.add(tf.keras.layers.Rescaling(1.0 / 255))

    # Conv block helper function
    def conv_block(filters):
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if USE_DRPT:
            model.add(tf.keras.layers.Dropout(0.2))

    convos = [16, 32, 64, 64, 128]
    """
    if NUM_CONVOS == 5:
        convos.append(128)
    """

    for f in convos:
        conv_block(f)

    # Flatten + dense
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    if USE_DRPT:
        model.add(tf.keras.layers.Dropout(0.3))

    if NUM_LAYERS == 3:
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        if USE_DRPT:
            model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model, convos


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


(no_aug_model, convos), augmentation = (
    create_no_aug_model(USE_DRPT, NUM_CONVOS, NUM_LAYERS),
    create_augmentation(),
)


"""
if USE_AUG:
    model = tf.keras.models.Sequential([augmentation, no_aug_model])
else:
    model = tf.keras.models.Sequential([no_aug_model])
"""
model = tf.keras.models.Sequential([augmentation, no_aug_model])


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

train_dataset_final = (
    train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
)

validation_dataset_final = validation_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(LR),
    metrics=["accuracy"],
)


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= 0.999 and logs.get("val_accuracy") >= 0.999:
            self.model.stop_training = True
            print(
                "\nReached 99.9% train accuracy and 99.9% validation accuracy, so cancelling training!"
            )


history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1,
    validation_data=validation_dataset,
    callbacks=[EarlyStoppingCallback()],
)

print_parameters(
    IMAGE_SIZE=IMAGE_SIZE,
    BATCH_SIZE=BATCH_SIZE,
    LR=LR,
    EPOCHS=EPOCHS,
    NUM_LAYERS=NUM_LAYERS,
    NUM_CONVOS=NUM_CONVOS,
    USE_AUG=USE_AUG,
    convos=convos,
    USE_DRPT=USE_DRPT,
    history=history,
)
