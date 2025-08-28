import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------
# Settings
# ---------------------------
dataset_path = (
    r"C:\Users\omerf\Desktop\PetImages"  # Path to folder containing 'Cat' and 'Dog'
)
categories = ["Cat", "Dog"]  # Folder names inside dataset_path
image_size = (64, 64)  # Resize images to 64*64
test_size = 0.2  # 20% of data will be test set
output_h5_file = "large_set_64.h5"

# ---------------------------
# Load and preprocess images
# ---------------------------
X = []
Y = []

for label, category in enumerate(categories):
    folder = os.path.join(dataset_path, category)
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(image_size)
                X.append(np.array(img))
                Y.append(label)
            except Exception:
                print(f"Skipped corrupted image: {img_path}")

X = np.array(X, dtype="float32") / 255.0  # normalize to [0,1]
Y = np.array(Y, dtype="int")

print(f"Total images loaded: {len(X)}")

# ---------------------------
# Shuffle and split dataset
# ---------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# ---------------------------
# Save to HDF5
# ---------------------------
with h5py.File(output_h5_file, "w") as hf:
    hf.create_dataset("X_train", data=X_train)
    hf.create_dataset("Y_train", data=Y_train)
    hf.create_dataset("X_test", data=X_test)
    hf.create_dataset("Y_test", data=Y_test)

print(f"HDF5 dataset saved to {output_h5_file}")
