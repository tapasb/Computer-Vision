import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths and parameters
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SIZE = 100
BATCH_SIZE = 64  # Adjust as needed
EPOCHS = 20  # Adjust as needed
NUM_CLASSES = 33

# Create a mapping of class names to indices
class_names = sorted(os.listdir(TRAIN_DIR))
class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

# Function to load and preprocess images
def load_and_preprocess_image(filepath, label=None):
    img = Image.open(filepath).convert("RGB") # Ensure all images are RGB
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    if label is not None:
      return img_array, label
    return img_array

# Load training data
training_data = []
training_labels = []
for class_name in class_names:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg"):
            filepath = os.path.join(class_dir, filename)
            image, label = load_and_preprocess_image(filepath, class_to_index[class_name])
            training_data.append(image)
            training_labels.append(label)

training_data = np.array(training_data)
training_labels = np.array(training_labels)


# Split training data into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data, training_labels, test_size=0.2, random_state=42
)

# Data augmentation (optional but recommended)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(train_data)


# Build the model (using a simple CNN as an example)
model = keras.Sequential(
    [
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5), # Add dropout for regularization
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model (using data augmentation)
model.fit(datagen.flow(train_data, train_labels, batch_size=BATCH_SIZE), epochs=EPOCHS, validation_data=(val_data, val_labels))


# Load and preprocess test data (without labels)
test_data = []
test_filenames = sorted(os.listdir(TEST_DIR)) # Important: Sort to maintain order
for filename in test_filenames:
    if filename.endswith(".jpg"):
        filepath = os.path.join(TEST_DIR, filename)
        image = load_and_preprocess_image(filepath)
        test_data.append(image)

test_data = np.array(test_data)

# Make predictions on the test data
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Create submission file (example: CSV)
import pandas as pd
submission = pd.DataFrame({'id': [int(f[:-4]) for f in test_filenames], 'label': predicted_labels}) # Extract IDs
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")

# Save the model
model.save("fruit_classification_model.h5") # Save the model for future use
print("Model saved as fruit_classification_model.h5")