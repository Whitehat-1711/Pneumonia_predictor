import tensorflow as tf
from tensorflow import keras
import os
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.preprocessing import image_dataset_from_directory
import pickle

os.makedirs('model', exist_ok=True)

# Load the dataset
train_ds = image_dataset_from_directory(
    "data/train/",
    labels="inferred",
    color_mode="grayscale",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True
)

test_ds = image_dataset_from_directory(
    "data/test/",
    labels="inferred",
    color_mode="grayscale",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True
)

def preprocess(image, label):
    """Preprocess images by normalizing pixel values."""
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# Build the model
model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=5, validation_data=test_ds)

# Save the trained model
model.save('model/trained_model.h5')

# Optional: Save training history if needed
with open('model/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
