import os
import subprocess
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np


# Kaggle dataset extraction
kaggle_dir = os.path.expanduser('~/.kaggle')
if not os.path.exists(kaggle_dir):
    os.mkdir(kaggle_dir)

# Move kaggle.json to ~/.kaggle (make sure you have 'kaggle.json' in the current directory)
subprocess.run(['mv', 'kaggle.json', kaggle_dir])

# Set the permissions of kaggle.json
os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

# Download the dataset from Kaggle
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'raiaone/olid-i'])

# Unzip the dataset into the 'dataset' folder
subprocess.run(['unzip', 'olid-i.zip', '-d', 'dataset'])

# Set up parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 25

# Dataset path
dataset_path = 'dataset'

# Prepare Image Data Generators for training and validation data
train_data_dir = os.path.join(dataset_path, 'train')  # Path to training data
test_data_dir = os.path.join(dataset_path, 'test')    # Path to testing data

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build the CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(57, activation='softmax'))  # Assuming you have 57 classes; adjust based on actual classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Get predictions for classification report
Y_pred = model.predict(test_generator, test_generator.samples // BATCH_SIZE + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Get ground truth labels
y_true = test_generator.classes

# Classification Report
print("Classification Report:")
target_names = list(test_generator.class_indices.keys())  # Class names
print(classification_report(y_true, y_pred, target_names=target_names))

# Save the trained model
model.save('plant_stress_classifier.h5')
