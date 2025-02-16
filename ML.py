import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set the dataset paths
train_dir = "/path/to/dataset/kag2"   # Change this to the correct directory
val_dir = "/path/to/dataset/crop_images"  # You can use crop_images as validation

# Define image size and batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Data Augmentation & Loading
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of train data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get class labels
class_names = list(train_generator.class_indices.keys())
print(f"Classes: {class_names}")

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Output layer (5 classes)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

model.save("crop_classifier.h5")  # Save as a normal model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("crop_classifier.tflite", "wb") as f:
    f.write(tflite_model)
