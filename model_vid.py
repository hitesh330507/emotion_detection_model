import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Step 1: Data Preparation

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,

    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Prepare data generators for training
train_generator = train_datagen.flow_from_directory(
    'train',  # Path to training images
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

# Assuming input_shape = (48, 48, 3) for RGB images
input_shape = (48, 48, 3)
num_classes = 7

model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape)) # Using 'same' padding initially
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same')) # Second conv before pooling can sometimes help
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25)) # Optional: Dropout after conv block

# Block 2
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25)) # Optional

# Block 3
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25)) # Optional

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(256)) # Increased dense layer size
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) # EFFECTIVE Dropout
model.add(Dense(num_classes, activation='softmax')) # Softmax for output

model.summary()


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model architecture
model.summary()

# Step 3: Train the Model (without validation data)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25
)

# Step 4: Save the Model
model.save('emotion_model.h5')

# Step 5: Plot Training History (only training accuracy and loss)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
