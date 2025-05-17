import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1️⃣ Load FER2013 CSV
data = pd.read_csv('fer2013.csv')

# 2️⃣ Preprocess data
pixels = data['pixels'].tolist()
width, height = 48, 48
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split()]
    face = np.asarray(face).reshape(width, height)
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)  # shape: (n_samples, 48, 48, 1)
faces /= 255.0

# One-hot encode labels
emotions = pd.get_dummies(data['emotion']).values

# Split train/test
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.1, random_state=42)

# 3️⃣ Build CNN model
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(emotions.shape[1], activation='softmax'))

# 4️⃣ Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5️⃣ Train model
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

# 6️⃣ Save model
model.save('video_emotion_model.h5')

print('✅ Training complete! Model saved as video_emotion_model.h5')
