import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to the RAVDESS dataset (change as needed)
DATA_DIR = 'audio_data'  # Replace with your path to the dataset folder

# Audio settings
SAMPLE_RATE = 22050  # Sample rate
DURATION = 3  # Duration of each audio sample in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION  # Total samples per audio
MFCC_FEATURES = 40  # Number of MFCC features

# Emotion labels for RAVDESS (make sure to map according to the dataset's emotion labels)
# You can modify these labels as per your dataset's description
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Load and preprocess audio data
def load_data(data_dir):
    features = []
    labels = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            # Extract label from the filename (first two digits are emotion)
            label_code = file_name.split('-')[2]  # Assuming the format: "01-01-01-01.wav"
            label = EMOTION_LABELS.get(label_code, 'unknown')
            
            file_path = os.path.join(data_dir, file_name)
            try:
                # Load audio file
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                if len(audio) < SAMPLES_PER_TRACK:
                    audio = np.pad(audio, (0, SAMPLES_PER_TRACK - len(audio)))
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)
                features.append(mfcc)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    return np.array(features), np.array(labels)

# Load and preprocess data
print("🚀 Loading data...")
X, y = load_data(DATA_DIR)
print(f"✅ Loaded {len(X)} samples")

# Add channel dimension → (samples, 40, time_frames, 1)
X = X[..., np.newaxis]

# Encode labels (emotion categories)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Check input shape
input_shape = X_train.shape[1:]
print(f"📏 Input shape: {input_shape}")

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("🚀 Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the trained model
model.save('emotion_detection_ravdess_model.h5')
print("✅ Model saved as emotion_detection_ravdess_model.h5")
