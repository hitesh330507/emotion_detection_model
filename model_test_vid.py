import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from collections import Counter

# Load the trained model
model = load_model('video_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

# List to store predicted emotions
predicted_emotions = []

print("⚡ Starting 15-second real-time webcam emotion detection...")

# Get start time
start_time = time.time()
duration = 15  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from webcam.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        predicted_class = np.argmax(prediction)
        emotion = emotion_labels[predicted_class]
        predicted_emotions.append(emotion)

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255,0,0), 2)

    cv2.imshow('15-Second Emotion Detection', frame)

    # Check if 15 seconds are over
    if time.time() - start_time > duration:
        print("\n⏰ 15 seconds complete!")
        break

    # Optional: Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 Interrupted by user (q pressed).")
        break

cap.release()
cv2.destroyAllWindows()

# Calculate majority emotion
if predicted_emotions:
    emotion_counts = Counter(predicted_emotions)
    majority_emotion = emotion_counts.most_common(1)[0][0]
    print(f"\n✅ Majority Emotion in 15 seconds: {majority_emotion}")
else:
    print("\n⚠️ No faces detected during the session.")
