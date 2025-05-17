import cv2
import numpy as np
import time
import threading
import librosa
import pyaudio
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load models
video_model = load_model('video_model.h5')
audio_model = load_model('audio_model.h5')

# Define emotion labels
video_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
audio_emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Label encoder for audio
le = LabelEncoder()
le.classes_ = np.array(audio_emotion_labels)

# Audio settings
SAMPLE_RATE = 22050
DURATION = 15  # match video
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
EXPECTED_TIME_FRAMES = 130
MFCC_FEATURES = 40

# Shared lists to store predictions
video_predictions = []
audio_prediction = None

# --------------------- VIDEO THREAD ---------------------
def video_thread():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    start_time = time.time()
    duration = DURATION

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            pred = video_model.predict(face)[0]
            video_predictions.append(pred)

            emotion = video_emotion_labels[np.argmax(pred)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Webcam Emotion Detection', frame)

        if time.time() - start_time > duration:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------- AUDIO THREAD ---------------------
def audio_thread():
    global audio_prediction
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)
    frames = []

    print("🎙️ Recording audio...")
    for _ in range(0, int(SAMPLE_RATE / 1024 * DURATION)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("✅ Audio recording complete.")

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))
    if len(audio_data) < SAMPLES_PER_TRACK:
        audio_data = np.pad(audio_data, (0, SAMPLES_PER_TRACK - len(audio_data)))

    mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
    if mfcc.shape[1] < EXPECTED_TIME_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, EXPECTED_TIME_FRAMES - mfcc.shape[1])), mode='constant')
    elif mfcc.shape[1] > EXPECTED_TIME_FRAMES:
        mfcc = mfcc[:, :EXPECTED_TIME_FRAMES]

    mfcc = mfcc[..., np.newaxis]
    mfcc = np.expand_dims(mfcc, axis=0)

    pred = audio_model.predict(mfcc)[0]
    audio_prediction = pred

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    vt = threading.Thread(target=video_thread)
    at = threading.Thread(target=audio_thread)

    vt.start()
    at.start()

    vt.join()
    at.join()

    print("\n⏰ 15 seconds complete!")

    # Handle video softmax
    if video_predictions:
        video_softmax = np.mean(video_predictions, axis=0)
    else:
        video_softmax = np.zeros(len(video_emotion_labels))

    # Handle audio softmax
    if audio_prediction is not None:
        audio_softmax = audio_prediction
    else:
        audio_softmax = np.zeros(len(audio_emotion_labels))

    # Resize vectors to match (basic alignment)
    min_len = min(len(video_softmax), len(audio_softmax))
    video_softmax = video_softmax[:min_len]
    audio_softmax = audio_softmax[:min_len]

    # Fuse predictions (average softmax)
    combined_softmax = (video_softmax + audio_softmax) / 2

    # Final predicted emotion
    final_emotion_idx = np.argmax(combined_softmax)
    final_emotion = video_emotion_labels[:min_len][final_emotion_idx]

    print(f"\n✅ Final Combined Emotion Prediction: {final_emotion}")
