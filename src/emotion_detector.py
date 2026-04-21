import cv2
from deepface import DeepFace
import numpy as np

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_map = {
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'neutral': 'neutral',
            'surprise': 'surprise',
            'fear': 'fear',
            'disgust': 'disgust'
        }

    def detect_emotion(self, frame):
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            emotion = result[0]['dominant_emotion']
            scores = result[0]['emotion']
            return self.emotion_map.get(emotion, 'neutral'), scores
        except Exception as e:
            return 'neutral', {}

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion, scores = self.detect_emotion(rgb_frame)
        faces = self.face_cascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, emotion.upper(),
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2
            )
        return frame, emotion, scores