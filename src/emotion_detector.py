import cv2
from deepface import DeepFace
import numpy as np

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
            
            if result and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                emotion_scores = result[0]['emotion']
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                return emotion, emotion_scores
            else:
                return None, {}
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return None, {}
    
    def process_frame(self, frame):
        emotion, scores = self.detect_emotion(frame)
        
        if emotion:
            return frame, emotion, scores
        else:
            return frame, None, {}