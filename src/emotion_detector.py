import cv2
from deepface import DeepFace
import numpy as np

class EmotionDetector:
    def __init__(self):
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