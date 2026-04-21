import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import random

class MusicRecommender:
    def __init__(self):
        self.df = pd.read_csv('data/songs.csv')
        self.emotion_weights = {
            'happy': {'happy': 1.0},
            'sad': {'sad': 1.0},
            'angry': {'angry': 1.0},
            'neutral': {'neutral': 1.0},
            'surprise': {'surprise': 1.0},
            'fear': {'fear': 1.0},
            'disgust': {'disgust': 1.0}
        }

    def get_recommendations(self, emotion, n=5):
        try:
            # Filter songs by emotion
            filtered = self.df[self.df['emotion'] == emotion]
            
            if filtered.empty:
                filtered = self.df[self.df['emotion'] == 'neutral']
            
            # Randomly sample n songs
            n = min(n, len(filtered))
            recommendations = filtered.sample(n=n)
            
            return recommendations[['song_name', 'artist', 'spotify_url']].to_dict('records')
        
        except Exception as e:
            print(f"Error: {e}")
            return []

    def get_emotion_message(self, emotion):
        messages = {
            'happy': "You're glowing! 🌟 Here's your vibe playlist!",
            'sad': "Sending you a warm hug 🤗 Music heals everything!",
            'angry': "Let it out! 🔥 These tracks get it!",
            'neutral': "Chill mode on 😌 Perfect background tracks!",
            'surprise': "Woah! 😮 Match your energy with these!",
            'fear': "You're safe 💙 Let this calm you down!",
            'disgust': "Shake it off 😤 Music will fix the mood!"
        }
        return messages.get(emotion, "Here's your playlist! 🎵")