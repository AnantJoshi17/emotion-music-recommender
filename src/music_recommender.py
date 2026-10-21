"""Recommend songs with KNN in a 2-D "mood space": valence x energy."""
from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
FEATURES = ["valence", "energy"]  # Spotify audio features, both on a 0-1 scale

# Where each emotion sits on the valence-arousal plane (Russell's circumplex model).
# valence = how positive the music sounds, energy = how intense / active it is.
EMOTION_POINTS = {
    "happy":    (0.85, 0.75),
    "surprise": (0.65, 0.85),
    "angry":    (0.30, 0.90),
    "fear":     (0.20, 0.65),
    "disgust":  (0.25, 0.55),
    "sad":      (0.15, 0.30),
    "neutral":  (0.50, 0.50),
}

MESSAGES = {
    "happy": "You're glowing! 🌟 Here's your vibe playlist!",
    "sad": "Sending you a warm hug 🤗 Music heals everything!",
    "angry": "Let it out! 🔥 These tracks get it!",
    "neutral": "Chill mode on 😌 Perfect background tracks!",
    "surprise": "Woah! 😮 Match your energy with these!",
    "fear": "You're safe 💙 Let this calm you down!",
    "disgust": "Shake it off 😤 Music will fix the mood!",
}


class MusicRecommender:
    def __init__(self, csv_path=DATA_PATH):
        self.songs = pd.read_csv(csv_path)
        # Both features are already 0-1, so no scaling is needed before KNN (Euclidean distance).
        self.knn = NearestNeighbors(n_neighbors=5).fit(self.songs[FEATURES])

    def target_point(self, scores, lift=False):
        """Turn the 7 emotion scores into one (valence, energy) point.

        It is a weighted average of the emotion points, so a face that is 70% happy and
        30% sad lands between "happy" and "sad". Lift mode moves the point halfway to "happy".
        """
        total = sum(scores.get(e, 0) for e in EMOTION_POINTS) or 1
        valence = sum(EMOTION_POINTS[e][0] * scores.get(e, 0) for e in EMOTION_POINTS) / total
        energy = sum(EMOTION_POINTS[e][1] * scores.get(e, 0) for e in EMOTION_POINTS) / total
        if lift:
            happy_valence, happy_energy = EMOTION_POINTS["happy"]
            valence, energy = (valence + happy_valence) / 2, (energy + happy_energy) / 2
        return round(valence, 3), round(energy, 3)

    def recommend(self, scores, n=5, lift=False):
        """Return the mood point and the n songs closest to it."""
        target = self.target_point(scores, lift)
        distances, indices = self.knn.kneighbors(pd.DataFrame([target], columns=FEATURES), n_neighbors=n)
        picks = self.songs.iloc[indices[0]].copy()
        picks["distance"] = distances[0].round(3)
        return target, picks

    @staticmethod
    def message(emotion):
        return MESSAGES.get(emotion, "Here's your playlist! 🎵")
