# 🎵 Moodify — AI Emotion Music Recommender

Take or upload a photo → Moodify reads your facial expression and recommends 5 songs that fit your mood.

🔗 **Live Demo:** [moodify-app.streamlit.app](https://moodify-app.streamlit.app)

---

## 🚀 Features

- Detects 7 facial emotions (happy, sad, angry, neutral, surprise, fear, disgust) with DeepFace
- Webcam snapshot or photo upload — works locally and on Streamlit Cloud
- **KNN recommendations in a "mood space"** — songs are matched on Spotify audio features (valence, energy), not hand-written labels
- **Lift my mood** toggle — match your mood, or nudge it towards happier songs
- Spotify players embedded right in the app (no API key needed)
- Mood map that shows exactly why each song was picked

---

## 📊 How It Works

```
Photo ──► DeepFace ──────────────► 7 emotion scores ──► mood point ──► KNN ──► 5 nearest songs
          (OpenCV face detection    (e.g. happy 80%,     (valence,              (Spotify embeds)
           + emotion CNN)            neutral 15%, ...)     energy)
```

1. **Face + emotion** — DeepFace finds the face (OpenCV Haar cascade), crops it to 48×48 grayscale and runs a CNN
   trained on FER-2013. It returns a confidence score for each of the 7 emotions. No face → the app says so instead of guessing.
2. **Mood point** — every emotion has a fixed spot on the valence–energy plane (based on Russell's circumplex model of affect):

   | Emotion | Valence (positive?) | Energy (intense?) |
   |---|---|---|
   | happy | 0.85 | 0.75 |
   | surprise | 0.65 | 0.85 |
   | angry | 0.30 | 0.90 |
   | fear | 0.20 | 0.65 |
   | disgust | 0.25 | 0.55 |
   | sad | 0.15 | 0.30 |
   | neutral | 0.50 | 0.50 |

   The mood point is the **weighted average** of these spots, using the 7 scores as weights — so a face that is
   70% happy and 30% sad lands between the two. **Lift my mood** moves the point halfway towards "happy".
3. **KNN** — scikit-learn `NearestNeighbors` finds the 5 songs closest to the mood point (Euclidean distance on
   valence and energy, both already on a 0–1 scale, so no scaling is needed).

---

## 🎧 Song Data

`data/songs.csv` has **477 popular songs** with Spotify's `valence` and `energy` audio features, taken from the
[TidyTuesday Spotify Songs dataset](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-01-21)
(collected from the Spotify API in Jan 2020). To cover every mood, the valence × energy plane was split into a
5×5 grid and the 20 most popular songs in each cell were kept (remixes and live versions removed).

Spotify no longer gives audio features to new API apps (since Nov 2024), which is why the features come from this dataset.

---

## 🎯 Accuracy

Measured with `evaluate.py` on the **full FER-2013 test set (7,178 images)**:

| Emotion | Precision | Recall | Images |
|---|---|---|---|
| happy | 0.77 | 0.77 | 1774 |
| surprise | 0.77 | 0.71 | 831 |
| neutral | 0.46 | 0.56 | 1233 |
| disgust | 0.64 | 0.44 | 111 |
| angry | 0.50 | 0.43 | 958 |
| fear | 0.40 | 0.42 | 1024 |
| sad | 0.43 | 0.41 | 1247 |
| **Overall accuracy** | | **56.4%** | 7178 |

FER-2013 is a hard dataset (tiny 48×48 grayscale images, noisy labels); human accuracy on it is estimated at
about 65%. The model is strongest on happy and surprise. The most common mistakes are sad ↔ neutral and
fear ↔ sad, which the confusion matrix from `evaluate.py` shows. Because the app blends all 7 scores into one
mood point, a wrong top label still gives sensible songs when the scores are close.

To reproduce: download [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013), unzip it, then run
`python evaluate.py fer2013/test`.

---

## 🛠️ Tech Stack

- **DeepFace** — pre-trained emotion CNN (transfer learning, no training needed) + OpenCV face detection
- **scikit-learn** — `NearestNeighbors` (KNN) for recommendations
- **Streamlit** — UI and deployment (Streamlit Community Cloud)
- **Pandas, NumPy, OpenCV, Pillow** — data and image handling

---

## 📁 Project Structure

```
app.py                    Streamlit UI
src/emotion_detector.py   DeepFace wrapper: emotion, scores, face box
src/music_recommender.py  Mood point + KNN over valence/energy
data/songs.csv            477 songs with valence and energy
data/build_songs.py       Script that built songs.csv from the public dataset
evaluate.py               Accuracy test on FER-2013
```

---

## 💻 Local Setup

```bash
git clone https://github.com/AnantJoshi17/emotion-music-recommender.git
cd emotion-music-recommender
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. The first run downloads the DeepFace emotion model (~6 MB).

---

## ⚠️ Limitations

- Facial expression ≠ how you feel inside; FER models struggle most with fear and disgust.
- Songs are from a 2020 dataset, so very recent releases are missing.
- The emotion → mood-point mapping is a hand-set design choice, not learned from user data.

---

## 📧 Contact

**Anant Joshi**
[LinkedIn](https://linkedin.com/in/anantjoshi17) | [GitHub](https://github.com/AnantJoshi17) | [LeetCode](https://leetcode.com/u/anantjoshi17)
