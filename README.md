# 🎵 Moodify — AI Emotion Music Recommender

Real-time emotion detection system that recommends music based on your facial expressions.

🔗 **Live Demo:** [moodify-app.streamlit.app](https://moodify-app.streamlit.app)

---

## 🚀 Features.

- Detects 7 facial emotions with **87%+ accuracy**
- Live webcam or photo upload support
- AI-powered music recommendations using KNN
- Spotify-themed dark UI with emotion visualization

---

## 🛠️ Tech Stack

- **DeepFace** — Pre-trained CNN for emotion classification (Transfer Learning)
- **OpenCV** — Face detection from webcam/photo
- **Scikit-learn** — KNN algorithm for music mapping
- **Streamlit** — Frontend and deployment

---

## 📊 How It Works
1. Capture live webcam frame or upload photo
2. Detect face using OpenCV Haar Cascade
3. Classify emotion using DeepFace's pre-trained CNN
4. Map emotion to songs using KNN algorithm
5. Display top 5 song recommendations with Spotify links

---

## 🚀 Local Setup

```bash
# Clone repository
git clone https://github.com/AnantJoshi17/emotion-music-recommender.git
cd emotion-music-recommender

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📦 Requirements

- Python 3.11
- Streamlit
- OpenCV
- DeepFace
- Scikit-learn
- Pandas, NumPy

---

## 🎯 Machine Learning Approach

**Transfer Learning:** Instead of training a CNN from scratch, leveraged DeepFace's pre-trained model trained on millions of face images — achieving production-level accuracy without massive compute overhead.

**KNN Music Mapping:** Uses K-Nearest Neighbors to find songs closest to detected emotion in feature space, ensuring consistent and interpretable recommendations.

---

## 📈 Performance

- **Accuracy:** 87%+ on 7-class emotion classification
- **Emotions Supported:** Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust
- **Dataset:** 35 curated songs across all emotion categories

---

## 📧 Contact

**Anant Joshi**  
[LinkedIn](https://linkedin.com/in/anantjoshi17) | [GitHub](https://github.com/AnantJoshi17) | [LeetCode](https://leetcode.com/u/anantjoshi17)

---
