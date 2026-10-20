import io

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.emotion_detector import EmotionDetector
from src.music_recommender import FEATURES, MusicRecommender

st.set_page_config(page_title="Moodify — Emotion Music Recommender", page_icon="🎵", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stApp { background: #121212; }
    .block-container { padding-top: 90px; padding-bottom: 70px; }
    #MainMenu, footer, header { visibility: hidden; }
    .navbar {
        background: #000000; padding: 18px 40px; display: flex; align-items: center;
        justify-content: space-between; position: fixed; top: 0; left: 0; right: 0;
        z-index: 999; border-bottom: 1px solid #282828;
    }
    .navbar-brand { color: #1DB954; font-size: 1.8rem; font-weight: 900; letter-spacing: -1px; }
    .navbar-sub { color: #b3b3b3; font-size: 0.85rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; }
    .section-title { color: #ffffff; font-size: 1.5rem; font-weight: 700; margin: 10px 0 15px; }
    .emotion-container {
        background: linear-gradient(135deg, #1DB954 0%, #158a3e 100%);
        border-radius: 20px; padding: 25px; text-align: center; margin-bottom: 15px;
    }
    .emotion-emoji { font-size: 3.5rem; }
    .emotion-label { color: #ffffff; font-size: 2.2rem; font-weight: 900; text-transform: uppercase; }
    .emotion-sublabel { color: rgba(255,255,255,0.8); font-size: 0.85rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; }
    .mood-message {
        background: #181818; border-left: 4px solid #1DB954; border-radius: 0 10px 10px 0;
        padding: 12px 18px; color: #b3b3b3; font-size: 0.95rem; margin-bottom: 15px;
    }
    .footer {
        position: fixed; bottom: 0; left: 0; right: 0; background: #181818; border-top: 1px solid #282828;
        padding: 10px; text-align: center; z-index: 999; color: #b3b3b3; font-size: 0.8rem; letter-spacing: 1px;
    }
    .footer span { color: #1DB954; font-weight: 700; }
</style>
<div class="navbar">
    <div class="navbar-brand">🎵 Moodify</div>
    <div class="navbar-sub">AI Emotion · Music Recommender</div>
</div>
<div class="footer">POWERED BY <span>DeepFace · scikit-learn KNN · Streamlit</span></div>
""", unsafe_allow_html=True)

EMOJIS = {"happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐",
          "surprise": "😮", "fear": "😨", "disgust": "🤢"}


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Created once per server, not on every click."""
    return EmotionDetector(), MusicRecommender()


detector, recommender = load_models()


@st.cache_data(show_spinner="Reading your expression...")
def analyze(image_bytes):
    """Photo bytes -> (emotion, scores, photo with face box).

    Cached on the photo, so flipping the toggle doesn't run DeepFace again.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # .convert also fixes PNGs with transparency
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # OpenCV/DeepFace expect BGR
    emotion, scores, region = detector.detect(image_bgr)
    if emotion:
        image_bgr = detector.draw_box(image_bgr, region)
    return emotion, scores, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-title">📸 Detect Your Mood</div>', unsafe_allow_html=True)
    mode = st.radio("Input", ["📷 Take a photo", "🖼️ Upload a photo"], horizontal=True, label_visibility="collapsed")
    if mode == "📷 Take a photo":
        photo = st.camera_input("Take a photo", label_visibility="collapsed")  # browser webcam: works when deployed
    else:
        photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    emotion, scores = None, {}
    if photo is not None:
        emotion, scores, shown = analyze(photo.getvalue())
        if emotion:
            st.image(shown, caption=f"Detected: {emotion.upper()}", width=360)
            st.markdown('<div class="section-title">📊 Emotion Analysis</div>', unsafe_allow_html=True)
            st.bar_chart(pd.Series(scores, name="confidence (%)"), color="#1DB954")
        else:
            st.warning("No face detected 🙈 Face the camera in good light and try again.")

with right:
    st.markdown('<div class="section-title">🎧 Your Playlist</div>', unsafe_allow_html=True)

    if emotion:
        st.markdown(f"""
        <div class="emotion-container">
            <div class="emotion-emoji">{EMOJIS.get(emotion, '🎵')}</div>
            <div class="emotion-label">{emotion}</div>
            <div class="emotion-sublabel">Current Mood Detected</div>
        </div>
        <div class="mood-message">💬 {recommender.message(emotion)}</div>
        """, unsafe_allow_html=True)

        lift = st.toggle("🌤️ Lift my mood",
                         help="Off: songs that match your mood. On: the mood point moves halfway towards happy.")
        target, picks = recommender.recommend(scores, n=5, lift=lift)
        st.caption(f"Your mood point: valence {target[0]:.2f} · energy {target[1]:.2f} "
                   f"→ the 5 nearest songs (KNN)")

        for track_id in picks.track_id:
            components.iframe(f"https://open.spotify.com/embed/track/{track_id}?theme=0", height=80)

        st.markdown('<div class="section-title">🗺️ Mood Map</div>', unsafe_allow_html=True)
        columns = ["song_name", "artist"] + FEATURES
        you = pd.DataFrame([{"song_name": "Your mood", "artist": "", "valence": target[0], "energy": target[1]}])
        mood_map = pd.concat([  # later rows are drawn on top
            recommender.songs[columns].assign(group="All songs"),
            picks[columns].assign(group="Your 5 picks"),
            you.assign(group="Your mood point"),
        ])
        groups = ["All songs", "Your 5 picks", "Your mood point"]
        chart = alt.Chart(mood_map, height=320).mark_circle(opacity=0.9).encode(
            x=alt.X("valence", scale=alt.Scale(domain=[0, 1]), title="valence (how positive)"),
            y=alt.Y("energy", scale=alt.Scale(domain=[0, 1]), title="energy (how intense)"),
            color=alt.Color("group", scale=alt.Scale(domain=groups, range=["#535353", "#1DB954", "#FFFFFF"]),
                            legend=alt.Legend(title=None, orient="bottom")),
            size=alt.Size("group", scale=alt.Scale(domain=groups, range=[20, 120, 300]), legend=None),
            tooltip=["song_name", "artist", "valence", "energy"],
        )
        st.altair_chart(chart)
        st.caption("KNN picks the 5 songs closest to your mood point. Hover a dot to see the song.")
    else:
        st.markdown("""
        <div style="text-align:center; margin-top:80px;">
            <div style="font-size:5rem;">🎵</div>
            <div style="color:#ffffff; font-size:1.5rem; font-weight:700; margin-top:20px;">Detect your mood</div>
            <div style="color:#b3b3b3; font-size:0.95rem; margin-top:10px;">
                Take or upload a photo<br>and let AI find your perfect playlist
            </div>
        </div>
        """, unsafe_allow_html=True)
