import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from src.emotion_detector import EmotionDetector
from src.music_recommender import MusicRecommender

st.set_page_config(
    page_title="Moodify — Emotion Music Recommender",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;900&display=swap');
    * { font-family: 'Montserrat', sans-serif; }
    .stApp { background: #121212 !important; }
    .navbar {
        background: #000000;
        padding: 18px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 999;
        border-bottom: 1px solid #282828;
    }
    .navbar-brand { color: #1DB954; font-size: 1.8rem; font-weight: 900; letter-spacing: -1px; }
    .navbar-sub { color: #b3b3b3; font-size: 0.85rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; }
    .main-content { margin-top: 80px; padding: 30px 20px; }
    .section-title { color: #ffffff; font-size: 1.5rem; font-weight: 700; margin-bottom: 20px; }
    .emotion-container {
        background: linear-gradient(135deg, #1DB954 0%, #158a3e 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(29,185,84,0.4); }
        70% { box-shadow: 0 0 0 20px rgba(29,185,84,0); }
        100% { box-shadow: 0 0 0 0 rgba(29,185,84,0); }
    }
    .emotion-emoji { font-size: 4rem; margin-bottom: 10px; }
    .emotion-label { color: #ffffff; font-size: 2.5rem; font-weight: 900; text-transform: uppercase; }
    .emotion-sublabel { color: rgba(255,255,255,0.8); font-size: 0.9rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-top: 5px; }
    .song-card {
        background: #181818;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    .song-card:hover { background: #282828; border-color: #1DB954; transform: translateX(5px); }
    .song-number { color: #b3b3b3; font-size: 1rem; font-weight: 600; width: 30px; }
    .song-info { flex: 1; margin-left: 15px; }
    .song-name { color: #ffffff; font-size: 1rem; font-weight: 700; }
    .song-artist { color: #b3b3b3; font-size: 0.85rem; margin-top: 3px; }
    .spotify-btn {
        background: #1DB954;
        color: white !important;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-decoration: none !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .mood-message { background: #181818; border-left: 4px solid #1DB954; border-radius: 0 10px 10px 0; padding: 15px 20px; color: #b3b3b3; font-size: 0.95rem; margin-bottom: 20px; }
    .camera-container { background: #181818; border-radius: 15px; padding: 20px; border: 1px solid #282828; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .now-playing {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: #181818;
        border-top: 1px solid #282828;
        padding: 12px 30px;
        text-align: center;
        z-index: 999;
    }
    .now-playing-text { color: #b3b3b3; font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase; }
    .now-playing-name { color: #1DB954; font-weight: 700; margin-left: 8px; }
</style>

<div class="navbar">
    <div class="navbar-brand">🎵 Moodify</div>
    <div class="navbar-sub">AI Emotion · Music Recommender</div>
</div>

<div class="now-playing">
    <span class="now-playing-text">Powered by</span>
    <span class="now-playing-name">DeepFace · OpenCV · Streamlit</span>
</div>
""", unsafe_allow_html=True)

detector = EmotionDetector()
recommender = MusicRecommender()

st.markdown('<div class="main-content">', unsafe_allow_html=True)

emotion_emojis = {
    'happy': '😊', 'sad': '😢', 'angry': '😠',
    'neutral': '😐', 'surprise': '😮', 'fear': '😨', 'disgust': '🤢'
}

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown('<div class="section-title">📸 Detect Your Mood</div>', unsafe_allow_html=True)
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)

    mode = st.radio("Choose input", ["📷 Live Webcam", "🖼️ Upload Photo"], horizontal=True, label_visibility="collapsed")

    emotion = None
    scores = {}

    if mode == "📷 Live Webcam":
        run = st.checkbox("🎥 Start Camera")
        FRAME_WINDOW = st.image([])

        if run:
            cap = cv2.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera not found!")
                    break
                processed_frame, detected_emotion, emotion_scores = detector.process_frame(frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_frame)
                emotion = detected_emotion
                scores = emotion_scores
                time.sleep(0.03)
            cap.release()

    else:
        uploaded = st.file_uploader("Upload photo", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            processed, emotion, scores = detector.process_frame(img_bgr)
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            st.image(processed_rgb, caption=f"Detected: {emotion.upper()}", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if scores:
        st.markdown('<div class="section-title" style="margin-top:25px;">📊 Emotion Analysis</div>', unsafe_allow_html=True)
        st.bar_chart(scores, color="#1DB954")

with col2:
    st.markdown('<div class="section-title">🎧 Your Playlist</div>', unsafe_allow_html=True)

    if emotion:
        emoji = emotion_emojis.get(emotion, '🎵')
        message = recommender.get_emotion_message(emotion)
        songs = recommender.get_recommendations(emotion, n=5)

        st.markdown(f"""
        <div class="emotion-container">
            <div class="emotion-emoji">{emoji}</div>
            <div class="emotion-label">{emotion}</div>
            <div class="emotion-sublabel">Current Mood Detected</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="mood-message">💬 {message}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🎵 Top Picks For You</div>', unsafe_allow_html=True)

        for i, song in enumerate(songs):
            st.markdown(f"""
            <div class="song-card">
                <div class="song-number">{i+1}</div>
                <div class="song-info">
                    <div class="song-name">{song['song_name']}</div>
                    <div class="song-artist">{song['artist']}</div>
                </div>
                <a href="{song['spotify_url']}" target="_blank" class="spotify-btn">▶ Play</a>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; margin-top:80px;">
            <div style="font-size:5rem;">🎵</div>
            <div style="color:#ffffff; font-size:1.5rem; font-weight:700; margin-top:20px;">Detect your mood</div>
            <div style="color:#b3b3b3; font-size:0.95rem; margin-top:10px;">
                Start camera or upload a photo<br>and let AI find your perfect playlist
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)