import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import webbrowser

# Streamlit UI Header
st.header("Emotion-Based Music Recommender with Hand Gesture Volume Control")

# Manage session state for the application
if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Input fields for language and singer
lang = st.text_input("Language")
singer = st.text_input("Singer")

# WebRTC streamer for emotion detection
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Button to trigger song recommendation
btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        search_and_play_song(lang, emotion, singer)
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
