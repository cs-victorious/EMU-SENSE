import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
from PIL import Image

model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# Load Animation
animation_symbol = "🎵"

st.markdown(
    f"""
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
    <div class="music">{animation_symbol}</div>
  
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.write("")

with col2:
    image = Image.open('logo.png')

    st.image(image, caption='The AI that turns on the music with your emotions')

with col3:
    st.write("")

hide_menu_style = """
    <style>
    MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    <style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not (emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")  # convert frame into array
        frm = cv2.flip(frm, 1)  # mirror image
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))  # detect landmarks on face and hand
        lst = []

        ############################################################################################
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (0, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")
        #############################################################################################


st.sidebar.header("Options")
lang = st.sidebar.text_input("Language")
singer = st.sidebar.text_input("singer")

if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

btn = st.sidebar.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
