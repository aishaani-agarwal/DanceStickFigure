#faces.py

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Face + Hands Detail", layout="wide")
st.title("🧠 Advanced Face + Hands")
st.markdown("Detailed hand and face landmark rendering. No body stick figure.")

# Initialize MediaPipe Holistic with face refinement
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

run = st.checkbox("✅ Turn ON Webcam", value=False)
FRAME_WINDOW = st.image([])
STICK_WINDOW = st.image([])

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def draw_face_and_hands(image, results):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.invert_yaxis()
    ax.axis("off")

    def extract_points(landmark_list, width, height):
        if not landmark_list:
            return []
        return [(int(lm.x * width), int(lm.y * height)) for lm in landmark_list]

    face = extract_points(results.face_landmarks.landmark if results.face_landmarks else [], 1280, 720)
    left_hand = extract_points(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [], 1280, 720)
    right_hand = extract_points(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [], 1280, 720)

    def draw_lines(points, pairs, color="black"):
        for a, b in pairs:
            if a < len(points) and b < len(points):
                ax.add_line(Line2D([points[a][0], points[b][0]], [points[a][1], points[b][1]], linewidth=2, color=color))

    HAND_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4),
                  (0, 5), (5, 6), (6, 7), (7, 8),
                  (5, 9), (9, 10), (10, 11), (11, 12),
                  (9, 13), (13, 14), (14, 15), (15, 16),
                  (13, 17), (17, 18), (18, 19), (19, 20)]

    draw_lines(left_hand, HAND_PAIRS, color="green")
    draw_lines(right_hand, HAND_PAIRS, color="blue")

    for pt in left_hand + right_hand:
        ax.add_patch(Circle(pt, radius=3, color='black'))

    for pt in face:
        ax.add_patch(Circle(pt, radius=1.5, color='orange'))

   

    # 👀 Eyeball tracking using iris
    if len(face) > 475:
        left_iris = face[468]
        right_iris = face[473]
        ax.add_patch(Circle(left_iris, radius=3, color='black'))
        ax.add_patch(Circle(right_iris, radius=3, color='black'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("❌ Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    FRAME_WINDOW.image(rgb)

    if results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
        stick_img = draw_face_and_hands(frame, results)
        STICK_WINDOW.image(stick_img)
    else:
        STICK_WINDOW.empty()
else:
    cap.release()
    st.info("🛑 Webcam turned off.")









