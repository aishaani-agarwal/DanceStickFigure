# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Polygon
# from matplotlib.lines import Line2D
# from PIL import Image
# import io

# st.set_page_config(page_title="Dance Stick Figure Maker - Live Version", layout="centered")

# st.title("🎥 Live Stick Figure (Styled EXACT)")
# st.write("See your stick figure in real-time, same as your original style!")

# run = st.checkbox('✅ Turn ON Webcam', value=False)

# FRAME_WINDOW = st.image([])
# STICK_WINDOW = st.image([])

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# POSE_PAIRS = [
#     (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
#     (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
#     (23, 25), (25, 27), (27, 29), (29, 31),
#     (24, 26), (26, 28), (28, 30), (30, 32),
#     (11, 12), (23, 24), (11, 23), (12, 24), (0, 11), (0, 12)
# ]

# # 🟢 Fixed output size: same as your pipeline
# OUTPUT_W = 1280
# OUTPUT_H = 720

# camera = cv2.VideoCapture(0)

# prev_points = None
# SMOOTHING_ALPHA = 0.4

# while run:
#     ret, frame = camera.read()
#     if not ret:
#         st.warning("⚠️ Failed to grab frame.")
#         break

#     frame = cv2.flip(frame, 1)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame_rgb)

#     # Resize to match pipeline scale
#     image_rgb = frame_rgb  # same for pose detection
#     results = pose.process(image_rgb)

#     fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
#     ax.set_xlim(0, OUTPUT_W)
#     ax.set_ylim(0, OUTPUT_H)
#     ax.invert_yaxis()
#     ax.axis('off')

#     points = []
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         for lm in landmarks:
#             cx, cy = lm.x * OUTPUT_W, lm.y * OUTPUT_H
#             points.append((cx, cy))

#         if prev_points:
#             smoothed_points = []
#             for p_curr, p_prev in zip(points, prev_points):
#                 x = SMOOTHING_ALPHA * p_curr[0] + (1 - SMOOTHING_ALPHA) * p_prev[0]
#                 y = SMOOTHING_ALPHA * p_curr[1] + (1 - SMOOTHING_ALPHA) * p_prev[1]
#                 smoothed_points.append((x, y))
#             points = smoothed_points

#         prev_points = points.copy()

#         if len(points) > 24:
#             l_shoulder = points[11]
#             r_shoulder = points[12]
#             l_hip = points[23]
#             r_hip = points[24]

#             torso = Polygon([l_shoulder, r_shoulder, r_hip, l_hip],
#                             closed=True, color='orange', alpha=0.5)
#             ax.add_patch(torso)

#             mid_shoulder = (
#                 (l_shoulder[0] + r_shoulder[0]) / 2,
#                 (l_shoulder[1] + r_shoulder[1]) / 2
#             )
#             chest = Circle(mid_shoulder, radius=10, color='red')
#             ax.add_patch(chest)

#             mid_hip = (
#                 (l_hip[0] + r_hip[0]) / 2,
#                 (l_hip[1] + r_hip[1]) / 2
#             )
#             stomach = Circle(mid_hip, radius=10, color='purple')
#             ax.add_patch(stomach)

#             head = points[0]

#             # Shortened neck
#             dx = mid_shoulder[0] - head[0]
#             dy = mid_shoulder[1] - head[1]
#             NECK_SCALE = 0.5
#             neck_end = (
#                 head[0] + dx * NECK_SCALE,
#                 head[1] + dy * NECK_SCALE
#             )

#             neck = Line2D(
#                 [head[0], neck_end[0]],
#                 [head[1], neck_end[1]],
#                 linewidth=4, color='brown'
#             )
#             ax.add_line(neck)

#         for pair in POSE_PAIRS:
#             a, b = pair
#             if a < len(points) and b < len(points):
#                 ax.add_line(Line2D([points[a][0], points[b][0]],
#                                    [points[a][1], points[b][1]],
#                                    linewidth=6, color='red'))
#                 ax.add_patch(Circle(points[a], radius=5, color='black'))
#                 ax.add_patch(Circle(points[b], radius=5, color='black'))

#         head = points[0]
#         HEAD_RADIUS = OUTPUT_H * 0.04  # ~30 in 720p
#         head_circle = Circle(head, radius=HEAD_RADIUS, fill=False, color='blue', linewidth=3)
#         ax.add_patch(head_circle)
#         eye_offset_x = HEAD_RADIUS * 0.3
#         eye_offset_y = HEAD_RADIUS * 0.3
#         ax.plot(head[0] - eye_offset_x, head[1] - eye_offset_y, 'bo')
#         ax.plot(head[0] + eye_offset_x, head[1] - eye_offset_y, 'bo')

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img = Image.open(buf)
#     STICK_WINDOW.image(img)
#     plt.close(fig)

# else:
#     camera.release()
#     st.info("✅ Webcam turned off. Check the box again to restart!")

import streamlit as st
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from PIL import Image
import io

st.set_page_config(page_title="Dance Stick Figure Maker - Live Fingers 🖐️", layout="centered")

st.title("🎥 Live Stick Figure (with Fingers!)")
st.write("Turn on your webcam to see a stick figure version of yourself with fingers!")

run = st.checkbox('✅ Turn ON Webcam', value=False)

FRAME_WINDOW = st.image([])  
STICK_WINDOW = st.image([])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

POSE_PAIRS = [
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (11, 12), (23, 24), (11, 23), (12, 24), (0, 11), (0, 12)
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9,10), (10,11), (11,12), # Middle
    (0,13), (13,14), (14,15), (15,16), # Ring
    (0,17), (17,18), (18,19), (19,20)  # Pinky
]

# 🟢 Fixed output size like your pipeline
OUTPUT_W, OUTPUT_H = 1280, 720

camera = cv2.VideoCapture(0)

prev_points = None
SMOOTHING_ALPHA = 0.4

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("⚠️ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    # ✅ Convert for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

    # ✅ Use RGB for MediaPipe
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    ax.set_xlim(0, OUTPUT_W)
    ax.set_ylim(0, OUTPUT_H)
    ax.invert_yaxis()
    ax.axis('off')

    points = []
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            cx, cy = lm.x * OUTPUT_W, lm.y * OUTPUT_H
            points.append((cx, cy))

        if prev_points:
            smoothed_points = []
            for p_curr, p_prev in zip(points, prev_points):
                x = SMOOTHING_ALPHA * p_curr[0] + (1 - SMOOTHING_ALPHA) * p_prev[0]
                y = SMOOTHING_ALPHA * p_curr[1] + (1 - SMOOTHING_ALPHA) * p_prev[1]
                smoothed_points.append((x, y))
            points = smoothed_points

        prev_points = points.copy()

        if len(points) > 24:
            l_shoulder = points[11]
            r_shoulder = points[12]
            l_hip = points[23]
            r_hip = points[24]

            torso = Polygon([l_shoulder, r_shoulder, r_hip, l_hip],
                            closed=True, color='orange', alpha=0.5)
            ax.add_patch(torso)

            mid_shoulder = (
                (l_shoulder[0] + r_shoulder[0]) / 2,
                (l_shoulder[1] + r_shoulder[1]) / 2
            )
            chest = Circle(mid_shoulder, radius=10, color='red')
            ax.add_patch(chest)

            mid_hip = (
                (l_hip[0] + r_hip[0]) / 2,
                (l_hip[1] + r_hip[1]) / 2
            )
            stomach = Circle(mid_hip, radius=10, color='purple')
            ax.add_patch(stomach)

            head = points[0]
            dx = mid_shoulder[0] - head[0]
            dy = mid_shoulder[1] - head[1]
            NECK_SCALE = 0.5
            neck_end = (
                head[0] + dx * NECK_SCALE,
                head[1] + dy * NECK_SCALE
            )

            neck = Line2D(
                [head[0], neck_end[0]],
                [head[1], neck_end[1]],
                linewidth=4, color='brown'
            )
            ax.add_line(neck)

        for pair in POSE_PAIRS:
            a, b = pair
            if a < len(points) and b < len(points):
                ax.add_line(Line2D([points[a][0], points[b][0]],
                                   [points[a][1], points[b][1]],
                                   linewidth=6, color='red'))
                ax.add_patch(Circle(points[a], radius=5, color='black'))
                ax.add_patch(Circle(points[b], radius=5, color='black'))

        head = points[0]
        HEAD_RADIUS = OUTPUT_H * 0.04  # ~30 in 720p
        head_circle = Circle(head, radius=HEAD_RADIUS, fill=False, color='blue', linewidth=3)
        ax.add_patch(head_circle)
        eye_offset_x = HEAD_RADIUS * 0.3
        eye_offset_y = HEAD_RADIUS * 0.3
        ax.plot(head[0] - eye_offset_x, head[1] - eye_offset_y, 'bo')
        ax.plot(head[0] + eye_offset_x, head[1] - eye_offset_y, 'bo')

    # ✅ Draw hands (fingers!)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            hand_points = []
            for lm in hand_landmarks.landmark:
                x, y = lm.x * OUTPUT_W, lm.y * OUTPUT_H
                hand_points.append((x, y))

            for a, b in HAND_CONNECTIONS:
                ax.add_line(Line2D(
                    [hand_points[a][0], hand_points[b][0]],
                    [hand_points[a][1], hand_points[b][1]],
                    linewidth=2, color='red'
                ))
                ax.add_patch(Circle(hand_points[a], radius=3, color='black'))
                ax.add_patch(Circle(hand_points[b], radius=3, color='black'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    STICK_WINDOW.image(img)
    plt.close(fig)

else:
    camera.release()
    st.info("✅ Webcam turned off. Check the box again to restart!")
