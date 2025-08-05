# #faces.py

# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# from PIL import Image
# from matplotlib.patches import Circle
# from matplotlib.lines import Line2D
# import matplotlib.pyplot as plt
# import io

# st.set_page_config(page_title="Face + Hands Detail", layout="wide")
# st.title("🧠 Advanced Face + Hands")
# st.markdown("Detailed hand and face landmark rendering. No body stick figure.")

# # Initialize MediaPipe Holistic with face refinement
# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=1,
#     enable_segmentation=False,
#     refine_face_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# run = st.checkbox("✅ Turn ON Webcam", value=False)
# FRAME_WINDOW = st.image([])
# STICK_WINDOW = st.image([])

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# def draw_face_and_hands(image, results):
#     fig, ax = plt.subplots(figsize=(12, 7))
#     ax.set_xlim(0, 1280)
#     ax.set_ylim(0, 720)
#     ax.invert_yaxis()
#     ax.axis("off")

#     def extract_points(landmark_list, width, height):
#         if not landmark_list:
#             return []
#         return [(int(lm.x * width), int(lm.y * height)) for lm in landmark_list]

#     face = extract_points(results.face_landmarks.landmark if results.face_landmarks else [], 1280, 720)
#     left_hand = extract_points(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [], 1280, 720)
#     right_hand = extract_points(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [], 1280, 720)

#     def draw_lines(points, pairs, color="black"):
#         for a, b in pairs:
#             if a < len(points) and b < len(points):
#                 ax.add_line(Line2D([points[a][0], points[b][0]], [points[a][1], points[b][1]], linewidth=2, color=color))

#     HAND_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4),
#                   (0, 5), (5, 6), (6, 7), (7, 8),
#                   (5, 9), (9, 10), (10, 11), (11, 12),
#                   (9, 13), (13, 14), (14, 15), (15, 16),
#                   (13, 17), (17, 18), (18, 19), (19, 20)]

#     draw_lines(left_hand, HAND_PAIRS, color="green")
#     draw_lines(right_hand, HAND_PAIRS, color="blue")

#     for pt in left_hand + right_hand:
#         ax.add_patch(Circle(pt, radius=3, color='black'))

#     for pt in face:
#         ax.add_patch(Circle(pt, radius=1.5, color='orange'))

   

#     # 👀 Eyeball tracking using iris
#     if len(face) > 475:
#         left_iris = face[468]
#         right_iris = face[473]
#         ax.add_patch(Circle(left_iris, radius=3, color='black'))
#         ax.add_patch(Circle(right_iris, radius=3, color='black'))

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img = Image.open(buf)
#     plt.close(fig)
#     return img

# while run:
#     ret, frame = cap.read()
#     if not ret:
#         st.warning("❌ Could not read from webcam.")
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = holistic.process(rgb)

#     FRAME_WINDOW.image(rgb)

#     if results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
#         stick_img = draw_face_and_hands(frame, results)
#         STICK_WINDOW.image(stick_img)
#     else:
#         STICK_WINDOW.empty()
# else:
#     cap.release()
#     st.info("🛑 Webcam turned off.")





import streamlit as st
import av
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Face + Hands Detail", layout="wide")
st.title("🧠 Advanced Face + Hands")
st.markdown("Detailed hand and face landmark rendering. No body stick figure.")
st.info("🌐 This now works in deployed versions using WebRTC!")

# Initialize MediaPipe Holistic with face refinement
mp_holistic = mp.solutions.holistic

OUTPUT_W, OUTPUT_H = 1280, 720

class FaceHandsProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Holistic
        results = self.holistic.process(img_rgb)
        
        # Create face and hands visualization
        face_hands_img = self.draw_face_and_hands(results, img.shape)
        
        return av.VideoFrame.from_ndarray(face_hands_img, format="bgr24")

    def draw_face_and_hands(self, results, img_shape):
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        ax.set_xlim(0, OUTPUT_W)
        ax.set_ylim(0, OUTPUT_H)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_facecolor('black')

        def extract_points(landmark_list, width, height):
            if not landmark_list:
                return []
            return [(int(lm.x * width), int(lm.y * height)) for lm in landmark_list]

        # Extract landmarks
        face = extract_points(results.face_landmarks.landmark if results.face_landmarks else [], OUTPUT_W, OUTPUT_H)
        left_hand = extract_points(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [], OUTPUT_W, OUTPUT_H)
        right_hand = extract_points(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [], OUTPUT_W, OUTPUT_H)

        def draw_lines(points, pairs, color="white", linewidth=2):
            for a, b in pairs:
                if a < len(points) and b < len(points):
                    ax.add_line(Line2D([points[a][0], points[b][0]], [points[a][1], points[b][1]], 
                                     linewidth=linewidth, color=color))

        # Hand connections
        HAND_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                      (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                      (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                      (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                      (13, 17), (17, 18), (18, 19), (19, 20)]  # Pinky

        # Draw hands with different colors
        draw_lines(left_hand, HAND_PAIRS, color="lime", linewidth=3)
        draw_lines(right_hand, HAND_PAIRS, color="cyan", linewidth=3)

        # Draw hand landmarks
        for pt in left_hand:
            ax.add_patch(Circle(pt, radius=4, color='lime'))
        for pt in right_hand:
            ax.add_patch(Circle(pt, radius=4, color='cyan'))

        # Draw face landmarks
        for pt in face:
            ax.add_patch(Circle(pt, radius=1.5, color='orange'))

        # Enhanced eyeball tracking using iris
        if len(face) > 475:
            left_iris = face[468] if 468 < len(face) else None
            right_iris = face[473] if 473 < len(face) else None
            
            if left_iris:
                ax.add_patch(Circle(left_iris, radius=5, color='red'))
            if right_iris:
                ax.add_patch(Circle(right_iris, radius=5, color='red'))

        # Add face outline if available
        if len(face) > 10:
            # Draw face contour
            face_oval_indices = [10, 151, 9, 8, 168, 6, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            face_contour = [face[i] for i in face_oval_indices if i < len(face)]
            if len(face_contour) > 2:
                draw_lines(face_contour, [(i, i+1) for i in range(len(face_contour)-1)], color="yellow", linewidth=2)

        # Convert matplotlib to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black')
        buf.seek(0)
        img_pil = Image.open(buf)
        img_array = np.array(img_pil)
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

st.markdown("### 🎥 Live Camera Feed")
st.markdown("📹 **Your live webcam feed:**")

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📷 Camera Input")
    camera_stream = webrtc_streamer(
        key="face-hands-camera",
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

with col2:
    st.markdown("#### 🎭 Face & Hands Analysis")
    analysis_stream = webrtc_streamer(
        key="face-hands-analysis",
        video_processor_factory=FaceHandsProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.markdown("---")
st.markdown("### 📊 Features:")
st.markdown("""
- 🟢 **Right Hand**: Lime green tracking
- 🔵 **Left Hand**: Cyan blue tracking  
- 🟠 **Face Landmarks**: Orange dots
- 🔴 **Eye Iris**: Red circles for precise eye tracking
- 🟡 **Face Contour**: Yellow outline
""")

st.info("� **Tip**: Make sure your hands and face are well-lit for better detection!")













