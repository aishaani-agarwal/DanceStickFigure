import streamlit as st
import av
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from PIL import Image
import io
import os
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Show Me The Moves - Live Fingers 🖐️", layout="centered")

st.title("🎥 Live Stick Figure (with Fingers!)")
st.write("Turn on your webcam to see a stick figure version of yourself with fingers!")
st.info("🌐 This now works in deployed versions using WebRTC!")

# Check if running in deployed environment
def is_deployed():
    return os.getenv("STREAMLIT_SERVER_PORT") is not None or os.getenv("STREAMLIT_SHARING_MODE") is not None

# MediaPipe setup with error handling
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

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

OUTPUT_W, OUTPUT_H = 640, 480  # Reduced resolution for better performance
SMOOTHING_ALPHA = 0.4

class StickFigureProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = None
        self.hands = None
        self.prev_points = None
        self.frame_count = 0
        self.init_error = None
        
        # Initialize MediaPipe with error handling
        try:
            # Set environment variables for MediaPipe
            os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
            
            # Create temporary directory for MediaPipe models if needed
            if is_deployed():
                temp_dir = tempfile.mkdtemp()
                os.environ['TMPDIR'] = temp_dir
            
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Use lightest model
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        except Exception as e:
            self.init_error = str(e)
            st.error(f"MediaPipe initialization failed: {e}")

    def recv(self, frame):
        # Return error frame if initialization failed
        if self.init_error or not self.pose or not self.hands:
            return self.create_error_frame(frame)
        
        try:
            self.frame_count += 1
            
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Mirror the image
            
            # Resize input for faster processing
            height, width = img.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results_pose = self.pose.process(img_rgb)
            results_hands = self.hands.process(img_rgb)
            
            # Create stick figure
            stick_img = self.create_stick_figure(results_pose, results_hands, img.shape)
            
            return av.VideoFrame.from_ndarray(stick_img, format="bgr24")
            
        except Exception as e:
            # Return error frame on processing failure
            return self.create_error_frame(frame, str(e))

    def create_error_frame(self, frame, error_msg="MediaPipe Error"):
        """Create a simple error message frame"""
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        
        # Create black frame with error text
        error_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Processing Error"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        cv2.putText(error_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        cv2.putText(error_frame, "Check console for details", (text_x - 50, text_y + 40), font, 0.5, (255, 255, 255), 1)
        
        return error_frame

    def create_stick_figure(self, results_pose, results_hands, img_shape):
        # Use smaller figure size for better performance
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.set_xlim(0, OUTPUT_W)
        ax.set_ylim(0, OUTPUT_H)
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_facecolor('black')

        points = []
        if results_pose.pose_landmarks:
            for lm in results_pose.pose_landmarks.landmark:
                cx, cy = lm.x * OUTPUT_W, lm.y * OUTPUT_H
                points.append((cx, cy))

            # Smoothing
            if self.prev_points and len(self.prev_points) == len(points):
                smoothed_points = []
                for p_curr, p_prev in zip(points, self.prev_points):
                    x = SMOOTHING_ALPHA * p_curr[0] + (1 - SMOOTHING_ALPHA) * p_prev[0]
                    y = SMOOTHING_ALPHA * p_curr[1] + (1 - SMOOTHING_ALPHA) * p_prev[1]
                    smoothed_points.append((x, y))
                points = smoothed_points

            self.prev_points = points.copy()

            # Draw body parts (simplified for performance)
            if len(points) > 24:
                l_shoulder = points[11]
                r_shoulder = points[12]
                l_hip = points[23]
                r_hip = points[24]

                # Simplified torso (just lines instead of polygon)
                ax.add_line(Line2D([l_shoulder[0], r_shoulder[0]], [l_shoulder[1], r_shoulder[1]], linewidth=8, color='orange'))
                ax.add_line(Line2D([l_hip[0], r_hip[0]], [l_hip[1], r_hip[1]], linewidth=8, color='orange'))
                ax.add_line(Line2D([l_shoulder[0], l_hip[0]], [l_shoulder[1], l_hip[1]], linewidth=6, color='orange'))
                ax.add_line(Line2D([r_shoulder[0], r_hip[0]], [r_shoulder[1], r_hip[1]], linewidth=6, color='orange'))

                # Simplified chest and stomach (smaller circles)
                mid_shoulder = (
                    (l_shoulder[0] + r_shoulder[0]) / 2,
                    (l_shoulder[1] + r_shoulder[1]) / 2
                )
                chest = Circle(mid_shoulder, radius=8, color='red')
                ax.add_patch(chest)

                mid_hip = (
                    (l_hip[0] + r_hip[0]) / 2,
                    (l_hip[1] + r_hip[1]) / 2
                )
                stomach = Circle(mid_hip, radius=8, color='purple')
                ax.add_patch(stomach)

                # Neck
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

            # Draw pose connections (simplified)
            for pair in POSE_PAIRS:
                a, b = pair
                if a < len(points) and b < len(points):
                    ax.add_line(Line2D([points[a][0], points[b][0]],
                                       [points[a][1], points[b][1]],
                                       linewidth=4, color='red'))  # Reduced linewidth
                    ax.add_patch(Circle(points[a], radius=3, color='black'))  # Smaller circles
                    ax.add_patch(Circle(points[b], radius=3, color='black'))

            # Draw head (simplified)
            if len(points) > 0:
                head = points[0]
                HEAD_RADIUS = OUTPUT_H * 0.06  # Slightly larger relative to smaller resolution
                head_circle = Circle(head, radius=HEAD_RADIUS, fill=False, color='blue', linewidth=3)
                ax.add_patch(head_circle)
                eye_offset_x = HEAD_RADIUS * 0.3
                eye_offset_y = HEAD_RADIUS * 0.3
                ax.plot(head[0] - eye_offset_x, head[1] - eye_offset_y, 'bo', markersize=6)
                ax.plot(head[0] + eye_offset_x, head[1] - eye_offset_y, 'bo', markersize=6)

        # Draw hands (simplified)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                hand_points = []
                for lm in hand_landmarks.landmark:
                    x, y = lm.x * OUTPUT_W, lm.y * OUTPUT_H
                    hand_points.append((x, y))

                # Draw only main finger connections for performance
                main_connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                                  (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                                  (0, 9), (9, 10), (10, 11), (11, 12)]  # Middle only
                
                for a, b in main_connections:
                    if a < len(hand_points) and b < len(hand_points):
                        ax.add_line(Line2D(
                            [hand_points[a][0], hand_points[b][0]],
                            [hand_points[a][1], hand_points[b][1]],
                            linewidth=2, color='lime'
                        ))
                        ax.add_patch(Circle(hand_points[a], radius=2, color='yellow'))  # Smaller circles

        # Convert matplotlib to image (optimized)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black', dpi=80)  # Reduced DPI
        buf.seek(0)
        img_pil = Image.open(buf)
        img_array = np.array(img_pil)
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr

# Enhanced WebRTC Configuration for better Streamlit Cloud compatibility
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {"urls": ["stun:openrelay.metered.ca:80"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject", 
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ],
    "iceCandidatePoolSize": 20,  # Increased for better connectivity
})

# Conservative media constraints for Streamlit Cloud
MEDIA_STREAM_CONSTRAINTS = {
    "video": {
        "width": {"min": 320, "ideal": 480, "max": 640},  # Lower ideal resolution
        "height": {"min": 240, "ideal": 360, "max": 480},
        "frameRate": {"min": 5, "ideal": 10, "max": 15},  # Lower frame rate
    },
    "audio": False
}

st.markdown("### 🎥 Live Camera Feed & Stick Figure")

# Add deployment-specific warnings
if is_deployed():
    st.warning("⚠️ **Running on Streamlit Cloud**: Connection may take 30-60 seconds. Please be patient!")
    st.info("💡 **If connection fails**: Try refreshing the page or use a different browser")
else:
    st.warning("⚠️ **Network Tips**: If connection fails, try refreshing the page or check your internet connection.")

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📷 Camera Input")
    try:
        camera_stream = webrtc_streamer(
            key="stick-figure-camera",
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            async_processing=False,
        )
    except Exception as e:
        st.error(f"Camera connection failed: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser (Chrome/Firefox recommended)")

with col2:
    st.markdown("#### 🤸 Stick Figure Output")
    try:
        stick_stream = webrtc_streamer(
            key="stick-figure-live",
            video_processor_factory=StickFigureProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            async_processing=True,
        )
    except Exception as e:
        st.error(f"Stick figure processing failed: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser (Chrome/Firefox recommended)")
        
        # Show detailed error for debugging
        if is_deployed():
            st.error(f"**Debug Info**: {str(e)}")
            st.info("This may be a MediaPipe model access issue on the server.")

st.markdown("---")
st.markdown("### 🎯 Features:")
st.markdown("""
- 🔴 **Body Structure**: Red lines and joints
- 🟠 **Torso**: Orange filled polygon  
- 🟢 **Hands/Fingers**: Lime green finger tracking
- 🟡 **Joint Points**: Yellow circles on fingers
- 🔵 **Head**: Blue circle with eyes
- 🟤 **Neck**: Brown connection line
""")

st.info("💡 **Tip**: Stand back from camera for full body detection. Good lighting helps!")

# Troubleshooting section
with st.expander("🔧 **Troubleshooting Connection Issues**"):
    st.markdown("""
    **If you see "Connection is taking longer than expected":**
    
    1. **Refresh the page** - Often fixes temporary connection issues
    2. **Use Chrome or Firefox** - Better WebRTC support than Safari/Edge
    3. **Check your internet** - Stable connection required for video streaming
    4. **Allow camera permissions** - Browser must have camera access
    5. **Disable VPN/Firewall** - May block WebRTC connections
    6. **Try incognito mode** - Bypasses browser extensions that might interfere
    
    **Network Requirements:**
    - Stable internet connection (minimum 1 Mbps upload)
    - Unrestricted UDP traffic
    - Browser with WebRTC support
    
    **Still having issues?** The app uses multiple STUN/TURN servers for best compatibility.
    """)

st.markdown("---")
st.markdown("**🌐 Optimized for deployed versions** - Uses multiple relay servers for better connectivity")


