import streamlit as st
import sounddevice as sd
import numpy as np
import soundfile as sf
import whisper
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import time

# -----------------------
# CONFIG
DURATION = 4  # seconds
FS = 44100    # sample rate

# -----------------------
# Load Whisper once
if "model" not in st.session_state:
    st.session_state.model = whisper.load_model("base")
if "action" not in st.session_state:
    st.session_state.action = None
if "trigger_time" not in st.session_state:
    st.session_state.trigger_time = None

st.title("🎙️ Voice Commands Stick Figure")
st.write("MOVES: spin, jump, wave, kick, dab, split, cartwheel, moonwalk")

# -----------------------
# Record + Save WAV
def record_audio():
    st.info("🎙️ Recording... Speak now!")
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype="float32")
    sd.wait()
    sf.write("recording.wav", recording, FS)
    st.success("✅ Done recording.")
    return "recording.wav"

# -----------------------
# Run Whisper
def recognize(path):
    result = st.session_state.model.transcribe(path)
    return result["text"].lower()

# -----------------------
# Handle command
if st.button("🎙️ Record Command"):
    audio_path = record_audio()
    text = recognize(audio_path)
    st.write(f"**You said:** `{text}`")

    if "spin" in text:
        st.session_state.action = "spin"
    elif "jump" in text:
        st.session_state.action = "jump"
    elif "wave" in text:
        st.session_state.action = "wave"
    elif "kick" in text:
        st.session_state.action = "kick"
    elif "dab" in text:
        st.session_state.action = "dab"
    elif "moonwalk" in text:
        st.session_state.action = "moonwalk"
    elif "cartwheel" in text:
        st.session_state.action = "cartwheel"
    elif "split" in text:
        st.session_state.action = "split"
    elif "headbang" in text:
        st.session_state.action = "headbang"
    else:
        st.warning("❌ Command not found. Try: spin, jump, wave, kick, dab, moonwalk, cartwheel, split, headbang.")
        st.session_state.action = None

    st.session_state.trigger_time = time.time()

# -----------------------
# Stick Figure Drawing
def draw_stick_figure(action=None, step=0):
    fig, ax = plt.subplots(figsize=(3, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.axis('off')

    # Base pose
    head = np.array([50, 130])
    neck = np.array([50, 110])
    left_shoulder = np.array([40, 105])
    right_shoulder = np.array([60, 105])
    left_hip = np.array([45, 70])
    right_hip = np.array([55, 70])
    left_hand = np.array([30, 90])
    right_hand = np.array([70, 90])
    left_knee = np.array([45, 50])
    right_knee = np.array([55, 50])
    left_foot = np.array([45, 30])
    right_foot = np.array([55, 30])

    if action == "jump":
        offset = 15 * np.sin(step / 10 * np.pi)
        for part in [head, neck, left_shoulder, right_shoulder, left_hip, right_hip,
                     left_hand, right_hand, left_knee, right_knee, left_foot, right_foot]:
            part[1] += offset

    elif action == "spin":
        angle = np.deg2rad(step * 36)
        center = (left_hip + right_hip) / 2
        def rot(p): return np.dot([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle),  np.cos(angle)]], p - center) + center
        for part in [head, neck, left_shoulder, right_shoulder, left_hip, right_hip,
                     left_hand, right_hand, left_knee, right_knee, left_foot, right_foot]:
            p_rot = rot(part)
            part[0], part[1] = p_rot[0], p_rot[1]

    elif action == "wave":
        right_hand[1] += 10 * np.sin(step / 2 * np.pi)

    elif action == "kick":
        kick_offset = 15 * np.sin(step / 2 * np.pi)
        right_knee[0] += kick_offset
        right_foot[0] += kick_offset

    elif action == "dab":
        head[0] += -5
        head[1] += -5
        left_hand[0] += -15
        left_hand[1] += 15
        right_hand[0] += 15
        right_hand[1] += -15

    elif action == "moonwalk":
        slide = 20 * (step / 20)
        tilt = np.deg2rad(-10)
        def rot(p): return np.dot([[np.cos(tilt), -np.sin(tilt)],
                                   [np.sin(tilt),  np.cos(tilt)]], p - neck) + neck
        for part in [head, neck, left_shoulder, right_shoulder, left_hip, right_hip,
                     left_hand, right_hand, left_knee, right_knee, left_foot, right_foot]:
            part[0] -= slide
            p_rot = rot(part)
            part[0], part[1] = p_rot[0], p_rot[1]

    elif action == "cartwheel":
        angle = np.deg2rad(step * 72)
        center = neck
        def rot(p): return np.dot([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle),  np.cos(angle)]], p - center) + center
        for part in [head, neck, left_shoulder, right_shoulder, left_hip, right_hip,
                     left_hand, right_hand, left_knee, right_knee, left_foot, right_foot]:
            p_rot = rot(part)
            part[0], part[1] = p_rot[0], p_rot[1]

    elif action == "split":
        left_foot[0] -= 20 * (step / 20)
        right_foot[0] += 20 * (step / 20)
        left_knee[0] -= 15 * (step / 20)
        right_knee[0] += 15 * (step / 20)

    elif action == "headbang":
        head[1] += 10 * np.sin(step / 2 * np.pi)

    # Draw
    ax.add_patch(Circle(head, radius=5, color='blue'))
    pairs = [
        (head, neck),
        (neck, left_shoulder),
        (neck, right_shoulder),
        (left_shoulder, left_hand),
        (right_shoulder, right_hand),
        (neck, left_hip),
        (neck, right_hip),
        (left_hip, left_knee),
        (left_knee, left_foot),
        (right_hip, right_knee),
        (right_knee, right_foot),
        (left_hip, right_hip)
    ]
    for a, b in pairs:
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color='black'))
    st.pyplot(fig)

# -----------------------
# Animate
if st.session_state.action:
    elapsed = time.time() - st.session_state.trigger_time
    if elapsed < 2:
        step = int((elapsed / 2) * 20)
        draw_stick_figure(st.session_state.action, step)
        st.rerun()
    else:
        st.session_state.action = None
        draw_stick_figure()
else:
    draw_stick_figure()
