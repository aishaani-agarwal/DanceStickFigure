#streamlit_app.py

import os
import streamlit as st
import subprocess
import time
import sys
from datetime import datetime


# ✅ Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_videos")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "stick_figure_with_audio.mp4")
MARKER_FILE = os.path.join(OUTPUT_DIR, "latest.txt")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Show Me The Moves 💃🕺", layout="centered")

st.title("💃🕺 Show Me The Moves")
st.write("Turn your dance moves into a custom stick figure animation!")

uploaded_file = st.file_uploader(
    "📹 **Upload your dance video** (MP4 or MOV)",
    type=['mp4', 'mov']
)

if uploaded_file is not None:
    input_path = os.path.join(INPUT_DIR, "dance2.mov")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Remove any old output video
    if os.path.exists(OUTPUT_VIDEO):
        os.remove(OUTPUT_VIDEO)

    st.info("✨ Generating your stick figure animation... Please wait!")

    # Progress bar + dynamic text
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Run pipeline
    process = subprocess.Popen(
        [sys.executable, "colab_pose_pipeline.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    total_frames = 0
    for line in process.stdout:
        line = line.strip()
        if "TOTAL_FRAMES:" in line:
            total_frames = int(line.split(":")[1].strip())
        if "FRAME_IDX:" in line and total_frames > 0:
            frame_idx = int(line.split(":")[1].strip())
            percent = int(min(frame_idx / total_frames, 1.0) * 100)
            progress_bar.progress(frame_idx / total_frames)
            progress_text.info(f"⏳ Processing: {percent}%")

    process.wait()
    progress_bar.progress(1.0)
    progress_text.success("✅ Processing done!")

    # ✅ Robust wait for output to appear & flush fully
    max_wait = 10  # seconds
    elapsed = 0
    while (not os.path.exists(OUTPUT_VIDEO) or os.path.getsize(OUTPUT_VIDEO) < 100_000) and elapsed < max_wait:
        time.sleep(1)
        elapsed += 1

    if process.returncode == 0 and os.path.exists(OUTPUT_VIDEO) and os.path.getsize(OUTPUT_VIDEO) > 100_000:
        st.success("✅ Stick figure video created successfully!")

        # ✅ Write freshness marker
        with open(MARKER_FILE, "w") as f:
            f.write(str(datetime.now().timestamp()))

        st.subheader("▶️ **Your Stick Figure Dance Video**")

        with open(OUTPUT_VIDEO, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            "⬇️ Download your stick figure video",
            data=video_bytes,
            file_name="stick_figure_tutorial.mp4",
            mime="video/mp4"
        )

        # ✅ Link to the Learn page!
        st.markdown("---")
        st.markdown("### 📚 Ready to learn the moves?")
        st.page_link("pages/learn.py", label="👉 Start Step-by-Step Tutorial")

    else:
        st.error("❌ Pipeline failed or output video missing/corrupted. Please check logs and try again.")

# ✅ Always show link to the Live Version too!
st.markdown("---")
st.markdown("### 🎥 Want to try it live right now?")
st.page_link("pages/live_version.py", label="🎥 Try Live Stick Figure")
st.page_link("pages/faces.py", label="🎭 Faces and Hands")









