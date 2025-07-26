
import os
import streamlit as st
import subprocess
import time

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_VIDEO = os.path.join(BASE_DIR, "../output_videos", "stick_figure_with_audio.mp4")
STEPS_DIR = os.path.join(BASE_DIR, "../output_videos/steps")
MARKER = os.path.join(BASE_DIR, "../output_videos/latest.txt")

# === Streamlit config ===
st.set_page_config(page_title="Learn Your Dance 🕺💃")

st.title("🕺💃 Learn Your Dance — Step by Step")
st.write("Practice your stick figure dance with clear steps and your original music!")

# === Check freshness ===
is_fresh = False
if os.path.exists(MARKER):
    with open(MARKER) as f:
        ts = float(f.read().strip())
        if time.time() - ts < 600:  # 10 mins window
            is_fresh = True

# === If stale, block & clean ===
if not os.path.exists(OUTPUT_VIDEO) or os.path.getsize(OUTPUT_VIDEO) < 100_000 or not is_fresh:
    if os.path.exists(STEPS_DIR):
        for f in os.listdir(STEPS_DIR):
            if f.endswith(".mp4"):
                os.remove(os.path.join(STEPS_DIR, f))
    st.info("📹 Please upload a video on the main page to generate your tutorial first.")
    st.stop()

# === Always clear old steps ===
os.makedirs(STEPS_DIR, exist_ok=True)
for f in os.listdir(STEPS_DIR):
    if f.endswith(".mp4"):
        os.remove(os.path.join(STEPS_DIR, f))

# === Split into slowed & mirrored steps WITH slowed audio ===
st.info("⏳ Splitting, slowing & mirroring your video into steps...")
split_cmd = [
    "ffmpeg",
    "-i", OUTPUT_VIDEO,
    "-filter_complex", "[0:v]hflip,setpts=2.0*PTS[v];[0:a]atempo=0.5[a]",
    "-map", "[v]",
    "-map", "[a]",
    "-f", "segment",
    "-segment_time", "5",
    os.path.join(STEPS_DIR, "step_%03d.mp4")
]

try:
    subprocess.run(split_cmd, check=True)
    st.success("✅ Steps created! Let’s start learning.")
except subprocess.CalledProcessError:
    st.error("⚠️ Failed to split video. Please check the input video and try again.")
    st.stop()

# === List steps ===
step_files = sorted([f for f in os.listdir(STEPS_DIR) if f.endswith(".mp4")])
if not step_files:
    st.warning("⚠️ No tutorial steps found. Please upload a video on the main page first.")
    st.stop()

# === Navigation ===
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

step_idx = st.session_state.current_step
step_path = os.path.join(STEPS_DIR, step_files[step_idx])

with open(step_path, "rb") as f:
    step_bytes = f.read()

st.video(step_bytes)

instructions = [
    "🟢 Step 1: Stand straight & get ready!",
    "🟢 Step 2: Raise your right arm smoothly.",
    "🟢 Step 3: Take a step back and bend knees.",
    "🟢 Step 4: Twist your hips gently.",
    "🟢 Step 5: Do a quick spin to the left.",
    "🟢 Step 6: Extend both arms out wide.",
    "🟢 Step 7: Step forward with style.",
    "🟢 Step 8: Dip down slightly.",
    "🟢 Step 9: Swing arms across the body.",
    "🟢 Step 10: Finish with a big pose!"
]

current_instruction = instructions[step_idx] if step_idx < len(instructions) else "🟢 Keep going!"
st.write(f"**Instruction:** {current_instruction}")
st.write(f"**Progress:** Step {step_idx + 1} of {len(step_files)}")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅️ Previous") and step_idx > 0:
        st.session_state.current_step -= 1
        st.rerun()

with col3:
    if st.button("Next ➡️") and step_idx < len(step_files) - 1:
        st.session_state.current_step += 1
        st.rerun()
