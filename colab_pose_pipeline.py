import sys
sys.stdout.reconfigure(line_buffering=True)  # ✅ Add this at the top

import cv2
import mediapipe as mp
import json
import os
import shutil
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_pose_pipeline(input_file="input_videos/dance2.mov"):
    input_path = os.path.join(BASE_DIR, input_file)

    poses_dir = os.path.join(BASE_DIR, "poses")
    output_frames_dir = os.path.join(BASE_DIR, "output_frames")
    output_videos_dir = os.path.join(BASE_DIR, "output_videos")

    if os.path.exists(poses_dir):
        shutil.rmtree(poses_dir)
    os.makedirs(poses_dir, exist_ok=True)

    if os.path.exists(output_frames_dir):
        shutil.rmtree(output_frames_dir)
    os.makedirs(output_frames_dir, exist_ok=True)

    os.makedirs(output_videos_dir, exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15
    print(f"VIDEO_FPS:{fps}", flush=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"TOTAL_FRAMES:{total_frames}", flush=True)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"FRAME_IDX:{frame_idx}", flush=True)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])

        with open(os.path.join(poses_dir, f"frame_{frame_idx:04d}_keypoints.json"), "w") as f:
            json.dump({"people": [{"pose_keypoints_2d": keypoints}]}, f)

        frame_idx += 1

    cap.release()

    POSE_PAIRS = [
        (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (11, 12), (23, 24), (11, 23), (12, 24), (0, 11), (0, 12)
    ]

    files = sorted([f for f in os.listdir(poses_dir) if f.endswith("_keypoints.json")])

    prev_points = None
    SMOOTHING_ALPHA = 0.4

    for idx, file in enumerate(files):
        with open(os.path.join(poses_dir, file)) as f:
            data = json.load(f)
            people = data.get("people", [])
            if not people:
                continue

            keypoints = people[0]["pose_keypoints_2d"]
            points = []
            for i in range(0, len(keypoints), 3):
                x = keypoints[i] * 1280
                y = keypoints[i + 1] * 720
                v = keypoints[i + 2]
                if v > 0.1:
                    points.append((x, y))
                else:
                    points.append(None)

            if prev_points:
                smoothed_points = []
                for p_curr, p_prev in zip(points, prev_points):
                    if p_curr and p_prev:
                        x = SMOOTHING_ALPHA * p_curr[0] + (1 - SMOOTHING_ALPHA) * p_prev[0]
                        y = SMOOTHING_ALPHA * p_curr[1] + (1 - SMOOTHING_ALPHA) * p_prev[1]
                        smoothed_points.append((x, y))
                    else:
                        smoothed_points.append(p_curr)
                points = smoothed_points

            prev_points = points.copy()

            fig, ax = plt.subplots(figsize=(5, 7))
            ax.set_xlim(0, 1280)
            ax.set_ylim(0, 720)
            ax.invert_yaxis()
            ax.axis('off')

            try:
                # ✅ Torso & chest only if all points present
                l_shoulder = points[11] if len(points) > 11 else None
                r_shoulder = points[12] if len(points) > 12 else None
                l_hip = points[23] if len(points) > 23 else None
                r_hip = points[24] if len(points) > 24 else None

                if all([l_shoulder, r_shoulder, l_hip, r_hip]):
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

                    head = points[0] if len(points) > 0 else None
                    if head:
                        neck = Line2D(
                            [head[0], mid_shoulder[0]],
                            [head[1], mid_shoulder[1]],
                            linewidth=4, color='brown'
                        )
                        ax.add_line(neck)

                # ✅ Limbs safe check
                for pair in POSE_PAIRS:
                    a, b = pair
                    if a < len(points) and b < len(points):
                        if points[a] and points[b]:
                            limb = Line2D(
                                [points[a][0], points[b][0]],
                                [points[a][1], points[b][1]],
                                linewidth=6, color='red'
                            )
                            ax.add_line(limb)
                            joint_a = Circle(points[a], radius=5, color='black')
                            joint_b = Circle(points[b], radius=5, color='black')
                            ax.add_patch(joint_a)
                            ax.add_patch(joint_b)

                # ✅ Head & eyes if head exists
                head = points[0] if len(points) > 0 else None
                if head:
                    head_circle = Circle((head[0], head[1]), 30,
                                         color='blue', fill=False, linewidth=3)
                    ax.add_patch(head_circle)
                    eye_offset_x = 30 * 0.3
                    eye_offset_y = 30 * 0.3
                    ax.plot(head[0] - eye_offset_x, head[1] - eye_offset_y, 'bo')
                    ax.plot(head[0] + eye_offset_x, head[1] - eye_offset_y, 'bo')

            except Exception as e:
                print(f"⚠️ Drawing error at frame {idx}: {e}")

            plt.savefig(os.path.join(output_frames_dir, f"frame_{idx:04d}.png"))
            plt.close(fig)

    frames = sorted([f for f in os.listdir(output_frames_dir) if f.endswith(".png")])
    if not frames:
        print("⚠️ No frames found!")
        return

    frame = cv2.imread(os.path.join(output_frames_dir, frames[0]))
    height, width, _ = frame.shape

    output_path = os.path.join(output_videos_dir, 'stick_figure_tutorial.mp4')
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for file in frames:
        img = cv2.imread(os.path.join(output_frames_dir, file))
        video.write(img)

    video.release()
    print(f"✅ Stick figure video created (no audio): {output_path}")

    output_with_audio = os.path.join(output_videos_dir, 'stick_figure_with_audio.mp4')

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", output_path,
        "-i", input_path,
        "-c:v", "libx264",  # re-encode with a proper codec
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_with_audio
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"✅ Stick figure video created with audio: {output_with_audio}")
    except subprocess.CalledProcessError:
        print("⚠️ No audio found. Adding silent audio track instead...")
        ffmpeg_silent = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-i", output_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_with_audio
        ]
        subprocess.run(ffmpeg_silent, check=True)
        print(f"✅ Stick figure video created with silent audio: {output_with_audio}")

        # ✅ Cleanup
 


if __name__ == "__main__":
    run_pose_pipeline()


