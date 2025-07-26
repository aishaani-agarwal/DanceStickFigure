import os
import json
import cv2
import matplotlib.pyplot as plt

# Paths
POSES_DIR = "poses"
OUTPUT_FRAMES_DIR = "output_frames"

# Make sure output folder exists
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Pairs of keypoints to draw the stick figure skeleton
POSE_PAIRS = [
    (1, 2), (1, 5),    # Neck to shoulders
    (2, 3), (3, 4),    # Right arm
    (5, 6), (6, 7),    # Left arm
    (1, 8), (8, 9), (9, 10),  # Spine + right leg
    (8, 12), (12, 13), (13, 14)  # Spine + left leg
]

# Load all JSON files in order
files = sorted([f for f in os.listdir(POSES_DIR) if f.endswith("_keypoints.json")])

for idx, file in enumerate(files):
    with open(os.path.join(POSES_DIR, file)) as f:
        data = json.load(f)
        people = data.get("people", [])
        if not people:
            continue

        # OpenPose gives 25 keypoints, each with x,y,confidence
        keypoints = people[0]["pose_keypoints_2d"]
        points = []
        for i in range(0, len(keypoints), 3):
            x = keypoints[i]
            y = keypoints[i + 1]
            confidence = keypoints[i + 2]
            if confidence > 0.1:
                points.append((x, y))
            else:
                points.append(None)

        # Plot stick figure
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.invert_yaxis()  # So the figure isn't upside down

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                x_values = [points[partA][0], points[partB][0]]
                y_values = [points[partA][1], points[partB][1]]
                ax.plot(x_values, y_values, 'ro-')

        ax.axis('off')
        plt.savefig(f"{OUTPUT_FRAMES_DIR}/frame_{idx:04d}.png")
        plt.close(fig)

print("✅ All stick figure frames saved!")

# Now make a video from frames
frame = cv2.imread(f"{OUTPUT_FRAMES_DIR}/frame_0000.png")
height, width, layers = frame.shape
video = cv2.VideoWriter('stick_figure_tutorial.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

for idx in range(len(files)):
    img = cv2.imread(f"{OUTPUT_FRAMES_DIR}/frame_{idx:04d}.png")
    video.write(img)

video.release()
print("✅ Stick figure tutorial video created: stick_figure_tutorial.mp4")
