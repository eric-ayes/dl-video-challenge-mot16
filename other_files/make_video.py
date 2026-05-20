import csv
import os

import cv2
import matplotlib.pyplot as plt
import torch
from tracker.data_track import MOT16Sequences

# -----------------------------
# Config
# -----------------------------
working_dir = "/Users/marcelhofmann/Deep_Learning_UAM/Deep_Learning_for_Video_Signal_Processing/MOT_Challenge/dlvsp_challenge_material"  # adjust
data_dir = os.path.join(working_dir, "data/MOT16")
seq_name = "MOT16-09"                  # sequence to visualize
results_file = os.path.join(working_dir, "output/MOT16-09_best.txt")
output_video = os.path.join(working_dir, "MOT16-09_best.mp4")
fps = 30  # use dataset fps if known

# -----------------------------
# Load results
# -----------------------------
tracks = {}  # frame_idx -> list of (track_id, x1, y1, x2, y2)

with open(results_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        frame_id = int(row[0]) - 1
        track_id = int(row[1])
        x1 = float(row[2]) - 1
        y1 = float(row[3]) - 1
        w = float(row[4])
        h = float(row[5])
        x2 = x1 + w
        y2 = y1 + h

        if frame_id not in tracks:
            tracks[frame_id] = []
        tracks[frame_id].append((track_id, x1, y1, x2, y2))

# -----------------------------
# Load MOT16 sequence
# -----------------------------
sequences = MOT16Sequences(seq_name, data_dir, load_seg=False)
seq = sequences[0]  # only one sequence

# Get video dimensions from first frame
first_frame = next(iter(seq))
img = first_frame['img'].mul(255).permute(1, 2, 0).byte().numpy()
height, width, _ = img.shape

# -----------------------------
# Setup OpenCV video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# -----------------------------
# Draw boxes and save video
# -----------------------------
for i, frame in enumerate(seq):
    img = frame['img'].mul(255).permute(1, 2, 0).byte().numpy()

    # Convert RGB (matplotlib) to BGR (OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if i in tracks:
        for track_id, x1, y1, x2, y2 in tracks[i]:
            color = (
                int(track_id * 37 % 255),
                int(track_id * 17 % 255),
                int(track_id * 29 % 255)
            )
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                img, str(track_id), (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    out.write(img)

out.release()
print(f"Saved tracked video to {output_video}")