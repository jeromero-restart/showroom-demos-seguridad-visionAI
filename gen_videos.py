import cv2
import numpy as np
import os

out_dir = "C:\\Showroom\\poc-sialar_live_cam_detection\\data\\demo_cameras"
os.makedirs(out_dir, exist_ok=True)

configs = [
    ("parking.mp4",  (255, 0, 0),   30, 45),
    ("street.mp4",   (0, 255, 0),   25, 60),
    ("building.mp4", (0, 0, 255),   30, 50),
]

for fname, color, fps, duration in configs:
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        print(f"EXISTS: {fname}")
        continue
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (1280, 720))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = color
    for _ in range(fps * duration):
        out.write(frame)
    out.release()
    print(f"CREATED: {fname}")

print("Done")
