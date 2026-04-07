import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Normalize landmarks
# -------------------------------
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])

    # shift to origin (wrist)
    pts -= pts[0]

    # scale normalize
    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts /= max_val

    return pts.flatten()

# -------------------------------
# MediaPipe setup
# -------------------------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = "../own_dataset"

data = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".jpg"):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            print(f"Processing {label}/{file}")

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:

                    row = normalize_landmarks(hand_landmarks)
                    row = list(row)
                    row.append(label)

                    data.append(row)

# Save dataset
df = pd.DataFrame(data)
df.to_csv("../data/final_dataset.csv", index=False, header=False)

print("Dataset created!")