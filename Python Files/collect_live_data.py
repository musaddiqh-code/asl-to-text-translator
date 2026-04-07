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
# Hand connections (manual)
# -------------------------------
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# -------------------------------
# MediaPipe setup (Tasks API)
# -------------------------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

label = input("Enter label (A/B/C...): ")

print("Press 's' to save sample, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape

            # 🔥 Draw points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # 🔥 Draw connections (skeleton)
            for start, end in connections:
                x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
                x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                row = normalize_landmarks(hand_landmarks)
                row = list(row)
                row.append(label)

                df = pd.DataFrame([row])
                df.to_csv("../data/live_data.csv", mode='a', header=False, index=False)

                print("✅ Saved sample")

    cv2.imshow("Collecting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Done collecting data")