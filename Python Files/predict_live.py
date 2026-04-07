import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Load trained model
# -------------------------------
with open("hand_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# MediaPipe setup
# -------------------------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -------------------------------
# Normalize landmarks (IMPORTANT)
# -------------------------------
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])

    # shift origin to wrist
    pts -= pts[0]

    # scale normalize
    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts /= max_val

    return pts.flatten()

# -------------------------------
# Smoothing (stability)
# -------------------------------
history = deque(maxlen=5)

def most_common(lst):
    return max(set(lst), key=lst.count)

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    display_text = "No hand"

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        # -------------------------------
        # Feature extraction (FIXED)
        # -------------------------------
        row = normalize_landmarks(hand_landmarks)

        # -------------------------------
        # Prediction + confidence
        # -------------------------------
        probs = model.predict_proba([row])[0]
        idx = np.argmax(probs)

        prediction = model.classes_[idx]
        confidence = probs[idx]

        # -------------------------------
        # Confidence filtering
        # -------------------------------
        if confidence > 0.7:
            history.append(prediction)

            if len(history) == 5:
                stable_pred = most_common(history)
                display_text = f"{stable_pred} ({confidence:.2f})"
        else:
            display_text = "Low confidence"

        # -------------------------------
        # Draw landmarks
        # -------------------------------
        for lm in hand_landmarks:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    # -------------------------------
    # Show prediction
    # -------------------------------
    cv2.putText(frame, f"Sign: {display_text}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()