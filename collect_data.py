"""
STEP 1 — DATA COLLECTOR  (MediaPipe Tasks API compatible)
==========================================================
Controls:
  Press a letter key (a-z) → records that sign
  Press 'q'               → quit

Saves samples to: data/landmarks.csv
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
import numpy as np
import csv
import os
import time
import urllib.request

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_PATH = os.path.join(DATA_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SAMPLES_PER_KEY = 100
COUNTDOWN_SEC = 2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]


def download_model():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def extract_landmarks(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    pts -= pts[0]
    scale = np.max(np.abs(pts)) or 1.0
    pts /= scale
    return pts.flatten().tolist()


def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + \
                [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]
            writer.writerow(header)


def append_sample(path, label, features):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([label] + features)


def draw_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 80), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 150, 60), 1)


def draw_ui(frame, status, label, collected, total):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "ASL Data Collector", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)
    cv2.putText(frame, "Press a letter key to record  |  Q = quit",
                (10, 56), cv2.FONT_HERSHEY_PLAIN, 1.1, (160, 160, 160), 1)
    color = (50, 220, 80) if status == "RECORDING" else (30, 160, 255)
    cv2.rectangle(frame, (0, h-75), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, status, (10, h-45),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    if label:
        cv2.putText(frame, f"Sign: '{label.upper()}'  [{collected}/{total}]",
                    (10, h-12), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 1)


def main():
    download_model()
    ensure_csv(CSV_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    recording, current_label, collected, countdown_end = False, "", 0, 0

    print(f"Collector ready. Data -> {CSV_PATH}")
    print("Press any letter (a-z) to record that sign.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks[0], w, h)

        now = time.time()

        if recording and now < countdown_end:
            secs = int(countdown_end - now) + 1
            cv2.putText(frame, str(secs), (w//2 - 40, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 5, (0, 200, 255), 7)
            draw_ui(frame, "GET READY...", current_label,
                    collected, SAMPLES_PER_KEY)

        elif recording:
            if result.hand_landmarks and collected < SAMPLES_PER_KEY:
                feats = extract_landmarks(result.hand_landmarks[0])
                append_sample(CSV_PATH, current_label, feats)
                collected += 1
            if collected >= SAMPLES_PER_KEY:
                print(f"  '{current_label}' — {collected} samples saved.")
                recording, current_label, collected = False, "", 0
            draw_ui(frame, "RECORDING", current_label,
                    collected, SAMPLES_PER_KEY)
        else:
            draw_ui(frame, "IDLE — press a letter", "", 0, 0)

        cv2.imshow("ASL Data Collector", frame)
        cv2.setWindowProperty("ASL Data Collector", cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key != 255 and key != 0xFF and not recording:
            try:
                ch = chr(key)
                if ch.isalpha():
                    current_label = ch.lower()
                    recording = True
                    collected = 0
                    countdown_end = time.time() + COUNTDOWN_SEC
                    print(
                        f"  Recording '{current_label}' in {COUNTDOWN_SEC}s...")
            except ValueError:
                pass

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\nDone! Data saved to {CSV_PATH}")
    print("Next -> run  train_model.py")


if __name__ == "__main__":
    main()
