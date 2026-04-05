"""
STEP 3 — REAL-TIME ASL TRANSLATOR  (MediaPipe Tasks API)
=========================================================
Controls:
  SPACE  → add current letter to word
  ENTER  → add word to sentence
  BKSP   → delete last letter
  C      → clear everything
  Q      → quit
"""

import mediapipe as mp
import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
import numpy as np
import pickle
import os
import time
import urllib.request
from collections import deque

MODEL_PATH_TASK = os.path.join("data", "hand_landmarker.task")
MODEL_PATH_CLF = os.path.join("data", "asl_model.pkl")
MODEL_URL = "hand_landmarker.task"
CONFIDENCE_THRESH = 0.70
SMOOTHING_FRAMES = 10
AUTO_ADD_DELAY = 2.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]


def download_model():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(MODEL_PATH_TASK):
        print("Downloading hand landmark model (~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH_TASK)
        print("Model downloaded.")


def load_classifier(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Classifier not found at '{path}'.\n"
            "Run collect_data.py then train_model.py first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoder"]


def extract_landmarks(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    pts -= pts[0]
    scale = np.max(np.abs(pts)) or 1.0
    pts /= scale
    return pts.flatten().reshape(1, -1).astype(np.float32)


def smooth_prediction(history):
    if not history:
        return None, 0.0
    labels = [h[0] for h in history]
    best = max(set(labels), key=labels.count)
    conf = np.mean([p for l, p in history if l == best])
    return best, conf


def draw_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 80), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 150, 60), 1)


def draw_bar(frame, x, y, w, h, value, color):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*min(value, 1.0)), y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (130, 130, 130), 1)


def draw_ui(frame, sign, conf, word, sentence, auto_prog, hand_present):
    h, w = frame.shape[:2]
    FONT = cv2.FONT_HERSHEY_DUPLEX
    PLAIN = cv2.FONT_HERSHEY_PLAIN

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "ASL -> Text Translator",
                (10, 30), FONT, 0.9, (255, 255, 255), 1)
    cv2.putText(frame, "SPACE=add letter  ENTER=word  BKSP=del  C=clear  Q=quit",
                (10, 56), PLAIN, 1.0, (150, 150, 150), 1)

    # Prediction card (top-right)
    cx, cy = w-200, 78
    cv2.rectangle(frame, (cx, cy), (w-8, cy+165), (25, 25, 25), -1)
    cv2.rectangle(frame, (cx, cy), (w-8, cy+165), (70, 70, 70), 1)

    if sign and conf >= CONFIDENCE_THRESH:
        cv2.putText(frame, sign.upper(), (cx+50, cy+100),
                    FONT, 3.5, (50, 220, 80), 5)
        cv2.putText(frame, f"conf: {conf*100:.0f}%",
                    (cx+8, cy+118), PLAIN, 1.1, (150, 150, 150), 1)
        draw_bar(frame, cx+8, cy+125, 175, 13, conf, (50, 220, 80))
        if auto_prog > 0:
            cv2.putText(frame, "Hold to auto-add...",
                        (cx+8, cy+150), PLAIN, 0.9, (30, 160, 255), 1)
            draw_bar(frame, cx+8, cy+155, 175, 8, auto_prog, (30, 160, 255))
    elif not hand_present:
        cv2.putText(frame, "No hand", (cx+25, cy+85),
                    FONT, 0.8, (130, 130, 130), 1)
    else:
        cv2.putText(frame, "Low conf", (cx+18, cy+85),
                    FONT, 0.8, (60, 60, 200), 1)

    # Word bar
    cv2.rectangle(frame, (8, h-115), (w-8, h-72), (20, 20, 20), -1)
    cv2.putText(frame, "Word: " + (word if word else "_"),
                (16, h-85), FONT, 0.8, (200, 200, 0), 1)

    # Sentence bar
    cv2.rectangle(frame, (8, h-65), (w-8, h-8), (20, 20, 20), -1)
    disp = sentence if sentence else "(sentence appears here)"
    if len(disp) > 60:
        disp = "..." + disp[-57:]
    cv2.putText(frame, disp, (16, h-25), FONT, 0.75,
                (255, 255, 255) if sentence else (110, 110, 110), 1)


def main():
    download_model()
    clf, le = load_classifier(MODEL_PATH_CLF)
    print(f"Classifier loaded. Signs: {list(le.classes_)}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH_TASK)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.65,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    history = deque(maxlen=SMOOTHING_FRAMES)
    word = ""
    sentence = ""
    sign_start = None
    last_added = ""
    cur_sign = None
    cur_conf = 0.0

    print("Translator running!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)

        hand_present = bool(result.hand_landmarks)
        auto_progress = 0.0

        if hand_present:
            hl = result.hand_landmarks[0]
            draw_landmarks(frame, hl, w, h)

            feat = extract_landmarks(hl)
            probs = clf.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            prob = float(probs[idx])
            label = le.classes_[idx]

            history.append((label, prob))
            cur_sign, cur_conf = smooth_prediction(history)

            if cur_conf >= CONFIDENCE_THRESH:
                if sign_start is None or last_added != cur_sign:
                    sign_start = time.time()
                    last_added = cur_sign

                held = time.time() - sign_start
                auto_progress = min(held / AUTO_ADD_DELAY, 1.0)

                if held >= AUTO_ADD_DELAY and not last_added.endswith("_done"):
                    word += cur_sign
                    last_added = cur_sign + "_done"
                    sign_start = None
                    print(f"  Auto-added '{cur_sign}'  word='{word}'")
            else:
                sign_start = None
        else:
            history.clear()
            cur_sign, cur_conf = None, 0.0
            sign_start = None

        draw_ui(frame, cur_sign, cur_conf, word,
                sentence, auto_progress, hand_present)
        cv2.imshow("ASL Translator", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            if cur_sign and cur_conf >= CONFIDENCE_THRESH:
                word += cur_sign
                sign_start = None
                last_added = cur_sign + "_done"
                print(f"  Added '{cur_sign}'  word='{word}'")
        elif key == 13:  # ENTER
            if word:
                sentence = (sentence + " " + word).strip()
                print(f"  Word: '{word}'  sentence: '{sentence}'")
                word = ""
                last_added = ""
                sign_start = None
        elif key == 8:   # BACKSPACE
            if word:
                word = word[:-1]
                last_added = ""
            elif sentence:
                parts = sentence.rsplit(" ", 1)
                sentence = parts[0] if len(parts) > 1 else ""
        elif key == ord("c"):
            word = sentence = ""
            last_added = ""
            sign_start = None

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    if sentence:
        print(f"\nFinal text: {sentence}")


if __name__ == "__main__":
    main()
