"""
ASL Translator — Web Server
============================
Streams annotated webcam feed as MJPEG and exposes a JSON state endpoint.

Run:
    pip install fastapi uvicorn opencv-python mediapipe numpy scikit-learn
    python server.py

Then open http://localhost:8000 in your browser.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
import numpy as np
import pickle
import os
import time
import threading
import urllib.request
from collections import deque

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH_TASK   = os.path.join("data", "hand_landmarker.task")
MODEL_PATH_CLF    = os.path.join("data", "asl_model.pkl")
MODEL_URL         = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CONFIDENCE_THRESH = 0.70
SMOOTHING_FRAMES  = 10
AUTO_ADD_DELAY    = 2.0

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ── Shared State (thread-safe via lock) ───────────────────────────────────────
state_lock = threading.Lock()
shared = {
    "sign":         None,
    "conf":         0.0,
    "word":         "",
    "sentence":     "",
    "auto_progress": 0.0,
    "hand_present": False,
    "last_added":   "",
    "sign_start":   None,
}

# ── Model helpers ─────────────────────────────────────────────────────────────
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
    best   = max(set(labels), key=labels.count)
    conf   = np.mean([p for l, p in history if l == best])
    return best, conf

def draw_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 90), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 180, 70), 1)

def draw_overlay(frame, sign, conf, auto_prog, hand_present):
    """Draw a minimal overlay on the video frame itself."""
    h, w = frame.shape[:2]

    # Semi-transparent top strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "ASL Translator — live feed",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1)

    # Prediction badge (top-right of the video frame)
    bx, by = w - 110, 10
    if sign and conf >= CONFIDENCE_THRESH:
        badge_col = (30, 200, 80)
        cv2.rectangle(frame, (bx, by), (w - 8, by + 50), (15, 15, 15), -1)
        cv2.rectangle(frame, (bx, by), (w - 8, by + 50), badge_col, 1)
        cv2.putText(frame, sign.upper(),
                    (bx + 18, by + 38), cv2.FONT_HERSHEY_DUPLEX, 1.6, badge_col, 3)
        # Auto-add progress bar inside badge
        if auto_prog > 0:
            bar_w = int(94 * auto_prog)
            cv2.rectangle(frame, (bx + 4, by + 46), (bx + 4 + bar_w, by + 52), (30, 160, 255), -1)
    elif not hand_present:
        cv2.rectangle(frame, (bx, by), (w - 8, by + 36), (20, 20, 20), -1)
        cv2.putText(frame, "no hand",
                    (bx + 6, by + 24), cv2.FONT_HERSHEY_PLAIN, 1.0, (100, 100, 100), 1)

# ── Video capture + inference loop (background thread) ────────────────────────
latest_frame_jpg = None
frame_lock = threading.Lock()

def capture_loop():
    global latest_frame_jpg

    download_model()
    clf, le = load_classifier(MODEL_PATH_CLF)
    print(f"Classifier loaded — signs: {list(le.classes_)}")

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
        print("ERROR: Cannot open webcam.")
        return

    history = deque(maxlen=SMOOTHING_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = detector.detect(mp_image)

        hand_present  = bool(result.hand_landmarks)
        auto_progress = 0.0
        cur_sign      = None
        cur_conf      = 0.0

        if hand_present:
            hl = result.hand_landmarks[0]
            draw_landmarks(frame, hl, w, h)

            feat  = extract_landmarks(hl)
            probs = clf.predict_proba(feat)[0]
            idx   = int(np.argmax(probs))
            prob  = float(probs[idx])
            label = le.classes_[idx]

            history.append((label, prob))
            cur_sign, cur_conf = smooth_prediction(history)

            with state_lock:
                if cur_conf >= CONFIDENCE_THRESH:
                    last = shared["last_added"]
                    if shared["sign_start"] is None or last != cur_sign:
                        shared["sign_start"] = time.time()
                        shared["last_added"] = cur_sign

                    held = time.time() - shared["sign_start"]
                    auto_progress = min(held / AUTO_ADD_DELAY, 1.0)

                    if held >= AUTO_ADD_DELAY and not (shared["last_added"] or "").endswith("_done"):
                        shared["word"] += cur_sign
                        shared["last_added"] = cur_sign + "_done"
                        shared["sign_start"] = None
                        print(f"  Auto-added '{cur_sign}'  word='{shared['word']}'")
                else:
                    shared["sign_start"] = None

                shared["sign"]          = cur_sign
                shared["conf"]          = cur_conf
                shared["auto_progress"] = auto_progress
                shared["hand_present"]  = True
        else:
            history.clear()
            with state_lock:
                shared["sign"]          = None
                shared["conf"]          = 0.0
                shared["auto_progress"] = 0.0
                shared["hand_present"]  = False
                shared["sign_start"]    = None

        draw_overlay(frame, cur_sign, cur_conf, auto_progress, hand_present)

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame_jpg = jpg.tobytes()

    cap.release()
    detector.close()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="ASL Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def mjpeg_generator():
    """Yields MJPEG multipart frames."""
    while True:
        with frame_lock:
            jpg = latest_frame_jpg
        if jpg is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(0.033)  # ~30 fps cap

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/state")
def get_state():
    with state_lock:
        return JSONResponse({
            "sign":          shared["sign"],
            "conf":          round(shared["conf"], 3),
            "word":          shared["word"],
            "sentence":      shared["sentence"],
            "auto_progress": round(shared["auto_progress"], 3),
            "hand_present":  shared["hand_present"],
        })

class Action(BaseModel):
    type: str  # "add_letter" | "add_word" | "backspace" | "clear"

@app.post("/action")
def do_action(action: Action):
    with state_lock:
        t = action.type
        if t == "add_letter":
            if shared["sign"] and shared["conf"] >= CONFIDENCE_THRESH:
                shared["word"] += shared["sign"]
                shared["sign_start"] = None
                shared["last_added"] = (shared["sign"] or "") + "_done"
        elif t == "add_word":
            if shared["word"]:
                shared["sentence"] = (shared["sentence"] + " " + shared["word"]).strip()
                shared["word"] = ""
                shared["last_added"] = ""
                shared["sign_start"] = None
        elif t == "backspace":
            if shared["word"]:
                shared["word"] = shared["word"][:-1]
                shared["last_added"] = ""
            elif shared["sentence"]:
                parts = shared["sentence"].rsplit(" ", 1)
                shared["sentence"] = parts[0] if len(parts) > 1 else ""
        elif t == "clear":
            shared["word"] = ""
            shared["sentence"] = ""
            shared["last_added"] = ""
            shared["sign_start"] = None
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def index():
    # Serve index.html from current directory
    if os.path.exists("index.html"):
        with open("index.html") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found — place it next to server.py</h1>")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start capture thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")