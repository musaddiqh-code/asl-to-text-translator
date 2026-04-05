from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
with open("hand_model.pkl", "rb") as f:
    model = pickle.load(f)

# MediaPipe
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# 🧠 Normalize landmarks (VERY IMPORTANT)
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])

    # shift origin to first point
    pts -= pts[0]

    # scale normalization
    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts /= max_val

    return pts.flatten()


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"prediction": "Invalid image received"}

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return {"prediction": "No hand detected"}

    try:
        hand_landmarks = result.hand_landmarks[0]

        # 🔥 normalize
        row = normalize_landmarks(hand_landmarks)

        # ensure correct shape
        if len(row) != 42:
            return {"prediction": "Invalid landmark size"}

        # 🔥 get confidence
        probs = model.predict_proba([row])[0]
        idx = int(np.argmax(probs))

        prediction = model.classes_[idx]
        confidence = float(probs[idx])

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"prediction": f"Error: {str(e)}"}