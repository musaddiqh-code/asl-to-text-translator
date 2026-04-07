# 🤟 SignBridge — ASL Translator

> Real-time American Sign Language ↔ Text translation in your browser, powered by MediaPipe and a Random Forest classifier.

![Python](https://img.shields.io/badge/Python-3.9--3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.30+-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-yellow?logo=scikitlearn)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## ✨ Features

- **ASL → Text** — Point your webcam at your hand, sign a letter, and watch it appear on screen in real time
- **Text → ASL** — Type any text and see each letter rendered as its ASL hand sign with animated playback
- **Voice Input** — Dictate text with your microphone via the Web Speech API
- **Translation History** — Sessions are saved locally in the browser for later review
- **Dark Mode** — Light/dark theme toggle, preference persisted in `localStorage`
- **Accessibility First** — Keyboard navigation, ARIA labels, focus rings, and screen-reader support throughout
- **On-device Processing** — No video frames or landmark data ever leave your browser

---

## 🖥️ Demo

```
Webcam frame
    └─► MediaPipe Hands  ──►  21 hand landmarks (x, y)
                                    │
                         Normalize (wrist = origin,
                           scale to unit bounding box)
                                    │
                       Random Forest Classifier (300 trees)
                                    │
                        Smoothed over 5-frame window
                                    │
                   Predicted ASL letter + confidence score
                                    │
                      Live overlay on webcam feed
```

---

## 🗂️ Project Structure

```
signbridge/
├── index.html                  # Frontend SPA (4 pages: Home, ASL→Text, Text→ASL, About)
├── style.css                   # Design tokens, components, dark mode, responsive layout
├── script.js                   # All frontend logic — camera, prediction, history, playback
├── database.json               # Word/letter → video asset mapping
│
├── app.py                      # FastAPI endpoint: POST /predict-image (per-frame inference)
├── server.py                   # Alternative server: MJPEG stream + /state + /action endpoints
│
├── collect_live_data.py        # Step 1a — Record live webcam samples into live_data.csv
├── image_to_csv.py             # Step 1b — Extract landmarks from an image dataset folder
├── merge_data_csv.py           # Step 2  — Combine final_dataset.csv + live_data.csv
├── train_hand_sign_model.py    # Step 3  — Train Random Forest, save hand_model.pkl
├── predict_live.py             # Step 4  — Standalone OpenCV real-time prediction (no browser)
│
├── final_dataset.csv           # Landmarks extracted from image dataset
├── live_data.csv               # Landmarks recorded live via webcam
├── combined_dataset.csv        # Merged training data
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/signbridge.git
cd signbridge

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download the MediaPipe hand landmark model

```bash
mkdir -p data
curl -L -o hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

### 3. Collect training data

**Option A — from your webcam (recommended for personalization):**

```bash
python collect_live_data.py
# Enter a label (e.g. A), show your hand sign, press S to save samples, Q to quit
# Repeat for every letter you want to support
```

**Option B — from an existing image dataset:**

```
own_dataset/
├── A/
│   ├── img001.jpg
│   └── ...
├── B/
│   └── ...
```

```bash
python image_to_csv.py
```

**Merge datasets (if you used both):**

```bash
python merge_data_csv.py
```

### 4. Train the model

```bash
python train_hand_sign_model.py
# Outputs: hand_model.pkl  +  data/confusion_matrix.png
```

> **Tip:** Aim for ≥ 100–200 samples per sign. Record in varied lighting and hand positions for better accuracy.

### 5. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Then open `index.html` in your browser (or serve it locally):

```bash
python -m http.server 5500
# → http://localhost:5500
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Hand tracking | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
| ML classifier | scikit-learn `RandomForestClassifier` (300 trees, balanced class weights) |
| Video capture | OpenCV |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Voice input | Web Speech API |
| Fonts | Outfit + JetBrains Mono (Google Fonts) |

---

## 🔬 How the Model Works

1. **Landmark extraction** — MediaPipe detects 21 (x, y) keypoints per hand per frame
2. **Normalization** — Wrist landmark shifted to origin; coordinates scaled to a unit bounding box so predictions are position- and size-invariant
3. **Feature vector** — 42 floats (21 landmarks × 2 axes) fed to the classifier
4. **Prediction smoothing** — Most-frequent prediction across a 5-frame rolling buffer is used to reduce flicker
5. **Confidence gating** — Only predictions above 0.70 confidence are accepted

---

## 📊 Training Tips

| Goal | Recommendation |
|---|---|
| Increase accuracy | 300+ samples per sign |
| Reduce confusion between similar letters (e.g. M/N, U/V) | Record extra samples for confusable pairs |
| Handle different users | Record with multiple people |
| Improve lighting robustness | Record in bright, dim, and backlit conditions |
| Swap the classifier | Replace `RandomForestClassifier` with `MLPClassifier` or a small Keras/PyTorch model |

---

## 🔌 API Reference

### `POST /predict-image`

Upload a JPEG frame and receive a predicted sign.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | JPEG image of a hand sign |

**Response**

```json
{
  "prediction": "A",
  "confidence": 0.94
}
```

---

## 🌐 Browser Compatibility

| Browser | ASL→Text camera | Voice input |
|---|---|---|
| Chrome / Edge | ✅ | ✅ |
| Firefox | ✅ | ⚠️ partial |
| Safari (macOS 14+) | ✅ | ⚠️ partial |

> Camera access requires HTTPS in production (or `localhost` for development).

---

## 🔒 Privacy

All inference runs **on your device**. No video frames, landmark data, or translated text is transmitted to any remote server. The camera feed never leaves your browser tab.

---

## 📁 Supported Python Versions

Python **3.9 – 3.11** (required for MediaPipe compatibility).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Commit your changes: `git commit -m "Add my improvement"`
4. Push and open a Pull Request

Bug reports, dataset contributions, and accuracy improvements are especially welcome.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the ASL community.*
