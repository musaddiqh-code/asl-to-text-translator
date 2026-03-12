# ASL → Text Translator
A real-time American Sign Language translator using MediaPipe hand landmarks,
a Random Forest classifier, and OpenCV for display.

---

## Project Structure
```
asl_translator/
├── collect_data.py      ← Step 1: Record training samples
├── train_model.py       ← Step 2: Train the classifier
├── asl_translator.py    ← Step 3: Run the real-time translator
├── requirements.txt
└── data/                ← Created automatically
    ├── landmarks.csv    ← Your recorded samples
    ├── asl_model.pkl    ← Trained model
    └── confusion_matrix.png
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Step 1 — Collect Training Data

```bash
python collect_data.py
```

- A webcam window opens.
- **Press any letter key** (e.g. `a`) → position your hand into that ASL sign → the app records 100 frames automatically after a 2-second countdown.
- Repeat for every sign you want to support (more signs = more expressive translator).
- Press `Q` to quit.

**Tip:** Record at least 100–200 samples per sign, in different lighting conditions and hand positions for best accuracy.

---

## Step 2 — Train the Model

```bash
python train_model.py
```

- Reads `data/landmarks.csv`.
- Trains a Random Forest classifier (200 trees).
- Prints per-class accuracy and saves a confusion matrix image.
- Saves the model to `data/asl_model.pkl`.

---

## Step 3 — Run the Translator

```bash
python asl_translator.py
```

### Controls
| Key       | Action                              |
|-----------|-------------------------------------|
| `SPACE`   | Manually add the current letter     |
| `ENTER`   | Accept current word → sentence      |
| `BKSP`    | Delete last letter                  |
| `C`       | Clear everything                    |
| `Q`       | Quit                                |

### Auto-add
Hold a sign steady for **2 seconds** and the letter is added automatically — no key press needed.

---

## How It Works

```
Webcam frame
    └─► MediaPipe Hands  ──►  21 landmarks (x, y, z)
                                    │
                             Normalise (wrist = origin,
                               scale to unit bounding box)
                                    │
                         Random Forest Classifier
                                    │
                          Smoothed over 10 frames
                                    │
                         Predicted ASL letter + confidence
                                    │
                         OpenCV UI overlay (letter, bar, word, sentence)
```

---

## Improving Accuracy
- Collect **more samples per sign** (300+ recommended).
- Record in **different lighting** and with **different hand sizes**.
- Try both left and right hands.
- In `train_model.py`, you can swap `RandomForestClassifier` for a **MLP** (`sklearn.neural_network.MLPClassifier`) or even a small Keras model for higher accuracy.

---

## Supported Python Version
Python 3.9 – 3.11 recommended (MediaPipe compatibility).
