
"""
STEP 2 — MODEL TRAINER
======================
Reads data/landmarks.csv, trains a Random Forest classifier, and saves
the model to data/asl_model.pkl.

Run:
    python train_model.py
"""

from sklearn.metrics           import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing     import LabelEncoder
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.ensemble          import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle


# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join("data", "landmarks.csv")
MODEL_PATH = os.path.join("data", "asl_model.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path, encoding="utf-8")
    if df.empty:
        raise ValueError("CSV is empty — run collect_data.py first.")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(
        f"✅  Loaded {len(df)} samples across {len(le.classes_)} classes: {list(le.classes_)}")
    return X, y_enc, le


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("\n🌲  Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test,  y_test)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    print(f"\n📊  Results:")
    print(f"     Train accuracy : {train_acc*100:.1f}%")
    print(f"     Test  accuracy : {test_acc*100:.1f}%")
    print(
        f"     5-Fold CV      : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    y_pred = clf.predict(X_test)
    return clf, X_test, y_test, y_pred


def plot_confusion(y_test, y_pred, le):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(max(8, len(le.classes_)),
                                    max(6, len(le.classes_))))
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("ASL Sign Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join("data", "confusion_matrix.png"), dpi=120)
    print("     Confusion matrix → data/confusion_matrix.png")
    plt.show()


def save_model(clf, le, path):
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": clf, "encoder": le}, f)
    print(f"\n💾  Model saved to {path}")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌  {CSV_PATH} not found — run collect_data.py first.")
        return

    X, y, le = load_data(CSV_PATH)

    # Need at least 2 classes
    if len(le.classes_) < 2:
        print("❌  Need samples for at least 2 signs to train. Keep collecting!")
        return

    clf, X_test, y_test, y_pred = train(X, y)

    print("\n📋  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plot_confusion(y_test, y_pred, le)
    save_model(clf, le, MODEL_PATH)

    print("\n🎉  Training complete!")
    print("     Next step → run  asl_translator.py")


if __name__ == "__main__":
    main()
