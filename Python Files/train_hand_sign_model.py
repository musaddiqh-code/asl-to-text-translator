import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels   # 🔥 NEW
import matplotlib.pyplot as plt
import pickle
import os

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("../data/combined_dataset.csv", header=None)

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------------
# Clean data
# -------------------------------
df = df.dropna()

X = df.iloc[:, :-1].astype(float)
y = df.iloc[:, -1]

# -------------------------------
# 🔥 Fix class imbalance
# -------------------------------
counts = y.value_counts()

print("\n📊 Class distribution BEFORE:")
print(counts)

# keep only classes with >=2 samples
valid_classes = counts[counts >= 2].index
df = df[df.iloc[:, -1].isin(valid_classes)]

# recompute X and y
X = df.iloc[:, :-1].astype(float)
y = df.iloc[:, -1]

print("\n📊 Class distribution AFTER:")
print(y.value_counts())

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n🌲 Training model...")
model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")

# Cross-validation
cv_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(cv_model, X, y, cv=3)  # 🔥 changed to 3 (more stable)
print(f"📊 Cross-val: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))  # 🔥 fix warning

# -------------------------------
# 🔥 FIXED Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

# get correct labels (letters instead of numbers)
labels = unique_labels(y_test, y_pred)

print("\n🔤 Labels used in confusion matrix:", labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels   # 🔥 THIS FIXES YOUR ISSUE
)

plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix (ASL Letters)")
plt.tight_layout()

os.makedirs("../data", exist_ok=True)
plt.savefig("../data/confusion_matrix.png")
plt.show()

print("📊 Confusion matrix saved → ../data/confusion_matrix.png")

# -------------------------------
# Save model
# -------------------------------
with open("hand_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n💾 Model saved as hand_model.pkl")