import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("combined_dataset.csv", header=None)

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------------
# Clean data
# -------------------------------
df = df.dropna()

# Ensure numeric features
X = df.iloc[:, :-1].astype(float)
y = df.iloc[:, -1]

# -------------------------------
# Train-test split (IMPORTANT)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=None   # 🔥 keeps class balance
)

# -------------------------------
# Better model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# -------------------------------
# Save model
# -------------------------------
with open("hand_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as hand_model.pkl")