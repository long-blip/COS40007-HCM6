import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve

# 1. Read parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SEED = params["seed"]
MIN_RECALL = params["min_recall"]

# 2. Read the TRAINING data only (the test set is never used to choose the threshold)
train_df = pd.read_csv("data/processed/train.csv")
y = train_df["Machine failure"]
X = train_df.drop(columns=["Machine failure"])

# 3. Same model settings as train.py
# Reuse the saved model's configuration without its fitted state.
model = clone(joblib.load("models/model.pkl"))

# 4. Out-of-fold probabilities: each row is predicted by a model that did not see it
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

# 5. Precision, recall and F1 for every possible threshold
precision, recall, thresholds = precision_recall_curve(y, y_prob)
precision, recall = precision[:-1], recall[:-1]   # last point has no threshold
f1 = 2 * precision * recall / (precision + recall + 1e-12)

# 6. Rule: highest F1 among thresholds that keep recall. >= MIN_RECALL
f1_allowed = np.where(recall >= MIN_RECALL, f1, -1)
best = int(np.argmax(f1_allowed))

result = {
    "threshold": float(thresholds[best]),
    "min_recall": MIN_RECALL,
    "cv_precision": round(float(precision[best]), 4),
    "cv_recall": round(float(recall[best]), 4),
    "cv_f1": round(float(f1[best]), 4),
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/threshold.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))

# 7. Plot precision and recall against the threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision, label="Precision")
plt.plot(thresholds, recall, label="Recall")
plt.plot(thresholds, f1, label="F1")
plt.axvline(result["threshold"], color="red", linestyle="--", label=f"Chosen = {result['threshold']:.2f}")
plt.axhline(MIN_RECALL, color="grey", linestyle=":", label=f"Min recall = {MIN_RECALL}")
plt.xlabel("Decision threshold")
plt.ylabel("Score (5-fold CV on training set)")
plt.title("Threshold selection")
plt.legend()
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/threshold.png", dpi=150)
print("Saved metrics/threshold.json and plots/threshold.png")