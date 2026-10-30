"""
COS40007 Applied Project - Task 2: baseline training pipeline (Group HCM-6)

Predicts machine failure on the AI4I 2020 Predictive Maintenance dataset.
Run by GitHub Actions on every push / pull request to main.

Outputs:
  - model_results.png : confusion matrix + feature importance
  - metrics.txt       : evaluation metrics on the held-out test set
"""

import matplotlib
matplotlib.use("Agg")  # no display on the GitHub Actions runner

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/ai4i2020.csv"
SEED = 42

# 1. Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

# 2. Preprocessing: drop identifiers and failure-mode columns (data leakage), encode Type
X = df.drop(columns=["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
y = df["Machine failure"]
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# 3. Feature engineering (physics-based features from the EDA)
X["temp_diff"] = X["Process temperature [K]"] - X["Air temperature [K]"]
X["power"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
X["strain"] = X["Tool wear [min]"] * X["Torque [Nm]"]

# 4. Stratified train/test split (keeps the 3.39% failure rate in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# 5. Train baseline model (class weights handle the class imbalance)
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Evaluate on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "pr_auc": average_precision_score(y_test, y_prob),
}

with open("metrics.txt", "w") as f:
    f.write("Model: RandomForestClassifier (class_weight='balanced', 200 trees)\n")
    f.write(f"Train size: {len(X_train)} | Test size: {len(X_test)} | Test failures: {int(y_test.sum())}\n")
    for name, value in metrics.items():
        f.write(f"{name}: {value:.4f}\n")
print(open("metrics.txt").read())

# 7. Save plots: confusion matrix + top feature importances
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["No failure", "Failure"], cmap="Blues", ax=axes[0], colorbar=False)
axes[0].set_title(f"Confusion matrix (Recall = {metrics['recall']:.2f}, F1 = {metrics['f1']:.2f})")

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", ax=axes[1], color="#4C72B0")
axes[1].set_title("Feature importance")
plt.tight_layout()
plt.savefig("model_results.png", dpi=150)
print("Saved model_results.png and metrics.txt")
