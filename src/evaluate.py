
import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, average_precision_score, precision_score, recall_score, f1_score

model = joblib.load("models/model.pkl")
test_df = pd.read_csv("data/processed/test.csv")

y_test = test_df["Machine failure"]
X_test = test_df.drop(columns=["Machine failure"])
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "pr_auc": average_precision_score(y_test, y_prob)
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["No failure", "Failure"], cmap="Blues", ax=axes[0], colorbar=False)
axes[0].set_title(f"Confusion matrix (Recall = {metrics['recall']:.2f}, F1 = {metrics['f1']:.2f})")

importance = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
importance.plot(kind="barh", ax=axes[1], color="#4C72B0")
axes[1].set_title("Feature importance")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/model_results.png", dpi=150)
print("Saved plots/model_results.png and metrics/metrics.json")
