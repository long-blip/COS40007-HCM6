import os
import sys
import json
import yaml
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from features import build_features

# Usage: python src/monitor.py data/new/batch_002.csv   (run detect_drift.py on the same batch first)

# 1. Inputs
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
MIN_RECALL = params["min_recall"]
TOLERANCE = params["perf_tolerance"]  # allowed shortfall; batches are small (~34 failures), so 1 missed failure = 3% recall

batch_path = sys.argv[1]
batch_name = os.path.splitext(os.path.basename(batch_path))[0]
batch = pd.read_csv(batch_path)

with open("metrics/drift_report.json", "r") as f:
    drift = json.load(f)
if drift["batch"] != batch_path:
    sys.exit(f"drift_report.json is for {drift['batch']}. Run: python src/detect_drift.py {batch_path}")

with open("metrics/metrics.json", "r") as f:
    baseline = json.load(f)  # test-set results of the current model
with open("metrics/threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]
model = joblib.load("models/model.pkl")

# 2. Performance check: score the current model on the new batch (same features as training)
X, y = build_features(batch)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)  # e.g. a Type missing from the batch
y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)

performance = {
    "precision": round(precision_score(y, y_pred, zero_division=0), 4),
    "recall": round(recall_score(y, y_pred, zero_division=0), 4),
    "f1": round(f1_score(y, y_pred, zero_division=0), 4),
    "baseline_f1": round(baseline["f1"], 4),
}
performance["f1_drop"] = round(performance["baseline_f1"] - performance["f1"], 4)

# 3. Alerts
alerts = []
if drift["drift_detected"]:
    alerts.append(f"Data drift in {', '.join(drift['drifted_features'])}")
if performance["f1_drop"] > TOLERANCE:
    alerts.append(f"F1 dropped by {performance['f1_drop']:.3f} (limit {TOLERANCE})")
if performance["recall"] < MIN_RECALL - TOLERANCE:
    alerts.append(f"Recall {performance['recall']:.3f} is below {MIN_RECALL - TOLERANCE:.2f}")
status = "ALERT" if alerts else "OK"

report = {
    "time": datetime.now().isoformat(timespec="seconds"),
    "batch": batch_path,
    "rows": len(batch),
    "failures_in_batch": int(y.sum()),
    "status": status,
    "alerts": alerts,
    "retrain_recommended": status == "ALERT",
    "drift_detected": drift["drift_detected"],
    "drifted_features": drift["drifted_features"],
    "performance": performance,
}
os.makedirs("metrics", exist_ok=True)
with open("metrics/monitoring_report.json", "w") as f:
    json.dump(report, f, indent=2)

# 4. Log: one row per monitoring run, appended to a history file
os.makedirs("logs", exist_ok=True)
log_path = "logs/monitoring_log.csv"
row = pd.DataFrame([{
    "time": report["time"], "batch": batch_path, "rows": len(batch), "status": status,
    "drifted_features": "; ".join(drift["drifted_features"]),
    "precision": performance["precision"], "recall": performance["recall"],
    "f1": performance["f1"], "f1_drop": performance["f1_drop"],
}])
row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

# 5. Drift report plot: reference vs batch distribution for each sensor
reference = pd.read_csv("data/ai4i2020.csv")
numeric = [c for c, r in drift["features"].items() if "psi" in r]
fig, axes = plt.subplots(1, len(numeric), figsize=(4 * len(numeric), 3.5))
for ax, col in zip(axes, numeric):
    r = drift["features"][col]
    ax.hist(reference[col], bins=30, density=True, alpha=0.5, label="Reference")
    ax.hist(batch[col], bins=30, density=True, alpha=0.5, label=batch_name)
    ax.set_title(f"{col}\nPSI={r['psi']:.2f}  {'DRIFT' if r['drift'] else 'ok'}",
                 color="red" if r["drift"] else "black", fontsize=10)
axes[0].legend()
fig.suptitle(f"{batch_name}: {status}  |  F1 {performance['f1']:.3f} (baseline {performance['baseline_f1']:.3f})")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/drift_{batch_name}.png", dpi=120)

# 6. Console summary
print(f"[{status}] {batch_path}: {len(batch)} rows, {int(y.sum())} failures")
print(f"  Precision {performance['precision']:.3f}  Recall {performance['recall']:.3f}  "
      f"F1 {performance['f1']:.3f}  (baseline F1 {performance['baseline_f1']:.3f})")
for a in alerts:
    print(f"  ALERT: {a}")
print(f"Saved metrics/monitoring_report.json, {log_path} and plots/drift_{batch_name}.png")