import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# 1. Read parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

KS_ALPHA = params["ks_alpha"]            # KS-test: p-value below this = distributions differ
PSI_THRESHOLD = params["psi_threshold"]  # PSI: value above this = the difference is large enough to matter
CHI2_ALPHA = params["chi2_alpha"]        # chi-square: p-value below this = category mix differs

NUMERIC = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CATEGORICAL = "Type"

# 2. Reference = data the current model was trained on; current = the new batch (path given in the command)
reference = pd.read_csv("data/ai4i2020.csv")
batch_path = sys.argv[1]
current = pd.read_csv(batch_path)


def psi(ref, cur, bins=10):
    """Population Stability Index: how much the share of data in each bin has moved."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, 1e-4, None)  # avoid log(0)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# 3. Numeric features: KS-test (is it different?) + PSI (how much?)
features = {}
for col in NUMERIC:
    p_value = ks_2samp(reference[col], current[col]).pvalue
    psi_value = psi(reference[col], current[col])
    features[col] = {
        "test": "KS + PSI",
        "ks_p_value": round(float(p_value), 4),
        "psi": round(psi_value, 4),
        "drift": bool(p_value < KS_ALPHA and psi_value >= PSI_THRESHOLD),
    }

# 4. Categorical feature: chi-square test on the counts of L / M / H
counts = pd.DataFrame({
    "reference": reference[CATEGORICAL].value_counts(),
    "current": current[CATEGORICAL].value_counts(),
}).fillna(0)
chi2_p = chi2_contingency(counts.T).pvalue
features[CATEGORICAL] = {
    "test": "chi-square",
    "chi2_p_value": round(float(chi2_p), 4),
    "drift": bool(chi2_p < CHI2_ALPHA),
}

# 5. Overall result and report
drifted = [name for name, r in features.items() if r["drift"]]
report = {
    "batch": batch_path,
    "rows": len(current),
    "drift_detected": len(drifted) > 0,
    "drifted_features": drifted,
    "features": features,
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/drift_report.json", "w") as f:
    json.dump(report, f, indent=2)

for name, r in features.items():
    print(f"{name:26s} drift={r['drift']}")
print(f"Drift detected: {report['drift_detected']} {drifted}")
print("Saved metrics/drift_report.json")