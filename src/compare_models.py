
import os
import json
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier

TARGET = "Machine failure"
METRICS = ["accuracy", "precision", "recall", "f1", "pr_auc"]

# 1. Parameters and data
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
SEED = params["seed"]

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

# 2. Models and hyperparameter grids
# XGBoost has no class_weight option; scale_pos_weight = negatives / positives
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance (negatives / positives) in training set: {scale:.1f}")

candidates = {
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2]},
    ),
    "XGBoost": (
        XGBClassifier(scale_pos_weight=scale, eval_metric="logloss", random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
    ),
}

# 3. Tune each model with 5-fold stratified CV, then evaluate on the test set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}
for name, (model, grid) in candidates.items():
    print(f"\nTuning {name} ...")
    search = GridSearchCV(model, grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_prob),
        "cv_f1": search.best_score_,
        "best_params": search.best_params_,
    }
    print(f"Best params: {search.best_params_}")
    print(f"CV F1 (5-fold): {search.best_score_:.4f}")
    for m in METRICS:
        print(f"  test {m}: {results[name][m]:.4f}")

# 4. Save results
os.makedirs("metrics", exist_ok=True)
with open("metrics/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

# 5. Grouped bar chart of test metrics
table = pd.DataFrame({name: {m: r[m] for m in METRICS} for name, r in results.items()})
fig, ax = plt.subplots(figsize=(9, 5))
table.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], width=0.75)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
ax.set_title("Random Forest vs XGBoost on the test set (tuned with 5-fold GridSearchCV)")
ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_ylim(0.7, 1.05)
ax.set_xticklabels([m.upper() if m != "pr_auc" else "PR-AUC" for m in METRICS], rotation=0)
ax.legend(title="Model", loc="upper right", ncol=2)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/model_comparison.png", dpi=150)

print("\nComparison table (test set):")
print(table.T.round(4).to_string())
print("Saved metrics/model_comparison.json and plots/model_comparison.png")