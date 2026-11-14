import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate(model_path, threshold_path, X, y):
    """Evaluate a model using its saved threshold."""
    model = joblib.load(model_path)

    with open(threshold_path) as f:
        threshold = float(json.load(f)["threshold"])

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    # Keep the feature names and order used during training.
    expected = list(model.feature_names_in_)
    if set(expected) != set(X.columns):
        raise ValueError("Evaluation features do not match model features.")

    positive_class = list(model.classes_).index(1)
    probabilities = model.predict_proba(X[expected])[:, positive_class]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true",
                        help="Evaluate the first champion when none exists.")
    args = parser.parse_args()

    with open("params.yaml") as f:
        rules = yaml.safe_load(f)["promotion"]

    min_recall = float(rules["min_recall"])
    min_improvement = float(rules["min_f1_improvement"])

    if not np.isfinite(min_recall) or not 0 <= min_recall <= 1:
        raise ValueError("promotion.min_recall must be between 0 and 1.")
    if not np.isfinite(min_improvement) or not 0 <= min_improvement <= 1:
        raise ValueError("promotion.min_f1_improvement must be between 0 and 1.")

    data = pd.read_csv("data/processed/test.csv")
    y = data["Machine failure"]
    X = data.drop(columns=["Machine failure"])

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Evaluation data must contain both binary classes.")

    candidate = evaluate(
        "models/model.pkl", "metrics/threshold.json", X, y
    )

    champion_model = Path("models/champion/model.pkl")
    champion_threshold = Path("models/champion/threshold.json")

    # An incomplete champion must not be treated as a first run.
    if champion_model.exists() != champion_threshold.exists():
        raise FileNotFoundError("Champion model or threshold is missing.")

    champion = None
    improvement = None

    if champion_model.exists():
        if args.bootstrap:
            raise ValueError("Champion already exists. Remove --bootstrap.")

        champion = evaluate(champion_model, champion_threshold, X, y)
        improvement = candidate["f1"] - champion["f1"]

        approved = (
            candidate["recall"] >= min_recall
            and improvement > min_improvement
        )
        reason = (
            "Candidate meets the recall and F1 improvement rules."
            if approved else
            "Candidate does not meet all promotion rules."
        )
    else:
        if not args.bootstrap:
            raise FileNotFoundError(
                "No champion found. Use --bootstrap for the first model."
            )

        approved = candidate["recall"] >= min_recall
        reason = (
            "First candidate meets the recall requirement."
            if approved else
            "First candidate does not meet the recall requirement."
        )

    report = {
        "promotion_approved": bool(approved),
        "reason": reason,
        "evaluation_data": "data/processed/test.csv",
        "rules": {
            "min_recall": min_recall,
            "min_f1_improvement": min_improvement,
        },
        "candidate": candidate,
        "champion": champion,
        "f1_improvement": improvement,
    }

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/promotion.json", "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()