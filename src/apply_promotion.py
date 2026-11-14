import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from promote_model import evaluate


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    with open("metrics/promotion.json") as f:
        report = json.load(f)

    # Keep the current champion when the candidate is rejected.
    if report["promotion_approved"] is not True:
        print("Promotion rejected. Champion unchanged.")
        return

    # Allow publishing only from main.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        allowed = (
            os.environ.get("GITHUB_REF") == "refs/heads/main"
            and os.environ.get("GITHUB_EVENT_NAME")
            in {"push", "schedule", "workflow_dispatch"}
        )
    else:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
        allowed = branch == "main"

    if not allowed:
        raise RuntimeError("Champion updates are allowed only on main.")

    # Recheck the decision against the current files and rules.
    with open("params.yaml") as f:
        rules = yaml.safe_load(f)["promotion"]

    data = pd.read_csv("data/processed/test.csv")
    X = data.drop(columns=["Machine failure"])
    y = data["Machine failure"]

    candidate = evaluate(
        "models/model.pkl", "metrics/threshold.json", X, y
    )
    champion = evaluate(
        "models/champion/model.pkl",
        "models/champion/threshold.json",
        X, y,
    )

    approved = (
        candidate["recall"] >= float(rules["min_recall"])
        and candidate["f1"] - champion["f1"]
        > float(rules["min_f1_improvement"])
    )

    if (
        not approved
        or candidate != report["candidate"]
        or champion != report["champion"]
    ):
        raise RuntimeError("Promotion decision is stale. Run the check again.")

    # Record the candidate's source and artifact hashes.
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    metadata = {
        "version": args.version,
        "source_commit": source_commit,
        "model_sha256": file_hash("models/model.pkl"),
        "threshold_sha256": file_hash("metrics/threshold.json"),
        "evaluation_data_sha256": file_hash("data/processed/test.csv"),
        "params_sha256": file_hash("params.yaml"),
        "candidate": candidate,
        "previous_champion": champion,
    }

    destination = Path("models/champion")

    # Prepare files before replacing the champion's files.
    with tempfile.TemporaryDirectory(dir="models") as temporary:
        staging = Path(temporary)
        shutil.copy2("models/model.pkl", staging / "model.pkl")
        shutil.copy2("metrics/threshold.json", staging / "threshold.json")
        shutil.copy2("metrics/promotion.json", staging / "promotion.json")

        with open(staging / "version.json", "w") as f:
            json.dump(metadata, f, indent=2, allow_nan=False)

        for name in (
            "model.pkl", "threshold.json", "promotion.json", "version.json"
        ):
            os.replace(staging / name, destination / name)

    print(f"Champion updated to {args.version}")


if __name__ == "__main__":
    main()
    