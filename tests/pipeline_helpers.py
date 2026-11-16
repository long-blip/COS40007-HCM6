"""Shared helpers for the pipeline tests: synthetic AI4I-style data and a temporary workspace."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def make_raw(n=1500, seed=0):
    """Raw rows with the same columns as data/ai4i2020.csv. Failures follow a simple, learnable rule."""
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300, 2, n).round(1)
    process = (air + 10 + rng.normal(0, 1, n)).round(1)
    speed = rng.normal(1500, 150, n).round().astype(int)
    torque = rng.normal(40, 10, n).clip(3).round(1)
    wear = rng.integers(0, 250, n)
    failure = ((wear > 200) & (torque > 45)).astype(int)
    return pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{t}{10000 + i}" for i, t in enumerate(types)],
        "Type": types,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Machine failure": failure,
        "TWF": failure, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    })


def shift(df, temperature=5.0):
    """Simulate machines running hotter (what batch_002 does)."""
    df = df.copy()
    df["Air temperature [K]"] += temperature
    df["Process temperature [K]"] += temperature
    return df


class Workspace:
    """A temporary project folder with a small params.yaml, so scripts can run without touching the repo."""

    def __init__(self, **param_overrides):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name)
        params = yaml.safe_load((ROOT / "params.yaml").read_text())
        params.update(n_estimators=50, max_depth=8)  # small model = fast tests
        params.update(param_overrides)
        (self.path / "params.yaml").write_text(yaml.safe_dump(params))

    def write_csv(self, relative, df):
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)
        return target

    def run(self, script, *args):
        """Run a pipeline script from src/ inside this workspace."""
        return subprocess.run([sys.executable, str(SRC / script), *map(str, args)],
                              cwd=self.path, capture_output=True, text=True)

    def cleanup(self):
        self._tmp.cleanup()