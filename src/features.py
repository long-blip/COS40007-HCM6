import numpy as np
import pandas as pd

TARGET = "Machine failure"

# Raw column names -> snake_case (XGBoost does not allow [, ] or < in feature names)
RENAME = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

# Identifiers and failure-mode columns (data leakage)
DROP = ["UDI", "Product ID", TARGET, "TWF", "HDF", "PWF", "OSF", "RNF"]


def build_features(df):
    """Turn raw AI4I rows into model features. Used by preprocess.py (training) and monitor.py (new batches)."""
    df = df.rename(columns=RENAME)
    X = df.drop(columns=DROP)
    X = pd.get_dummies(X, columns=["Type"], dtype=int)
    X["temp_diff"] = X["process_temp_k"] - X["air_temp_k"]
    X["power"] = X["torque_nm"] * X["rot_speed_rpm"] * 2 * np.pi / 60
    X["strain"] = X["tool_wear_min"] * X["torque_nm"]
    y = df[TARGET]
    return X, y