import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Simulate new sensor data arriving from the factory.
# Batches are taken from the TEST split only (same split as preprocess.py), so the model has never seen these rows
# and the performance check in monitor.py is fair.
# batch_001: normal data                                      -> no drift expected
# batch_002: machines run hotter and tools are worn longer    -> drift expected
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SEED = params["seed"]
df = pd.read_csv("data/ai4i2020.csv")
_, unseen = train_test_split(df, test_size=params["test_size"],
                             stratify=df["Machine failure"], random_state=SEED)

normal, shifted = train_test_split(unseen, test_size=0.5,
                                   stratify=unseen["Machine failure"], random_state=SEED)
os.makedirs("data/new", exist_ok=True)
normal.to_csv("data/new/batch_001.csv", index=False)

shifted = shifted.copy()
shifted["Air temperature [K]"] += 2
shifted["Process temperature [K]"] += 2
shifted["Tool wear [min]"] = (shifted["Tool wear [min]"] * 1.3).round()
shifted.to_csv("data/new/batch_002.csv", index=False)

print(f"Saved data/new/batch_001.csv (normal, {len(normal)} rows) and data/new/batch_002.csv (shifted, {len(shifted)} rows)")