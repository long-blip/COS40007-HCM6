import os
import pandas as pd

# Simulate new sensor data arriving from the factory.
# batch_001: normal data (same distribution as the original dataset)  -> no drift expected
# batch_002: machines run hotter and tools are worn longer            -> drift expected
SEED = 42
df = pd.read_csv("data/ai4i2020.csv")
os.makedirs("data/new", exist_ok=True)

normal = df.sample(n=1000, random_state=SEED)
normal.to_csv("data/new/batch_001.csv", index=False)

shifted = df.sample(n=1000, random_state=SEED + 1).copy()
shifted["Air temperature [K]"] += 2
shifted["Process temperature [K]"] += 2
shifted["Tool wear [min]"] = (shifted["Tool wear [min]"] * 1.3).round()
shifted.to_csv("data/new/batch_002.csv", index=False)

print("Saved data/new/batch_001.csv (normal) and data/new/batch_002.csv (shifted)")