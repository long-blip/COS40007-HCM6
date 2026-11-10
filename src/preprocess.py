import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from features import build_features

DATA_PATH = "data/ai4i2020.csv"

# 1. Load data
#read seed and test_size from params.yaml, no hard-coding
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SEED = params["seed"]
TEST_SIZE = params["test_size"]

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

#Create the directory before writing: os.makedirs("data/processed", exist_ok=True).
os.makedirs("data/processed", exist_ok=True)

# 2-3. Drop leakage columns, encode Type and add physics-based features (shared with monitor.py)
X, y = build_features(df)

# 4. Stratified train/test split (keeps the 3.39% failure rate in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

# combine x and y into a single dataframe for train and test sets then save to data/processed/train.csv and data/processed/test.csv
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

#Print the sizes of the two sets and the proportion of faulty samples in each set to verify that the stratification is correct.
print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Proportion of faulty samples in train set: {y_train.mean():.2%}")
print(f"Proportion of faulty samples in test set: {y_test.mean():.2%}")
