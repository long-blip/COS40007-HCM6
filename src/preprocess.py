import os
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from features import build_features

DATA_PATH = "data/validated/ai4i2020.csv"

NEW_DIR = Path("data/validated/new")  # validated new batches (see src/ingest_batches.py)

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

# 5. New labelled batches: add them with their own 80/20 split.
# The original rows keep the split above, so the champion is never evaluated on rows it was trained on.
new_files = sorted(Path(NEW_DIR).glob("*.csv"))
if new_files:
    new = pd.concat([pd.read_csv(f) for f in new_files], ignore_index=True)
    # Drop rows that are exact copies of the original data (same values, only UDI differs)
    key = [c for c in df.columns if c != "UDI"]
    new = new.drop_duplicates(subset=key)
    new = new.merge(df[key], on=key, how="left", indicator=True)
    new = new[new["_merge"] == "left_only"].drop(columns="_merge")
    print(f"New batches: {len(new_files)} file(s), {len(new)} new rows")

    if len(new) > 0:
        X_new, y_new = build_features(new)
        X_new = X_new.reindex(columns=X.columns, fill_value=0)  # e.g. a Type missing from the batch
        stratify = y_new if y_new.value_counts().min() >= 2 and y_new.nunique() == 2 else None
        Xn_train, Xn_test, yn_train, yn_test = train_test_split(
            X_new, y_new, test_size=TEST_SIZE, stratify=stratify, random_state=SEED)
        X_train, y_train = pd.concat([X_train, Xn_train]), pd.concat([y_train, yn_train])
        X_test, y_test = pd.concat([X_test, Xn_test]), pd.concat([y_test, yn_test])

# combine x and y into a single dataframe for train and test sets then save to data/processed/train.csv and data/processed/test.csv
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("data/processed/train.csv", index=False, lineterminator="\n")
test_df.to_csv("data/processed/test.csv", index=False, lineterminator="\n")

#Print the sizes of the two sets and the proportion of faulty samples in each set to verify that the stratification is correct.
print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Proportion of faulty samples in train set: {y_train.mean():.2%}")
print(f"Proportion of faulty samples in test set: {y_test.mean():.2%}")
