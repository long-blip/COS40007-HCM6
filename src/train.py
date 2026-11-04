import os
import joblib
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Read parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SEED = params["seed"]
N_ESTIMATORS = params["n_estimators"]

# Read the training data
train_df = pd.read_csv("data/processed/train.csv")

# Separate features and target
y = train_df["Machine failure"]
X = train_df.drop(columns=["Machine failure"])

# Train the model
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)
model.fit(X, y)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
joblib.dump(model, "models/model.pkl")
print("Saved models/model.pkl")