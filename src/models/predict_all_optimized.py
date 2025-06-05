import os
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.Features_updated import create_features

DATA_PATH = "assets/raw/gold_prices.csv"
MODEL_DIR = "src/models/data"
SAVE_PATH = "assets/predictions/pred.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)
df_feat = create_features(df)
X = df_feat.drop(columns=["date", "close", "volume"])
results = df_feat[["date"]].copy()

targets = ["close", "volume"]

for target in targets:
    model = joblib.load(os.path.join(MODEL_DIR, f"{target}_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{target}_scaler.pkl"))

    y_scaled = model.predict(X).reshape(-1, 1)

    if target == "volume":
        y_pred = np.expm1(scaler.inverse_transform(y_scaled)).flatten()
    else:
        y_pred = scaler.inverse_transform(y_scaled).flatten()

    results[f"pred_{target}"] = y_pred

results.to_csv(SAVE_PATH, index=False)
print("Predictions saved to:", SAVE_PATH)