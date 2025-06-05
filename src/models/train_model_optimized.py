import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.Features_updated import create_features

DATA_PATH = "assets/raw/gold_prices.csv"
SAVE_DIR = "src/models/data"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = create_features(df)

targets = ["close", "volume"]

for target in targets:
    print(f"Training model for: {target}")
    X = df.drop(columns=["date", "close", "volume"])
    y = df[target].values.reshape(-1, 1)

    if target == "volume":
        y = np.log1p(y)
        log_flag = True
    else:
        log_flag = False

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)
    joblib.dump(scaler, os.path.join(SAVE_DIR, f"{target}_scaler.pkl"))

    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
    model.fit(X, y_scaled.ravel())
    joblib.dump(model, os.path.join(SAVE_DIR, f"{target}_model.pkl"))
    print(f"Saved model and scaler for {target}")