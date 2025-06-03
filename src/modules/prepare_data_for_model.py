from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_dataset(df: pd.DataFrame, window_size: int = None):
    df = df.copy()
    features = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_Signal"]
    df = df[features].dropna()
    if not window_size:
        window_size = len(df) - 1  # Sử dụng toàn bộ làm input

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i : i + window_size])
        y.append(scaled_data[i + window_size][3])  # giá Close làm target

    return np.array(X), np.array(y), scaler


def save_model_data(df: pd.DataFrame):
    path = Path("assets/model_data")
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "enriched_gold_data.csv", index=False)
    print("Enriched model data saved.")
