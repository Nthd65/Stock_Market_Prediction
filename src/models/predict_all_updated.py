import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.Model_Predict.Features import custom_feature_func

# ==== Đường dẫn ====
DATA_PATH = "assets/raw/gold_prices.csv"
MODEL_DIR = "src/models/data"
SAVE_PATH = "assets/predictions/pred.csv"
os.makedirs("assets/predictions", exist_ok=True)

# ==== Load và tạo features ====
df = pd.read_csv(DATA_PATH)
df_feat = custom_feature_func(df)

drop_cols = ["date", "close", "volume"]
X = df_feat.drop(columns=drop_cols)
results = df_feat[["date"]].copy()

targets = ["close", "volume"]

for target in targets:
    model = joblib.load(os.path.join(MODEL_DIR, f"{target}_model.pkl"))
    y_scaler = joblib.load(os.path.join(MODEL_DIR, f"{target}_scaler.pkl"))

    y_pred_scaled = model.predict(X)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    results[f"pred_{target}"] = y_pred

# Lưu kết quả
results.to_csv(SAVE_PATH, index=False)
print("Dự đoán hoàn tất và lưu vào:", SAVE_PATH)
