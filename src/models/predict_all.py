import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pickle

import pandas as pd

MODEL_DIR = "models/data"
DATA_PATH = "assets/processed/gold_prices_processed.csv"
PRED_PATH = "assets/predictions/gold_predictions.csv"
os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)

# Đọc dữ liệu
raw = pd.read_csv(DATA_PATH)
raw.columns = [c.lower() for c in raw.columns]
if "date" in raw.columns:
    raw["date"] = pd.to_datetime(raw["date"])

from src.models.Model_Predict.Features import custom_feature_func

features = custom_feature_func(raw.copy())

results = features[["date"]].copy()

for target in ["close", "volume"]:
    model_path = os.path.join(MODEL_DIR, f"xgb_{target}.pkl")
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model {model_path}, bỏ qua {target}")
        continue
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = features.drop(["date", "close", "volume"], axis=1, errors="ignore")
    results[f"pred_{target}"] = model.predict(X)

results.to_csv(PRED_PATH, index=False)
print(f"Đã lưu dự đoán vào {PRED_PATH}")
