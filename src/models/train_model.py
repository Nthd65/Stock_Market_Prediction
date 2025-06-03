import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
DATA_PATH = "assets/processed/gold_prices_processed.csv"
MODEL_DIR = "models/data"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = ["close", "volume"]

# Đọc dữ liệu
raw = pd.read_csv(DATA_PATH)
raw.columns = [c.lower() for c in raw.columns]
if "date" in raw.columns:
    raw["date"] = pd.to_datetime(raw["date"])

# Tự động enrich lại feature nếu cần
from src.models.Model_Predict.Features import custom_feature_func

features = custom_feature_func(raw.copy())

# Tách X, y cho từng target
for target in TARGETS:
    if target not in features.columns:
        print(f"Cảnh báo: Không tìm thấy cột {target} trong dữ liệu!")
        continue
    X = features.drop(["date", "volume", "close"], axis=1, errors="ignore")
    y = features[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print(f"\n--- Training model for {target} ---")
    model = XGBRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    # Lưu model
    model_path = os.path.join(MODEL_DIR, f"xgb_{target}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Đã lưu model {target} tại {model_path}")
