import os
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.Model_Predict.Features import custom_feature_func

# ==== Đường dẫn ====
DATA_PATH = "assets/processed/gold_prices_processed.csv"
SAVE_DIR = "src/models/data"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== Load và tạo features ====
df = pd.read_csv(DATA_PATH)
print(df.columns)
df = custom_feature_func(df)

# ==== Mục tiêu cần dự đoán ====
targets = ["close", "volume"]

for target in targets:
    print(f"Training model for: {target}")

    # Tạo X, y
    drop_cols = ["date", "close", "volume"]
    X = df.drop(columns=drop_cols)
    y = df[target].values.reshape(-1, 1)

    # ==== Chuẩn hóa y ====
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    # Lưu y_scaler
    joblib.dump(y_scaler, os.path.join(SAVE_DIR, f"{target}_scaler.pkl"))

    # ==== Train model ====
    model = XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, y_scaled.ravel())

    # Lưu mô hình
    joblib.dump(model, os.path.join(SAVE_DIR, f"{target}_model.pkl"))
    print(f"Model and scaler saved for: {target}")
