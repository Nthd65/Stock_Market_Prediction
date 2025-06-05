import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.configs.path_dir import DATA_PATH, MODEL_DIR, PRED_PATH
from src.models.Model_Predict.Features import custom_feature_func

os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)

# Đọc dữ liệu
raw = pd.read_csv(DATA_PATH)
raw.columns = [c.lower() for c in raw.columns]
if "date" in raw.columns:
    raw["date"] = pd.to_datetime(raw["date"])


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

# Thêm logic dự đoán cho ngày mai
latest_date = raw['date'].max()
next_date = latest_date + pd.Timedelta(days=1)

# Dự đoán cho ngày mai
next_features = features[features['date'] == latest_date].copy()
next_features['date'] = next_date
X_next = next_features.drop(['date', 'close', 'volume'], axis=1, errors='ignore')

for target in ['close', 'volume']:
    model_path = os.path.join(MODEL_DIR, f"xgb_{target}.pkl")
    if not os.path.exists(model_path):
        print(f"Khong tim thay model {model_path}, bo qua {target}")
        continue
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    results[f"pred_{target}_next"] = model.predict(X_next)

results.to_csv(PRED_PATH, index=False)
print(f"Đã lưu dự đoán vào {PRED_PATH}")
