import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ==== Đường dẫn ====
DATA_PATH = "assets/processed/gold_prices_processed.csv"
SAVE_DIR = "src/models/data"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== Load và tạo features ====
df = pd.read_csv(DATA_PATH)
print(df.columns)
