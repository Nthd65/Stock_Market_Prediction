from fontTools.misc.plistlib import end_date

from Model_Predict.Features import custom_feature_func, preprocess_data
from datetime import datetime, timedelta

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def pipeline(trained_model_file):
    # Load pipeline tiền xử lý
    pipe_path = os.path.join(base_dir, "scaler_pipeline.pkl")
    with open(pipe_path, "rb") as pipe_file:
        preprocess_pipe = pickle.load(pipe_file)

    # Load mô hình
    model = os.path.join(base_dir,trained_model_file)
    if os.path.splitext(trained_model_file)[1] == '.pkl':
        with open(model, "rb") as f:
            trained_model = pickle.load(f)
    else:
        trained_model = XGBRegressor()
        trained_model.load_model(model)

    # Gán model mới vào pipeline
    preprocess_pipe.set_params(model=trained_model)
    return preprocess_pipe


def get_data(start: str, end: str, warning=True):
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    end_plus_1 = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download("GC=F", start=start, end=end_plus_1)
    df = df.reset_index()

    df.columns = [col[0] for col in df.columns]
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])

    df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
    df = df.sort_values(by='date', ascending=True)

    last_date = df['date'].iloc[-1].date()
    if warning:
        if last_date != end_dt:
            print(f"***CẢNH BÁO: Ngày cuối cùng trong data lấy được là {last_date}, KHÔNG trùng với ngày yêu cầu ({end_dt})***")
    return df


def predict_volume(model_file, date_target: str, history_window: int = 30, timedelta_days: int = 60):
    """
    Dự đoán volume cho đúng một ngày `date_target`.
    - Nếu `date_target` <= ngày cuối có dữ liệu thực, sẽ sử dụng data tới và bao gồm `date_target` để tính feature, rồi predict và so với giá trị thật.
    - Nếu `date_target` > ngày cuối có dữ liệu thực, sẽ dự đoán tuần tự từ ngày kế tiếp của dữ liệu thực đến `date_target`, mỗi bước lặp dùng volume dự đoán từ ngày trước để xây feature, rồi chỉ trả kết quả cuối cùng cho `date_target`.
    Trả về dict:
        {
            'date': 'YYYY-MM-DD',
            'predicted_volume': int,
            # nếu target ở quá khứ (có giá trị thật):
            'real_volume': int,
            'residual': int   # real_volume - predicted_volume
        }
    """

    date_target_dt = datetime.strptime(date_target, "%Y-%m-%d").date()
    start_date = (date_target_dt - timedelta(days=timedelta_days)).strftime("%Y-%m-%d")
    end_date = date_target_dt.strftime("%Y-%m-%d")

    df_raw = get_data(start=start_date, end=end_date)
    # Pipeline đã load cả scaler lẫn model
    pipeline_model = pipeline(model_file)

    df_raw = df_raw.sort_values(by="date", ascending=True).reset_index(drop=True)

    if len(df_raw) < history_window:
        raise ValueError("Không đủ dữ liệu thô để tính window khởi tạo.")

    # Xác định ngày cuối có dữ liệu thực
    last_real_date = df_raw["date"].max().date()
    today = datetime.today().date()

    if date_target_dt <= last_real_date:
        window_raw = df_raw[df_raw["date"] <= pd.Timestamp(date_target_dt)].tail(history_window)
        if len(window_raw) < history_window:
            raise ValueError(f"Không đủ {history_window} ngày raw trước và tính cả {date_target} để dự đoán.")

        window_df = preprocess_data(window_raw.copy())
        X = window_df.drop(["date", "volume"], axis=1).iloc[[-1]]
        pred_volume = int(pipeline_model.predict(X)[0])


        actual_row = df_raw[df_raw["date"] == pd.Timestamp(date_target_dt)]
        if actual_row.empty:
            raise ValueError(f"Data raw ngày {date_target} không tồn tại để so sánh.")
        actual_volume = int(actual_row["volume"].values[0])
        return {
            "date": date_target_dt.strftime("%Y-%m-%d"),
            "predicted_volume": pred_volume,
            "real_volume": actual_volume,
            "residual": actual_volume - pred_volume
        }

    df_work = df_raw.copy().reset_index(drop=True)
    current_date = last_real_date + timedelta(days=1)

    # Kiểm tra đủ dữ liệu raw để lần đầu xây cửa sổ:
    if len(df_work) < history_window:
        raise ValueError("Không đủ dữ liệu thô để khởi tạo cửa sổ dự đoán tương lai.")

    pred_volume = None
    while current_date <= date_target_dt:
        window_raw = df_work[df_work["date"] < pd.Timestamp(current_date)].tail(history_window)
        if len(window_raw) < history_window:
            raise ValueError(f"Ngày {current_date} thiếu dữ liệu để build window.")
        # Tính feature trên window_raw
        window_df = preprocess_data(window_raw.copy())
        X = window_df.drop(["date", "volume"], axis=1).iloc[[-1]]
        pred_volume = int(pipeline_model.predict(X)[0])

        # Nếu cần tiếp tục lặp (chưa tới target), ta thêm hàng giả lập raw cho current_date
        if current_date < date_target_dt:
            last_raw_row = window_raw.iloc[-1].copy()
            new_raw = {
                "date": pd.Timestamp(current_date),
                "close": last_raw_row["close"],
                "high": last_raw_row["high"],
                "low": last_raw_row["low"],
                "open": last_raw_row["open"],
                "volume": pred_volume
            }
            df_work = pd.concat([df_work, pd.DataFrame([new_raw])], ignore_index=True)
        else:
            break

        current_date += timedelta(days=1)
    return {
        "date": date_target_dt.strftime("%Y-%m-%d"),
        "predicted_volume": pred_volume
    }
