import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "assets/raw/gold_prices.csv"
PROCESSED_PATH = "assets/processed/gold_prices_processed.csv"


def remove_missing(df):
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    print(
        f"[Missing] Số bản ghi trước: {before}, sau: {after}, đã loại: {before - after}"
    )
    return df_clean, before, after


def remove_outliers_iqr(df, cols):
    before = len(df)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    after = len(df)
    print(
        f"[Outlier] Số bản ghi trước: {before}, sau: {after}, đã loại: {before - after}"
    )
    return df, before, after


def standardize(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def main():
    # Đọc dữ liệu thô
    df = pd.read_csv(RAW_PATH)
    # Bỏ dòng tên mã nếu có
    if df.iloc[0].isnull().sum() > 0 or (df.iloc[0] == df.columns[1]).all():
        df = df.iloc[1:].reset_index(drop=True)
    # Đảm bảo đúng kiểu dữ liệu
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Bỏ missing values
    df, n_before, n_after = remove_missing(df)
    # Xử lý outlier bằng IQR
    df, n_before2, n_after2 = remove_outliers_iqr(
        df, ["Open", "High", "Low", "Close", "Volume"]
    )
    # Chuẩn hóa
    df = standardize(df, ["Open", "High", "Low", "Close", "Volume"])
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    # Lưu file
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Đã lưu dữ liệu đã xử lý vào {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
