import os

import pandas as pd
import yfinance as yf


def collect_gold_data(
    ticker: str = "GC=F",
    start: str = "2000-01-01",
    end: str = None,
    save_path: str = "assets/raw/gold_prices.csv",
):
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    # Tải dữ liệu từ Yahoo Finance
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    # Đổi tên cột cho đúng yêu cầu
    df = df.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )
    # Chỉ giữ các cột cần thiết
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Lưu file csv
    df.to_csv(save_path, index=False)
    print(f"Đã lưu dữ liệu vào {save_path}")


if __name__ == "__main__":
    collect_gold_data()
