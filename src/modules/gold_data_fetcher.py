from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_gold_data(ticker="GC=F", period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df


def save_data_to_csv(df: pd.DataFrame, filename: str):
    output_path = Path("assets") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
