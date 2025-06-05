import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])

    # Lag features
    df["close_lag_1"] = df["close"].shift(1)
    df["volume_lag_1"] = df["volume"].shift(1)

    # Rolling stats
    df["rolling_mean_5"] = df["close"].rolling(5).mean()
    df["rolling_std_5"] = df["close"].rolling(5).std()

    # Momentum
    df["price_change"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_diff"] = df["macd"] - df["signal"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
