import numpy as np
import pandas as pd


# Các hàm hỗ trợ
def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def force_index(df: pd.DataFrame, span: int = 1) -> pd.Series:
    fi = (df["close"] - df["close"].shift(1)) * df["volume"]
    return fi.ewm(span=span, adjust=False).mean() if span > 1 else fi


def vroc(series: pd.Series, window: int = 5) -> pd.Series:
    prev = series.shift(window)
    return (series - prev) / prev.replace(0, np.nan)


def range_and_return(df: pd.DataFrame) -> pd.DataFrame:
    price_range = df["high"] - df["low"]
    daily_return = df["close"].pct_change()
    return pd.DataFrame({"Range": price_range, "Return": daily_return})


def volume_momentum(series: pd.Series, window: int = 5) -> pd.Series:
    return series - series.shift(window)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, span_short: int = 12, span_long: int = 26, span_signal: int = 9
) -> pd.Series:
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line, signal_line


# Áp dụng các chỉ báo vào DataFrame
def custom_feature_func(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Tạo thêm các feature từ volume, giá và thời gian.
    """
    # ——————————————
    # 1. Feature giá
    # Đảm bảo 'date' là datetime
    df["date"] = pd.to_datetime(df["date"])
    df["Close_Lag1"] = df["close"].shift(1)
    df["Close_Change"] = df["close"].pct_change()
    df["Close_MA5"] = moving_average(df["close"], 5)

    df["Candle_Body"] = (df["close"] - df["open"]).abs()
    df["Upper_Shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["Lower_Shadow"] = df[["close", "open"]].min(axis=1) - df["low"]

    # ——————————————
    # 2. Feature volume cơ bản
    df["Volume_MA"] = moving_average(df["volume"], window)
    df["Volume_EMA"] = exponential_moving_average(df["volume"], window)
    df["Volume_STD"] = df["volume"].rolling(window=window).std()
    df["Volume_Momentum"] = volume_momentum(df["volume"], window)
    df["VROC"] = vroc(df["volume"], window=window)

    # ——————————————
    # 3. Chỉ báo khác từ volume và giá
    df["OBV"] = obv(df)
    df["Force_Index"] = force_index(df, span=window)

    rr = range_and_return(df)
    df["Range"] = rr["Range"]
    df["Return"] = rr["Return"]

    # ——————————————
    # 4. Feature lag và rolling trên volume
    df["Volume_Lag1"] = df["volume"].shift(1)
    df["Volume_Rolling_Max_30"] = df["volume"].rolling(window=30).max()
    df["Volume_Rolling_Mean_30"] = df["volume"].rolling(window=30).mean()
    df["Volume_Zscore_30"] = (df["volume"] - df["Volume_Rolling_Mean_30"]) / df[
        "volume"
    ].rolling(window=30).std()
    df["Volume_Percentile_30"] = (
        df["volume"]
        .rolling(window=30)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # ——————————————
    # 5. Feature thời gian
    df["Day_of_Week"] = df["date"].dt.dayofweek
    df["Is_Month_End"] = df["date"].dt.is_month_end.astype(int)
    df["Is_Month_Start"] = df["date"].dt.is_month_start.astype(int)

    # Thêm RSI
    df["RSI_14"] = rsi(df["close"], window=14)
    # Thêm MACD
    macd_line, signal_line = macd(df["close"])
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Gọi custom_feature_func để tạo toàn bộ feature.
    - Loại bỏ cột giá gốc, giữ lại date, volume và các feature mới.
    - Xóa NaN và sắp xếp theo date.
    """
    df = df.copy()
    df = custom_feature_func(df)
    # Không drop cột 'close' vì cần cho hiển thị Actual Price trên UI
    df.drop(["open", "high", "low"], axis=1, inplace=True, errors="ignore")
    df.dropna(inplace=True)
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    return df
