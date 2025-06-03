def calculate_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def enrich_features(df):
    df = calculate_rsi(df)
    df = calculate_macd(df)
    return df
