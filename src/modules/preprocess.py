import pandas as pd


def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df.dropna(inplace=True)
    numeric_cols = [
        col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns
    ]

    # Outlier removal using IQR
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df
