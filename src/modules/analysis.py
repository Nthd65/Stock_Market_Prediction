import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROCESSED_PATH = "assets/processed/gold_prices_processed.csv"
EDA_RESULT_DIR = "src/modules/eda_results"


def save_describe(df):
    desc = df.describe().T
    os.makedirs(EDA_RESULT_DIR, exist_ok=True)
    desc_path = os.path.join(EDA_RESULT_DIR, "describe.csv")
    desc.to_csv(desc_path)
    print(f"Đã lưu thống kê mô tả tại {desc_path}")
    return desc


def plot_distributions(df):
    os.makedirs(EDA_RESULT_DIR, exist_ok=True)
    for col in df.columns:
        if col == "Date":
            continue
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Phân phối của {col}")
        plt.xlabel(col)
        plt.ylabel("Tần suất")
        img_path = os.path.join(EDA_RESULT_DIR, f"dist_{col}.png")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        print(f"Đã lưu biểu đồ phân phối: {img_path}")


def main():
    df = pd.read_csv(PROCESSED_PATH)
    print("\n===== Thống kê mô tả =====")
    desc = save_describe(df)
    print(desc)
    print("\n===== Phân tích phân phối các cột =====")
    plot_distributions(df)


if __name__ == "__main__":
    main()
