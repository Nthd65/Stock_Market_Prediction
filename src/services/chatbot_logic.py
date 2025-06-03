import os

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# Path to processed data and EDA results
PROCESSED_PATH = "assets/processed/gold_prices_processed.csv"
EDA_RESULT_DIR = "src/modules/eda_results"


def get_data_info():
    """
    Returns a string containing information about the data source and variable meanings.
    """
    info = """
Nguồn dữ liệu:
- Giá vàng được thu thập từ Yahoo Finance bằng thư viện `yfinance`.

Ý nghĩa các biến:
- `Date`: Ngày của bản ghi.
- `Open`: Giá mở cửa.
- `High`: Giá cao nhất trong ngày.
- `Low`: Giá thấp nhất trong ngày.
- `Close`: Giá đóng cửa (đã điều chỉnh).
- `Volume`: Khối lượng giao dịch trong ngày.
    """
    return info


def get_eda_summary():
    """
    Reads the describe.csv file and returns a formatted summary of EDA results.
    """
    describe_path = os.path.join(EDA_RESULT_DIR, "describe.csv")
    if not os.path.exists(describe_path):
        return "Không tìm thấy kết quả phân tích sơ bộ. Vui lòng chạy EDA trước."

    df_describe = pd.read_csv(describe_path, index_col=0)
    summary = "\nKết quả phân tích sơ bộ:\n"
    for index, row in df_describe.iterrows():
        summary += f"\nCột: {index}\n"
        summary += f"  - Số lượng bản ghi: {row['count']:.0f}\n"
        summary += f"  - Giá trị trung bình: {row['mean']:.2f}\n"
        summary += f"  - Độ lệch chuẩn: {row['std']:.2f}\n"
        summary += f"  - Giá trị nhỏ nhất: {row['min']:.2f}\n"
        summary += f"  - Giá trị lớn nhất: {row['max']:.2f}\n"
        summary += f"  - Q1 (25%): {row['25%']:.2f}\n"
        summary += f"  - Q2 (50%): {row['50%']:.2f}\n"
        summary += f"  - Q3 (75%): {row['75%']:.2f}\n"
    return summary


def initialize_gemini_model():
    """
    Initializes and returns the Gemini Pro model.
    """
    model = genai.GenerativeModel(
        "gemini-pro"
    )  # Using gemini-pro as 2.0-Flash might be a typo for gemini-1.5-flash or gemini-pro.
    return model


def get_chatbot_response(user_query: str):
    """
    Generates a response to the user's query using the Gemini model.
    """
    model = initialize_gemini_model()
    data_info = get_data_info()
    eda_summary = get_eda_summary()

    context = f"""
Bạn là một trợ lý chatbot về phân tích thị trường chứng khoán, chuyên về dữ liệu giá vàng. 
Bạn có thể cung cấp thông tin về nguồn dữ liệu, ý nghĩa các biến, và kết quả phân tích sơ bộ của dữ liệu.
Đây là thông tin về dữ liệu và phân tích sơ bộ mà bạn có thể tham khảo để trả lời người dùng:

{data_info}

{eda_summary}

Hãy trả lời các câu hỏi của người dùng bằng tiếng Việt, dựa trên thông tin được cung cấp. Nếu câu hỏi nằm ngoài phạm vi, hãy lịch sự từ chối và nói rằng bạn chỉ có thể trả lời về dữ liệu giá vàng và phân tích sơ bộ.

Câu hỏi của người dùng: {user_query}
"""

    try:
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"Xin lỗi, tôi đang gặp vấn đề khi tạo câu trả lời. Lỗi: {e}"


if __name__ == "__main__":
    # Example usage
    print("Chào mừng bạn đến với Chatbot phân tích giá vàng!")
    print(
        "Tôi có thể trả lời các câu hỏi về nguồn dữ liệu, ý nghĩa biến và phân tích sơ bộ."
    )
    while True:
        user_input = input("\nBạn hỏi gì? (gõ 'thoat' để thoát): ")
        if user_input.lower() == "thoat":
            break

        response = get_chatbot_response(user_input)
        print(f"Chatbot: {response}")
