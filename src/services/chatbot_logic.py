import os
import sys

import google.generativeai as genai
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.configs.models import API_KEY, MODEL_NAME
from src.configs.path_dir import EDA_RESULT_DIR


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


def get_chatbot_response(user_query: str):
    """
    Generates a response to the user's query using the Gemini model.
    """
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
    # Cấu hình Google Generative AI
    # Kiểm tra kết nối API
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        return f"Không thể kết nối tới API Gemini. Vui lòng kiểm tra API_KEY và MODEL_NAME. Lỗi: {e}"

    try:
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"Xin lỗi, tôi đang gặp vấn đề khi tạo câu trả lời. Lỗi: {e}"


if __name__ == "__main__":
    # Example usage
    print("Chào mừng bạn đến với Chatbot phân tích giá vàng!")
    while True:
        user_input = input("\nBạn hỏi gì? (gõ 'thoat' để thoát): ")
        if user_input.lower() == "thoat":
            break

        response = get_chatbot_response(user_input)
        print(f"Chatbot: {response}")
