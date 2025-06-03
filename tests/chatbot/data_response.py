data_source_info = """
Dữ liệu được thu thập từ Yahoo Finance thông qua thư viện yfinance.
Nguồn dữ liệu là giá vàng theo ngày, bao gồm các trường: Open, High, Low, Close, Volume.
"""

feature_description = """
Các biến trong tập dữ liệu:
- Date: Ngày giao dịch
- Open: Giá mở cửa
- High: Giá cao nhất trong ngày
- Low: Giá thấp nhất trong ngày
- Close: Giá đóng cửa
- Volume: Khối lượng giao dịch trong ngày
- RSI, MACD: Các chỉ số kỹ thuật được tính toán để hỗ trợ dự đoán
"""

summary_analysis = """
Phân tích sơ bộ dữ liệu cho thấy:
- Khối lượng giao dịch có xu hướng tăng nhẹ vào các ngày đầu tuần.
- Giá vàng có sự biến động nhẹ, không có đột biến lớn.
- RSI và MACD có sự dao động rõ rệt, hỗ trợ tốt cho các mô hình ML.
"""

# Hàm gộp để gọi từ custom action hoặc Streamlit


def get_chatbot_answer(intent: str) -> str:
    if intent == "hoi_nguon":
        return data_source_info
    elif intent == "hoi_bien":
        return feature_description
    elif intent == "phan_tich":
        return summary_analysis
    else:
        return "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể hỏi về nguồn dữ liệu, ý nghĩa các biến, hoặc phân tích sơ bộ."
