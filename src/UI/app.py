import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Đường dẫn dữ liệu
PROCESSED_PATH = "assets/processed/gold_prices_processed.csv"
PRED_PATH = "assets/predictions/gold_predictions.csv"
EDA_DESCRIBE_PATH = "src/modules/eda_results/describe.csv"

# --- 1. Trợ lý AI - Chatbot ---
from src.services.chatbot_logic import get_chatbot_response

st.set_page_config(page_title="Gold Price AI App", layout="wide")
st.title("💡 AI-powered Gold Price App")

st.header("1. Trợ lý AI - Chatbot")
with st.expander("Trò chuyện với AI về dữ liệu, biến, phân tích sơ bộ..."):
    user_q = st.text_input("Bạn hỏi gì về dữ liệu?", "Nguồn dữ liệu là gì?")
    if st.button("Gửi câu hỏi", key="chatbot"):
        st.write(get_chatbot_response(user_q))

# --- 2. Chọn ngày kết thúc ---
st.header("2. Chọn ngày kết thúc dự đoán")
processed = pd.read_csv(PROCESSED_PATH)
processed.columns = [c.lower() for c in processed.columns]
if "date" in processed.columns:
    processed["date"] = pd.to_datetime(processed["date"])

pred = pd.read_csv(PRED_PATH)
pred.columns = [c.lower() for c in pred.columns]
if "date" in pred.columns:
    pred["date"] = pd.to_datetime(pred["date"])

min_date = processed["date"].min()
max_date = processed["date"].max()
end_date = st.date_input(
    "Chọn ngày kết thúc",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY",
)

# --- 3. Dự đoán giá và khối lượng ngày mai ---
st.header("3. Dự đoán giá đóng cửa và khối lượng ngày mai")
last_row = pred[pred["date"] == pd.to_datetime(end_date)]
if not last_row.empty:
    st.metric("Giá đóng cửa dự đoán", f"{last_row['pred_close'].values[0]:,.2f}")
    st.metric("Khối lượng dự đoán", f"{int(last_row['pred_volume'].values[0]):,}")
else:
    st.warning("Không có dữ liệu dự đoán cho ngày này!")

# --- 4. Phân tích sơ bộ ---
st.header("4. Phân tích sơ bộ dữ liệu")
if os.path.exists(EDA_DESCRIBE_PATH):
    eda = pd.read_csv(EDA_DESCRIBE_PATH, index_col=0)
    st.dataframe(eda)
else:
    st.info("Chưa có file phân tích sơ bộ.")

# --- 5. Biểu đồ giá và khối lượng ---
st.header("5. Biểu đồ giá và khối lượng")
show_df = processed[processed["date"] <= pd.to_datetime(end_date)]
show_pred = pred[pred["date"] <= pd.to_datetime(end_date)]

# Biểu đồ Actual vs Predicted Close
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=show_df["date"], y=show_df["close"], mode="lines", name="Actual Close")
)
if "pred_close" in show_pred.columns:
    fig.add_trace(
        go.Scatter(
            x=show_pred["date"],
            y=show_pred["pred_close"],
            mode="lines",
            name="Predicted Close",
        )
    )
fig.update_layout(
    title="Actual vs Predicted Close Price",
    xaxis_title="Date",
    yaxis_title="Close Price",
)
st.plotly_chart(fig, use_container_width=True)

# Bar chart khối lượng
st.subheader("Bar chart khối lượng giao dịch")
bar_df = show_df.copy()
if "pred_volume" in show_pred.columns:
    bar_df = bar_df.merge(show_pred[["date", "pred_volume"]], on="date", how="left")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=bar_df["date"], y=bar_df["volume"], name="Actual Volume"))
if "pred_volume" in bar_df.columns:
    fig2.add_trace(
        go.Bar(x=bar_df["date"], y=bar_df["pred_volume"], name="Predicted Volume")
    )
fig2.update_layout(
    barmode="group",
    title="Actual vs Predicted Volume",
    xaxis_title="Date",
    yaxis_title="Volume",
)
st.plotly_chart(fig2, use_container_width=True)
