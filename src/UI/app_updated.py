import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.configs.path_dir import EDA_DESCRIBE_PATH, PRED_PATH, RAW_DATA_PATH
from src.services.chatbot_logic import get_chatbot_response

# --- Cấu hình session state ---
if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False

# --- Cấu hình Streamlit ---
st.set_page_config(page_title="LAB 01", layout="wide")
st.title("Stock Market Prediction App")

# --- Sidebar: Cấu hình ---
st.sidebar.header("Cấu hình phân tích")
asset_type = st.sidebar.selectbox("Chọn loại tài sản", ["Gold"], index=0)

try:
    processed = pd.read_csv(RAW_DATA_PATH)
    pred = pd.read_csv(PRED_PATH)
    eda = (
        pd.read_csv(EDA_DESCRIBE_PATH, index_col=0)
        if os.path.exists(EDA_DESCRIBE_PATH)
        else None
    )

    processed.columns = [c.lower() for c in processed.columns]
    pred.columns = [c.lower() for c in pred.columns]
    processed["date"] = pd.to_datetime(processed["date"])
    pred["date"] = pd.to_datetime(pred["date"])

    min_date = processed["date"].min()
    max_date = processed["date"].max()
    # max_date = pd.to_datetime("today")  # Lấy ngày hiện tại làm ngày kết thúc
    selected_date = st.sidebar.date_input(
        "Chọn ngày kết thúc",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )

    if st.sidebar.button("Bắt đầu phân tích"):
        st.session_state["analysis_ready"] = True

except Exception as e:
    st.error(f"Lỗi đọc dữ liệu: {e}")
    st.stop()

# --- Main Content ---
if st.session_state["analysis_ready"]:
    st.success(
        f"Phân tích dữ liệu {asset_type} đến ngày {selected_date.strftime('%d/%m/%Y')}"
    )

    # --- 1. Chatbot ---
    st.subheader("Trợ lý AI - Chatbot")
    with st.expander("Hỏi trợ lý AI về dữ liệu", expanded=True):
        user_q = st.text_input("Bạn muốn hỏi gì?", "Nguồn dữ liệu là gì?")
        if st.button("Gửi", key="chatbot_send"):
            st.info(get_chatbot_response(user_q))

    # --- 2. Dự báo ---
    st.subheader("Dự báo giá và khối lượng ngày mai")
    latest_row = pred[pred["date"] == pd.to_datetime(selected_date)]
    if not latest_row.empty:
        st.metric("Giá đóng cửa dự báo", f"{latest_row['pred_close'].values[0]:,.2f}")
        st.metric("Khối lượng dự báo", f"{int(latest_row['pred_volume'].values[0]):,}")
    else:
        st.warning("Không có dữ liệu dự báo cho ngày này!")

    # --- 3. Phân tích sơ bộ ---
    st.subheader("Phân tích sơ bộ dữ liệu")
    if eda is not None:
        st.dataframe(eda, use_container_width=True)
    else:
        st.info("Chưa có kết quả phân tích sơ bộ.")

    # --- 4. Biểu đồ ---
    st.subheader("Biểu đồ giá và khối lượng")
    show_df = processed[processed["date"] <= pd.to_datetime(selected_date)]
    show_pred = pred[pred["date"] <= pd.to_datetime(selected_date)]

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=show_df["date"], y=show_df["close"], mode="lines", name="Actual Close"
        )
    )
    if "pred_close" in show_pred.columns:
        fig1.add_trace(
            go.Scatter(
                x=show_pred["date"],
                y=show_pred["pred_close"],
                mode="lines",
                name="Predicted Close",
            )
        )
    fig1.update_layout(
        title="Actual vs Predicted Close Price",
        xaxis_title="Date",
        yaxis_title="Close Price",
    )
    st.plotly_chart(fig1, use_container_width=True)

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
else:
    st.info(
        "💡 Vui lòng chọn loại tài sản, ngày kết thúc và nhấn **Phân tích dữ liệu** để bắt đầu."
    )
