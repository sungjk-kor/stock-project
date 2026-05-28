from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils.preprocessing import load_processed_dataset


st.set_page_config(page_title="거시 변수", layout="wide")
st.title("거시 변수 탐색")

df = load_processed_dataset()
if df.empty:
    st.info("먼저 `data/processed/final_dataset.csv`를 준비해주세요.")
    st.stop()

numeric_cols = [col for col in df.select_dtypes("number").columns if col != "sp500_log_return"]
selected = st.multiselect("표시할 변수", numeric_cols, default=numeric_cols[:4])

if selected:
    chart_df = df[["date", *selected]].melt(id_vars="date", var_name="variable", value_name="value")
    st.plotly_chart(px.line(chart_df, x="date", y="value", color="variable"), use_container_width=True)

st.subheader("상관관계")
corr_cols = ["sp500_log_return", *selected] if "sp500_log_return" in df.columns else selected
if len(corr_cols) >= 2:
    corr = df[corr_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto"), use_container_width=True)
else:
    st.info("상관관계를 보려면 두 개 이상의 숫자형 변수가 필요합니다.")
