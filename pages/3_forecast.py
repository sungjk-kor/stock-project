from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.metrics import direction_accuracy, mape, rmse
from utils.model_sarimax import fit_sarimax, forecast_sarimax, train_test_split_by_ratio
from utils.preprocessing import load_processed_dataset


st.set_page_config(page_title="예측", layout="wide")
st.title("SARIMAX 예측")

df = load_processed_dataset()
if df.empty:
    st.info("먼저 `data/processed/final_dataset.csv`를 준비해주세요.")
    st.stop()

if "sp500_log_return" not in df.columns:
    st.error("최종 데이터셋에 `sp500_log_return` 컬럼이 필요합니다.")
    st.stop()

df = df.dropna().reset_index(drop=True)
numeric_cols = [col for col in df.select_dtypes("number").columns if col != "sp500_log_return"]

with st.sidebar:
    st.header("모델 설정")
    train_ratio = st.slider("학습 데이터 비율", 0.5, 0.9, 0.8, 0.05)
    p = st.number_input("AR(p)", min_value=0, max_value=5, value=1)
    d = st.number_input("차분(d)", min_value=0, max_value=2, value=0)
    q = st.number_input("MA(q)", min_value=0, max_value=5, value=1)
    exog_cols = st.multiselect("외생변수", numeric_cols, default=numeric_cols[:5])

run = st.button("학습 및 백테스트", type="primary")

if run:
    train, test = train_test_split_by_ratio(df, train_ratio=train_ratio)
    model = fit_sarimax(train, exog_cols=exog_cols, order=(int(p), int(d), int(q)))
    future_exog = test[exog_cols] if exog_cols else None
    pred = forecast_sarimax(model, steps=len(test), future_exog=future_exog)
    pred = pd.Series(pred.to_numpy(), index=test.index, name="prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse(test['sp500_log_return'], pred):.4f}")
    col2.metric("MAPE", f"{mape(test['sp500_log_return'], pred):.2f}%")
    col3.metric("방향성 정확도", f"{direction_accuracy(test['sp500_log_return'], pred):.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["date"], y=test["sp500_log_return"], name="actual"))
    fig.add_trace(go.Scatter(x=test["date"], y=pred, name="prediction"))
    fig.update_layout(xaxis_title="date", yaxis_title="weekly log return")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("사이드바에서 설정을 확인한 뒤 `학습 및 백테스트`를 누르세요.")
