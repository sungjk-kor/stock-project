from __future__ import annotations

import streamlit as st

from utils.preprocessing import load_processed_dataset


st.set_page_config(
    page_title="Stock Index Forecast",
    page_icon="📈",
    layout="wide",
)


st.title("주가지수 중기 추세 예측")
st.caption("시장, 금리, 거시, 유동성 변수를 활용한 SARIMAX 기반 S&P500 주간 추세 예측")

st.sidebar.header("프로젝트 메뉴")
st.sidebar.page_link("app.py", label="홈")
st.sidebar.page_link("pages/1_data_view.py", label="데이터 확인")
st.sidebar.page_link("pages/2_macro_view.py", label="거시 변수")
st.sidebar.page_link("pages/3_forecast.py", label="예측")

dataset = load_processed_dataset()

col1, col2, col3 = st.columns(3)
col1.metric("목표 변수", "S&P500 주간 로그수익률")
col2.metric("모델", "SARIMAX")
col3.metric("평가 지표", "RMSE / MAPE / 방향성")

st.subheader("프로젝트 목표")
st.write(
    """
    외생변수의 흐름을 확인하고, 과거 주가지수와의 관계를 학습해 중기 시장 추세를 예측합니다.
    현재 앱 골격은 CSV 데이터 확인, 변수 탐색, SARIMAX 예측 화면으로 구성되어 있습니다.
    """
)

st.subheader("현재 데이터 상태")
if dataset.empty:
    st.info("아직 `data/processed/final_dataset.csv`가 없습니다. 데이터 수집 후 파일을 추가하면 화면에 자동 반영됩니다.")
else:
    st.dataframe(dataset.tail(10), use_container_width=True)

st.subheader("입력 변수 그룹")
st.table(
    {
        "그룹": ["시장", "금리", "거시", "유동성/달러", "이벤트"],
        "예시": [
            "S&P500, NASDAQ, DOW, VIX, 거래량",
            "Fed Funds, 2Y, 10Y, 장단기 금리차",
            "CPI, 실업률, 비농업고용, ISM PMI",
            "DXY, M2, 금융상황지수, 원유, 금",
            "FOMC, CPI 발표, 고용지표, 실적발표",
        ],
        "사용 방식": ["lag 1~12주", "lag 1~12주", "발표일 기준 주간화", "lag 적용", "더미/강도 변수"],
    }
)
