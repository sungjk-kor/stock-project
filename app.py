from __future__ import annotations

import streamlit as st

from utils.preprocessing import load_processed_dataset


st.set_page_config(
    page_title="Stock Index Forecast",
    page_icon="📈",
    layout="wide",
)

# ── 일관된 스타일 (data_view.py, 3_forecast.py와 통일) ──────────────────────
ACCENT = "#f97316"
st.markdown(
    f"""
    <style>
    .home-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 24px 0;
    }}
    .home-kpi {{
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
    }}
    .home-kpi-label {{
        color: #64748b;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .home-kpi-value {{
        color: #111827;
        font-size: 20px;
        font-weight: 700;
        margin-top: 8px;
    }}
    .home-note {{
        border-left: 4px solid {ACCENT};
        background: #fff7ed;
        padding: 12px 14px;
        border-radius: 6px;
        color: #475569;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 16px;
    }}
    @media (max-width: 900px) {{
        .home-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 주가지수 중기 추세 예측")
st.caption("시장, 금리, 거시, 유동성 변수를 활용한 SARIMAX 기반 S&P500 주간 추세 예측")

# ── 사이드바 메뉴 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗂️ 프로젝트 메뉴")
    st.page_link("app.py", label="📊 홈")
    st.page_link("pages/1_data_view.py", label="📥 데이터 확인")
    st.page_link("pages/3_forecast.py", label="🔮 예측")
    st.page_link("pages/4_sector_etf_monitor.py", label="📈 섹터 ETF 모니터링")
    st.page_link("pages/5_similar_periods.py", label="🔍 유사 시점 탐색기")

dataset = load_processed_dataset()

# ── KPI 카드 그리드 (스타일 통일) ────────────────────────────────────────────
kpi_cards = f"""
<div class="home-kpi-grid">
    <div class="home-kpi">
        <div class="home-kpi-label">목표 변수</div>
        <div class="home-kpi-value">S&P500 주간 로그수익률</div>
    </div>
    <div class="home-kpi">
        <div class="home-kpi-label">모델</div>
        <div class="home-kpi-value">SARIMAX</div>
    </div>
    <div class="home-kpi">
        <div class="home-kpi-label">평가 지표</div>
        <div class="home-kpi-value">RMSE / MAPE</div>
    </div>
</div>
"""
st.markdown(kpi_cards, unsafe_allow_html=True)

st.markdown(
    '<div class="home-note">외생변수의 흐름을 확인하고, 과거 주가지수와의 관계를 학습해 중기 시장 추세를 예측합니다.</div>',
    unsafe_allow_html=True,
)

st.markdown("### 📋 사용 흐름")
st.markdown(
    """
    아래 순서대로 진행하면 됩니다. 먼저 데이터를 수집하고, 그래프로 이상 여부와 상관관계를 확인한 다음, 최종 데이터셋을 만들어 SARIMAX 예측을 실행합니다.
    """
)

st.markdown(
    """
```mermaid
flowchart LR
    A["1. 데이터 확인 페이지"] --> B["기간 선택"]
    B --> C["원천 데이터 수집"]
    C --> D["최종 데이터셋 생성"]
    D --> E["2. 예측 페이지"]
    E --> F["최종 데이터셋 그래프 확인"]
    F --> G["시장 국면 진단 또는 SARIMAX 실행"]
```
    """
)

st.table(
    {
        "단계": ["1", "2", "3", "4", "5"],
        "메뉴": ["데이터 확인", "데이터 확인", "예측", "예측", "섹터 ETF 모니터링"],
        "할 일": [
            "시작일/종료일 선택",
            "`전체 파이프라인 실행`으로 수집 및 최종 데이터셋 생성",
            "최종 데이터셋 그래프로 입력 변수 확인",
            "시장 국면 진단 또는 SARIMAX 선택",
            "섹터 ETF 흐름과 이벤트 점검",
        ],
        "결과": [
            "분석 기간 확정",
            "`data/raw/*.csv`, `data/processed/final_dataset.csv` 생성",
            "누락/이상치 확인",
            "외생변수 후보 선정",
            "섹터별 수익률 및 이벤트 코멘트 확인",
        ],
    }
)

st.markdown("### 🔧 페이지별 사용법")
with st.expander("**1️⃣ 데이터 확인** — 시작일/종료일 선택 및 원천 데이터 수집", expanded=False):
    st.markdown(
        """
        - 시작일과 종료일을 고른 뒤 `전체 파이프라인 실행` 버튼을 클릭합니다.
        - 원천 CSV 파일(`data/raw/*.csv`)과 최종 데이터셋(`data/processed/final_dataset.csv`)이 생성됩니다.
        - 각 데이터 소스의 커버리지, 누락값, 수집 상태를 확인할 수 있습니다.
        """
    )

with st.expander("**2️⃣ 예측** — 최종 데이터셋 검토 및 모델 선택", expanded=False):
    st.markdown(
        """
        - 상단 최종 데이터셋 그래프로 입력 변수의 흐름과 누락값을 확인합니다.
        - 사이드바의 라디오 버튼에서 **시장 국면 진단** 또는 **SARIMAX**를 선택합니다.
        """
    )

with st.expander("**3️⃣ SARIMAX 모델** — 시계열 예측 실행", expanded=False):
    st.markdown(
        """
        - 예측 기간(horizon), 타겟 변수, ARIMA 차수(p, d, q)를 설정합니다.
        - 외생변수(exogenous variables)를 선택합니다.
        - **🚀 Train and forecast** 버튼으로 학습을 실행합니다.
        """
    )

with st.expander("**4️⃣ 섹터 ETF 모니터링** — 주요 섹터별 추세 확인", expanded=False):
    st.markdown(
        """
        - 주요 섹터 ETF(에너지, 금융, 경기소비, 필수소비, 산업재, 기술, 통신, 유틸리티, 부동산, 의료)를 모니터링합니다.
        - 주간 수익률, 이벤트 코멘트, 이동평균을 확인합니다.
        """
    )

with st.expander("📝 데이터 입력 메모"):
    st.markdown(
        """
        - **FRED 데이터**: `FRED_API_KEY` 환경변수 설정 필요 (금리, CPI, 실업률, M2 등)
        - **ISM PMI**: yfinance 미지원 → `data/raw/ism_pmi.csv`에 직접 입력
        - **ISM 대체 지표**: `XLI`(산업재 ETF), `^SOX`(필라델피아 반도체)의 주간 로그수익률 수집
        - **이벤트 변수**: `data/raw/events.csv`에 날짜/종류 추가 → 자동 주간화
        - **이벤트 종류**: `fomc`, `cpi_release`, `employment_release`, `bigtech_earnings`
        - **섹터 ETF**: 별도 경로 `data/raw/sector_etf_weekly.csv` 저장
        """
    )

st.markdown("### 📊 현재 데이터 상태")
if dataset.empty:
    st.warning("⚠️ 아직 `data/processed/final_dataset.csv`가 없습니다. 데이터 확인 페이지에서 수집을 진행하세요.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 행 개수", f"{len(dataset):,}")
    with col2:
        st.metric("📋 컬럼 개수", f"{len(dataset.columns)}")
    with col3:
        st.metric("⏱️ 최신 날짜", str(dataset["date"].max().date()) if "date" in dataset.columns else "—")

    st.dataframe(dataset.tail(10), use_container_width=True, hide_index=True)

st.markdown("### 📚 입력 변수 분류")
st.table(
    {
        "그룹": ["시장", "금리", "거시", "유동성/달러", "이벤트"],
        "예시": [
            "S&P500, NASDAQ, DOW, VIX, XLI, SOX, 거래량",
            "Fed Funds, 2Y, 10Y, 장단기 금리차",
            "CPI, 실업률, 비농업고용, ISM PMI",
            "DXY, M2, 금융상황지수, 원유, 금",
            "FOMC, CPI 발표, 고용지표, 실적발표",
        ],
        "사용 방식": ["lag 1~12주", "lag 1~12주", "발표일 기준 주간화", "lag 적용", "더미/강도 변수"],
    }
)
