from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.sector_etf_data import (
    SECTOR_ETF_DATA_PATH,
    SECTOR_ETFS,
    SECTOR_ETFS_GROUP2,
    add_sector_event_features,
    collect_sector_etf_weekly,
    detect_sector_etf_events,
    load_sector_etf_weekly,
)


st.set_page_config(page_title="섹터 ETF 모니터링", layout="wide")
st.title("섹터 ETF 모니터링")
st.caption("미국 주요 섹터 ETF의 주간 가격·거래량 흐름과 주목 이벤트를 추적합니다.")

BASE_DATE = pd.Timestamp("2020-01-03")
ETF_COLORS = [
    "#2563eb",
    "#16a34a",
    "#7c3aed",
    "#f97316",
    "#0891b2",
    "#ca8a04",
    "#0f766e",
    "#9333ea",
    "#4f46e5",
    "#65a30d",
    "#ea580c",
]

st.markdown(
    f"""
    <style>
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: #f97316 !important;
        border-color: #f97316 !important;
        box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.16) !important;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"]:focus,
    .stSlider [data-baseweb="slider"] div[role="slider"]:hover {{
        box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.22) !important;
    }}
    .stSlider [data-baseweb="slider"] > div > div {{
        background-color: #fed7aa !important;
    }}
    .stSlider [data-baseweb="slider"] > div > div > div {{
        background-color: #f97316 !important;
    }}
    .stSlider [data-testid="stThumbValue"],
    .stSlider [data-testid="stTickBar"] * {{
        color: #f97316 !important;
    }}
    [data-baseweb="select"] [data-baseweb="tag"] {{
        color: #ffffff !important;
        border: 0 !important;
    }}
    [data-baseweb="select"] [data-baseweb="tag"] svg {{
        fill: #ffffff !important;
    }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(1),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(1) {{ background-color: {ETF_COLORS[0]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(2),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(2) {{ background-color: {ETF_COLORS[1]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(3),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(3) {{ background-color: {ETF_COLORS[2]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(4),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(4) {{ background-color: {ETF_COLORS[3]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(5),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(5) {{ background-color: {ETF_COLORS[4]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(6),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(6) {{ background-color: {ETF_COLORS[5]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(7),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(7) {{ background-color: {ETF_COLORS[6]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(8),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(8) {{ background-color: {ETF_COLORS[7]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(9),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(9) {{ background-color: {ETF_COLORS[8]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(10),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(10) {{ background-color: {ETF_COLORS[9]} !important; }}
    [data-baseweb="select"] [data-baseweb="tag"]:nth-of-type(11),
    [data-baseweb="select"] [data-baseweb="tag"]:nth-child(11) {{ background-color: {ETF_COLORS[10]} !important; }}
    .sector-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px 0;
    }}
    .sector-kpi {{
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }}
    .sector-kpi-label {{
        color: #475569;
        font-size: 14px;
        line-height: 1.25;
        white-space: nowrap;
    }}
    .sector-kpi-value {{
        color: #111827;
        font-size: 18px;
        font-weight: 700;
        line-height: 1.25;
        margin-top: 6px;
        overflow-wrap: anywhere;
    }}
    @media (max-width: 900px) {{
        .sector-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_pct(value: float | None) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values("date").copy()
    direction = result["close"].diff().fillna(0).apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    result["obv"] = (direction * result["volume"]).cumsum()

    delta = result["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.mask(loss == 0)
    result["rsi_14w"] = 100 - (100 / (1 + rs))
    result.loc[(loss == 0) & (gain > 0), "rsi_14w"] = 100
    result.loc[(loss == 0) & (gain == 0), "rsi_14w"] = 50
    return result


def _make_cumulative_return_chart(
    df: pd.DataFrame,
    tickers: list[str],
    chart_start: pd.Timestamp,
    chart_end: pd.Timestamp,
) -> go.Figure:
    source_df = df[df["ticker"].isin(tickers)].copy()
    chart_df = source_df[(source_df["date"] >= chart_start) & (source_df["date"] <= chart_end)].copy()
    fig = go.Figure()
    ticker_color_map = {
        ticker: ETF_COLORS[index % len(ETF_COLORS)]
        for index, ticker in enumerate(sorted(tickers))
    }

    for ticker, full_part in source_df.groupby("ticker"):
        full_part = full_part.sort_values("date").copy()
        base_candidates = full_part[full_part["date"] >= BASE_DATE]
        if base_candidates.empty:
            continue
        base_close = float(base_candidates.iloc[0]["close"])
        if base_close == 0:
            continue

        part = chart_df[chart_df["ticker"] == ticker].sort_values("date").copy()
        if part.empty:
            continue

        sector = part["sector"].iloc[-1]
        part["cumulative_return"] = (part["close"] / base_close - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=part["date"],
                y=part["cumulative_return"],
                mode="lines",
                name=f"{ticker} {sector}",
                line=dict(color=ticker_color_map[ticker], width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>누적상승률 %{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        height=520,
        hovermode="x unified",
        xaxis_title="날짜",
        yaxis_title="2020-01-03 기준 누적상승률 (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.add_hline(y=0, line_width=1, line_color="gray", line_dash="dot")
    return fig


def _make_event_detail_chart(df: pd.DataFrame, event_row: pd.Series) -> go.Figure:
    ticker = event_row["ticker"]
    event_date = pd.to_datetime(event_row["date"])
    part = add_sector_event_features(df[df["ticker"] == ticker]).sort_values("date")
    start = event_date - pd.Timedelta(weeks=80)
    part = part[part["date"] >= start]
    part = _add_technical_indicators(part)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.56, 0.26, 0.18],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Scatter(x=part["date"], y=part["close"], mode="lines", name="종가", line=dict(color="#2563eb")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=part["date"], y=part["ma_12w"], mode="lines", name="12주 MA", line=dict(color="#16a34a", dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=part["date"], y=part["ma_26w"], mode="lines", name="26주 MA", line=dict(color="#f97316", dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=part["date"], y=part["volume"], name="거래량", marker_color="#94a3b8"),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=part["date"], y=part["volume_ma_12w"], mode="lines", name="12주 평균 거래량", line=dict(color="#475569")),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=part["date"], y=part["obv"], mode="lines", name="OBV", line=dict(color="#7c3aed")),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=part["date"], y=part["rsi_14w"], mode="lines", name="14주 RSI", line=dict(color="#f97316")),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_width=1, line_color="#f97316", line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_width=1, line_color="#f97316", line_dash="dot", row=3, col=1)

    fig.add_vline(
        x=event_date.timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        line_color="#111827",
        annotation_text=event_row["event"],
        annotation_position="top right",
    )
    fig.update_layout(
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="OBV", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_xaxes(title_text="날짜", row=3, col=1)
    return fig


def _render_kpis(items: list[tuple[str, str]]) -> None:
    cards = "\n".join(
        f'<div class="sector-kpi"><div class="sector-kpi-label">{label}</div>'
        f'<div class="sector-kpi-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="sector-kpi-grid">{cards}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("데이터 수집 설정")
    st.caption("데이터 단위: 주간 종가, 주간 거래량 합계")
    start_date = st.date_input("시작일", value=pd.Timestamp("2020-01-01").date())
    end_date = st.date_input("종료일", value=pd.Timestamp.today().normalize().date())
    volume_multiple = st.slider(
        "거래량 급증 기준",
        min_value=1.2,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="해당 주의 거래량이 직전 12주 평균 주간 거래량의 몇 배 이상이면 급증으로 볼지 정합니다.",
    )
    st.caption(f"현재 기준: 주간 거래량이 12주 평균의 {volume_multiple:.1f}배 이상이면 이벤트로 표시")
    lookback_weeks = st.slider(
        "이벤트 표시 기간",
        min_value=4,
        max_value=104,
        value=26,
        step=2,
        help="최근 몇 주 동안 발생한 이벤트를 표에 보여줄지 정합니다. 예: 26은 최근 26주입니다.",
    )
    st.caption(f"현재 기준: 최근 {lookback_weeks}주 안에 발생한 이벤트 표시")
    collect_btn = st.button("섹터 ETF 데이터 수집/갱신", type="primary")

if start_date > end_date:
    st.error("시작일이 종료일보다 늦습니다.")
    st.stop()

if collect_btn:
    with st.spinner("Yahoo Finance에서 섹터 ETF 주간 데이터를 수집 중입니다..."):
        fetch_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        collected = collect_sector_etf_weekly(start=str(start_date), end=str(fetch_end.date()))
    if collected.empty:
        st.warning("수집된 데이터가 없습니다. 인터넷 연결 또는 Yahoo Finance 응답을 확인하세요.")
    else:
        st.success(
            f"`{SECTOR_ETF_DATA_PATH.as_posix()}` 저장 완료: "
            f"{len(collected):,}행, {collected['ticker'].nunique()}개 ETF"
        )
        st.rerun()

df = load_sector_etf_weekly()
if df.empty:
    st.info("아직 섹터 ETF 데이터가 없습니다. 왼쪽의 `섹터 ETF 데이터 수집/갱신` 버튼을 눌러 데이터를 수집하세요.")
    st.stop()

df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()
if df.empty:
    st.warning("선택한 기간에 표시할 데이터가 없습니다.")
    st.stop()

events = detect_sector_etf_events(df, volume_spike_multiple=volume_multiple)
latest_date = df["date"].max()
recent_start = latest_date - pd.Timedelta(weeks=int(lookback_weeks))
recent_events = events[events["date"] >= recent_start].copy() if not events.empty else pd.DataFrame()

_render_kpis(
    [
        ("◆ ETF 수", f"{df['ticker'].nunique():,}"),
        ("▣ 데이터 기간", f"{df['date'].min().date()} ~ {latest_date.date()}"),
        ("● 발생 이벤트", f"{len(recent_events):,}"),
        ("▤ 데이터 파일", SECTOR_ETF_DATA_PATH.name),
    ]
)

st.divider()
st.subheader("주요 섹터 ETF별 누적상승률 추이")

# ETF 그룹 선택
etf_group = st.radio(
    "ETF 그룹 선택",
    options=["그룹1 (기본 섹터)", "그룹2 (테마·전략형)"],
    horizontal=True,
    index=0,
)

# 선택한 그룹에 맞는 ETF 목록 결정
if etf_group == "그룹1 (기본 섹터)":
    selected_group_etfs = SECTOR_ETFS
    group_label = "기본 섹터 ETF"
else:
    selected_group_etfs = SECTOR_ETFS_GROUP2
    group_label = "테마·전략형 ETF"

# 선택한 그룹에 포함된 ETF 중 데이터에 있는 것만 필터링
available_tickers = df[df["ticker"].isin(selected_group_etfs.keys())]["ticker"].unique().tolist()
ticker_labels = {
    ticker: f"{ticker} - {selected_group_etfs.get(ticker, '알 수 없음')}"
    for ticker in available_tickers
}

selected_tickers = st.multiselect(
    f"표시할 {group_label}",
    options=sorted(ticker_labels.keys()),
    default=sorted(ticker_labels.keys()),
    format_func=lambda ticker: ticker_labels[ticker],
)

chart_min_date = max(df["date"].min().date(), BASE_DATE.date())
chart_max_date = df["date"].max().date()
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    chart_start_date = st.date_input(
        "그래프 시작일",
        value=chart_min_date,
        min_value=chart_min_date,
        max_value=chart_max_date,
        key="sector_chart_start",
    )
with chart_col2:
    chart_end_date = st.date_input(
        "그래프 종료일",
        value=chart_max_date,
        min_value=chart_min_date,
        max_value=chart_max_date,
        key="sector_chart_end",
    )

if chart_start_date > chart_end_date:
    st.error("그래프 시작일이 종료일보다 늦습니다.")
    st.stop()

if selected_tickers:
    st.caption("각 ETF의 2020-01-03 종가를 기준점(0%)으로 두고, 선택한 기간의 누적상승률을 표시합니다.")
    st.plotly_chart(
        _make_cumulative_return_chart(
            df,
            selected_tickers,
            pd.Timestamp(chart_start_date),
            pd.Timestamp(chart_end_date),
        ),
        use_container_width=True,
    )
else:
    st.info("그래프에 표시할 ETF를 1개 이상 선택하세요.")

st.divider()
st.subheader("주요 섹터 ETF별 이벤트 발생 코멘트")

if recent_events.empty:
    st.info(f"최근 {lookback_weeks}주 동안 설정한 기준에 해당하는 이벤트가 없습니다.")
else:
    display_events = recent_events.copy()
    display_events["date"] = display_events["date"].dt.date
    display_events["weekly_return"] = display_events["weekly_return"].apply(_format_pct)
    display_events = display_events[
        ["date", "ticker", "sector", "event", "comment", "close", "volume", "weekly_return"]
    ].rename(
        columns={
            "date": "날짜",
            "ticker": "ETF",
            "sector": "섹터",
            "event": "이벤트",
            "comment": "코멘트",
            "close": "종가",
            "volume": "거래량",
            "weekly_return": "주간 수익률",
        }
    )

    selected = st.dataframe(
        display_events,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "코멘트": st.column_config.TextColumn(width="large"),
            "종가": st.column_config.NumberColumn(format="%.2f"),
            "거래량": st.column_config.NumberColumn(format="%d"),
        },
    )

    selection = getattr(selected, "selection", None)
    selected_rows = getattr(selection, "rows", []) if selection is not None else []
    if selected_rows:
        event_row = recent_events.iloc[selected_rows[0]]
        st.markdown(f"#### 선택 이벤트 상세: {event_row['comment']}")
        st.plotly_chart(_make_event_detail_chart(df, event_row), use_container_width=True)
    else:
        st.caption("코멘트 표의 행을 클릭하면 해당 ETF의 가격·거래량 상세 차트가 표시됩니다.")

with st.expander("최근 원천 데이터 미리보기"):
    preview = df.sort_values(["date", "ticker"]).tail(80).copy()
    preview["date"] = preview["date"].dt.date
    st.dataframe(preview, use_container_width=True, hide_index=True)
