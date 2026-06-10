from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from utils.metrics import direction_accuracy, mape, rmse
from utils.model_sarimax import (
    add_log_price_target,
    build_future_exog,
    find_auto_arima_order,
    fit_sarimax,
)
from utils.preprocessing import load_processed_dataset


st.set_page_config(page_title="Forecast", layout="wide")

ACCENT = "#f97316"
CHART_COLORS = ["#2563eb", "#16a34a", "#7c3aed", "#f97316", "#0891b2", "#ca8a04"]

st.markdown(
    f"""
    <style>
    .forecast-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px 0;
    }}
    .forecast-kpi {{
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }}
    .forecast-kpi-label {{
        color: #475569;
        font-size: 14px;
        line-height: 1.25;
        white-space: nowrap;
    }}
    .forecast-kpi-value {{
        color: #111827;
        font-size: 18px;
        font-weight: 700;
        line-height: 1.25;
        margin-top: 6px;
        overflow-wrap: anywhere;
    }}
    .forecast-note {{
        border-left: 4px solid {ACCENT};
        background: #fff7ed;
        padding: 12px 14px;
        border-radius: 8px;
        color: #334155;
        margin-bottom: 12px;
    }}
    @media (max-width: 900px) {{
        .forecast-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _columns_matching(columns: list[str], bases: list[str], lags: range) -> list[str]:
    selected: list[str] = []
    for base in bases:
        selected.extend([f"{base}_lag_{lag}" for lag in lags if f"{base}_lag_{lag}" in columns])
    return selected


def _top_correlated_lag_columns(
    df: pd.DataFrame, columns: list[str], target_col: str, limit: int = 12
) -> list[str]:
    lag_cols = [col for col in columns if "_lag_" in col]
    if not lag_cols:
        return columns[:limit]
    corr = (
        df[[target_col, *lag_cols]]
        .corr(numeric_only=True)[target_col]
        .drop(index=target_col, errors="ignore")
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(ascending=False)
    )
    return [col for col in corr.index if col in columns][:limit]


def _recommended_exog_sets(df: pd.DataFrame, columns: list[str], target_col: str) -> dict[str, list[str]]:
    presets = {
        # 외생변수 없음 — 순수 SARIMA 기준선
        "Baseline (외생변수 없음)": [],
        # financial_conditions lag1 + vix lag1 — 최대 2개
        "Core leads": _columns_matching(
            columns,
            ["financial_conditions", "vix"],
            range(1, 2),  # lag=1만 사용
        ),
    }
    # Baseline은 빈 리스트이므로 if cols 필터 제거 — 그대로 반환
    return {name: cols[:16] for name, cols in presets.items()}


def _fetch_actual_sp500_weekly(
    future_dates: pd.DatetimeIndex,
    last_train_price: float,
) -> tuple[pd.Series | None, pd.Series | None]:
    try:
        fetch_start = (future_dates[0] - pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
        fetch_end = (future_dates[-1] + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
        raw = yf.download("^GSPC", start=fetch_start, end=fetch_end, progress=False, auto_adjust=True)
        if raw.empty:
            return None, None

        close = raw[("Close", "^GSPC")] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        close.index = pd.to_datetime(close.index).normalize()
        weekly: pd.Series = close.resample("W-FRI").last().dropna()
        weekly.index = weekly.index.normalize()
        future_norm = future_dates.normalize()
        actual_prices = weekly.reindex(future_norm)

        if actual_prices.isna().all():
            return None, None

        prev_price = last_train_price
        log_returns = []
        for price in actual_prices:
            if pd.isna(price):
                log_returns.append(np.nan)
            else:
                log_returns.append(np.log(price) - np.log(prev_price))
                prev_price = price
        actual_log_returns = pd.Series(log_returns, index=actual_prices.index)
        return actual_prices, actual_log_returns
    except Exception:
        return None, None


def _make_timeseries_chart(df: pd.DataFrame, columns: list[str], normalized: bool = False) -> go.Figure:
    chart_df = df[["date", *columns]].copy()
    if normalized:
        for col in columns:
            std = chart_df[col].std()
            chart_df[col] = (chart_df[col] - chart_df[col].mean()) / std if pd.notna(std) and std != 0 else 0.0

    fig = go.Figure()
    for index, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df[col],
                mode="lines",
                name=col,
                line=dict(color=CHART_COLORS[index % len(CHART_COLORS)], width=1.8),
                connectgaps=False,
            )
        )
    fig.update_layout(
        height=420,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Z-score" if normalized else "Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=32, b=20),
    )
    return fig


def _render_kpis(items: list[tuple[str, str]]) -> None:
    cards = "\n".join(
        f'<div class="forecast-kpi"><div class="forecast-kpi-label">{label}</div>'
        f'<div class="forecast-kpi-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="forecast-kpi-grid">{cards}</div>', unsafe_allow_html=True)


def _render_final_dataset_graph(df: pd.DataFrame) -> None:
    st.subheader("Final Dataset Chart")
    st.caption("Check the processed weekly dataset before running the forecast model.")

    numeric_cols = list(df.select_dtypes("number").columns)
    if not numeric_cols:
        st.info("No numeric columns are available in the final dataset.")
        return

    preferred = ["sp500_log_return", "sp500", "vix", "financial_conditions", "term_spread_10y_2y"]
    defaults = [col for col in preferred if col in numeric_cols][:3] or numeric_cols[:3]
    c1, c2 = st.columns([3, 1])
    with c1:
        selected = st.multiselect(
            "Chart variables",
            options=numeric_cols,
            default=defaults,
            max_selections=5,
            key="forecast_top_dataset_chart_cols",
        )
    with c2:
        normalized = st.toggle("Normalize", value=False, help="Show selected variables as z-scores.")

    if selected:
        st.plotly_chart(_make_timeseries_chart(df, selected, normalized=normalized), use_container_width=True)
    else:
        st.info("Select at least one variable to display the chart.")


def _latest_value(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    series = df[col].dropna()
    return float(series.iloc[-1]) if not series.empty else None


def _render_market_regime(df: pd.DataFrame) -> None:
    """
    시장 국면 진단: 국면 이탈 점수 대시보드 (정적 HTML)

    동시적 지표(coincident)로, 과거 데이터 기준의 국면 분류 및
    현재 상태 측정을 제공합니다. 예측이 아닙니다.
    """
    st.subheader("📊 시장 국면 진단 — 이탈 점수 대시보드")

    st.markdown(
        """
        <div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; font-size: 13px; color: #78350f; line-height: 1.6;">
            <strong>⚠️ 고지사항</strong><br>
            본 대시보드는 <strong>현재 시점의 시장 국면을 측정</strong>하는 동시적 지표(coincident indicator)입니다.
            미래 수익률을 예측하지 않으며, 투자 권유가 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from utils.deviation_dashboard import generate_deviation_dashboard_html

        # HTML 대시보드 자동 생성
        dashboard_path = "analysis/dashboard/index.html"

        with st.spinner("📊 대시보드 데이터 준비 중..."):
            success, message = generate_deviation_dashboard_html(
                csv_path="data/processed/final_dataset.csv",
                output_path=dashboard_path
            )

        if success:
            st.success(message)

            # HTML 파일 로드 및 표시
            try:
                with open(dashboard_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # Streamlit components를 사용해서 HTML 표시
                st.components.v1.html(html_content, height=2200, scrolling=True)

            except Exception as e:
                st.error(f"⚠️ HTML 표시 실패: {e}")
                st.markdown(f"**💡 해결책**: 다음 경로에서 직접 열어보세요\n```\n{dashboard_path}\n```")

        else:
            st.error(f"❌ 대시보드 생성 실패\n\n{message}")
            st.markdown("**🔧 확인 사항**:")
            st.markdown("- `data/processed/final_dataset.csv` 파일 존재 여부")
            st.markdown("- 필수 컬럼: `dxy`, `treasury_10y`, `financial_conditions`, `credit_spread`, `vix`")

    except ImportError:
        st.error("❌ 대시보드 모듈을 불러올 수 없습니다.")
        st.code("pip install -r requirements.txt")
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
        import traceback
        st.code(traceback.format_exc())

    # ── MS-AR P(약세) KPI + 히트 스트립 ─────────────────────────────────────
    # HTML 이탈 점수 대시보드와 같은 화면에, x축 공유 히트 스트립을 아래에 추가한다.
    # 기존 코드(_render_ms_ar_current_status, _render_ms_ar_overlay_chart)를 그대로 재사용.
    st.divider()
    st.markdown("#### 📈 MS-AR P(약세) — 히트 스트립 (이탈 점수와 비교)")
    st.caption(
        "이탈 점수(거시 기반)와 MS-AR P(약세)(수익률 기반)를 같은 시간축으로 비교합니다.  "
        "**현재 상태 측정(coincident) · 미래 예측 아님 · 투자권유 아님**"
    )

    try:
        from utils.deviation_dashboard import DeviationDashboard
        from utils.ms_ar_model import align_with_deviation_score, fit_ms_ar_model

        # DeviationDashboard는 캐시 없이 직접 계산 (stats 포함)
        with st.spinner("MS-AR 모델 및 이탈 점수 계산 중... (최초 1회)"):
            dd_ms = DeviationDashboard()
            if not (
                dd_ms.load_data()
                and dd_ms.calculate_features()
                and dd_ms.fit_clusters()
                and dd_ms.calculate_scores()
                and dd_ms.calculate_stats()
            ):
                st.error("❌ 이탈 점수 계산에 실패했습니다.")
                return

        # fit_ms_ar_model은 @st.cache_data 적용 — 이미 계산된 경우 캐시 재사용
        ms_ar = fit_ms_ar_model()
        merged = align_with_deviation_score(ms_ar["prob_bear_series"])

        # 현재 값 KPI 카드 + 일치/불일치 판정
        _render_ms_ar_current_status(dd_ms.stats, ms_ar)
        st.divider()

        # 히트 스트립 차트 (기간 선택·spike_mode 포함)
        # key 접두어를 "regime_" 로 구분 — "3) 교차검증" 화면의 key와 충돌 방지
        _render_ms_ar_overlay_chart_in_regime(merged, dd_ms)

    except ImportError:
        st.warning("⚠️ statsmodels 등 필요 패키지가 없어 MS-AR 히트 스트립을 표시할 수 없습니다.")
    except Exception as e:
        st.error(f"❌ MS-AR 계산 중 오류: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_ms_ar_overlay_chart_in_regime(merged: pd.DataFrame, dd) -> None:
    """_render_ms_ar_overlay_chart의 복사본 — key 접두어만 'regime_'로 변경.

    '1) Market Regime Diagnosis' 화면에서 호출 시 Streamlit widget key가
    '3) 교차검증' 화면의 key와 충돌하지 않도록 별도 함수로 분리한다.
    로직은 _render_ms_ar_overlay_chart와 완전히 동일.
    """
    st.caption("두 신호 모두 '높을수록 스트레스'를 의미합니다 — 같이 오르내리면 두 신호가 일관됨을 뜻합니다.")

    # 표시 기간 선택
    c1, c2 = st.columns([1, 2])
    with c1:
        range_mode = st.radio(
            "표시 기간",
            ["전체 기간", "최근 N주"],
            horizontal=True,
            key="regime_ms_ar_range_mode",        # ★ key 접두어: regime_
        )
    with c2:
        n_weeks = st.number_input(
            "최근 주 수",
            min_value=4,
            max_value=int(len(merged)),
            value=min(104, int(len(merged))),
            step=4,
            disabled=(range_mode == "전체 기간"),
            key="regime_ms_ar_n_weeks",            # ★ key 접두어: regime_
            help="예: 52주 ≈ 1년, 104주 ≈ 2년",
        )

    view = merged.tail(int(n_weeks)).reset_index(drop=True) if range_mode == "최근 N주" else merged
    range_desc = f"최근 {int(n_weeks)}주" if range_mode == "최근 N주" else "전체 기간"

    spike_mode_label = st.selectbox(
        "단발 스파이크 표시 방식 (모두 비후행·대칭)",
        ["원본 그대로 (raw, 기본)", "2주 연속 임계만 강조", "중심 3주 평활 (과거 전용)"],
        key="regime_ms_ar_spike_mode",             # ★ key 접두어: regime_
        help="히트 스트립의 색칠 강도만 바꾸는 표시 옵션입니다.",
    )
    spike_mode = {
        "원본 그대로 (raw, 기본)": "raw",
        "2주 연속 임계만 강조": "consec2",
        "중심 3주 평활 (과거 전용)": "centered3",
    }[spike_mode_label]

    st.plotly_chart(
        _build_ms_ar_heatstrip_figure(view, dd, spike_mode=spike_mode, height=540),
        use_container_width=True,
        key="regime_ms_ar_chart",                  # ★ key 접두어: regime_
    )
    st.caption(
        f"표시 범위: {range_desc} ({view['date'].min().date()} ~ {view['date'].max().date()}, {len(view)}주)  ·  "
        f"하단 색 띠 = P(약세) 원값(raw filtered, 지연 없음)  ·  "
        f"점선 = 기존 이벤트(events.csv 13개)  ·  KPI '현재 P(약세)'는 항상 최신 filtered 원값"
    )


def _render_ms_ar_current_status(dashboard_stats: dict, ms_ar: dict) -> None:
    """① 현재 상태: 이탈 점수·P(약세) 현재값 + 한 줄 일치/불일치 판정"""
    current_score = dashboard_stats["current_score"]
    current_pct = dashboard_stats["current_pct"]
    current_prob_bear = ms_ar["current_prob_bear"]

    # 임계값: 이탈 점수는 비위기 90백분위, P(약세)는 0.5 (③ 일치/불일치 분석과 동일 기준)
    score_flag = current_pct >= 90
    prob_flag = current_prob_bear > 0.5

    if score_flag and prob_flag:
        verdict = "🔴 일치 — 두 신호 모두 '경계' 신호"
    elif (not score_flag) and (not prob_flag):
        verdict = "🟢 일치 — 두 신호 모두 '평온' 신호"
    else:
        verdict = "🟡 불일치 — 신호가 엇갈림 (③ 메뉴 참고)"

    _render_kpis(
        [
            ("이탈 점수 (백분위)", f"{current_score:+.4f}  ({current_pct:.1f}%ile)"),
            ("MS-AR P(약세)", f"{current_prob_bear * 100:.1f}%"),
            ("종합 판정", verdict),
        ]
    )
    st.caption(
        f"기준일 — 이탈 점수: {dashboard_stats['current_date']}  ·  MS-AR: {ms_ar['current_date']}  "
        f"(판정 임계값: 이탈 점수 비위기 90백분위 이상 & P(약세) > 0.5 동시 충족 여부)"
    )


def _find_high_regions(dates: pd.Series, mask: pd.Series) -> list[tuple]:
    """연속된 True 구간의 (시작일, 종료일) 리스트 반환 — 배경 음영(vrect)용"""
    dates_list = dates.tolist()
    mask_list = mask.tolist()
    regions = []
    start = None
    for i, (is_high, d) in enumerate(zip(mask_list, dates_list)):
        if is_high and start is None:
            start = d
        elif not is_high and start is not None:
            regions.append((start, dates_list[i - 1]))
            start = None
    if start is not None:
        regions.append((start, dates_list[-1]))
    return regions


def _build_ms_ar_overlay_figure(view: pd.DataFrame, dd, *, height: int = 440) -> go.Figure:
    """[변경 전] 이탈 점수(좌축) + P(약세) 8주 후행 압력선(우축) + 0.5 초과 음영.

    ★ 이 8주 '후행' 이동평균은 본질적으로 약 (8-1)/2 ≈ 3~4주의 시차를 갖기 때문에,
      이탈 점수와 겹쳐 보면 실제로는 없는 '선행/후행 관계'처럼 보이는 착시를 만든다
      (= 비교용으로만 남겨둔 '변경 전' 버전 — 프로덕션에서는 히트 스트립으로 대체).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    raw_prob = view["prob_bear"]
    pressure_8w = raw_prob.rolling(8, min_periods=1).mean()

    for r_start, r_end in _find_high_regions(view["date"], raw_prob > 0.5):
        fig.add_vrect(x0=r_start, x1=r_end, fillcolor="#fecaca", opacity=0.35, line_width=0, layer="below")

    fig.add_trace(
        go.Scatter(x=view["date"], y=view["score"], name="이탈 점수 (좌축)", line=dict(color="#2563eb", width=1.2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=view["date"], y=pressure_8w, name="P(약세) 8주 후행 압력선 (우축)", line=dict(color="#b91c1c", width=1.8)),
        secondary_y=True,
    )

    date_min, date_max = view["date"].min(), view["date"].max()
    for ev in dd.load_events():
        ev_date = pd.Timestamp(ev["date"])
        if date_min <= ev_date <= date_max:
            fig.add_vline(x=ev_date.timestamp() * 1000, line_width=1, line_dash="dot", line_color="gray", opacity=0.35)

    fig.update_yaxes(title_text="이탈 점수", secondary_y=False)
    fig.update_yaxes(title_text="P(약세)", range=[0, 1.05], secondary_y=True)
    fig.update_layout(
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=55),
    )
    return fig


def _build_ms_ar_heatstrip_figure(view: pd.DataFrame, dd, *, spike_mode: str = "raw", height: int = 440) -> go.Figure:
    """[변경 후] 이탈 점수(상단 패널, 선) + P(약세)(하단 패널, 히트 스트립) — 시간축 공유.

    ★ 히트 스트립 색상은 항상 '주간 filtered 원값'을 기준으로 한다 — 이동평균 등
      후행(trailing) 집계를 전혀 쓰지 않으므로 시차(lag)가 생기지 않는다.
      spike_mode는 단발 스파이크의 시각적 강조 정도만 조절하는 표시 옵션이며,
      두 옵션 모두 대칭(symmetric) 방식이라 한쪽으로 쏠리는 후행 지연이 없다.

    Args:
        spike_mode: "raw"(원값 그대로, 기본) |
                    "consec2"(0.5 초과가 2주 이상 연속될 때만 강조 — 단발 스파이크는 옅게) |
                    "centered3"(중심 3주 평활 — 대칭이라 지연 0, 과거 차트 전용)
    """
    raw_prob = view["prob_bear"]

    if spike_mode == "centered3":
        strip_values = raw_prob.rolling(3, center=True, min_periods=1).mean()
        strip_label = "P(약세) 중심 3주 평활"
    elif spike_mode == "consec2":
        high = raw_prob > 0.5
        run_id = (high != high.shift()).cumsum()
        run_len = high.groupby(run_id).transform("size")
        in_long_run = high & (run_len >= 2)
        strip_values = raw_prob.where(in_long_run, raw_prob * 0.3)
        strip_label = "P(약세) (2주 연속 임계만 강조)"
    else:
        strip_values = raw_prob
        strip_label = "P(약세) 원값"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.76, 0.24], vertical_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(x=view["date"], y=view["score"], name="이탈 점수", line=dict(color="#2563eb", width=1.3)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=[strip_values.values],
            x=view["date"],
            y=["P(약세)"],
            customdata=raw_prob.values.reshape(1, -1),
            colorscale=[[0.0, "#fff5f5"], [0.5, "#fca5a5"], [1.0, "#b91c1c"]],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(title="P(약세)", thickness=10, len=0.45, y=0.05, yanchor="bottom"),
            hovertemplate="%{x|%Y-%m-%d}<br>P(약세) 원값(raw, filtered): %{customdata:.1%}<extra></extra>",
            name=strip_label,
        ),
        row=2, col=1,
    )

    date_min, date_max = view["date"].min(), view["date"].max()
    for ev in dd.load_events():
        ev_date = pd.Timestamp(ev["date"])
        if date_min <= ev_date <= date_max:
            fig.add_vline(x=ev_date.timestamp() * 1000, line_width=1, line_dash="dot", line_color="gray", opacity=0.35, row="all", col=1)

    fig.update_yaxes(title_text="이탈 점수", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=55),
    )
    return fig


def _lag_correlation(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """a 대비 b를 lag주 이동했을 때의 상관계수 (양수 lag = b가 과거값과 비교됨 = a가 b를 후행)"""
    if lag < 0:
        x, y = a[:lag], b[-lag:]
    elif lag > 0:
        x, y = a[lag:], b[:-lag]
    else:
        x, y = a, b
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 8) -> tuple[int, float]:
    scored = [(lag, _lag_correlation(a, b, lag)) for lag in range(-max_lag, max_lag + 1)]
    return max(scored, key=lambda r: r[1])


def _render_ms_ar_overlay_chart(merged: pd.DataFrame, dd) -> None:
    """② 추이: 이탈 점수 · P(약세) 표시 — 변경 전(8주 후행 압력선) vs 변경 후(raw 히트 스트립) 비교"""
    st.markdown("#### 📈 추이 — 이탈 점수 vs MS-AR P(약세)")
    st.caption("두 신호 모두 '높을수록 스트레스'를 의미합니다 — 같이 오르내리면 두 신호가 일관됨을 뜻합니다.")

    # 표시 기간 선택: 전체 기간 또는 최근 N주
    c1, c2 = st.columns([1, 2])
    with c1:
        range_mode = st.radio(
            "표시 기간",
            ["전체 기간", "최근 N주"],
            horizontal=True,
            key="ms_ar_overlay_range_mode",
        )
    with c2:
        n_weeks = st.number_input(
            "최근 주 수",
            min_value=4,
            max_value=int(len(merged)),
            value=min(104, int(len(merged))),
            step=4,
            disabled=(range_mode == "전체 기간"),
            key="ms_ar_overlay_n_weeks",
            help="예: 52주 ≈ 1년, 104주 ≈ 2년",
        )

    view = merged.tail(int(n_weeks)).reset_index(drop=True) if range_mode == "최근 N주" else merged
    range_desc = f"최근 {int(n_weeks)}주" if range_mode == "최근 N주" else "전체 기간"

    spike_mode_label = st.selectbox(
        "단발 스파이크 표시 방식 (모두 비후행·대칭)",
        ["원본 그대로 (raw, 기본)", "2주 연속 임계만 강조", "중심 3주 평활 (과거 전용)"],
        key="ms_ar_heatstrip_spike_mode",
        help="히트 스트립의 색칠 강도만 바꾸는 표시 옵션입니다. 모두 대칭(symmetric) 방식이라 "
             "후행 지연이 생기지 않습니다 — KPI의 현재 값과는 무관합니다.",
    )
    spike_mode = {
        "원본 그대로 (raw, 기본)": "raw",
        "2주 연속 임계만 강조": "consec2",
        "중심 3주 평활 (과거 전용)": "centered3",
    }[spike_mode_label]

    st.plotly_chart(
        _build_ms_ar_heatstrip_figure(view, dd, spike_mode=spike_mode, height=540),
        use_container_width=True,
        key="ms_ar_overlay_chart",
    )
    st.caption(
        f"표시 범위: {range_desc} ({view['date'].min().date()} ~ {view['date'].max().date()}, {len(view)}주)  ·  "
        f"하단 색 띠 = P(약세) 원값(raw filtered, 지연 없음) 강도  ·  "
        f"기존 events.csv 13개 이벤트는 점선으로 표시됩니다 (새 이벤트 생성 없음).  ·  "
        f"KPI의 '현재 P(약세)'는 항상 최신 filtered 원값 그대로입니다."
    )


def _render_ms_ar_agreement(merged: pd.DataFrame, agreement: dict) -> None:
    """③ 두 신호 일치/불일치: 상관·자카드·리드래그·불일치 구간표 + 산점도"""
    st.markdown("#### 🔍 두 신호 일치/불일치 분석")
    st.caption("이탈 점수(거시 스트레스 기반)와 P(약세)(수익률 기반)가 같은 시점에 함께 반응하는지 정량적으로 점검합니다.")

    best_lag = agreement["best_lag"]
    _render_kpis(
        [
            ("상관계수 (Pearson)", f"{agreement['correlation']:.3f}"),
            ("동시 'high' 비율 (자카드)", f"{agreement['jaccard'] * 100:.1f}%"),
            ("최대 상관 시차", f"lag = {best_lag['lag']}주 (r = {best_lag['correlation']:.3f})"),
            ("불일치 주 수", f"점수만 {agreement['n_only_score_high']}주 · P(약세)만 {agreement['n_only_prob_high']}주"),
        ]
    )

    st.markdown(
        f"- 이탈 점수 임계값 (비위기 90백분위): **{agreement['score_threshold']:.4f}** → high **{agreement['score_high_weeks']}주**\n"
        f"- P(약세) 임계값: **{agreement['prob_threshold']:.2f}** → high **{agreement['prob_high_weeks']}주**\n"
        f"- 교집합(둘 다 high): **{agreement['intersection']}주**  ·  합집합(둘 중 하나라도 high): **{agreement['union']}주**"
    )

    st.markdown("##### 리드-래그 교차상관 (±6주)")
    lag_df = pd.DataFrame(agreement["lag_results"])
    lag_df["해석"] = lag_df["lag"].apply(
        lambda l: "이탈 점수가 먼저 변함 (lead)" if l < 0 else ("P(약세)가 먼저 변함 (lead)" if l > 0 else "동시")
    )
    lag_df = lag_df.rename(columns={"lag": "lag(주)", "correlation": "상관계수"})
    lag_df["상관계수"] = lag_df["상관계수"].round(4)
    st.dataframe(lag_df, use_container_width=True, hide_index=True)
    st.caption(
        "※ 이탈 점수와 P(약세) 모두 '현재 상태'를 측정하는 동시적(coincident) 지표이므로, "
        "큰 리드/래그는 기대하지 않으며 lag=0 부근의 상관이 가장 클수록 두 신호가 동시적임을 재확인하는 참고 정보입니다."
    )

    st.markdown("##### 불일치 구간 (한쪽만 임계 초과)")
    divergence = agreement["divergence"]
    if not divergence.empty:
        show = divergence.copy()
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
        show["score"] = show["score"].round(4)
        show["prob_bear"] = show["prob_bear"].round(4)
        show = show.rename(columns={"date": "날짜", "score": "이탈 점수", "prob_bear": "P(약세)", "구분": "유형"})
        st.dataframe(show[["날짜", "이탈 점수", "P(약세)", "유형"]], use_container_width=True, hide_index=True)
    else:
        st.info("선택한 임계값 기준으로는 한쪽만 'high'인 불일치 구간이 없습니다.")
    st.caption(
        "① 이탈 점수만 높음 → 거시 스트레스는 상승했지만 수익률 국면은 아직 평온한 '위기 전조형'  ·  "
        "② P(약세)만 높음 → 수익률은 급변했지만 거시 스트레스는 낮은 '가격 충격형'. "
        "두 신호가 서로 다른 정보(거시 vs 가격)에 반응하는 구간을 보여줍니다."
    )

    st.markdown("##### 산점도 — 이탈 점수 vs P(약세)")
    nc = merged.loc[~merged["crisis"]]
    cr = merged.loc[merged["crisis"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nc["score"], y=nc["prob_bear"], mode="markers", name="비위기 구간",
            marker=dict(size=5, color="#2563eb", opacity=0.45),
            text=nc["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="%{text}<br>이탈 점수=%{x:.4f}<br>P(약세)=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cr["score"], y=cr["prob_bear"], mode="markers", name="위기 구간 (2008-09, 2020 등)",
            marker=dict(size=6, color="#dc2626", opacity=0.75),
            text=cr["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="%{text}<br>이탈 점수=%{x:.4f}<br>P(약세)=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=agreement["prob_threshold"], line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="P(약세) 임계 0.5", annotation_position="top left")
    fig.add_vline(x=agreement["score_threshold"], line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="이탈 점수 90%ile", annotation_position="top right")
    fig.update_layout(
        xaxis_title="이탈 점수",
        yaxis_title="P(약세)",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("점선은 ①에서 사용한 임계값(이탈 점수 비위기 90백분위, P(약세) 0.5)이며, 우상단에 모일수록 두 신호가 동시에 '경계' 신호를 보냄을 뜻합니다.")


def _render_ms_ar_methodology(ms_ar: dict) -> None:
    """③ 방법·주의: 방법론·데이터 출처·한계·면책"""
    st.markdown("#### 📚 방법론 & 데이터 출처")

    st.markdown("##### 1) 이탈 점수 (Deviation Score)")
    st.markdown(
        "- 5개 표준화 지표(DXY 13주 변화율, 10Y 금리 변화, NFCI 금융여건, HY 신용스프레드, VIX)를 사용\n"
        "- **비위기 구간**(2008-09~2009-06, 2020-02~06 제외)에서 학습한 StandardScaler·KMeans 센트로이드를 "
        "**'동결'한 채 전체 표본(위기 포함)에 적용** → '평온강세' 클러스터로부터의 표준화 거리\n"
        "- 산출: `utils/deviation_dashboard.py`의 `DeviationDashboard` (이번 통합에서 재구현 없이 그대로 호출)"
    )

    st.markdown("##### 2) MS-AR 약세 확률 P(약세)")
    st.markdown(
        "- S&P500 **주간 로그수익률**에 2국면 Markov-Switching AR(1) 모델 적합 (`switching_variance=True`)\n"
        "- **약세 국면 = 평균이 더 낮은 국면**으로 식별 (라벨 스위칭에 안전하도록 평균 기준 정렬)\n"
        "- 국소최적 회피를 위해 `search_reps=20` + 시드 고정(`seed=42`)으로 재현성 확보, 전체 표본(위기 포함)으로 적합\n"
        "- 산출: `utils/ms_ar_model.py::fit_ms_ar_model` (독립 분석 스크립트 `ms_ar_analysis.py`의 로직을 캐싱과 함께 재사용)"
    )

    conv = "수렴함 ✓" if ms_ar["converged"] else "⚠ 수렴 실패 — 해석에 주의 필요"
    st.markdown(f"**적합된 모델 정보** — 로그우도: `{ms_ar['log_likelihood']:.2f}`  ·  수렴 여부: {conv}")
    st.markdown(
        "| 국면 | 평균 (주간 로그수익률) | 분산 | 기대 지속기간 |\n"
        "|---|---|---|---|\n"
        f"| 약세(bear) | {ms_ar['mean_bear']:+.4f} | {ms_ar['var_bear']:.6f} | "
        f"{ms_ar['duration_bear_weeks']:.1f}주 (약 {ms_ar['duration_bear_weeks'] / 4.33:.1f}개월) |\n"
        f"| 강세(bull) | {ms_ar['mean_bull']:+.4f} | {ms_ar['var_bull']:.6f} | "
        f"{ms_ar['duration_bull_weeks']:.1f}주 (약 {ms_ar['duration_bull_weeks'] / 4.33:.1f}개월) |"
    )
    st.caption(
        f"약세/강세 분산비 = {ms_ar['var_ratio']:.2f}배 — 1보다 뚜렷이 클수록 "
        f"'약세 = 낮은 평균 + 높은 분산(고변동성 위기 국면)'으로 해석할 근거가 됩니다."
    )

    st.markdown("##### 3) 데이터 출처")
    st.markdown(
        "- `data/processed/final_dataset.csv` — 시장·금리·거시·유동성 주간 데이터 (이탈 점수·MS-AR 공통 입력)\n"
        "- `data/raw/events.csv` — 역사적 이벤트 13개 (금요일 매칭, 추이 차트의 이벤트 마커에 그대로 사용. 새로 생성한 이벤트 없음)"
    )

    st.markdown("##### ⚠️ 한계 및 해석 시 유의사항")
    st.markdown(
        "1. **동시적(coincident) 지표** — 이탈 점수와 P(약세) 모두 '현재 상태'를 측정하는 지표이며, 미래 수익률을 예측하지 않습니다.\n"
        "2. **일치도 분석은 '일관성' 검증** — 두 독립 신호(서로 다른 데이터: 거시지표 vs 가격수익률)가 같은 시점에 함께 반응하는지를 "
        "보는 교차 확증(corroboration) 점검이며, 예측력의 증거가 아닙니다.\n"
        "3. **약세 국면 표본의 희소성** — 약세 국면은 위기(2008, 2020 등)에 집중된 희소 사건이라, MS-AR의 평균·분산·전이확률 "
        "추정치가 제한된 표본에서 산출되어 불안정할 수 있습니다.\n"
        "4. **동결 표준화·센트로이드의 한계** — 이탈 점수는 비위기 구간 기준으로 학습한 파라미터를 동결해 전체 표본에 적용하므로, "
        "시장 구조가 크게 바뀌면 주기적인 재학습이 필요할 수 있습니다.\n"
        "5. **라벨 스위칭(label switching)** — MS-AR의 국면 번호는 적합 시마다 임의로 배정될 수 있어, "
        "항상 '평균이 낮은 국면 = 약세'로 재식별한 값을 사용합니다."
    )

    st.markdown(
        """
        <div style="background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px 14px; margin-top: 12px; font-size: 13px; color: #374151; line-height: 1.6; text-align: center;">
            <strong>현재 상태 측정(coincident) · 미래 예측 아님 · 투자권유 아님</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_ms_ar_cross_validation(df: pd.DataFrame) -> None:
    """
    이탈 점수 ↔ MS-AR 약세 확률 교차검증

    거시지표 기반 이탈 점수(coincident)와 주간 수익률 기반 MS-AR P(약세)(coincident),
    독립적으로 산출된 두 신호의 '동시성(일관성)'을 점검합니다. 예측력 검증이 아닙니다.
    """
    st.subheader("🔬 이탈 점수 ↔ MS-AR 약세 확률 — 교차검증")

    st.markdown(
        """
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; font-size: 13px; color: #1e3a8a; line-height: 1.6;">
            <strong>ℹ️ 무엇을 보는 화면인가요?</strong><br>
            거시지표(DXY·금리·NFCI·신용스프레드·VIX) 기반 <strong>이탈 점수</strong>와
            S&P500 주간수익률의 <strong>MS-AR(2국면) 약세 확률 P(약세)</strong>는
            서로 다른 데이터로, 서로 독립적으로 산출됩니다.
            두 신호가 같은 시점에 함께 반응하는지를 보면, 현재 국면 판단에 대한
            <strong>교차 확증(corroboration)</strong> 근거를 얻을 수 있습니다.
            ⚠️ 이는 '동시성(일관성)' 검증이며, <strong>미래 예측력의 증거가 아닙니다.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sub_menu = st.radio(
        "세부 메뉴",
        ["① 현재 상태 & 추이", "② 두 신호 일치/불일치", "③ 방법·주의"],
        horizontal=True,
        key="ms_ar_sub_menu",
    )

    try:
        from utils.deviation_dashboard import DeviationDashboard
        from utils.ms_ar_model import align_with_deviation_score, compute_agreement, fit_ms_ar_model

        with st.spinner("📊 이탈 점수 계산 중..."):
            dd = DeviationDashboard()
            if not (
                dd.load_data()
                and dd.calculate_features()
                and dd.fit_clusters()
                and dd.calculate_scores()
                and dd.calculate_stats()
            ):
                st.error("❌ 이탈 점수 계산에 실패했습니다.")
                return

        ms_ar = fit_ms_ar_model()
        merged = align_with_deviation_score(ms_ar["prob_bear_series"])
        agreement = compute_agreement(merged)
    except ImportError:
        st.error("❌ 필요한 모듈을 불러올 수 없습니다. `pip install -r requirements.txt`로 statsmodels 등을 설치하세요.")
        return
    except Exception as e:
        st.error(f"❌ 계산 중 오류가 발생했습니다: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    if sub_menu == "① 현재 상태 & 추이":
        _render_ms_ar_current_status(dd.stats, ms_ar)
        st.divider()
        _render_ms_ar_overlay_chart(merged, dd)
    elif sub_menu == "② 두 신호 일치/불일치":
        _render_ms_ar_agreement(merged, agreement)
    else:
        _render_ms_ar_methodology(ms_ar)


def _render_sarimax(df: pd.DataFrame) -> None:
    # ── 제목 ──────────────────────────────────────────────────────────────────
    st.subheader("SARIMAX Forecast")

    # ── 데이터 검증 ────────────────────────────────────────────────────────────
    if "sp500_log_return" not in df.columns:
        st.error("The final dataset must include `sp500_log_return`.")
        return
    if "sp500" not in df.columns:
        st.error("The final dataset must include `sp500`.")
        return

    model_df = df.dropna().reset_index(drop=True)
    if model_df.empty:
        st.error("No usable rows remain after dropping missing values.")
        return

    model_df["date"] = pd.to_datetime(model_df["date"])
    model_df = add_log_price_target(model_df)

    target_options = {
        "Forecast log price, then convert to S&P500 level": "sp500_log_price",
        "Forecast weekly log return": "sp500_log_return",
    }
    target_exclusions = set(target_options.values())
    numeric_cols = [col for col in model_df.select_dtypes("number").columns if col not in target_exclusions]
    default_exog_cols = [col for col in numeric_cols if "_lag_" in col][:10] or numeric_cols[:5]

    last_date = model_df["date"].iloc[-1]
    today = pd.Timestamp.today().normalize()

    # ── 기본값: 제목 아래 즉시 버튼 ───────────────────────────────────────────
    # (아직 설정이 없으므로 일단 disabled=True로 표시)
    placeholder_run = st.container()

    # ── KPI 카드 & 안내 메시지 ──────────────────────────────────────────────
    future_dates_preview = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=12, freq="W-FRI")
    _render_kpis(
        [
            ("◆ Training end", str(last_date.date())),
            ("▣ Rows", f"{len(model_df):,} weeks"),
            ("● Default target", "S&P500"),
            ("▤ Next week", str(future_dates_preview[0].date())),
        ]
    )
    st.markdown(
        '<div class="forecast-note">Configure the model settings below, then click the training button above.</div>',
        unsafe_allow_html=True,
    )

    # ── 2열 Model Settings 배치 ────────────────────────────────────────────────
    st.markdown("#### ⚙️ Model Settings")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**Forecast & Target**")
        forecast_horizon = st.number_input(
            "Forecast horizon (weeks)",
            min_value=1,
            max_value=52,
            value=12,
            help="Number of weeks to forecast into the future"
        )
        target_label = st.radio(
            "Target transform",
            list(target_options.keys()),
            help="Choose what to forecast"
        )
        target_col = target_options[target_label]

    with right_col:
        st.markdown("**ARIMA Order**")
        use_auto_arima = st.checkbox(
            "Optimize order with Auto-ARIMA",
            value=False,
            help="Let the algorithm find the best ARIMA order"
        )
        p = st.number_input("AR(p)", min_value=0, max_value=5, value=1, disabled=use_auto_arima)
        d = st.number_input("Diff(d)", min_value=0, max_value=2, value=0, disabled=use_auto_arima)
        q = st.number_input("MA(q)", min_value=0, max_value=5, value=1, disabled=use_auto_arima)

    # ── 옵션: Seasonal & Exogenous ──────────────────────────────────────────────
    left_col2, right_col2 = st.columns(2)

    with left_col2:
        st.markdown("**Seasonal Order**")
        use_seasonal = st.checkbox("Include seasonal order", value=False)
        seasonal_m = st.number_input(
            "Seasonal period (m)",
            min_value=1,
            max_value=52,
            value=52,
            disabled=not use_seasonal
        )

    with right_col2:
        st.markdown("**Exogenous Variables**")
        scale_external = st.checkbox("Scale exogenous variables", value=True)
        exog_presets = _recommended_exog_sets(model_df, numeric_cols, target_col)
        if not exog_presets:
            exog_presets = {"Default numeric variables": default_exog_cols}
        preset_name = st.selectbox("Exogenous preset", list(exog_presets.keys()))
        selected_preset = exog_presets[preset_name]
        st.caption(f"📊 {len(selected_preset)} variables selected")

    # ── 변수 선택 (전폭) ─────────────────────────────────────────────────────────
    exog_cols = st.multiselect(
        "Fine-tune exogenous variables",
        numeric_cols,
        default=selected_preset,
        help="Add or remove specific variables from the model"
    )

    # ── 버튼 배치: 상단의 placeholder에 넣기 ──────────────────────────────────────
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=int(forecast_horizon), freq="W-FRI")
    forecast_end_date = future_dates[-1]
    is_past_forecast = forecast_end_date < today

    with placeholder_run:
        run = st.button(f"🚀 Train and forecast {int(forecast_horizon)} weeks", type="primary", use_container_width=True)

    if not run:
        st.info("Adjust the settings in the upper-right panel and click the training button.")
        return

    train = model_df.copy()
    model_train = train.copy()
    scaler: StandardScaler | None = None
    if exog_cols and scale_external:
        scaler = StandardScaler()
        model_train[exog_cols] = scaler.fit_transform(train[exog_cols])

    order = (int(p), int(d), int(q))
    seasonal_order = (0, 0, 0, 0)

    if use_auto_arima:
        with st.spinner("Searching ARIMA order by AIC..."):
            try:
                order, seasonal_order, aic = find_auto_arima_order(
                    model_train,
                    target_col=target_col,
                    exog_cols=exog_cols,
                    seasonal=use_seasonal,
                    m=int(seasonal_m),
                )
            except ImportError as exc:
                st.error(str(exc))
                return
        st.success(f"Selected order: order={order}, seasonal_order={seasonal_order}, AIC={aic:.2f}")

    with st.spinner(f"Training SARIMAX on {len(model_df):,} weeks and forecasting {forecast_horizon} weeks..."):
        fitted = fit_sarimax(
            model_train,
            target_col=target_col,
            exog_cols=exog_cols,
            order=order,
            seasonal_order=seasonal_order,
        )
        if exog_cols:
            future_exog_raw = build_future_exog(train, exog_cols, int(forecast_horizon))
            future_exog = (
                pd.DataFrame(scaler.transform(future_exog_raw), columns=exog_cols)
                if scaler is not None
                else future_exog_raw
            )
        else:
            future_exog = None

        forecast_result = fitted.get_forecast(steps=int(forecast_horizon), exog=future_exog)
        pred_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

    if target_col == "sp500_log_price":
        prediction = np.exp(pred_mean.to_numpy())
        ci_lower = np.exp(conf_int.iloc[:, 0].to_numpy())
        ci_upper = np.exp(conf_int.iloc[:, 1].to_numpy())
        yaxis_title = "S&P500"
    else:
        prediction = pred_mean.to_numpy()
        ci_lower = conf_int.iloc[:, 0].to_numpy()
        ci_upper = conf_int.iloc[:, 1].to_numpy()
        yaxis_title = "Weekly log return"

    last_train_price = float(model_df["sp500"].iloc[-1])
    actual_prices: pd.Series | None = None
    actual_log_returns: pd.Series | None = None
    has_actuals = False

    with st.spinner("Fetching actual S&P500 values for the forecast period...") if is_past_forecast else st.empty():
        if is_past_forecast:
            actual_prices, actual_log_returns = _fetch_actual_sp500_weekly(future_dates, last_train_price)
            has_actuals = actual_prices is not None and actual_prices.notna().any()

    if has_actuals:
        actual_series = actual_prices if target_col == "sp500_log_price" else actual_log_returns
        valid_mask = actual_series.notna().values
        pred_series = pd.Series(prediction, index=future_dates.normalize())
        actual_valid = actual_series[valid_mask]
        pred_valid = pred_series[valid_mask]
    else:
        actual_series = None
        valid_mask = np.array([False] * len(future_dates))

    _render_kpis(
        [
            ("◆ Training end", str(last_date.date())),
            ("▣ Forecast start", str(future_dates[0].date())),
            ("● Forecast end", str(future_dates[-1].date())),
            ("▤ Actuals", f"{int(valid_mask.sum())} / {forecast_horizon} weeks" if has_actuals else "Future period"),
        ]
    )

    if has_actuals and len(actual_valid) > 0 and len(pred_valid) > 0:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("RMSE", f"{rmse(actual_valid, pred_valid):.4f}")
        m_col2.metric("MAPE", f"{mape(actual_valid, pred_valid):.2f}%")
        dir_actual = actual_valid.diff().dropna()
        dir_pred = pred_valid.diff().dropna()
        if len(dir_actual) > 0:
            m_col3.metric("Direction accuracy", f"{direction_accuracy(dir_actual, dir_pred):.2f}%")

    hist_n = min(52, len(model_df))
    hist_x = model_df["date"].iloc[-hist_n:]
    hist_y = (model_df["sp500"] if target_col == "sp500_log_price" else model_df["sp500_log_return"]).iloc[-hist_n:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=hist_y, name="Actual training period", line=dict(color="#2563eb")))
    if has_actuals and actual_series is not None:
        fig.add_trace(
            go.Scatter(
                x=actual_series.index,
                y=actual_series.values,
                name="Actual forecast period",
                line=dict(color="#0891b2", dash="dot"),
                mode="lines+markers",
                marker=dict(symbol="circle-open", size=7),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=prediction,
            name="Forecast",
            mode="lines+markers",
            line=dict(color=ACCENT, dash="dash"),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill="toself",
            fillcolor="rgba(249, 115, 22, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% confidence interval",
        )
    )
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_width=1,
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast start",
        annotation_position="top right",
    )
    fig.update_layout(
        height=520,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=32, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    forecast_df = pd.DataFrame(
        {
            "date": future_dates.date,
            "forecast": prediction,
            "lower_95": ci_lower,
            "upper_95": ci_upper,
        }
    )
    if has_actuals and actual_series is not None:
        actual_arr = actual_series.reindex(future_dates.normalize()).values
        forecast_df["actual"] = actual_arr
        forecast_df["error_forecast_minus_actual"] = forecast_df["forecast"] - forecast_df["actual"]

    st.subheader("Forecast Details")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)


with st.sidebar:
    st.header("📊 Forecast Model")
    # ── 라디오 버튼으로 모델 선택 (selectbox 대신) ──
    model_choice = st.radio(
        "Select model:",
        [
            "1) Market Regime Diagnosis",
            "2) SARIMAX",
            "3) 이탈 점수 ↔ MS-AR 교차검증",
            "4) 유사 시점 탐색기",
        ],
        index=0,
        help="Choose which forecast model to run"
    )

st.title("S&P500 Forecast")
st.caption("Monitor the processed dataset, diagnose market regime, and run SARIMAX forecasts with lagged exogenous variables.")

df = load_processed_dataset()
if df.empty:
    st.info("Please build `data/processed/final_dataset.csv` from the Data View page first.")
    st.stop()

if "date" not in df.columns:
    st.error("The final dataset must include a `date` column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ★ 시장 레짐 진단/교차검증 시에는 final dataset 차트 생략 (각 화면이 별도 표시)
if model_choice == "1) Market Regime Diagnosis":
    _render_market_regime(df)
elif model_choice == "3) 이탈 점수 ↔ MS-AR 교차검증":
    _render_ms_ar_cross_validation(df)
elif model_choice == "4) 유사 시점 탐색기":
    from utils.similar_periods import render_similar_periods
    render_similar_periods()
else:
    # SARIMAX 시에만 final dataset 그래프 표시
    _render_final_dataset_graph(df)
    st.divider()
    _render_sarimax(df)
