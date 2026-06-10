"""
utils/similar_periods.py — 과거 유사 시점 탐색기 로직 모듈

pages/3_forecast.py 의 "4) 유사 시점 탐색기" 메뉴와
pages/5_similar_periods.py 독립 페이지 양쪽에서 공유한다.
UI 코드(render_similar_periods)와 데이터 함수를 한 곳에 유지해 중복을 없앤다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

# ── 상수 ──────────────────────────────────────────────────────────────────────
# 유사 풀에서 제외할 극단 사건 구간
EXCLUDE_PERIODS: list[tuple[str, str, str]] = [
    ("2008-09-01", "2009-06-30", "글로벌 금융위기(GFC)"),
    ("2020-02-15", "2020-06-30", "COVID-19 충격"),
]

# 유사도 계산 7개 표준화 피처 컬럼명
FEAT_COLS: list[str] = [
    "std_score",
    "std_prob_bear",
    "std_dxy_chg",
    "std_y10_chg",
    "std_financial_conditions",
    "std_credit_spread",
    "std_vix",
]

# 화면 표시용 레이블
FEAT_LABEL: dict[str, str] = {
    "std_score":                "이탈 점수",
    "std_prob_bear":            "P(약세)",
    "std_dxy_chg":              "DXY 변화율",
    "std_y10_chg":              "10Y 금리 변화",
    "std_financial_conditions": "NFCI",
    "std_credit_spread":        "HY 스프레드",
    "std_vix":                  "VIX",
}

# 지표 비교 카드용 (컬럼명, 표시명, 포맷 함수)
COMPARE_METRICS: list[tuple] = [
    ("score",                "이탈 점수",     lambda v: f"{v:.4f}"),
    ("prob_bear",            "P(약세)",       lambda v: f"{v * 100:.1f}%"),
    ("dxy_chg",              "DXY 변화율",    lambda v: f"{v * 100:.2f}%"),
    ("y10_chg",              "10Y 금리 변화", lambda v: f"{v:.3f}pp"),
    ("financial_conditions", "NFCI",          lambda v: f"{v:.4f}"),
    ("credit_spread",        "HY 스프레드",   lambda v: f"{v:.2f}%"),
    ("vix",                  "VIX",           lambda v: f"{v:.1f}"),
]

# 이후 수익률 horizon (주 단위)
HORIZONS: list[int] = [4, 13, 26]


# ── 데이터 구축 (캐싱) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="유사 시점 탐색 데이터 준비 중… (최초 1회)")
def _build_similarity_dataset(csv_path: str = "data/processed/final_dataset.csv") -> tuple[pd.DataFrame, list[str]]:
    """
    7개 표준화 피처 + 원값 + S&P500 가격이 담긴 분석용 DataFrame 구축.

    표준화 원칙 (동결 일관성):
      - 거시 5종 : DeviationDashboard의 비위기 기준 StandardScaler 그대로 재사용
      - 이탈 점수 : 비위기 구간만으로 별도 StandardScaler fit 후 전체 적용
      - P(약세)   : 동일 방식
    """
    from utils.deviation_dashboard import DeviationDashboard
    from utils.ms_ar_model import fit_ms_ar_model

    # 1. DeviationDashboard 실행
    dd = DeviationDashboard(csv_path=csv_path)
    if not (dd.load_data() and dd.calculate_features() and dd.fit_clusters() and dd.calculate_scores()):
        raise RuntimeError("DeviationDashboard 계산 실패")

    # 2. MS-AR filtered P(약세) 시계열 (캐시 재사용)
    ms_ar = fit_ms_ar_model(csv_path)
    prob_df = ms_ar["prob_bear_series"].dropna(subset=["prob_bear"])

    # 3. 이탈 점수 + 표준화 거시 피처 + 원값 + S&P500 취합
    std_macro = [
        "__std_dxy_chg", "__std_y10_chg", "__std_financial_conditions",
        "__std_credit_spread", "__std_vix",
    ]
    raw_macro = ["dxy_chg", "y10_chg", "financial_conditions", "credit_spread", "vix"]
    base = ["date", "score", "crisis", "p_peace", "p_danger"] + std_macro + raw_macro
    if "sp500" in dd.df_valid.columns:
        base.append("sp500")

    score_df = dd.df_valid[base].copy()
    score_df = score_df.rename(columns={c: c.replace("__std_", "std_") for c in std_macro})

    # 4. P(약세) 날짜 기준 병합 (inner join)
    df = pd.merge(score_df, prob_df, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    # 5. 이탈 점수·P(약세) → 비위기 기준 표준화 (동결)
    nc = ~df["crisis"]
    sc_score = StandardScaler()
    sc_score.fit(df.loc[nc, ["score"]])
    df["std_score"] = sc_score.transform(df[["score"]]).ravel()

    sc_prob = StandardScaler()
    sc_prob.fit(df.loc[nc, ["prob_bear"]])
    df["std_prob_bear"] = sc_prob.transform(df[["prob_bear"]]).ravel()

    # 6. 국면 라벨 (KMeans 결과 기반, 수동 인덱싱 금지)
    std_for_km = ["std_dxy_chg", "std_y10_chg", "std_financial_conditions", "std_credit_spread", "std_vix"]
    df["cluster_km"] = dd.km.predict(df[std_for_km].values)
    df["regime_label"] = df["cluster_km"].apply(
        lambda c: "평온강세" if c == dd.peacetime_label
        else ("위험주의" if c == dd.danger_label else "중간국면")
    )

    return df, FEAT_COLS


# ── 유사 시점 탐색 ─────────────────────────────────────────────────────────────
def _find_similar(
    df: pd.DataFrame,
    feat_cols: list[str],
    k: int = 6,
    min_gap_weeks: int = 8,
) -> tuple[pd.DataFrame, float]:
    """
    현재(최신 행)와 가장 유사한 과거 K개 시점을 유클리드 거리로 탐색.
    greedy 선정: 거리 오름차순 + 선택 시점 간 최소 min_gap_weeks 보장.
    """
    excl = pd.Series(False, index=df.index)
    for s, e, _ in EXCLUDE_PERIODS:
        excl |= (df["date"] >= pd.Timestamp(s)) & (df["date"] <= pd.Timestamp(e))
    nc_p99 = float(df.loc[~df["crisis"], "score"].quantile(0.99))
    excl |= df["score"] > nc_p99

    cur_vec = df.iloc[-1][feat_cols].values.astype(float)

    pool = df.iloc[:-1].copy()
    pool = pool[~excl.reindex(pool.index, fill_value=False)].dropna(subset=feat_cols).reset_index(drop=True)

    X = pool[feat_cols].values.astype(float)
    pool["dist"] = np.sqrt(((X - cur_vec) ** 2).sum(axis=1))
    pool = pool.sort_values("dist").reset_index(drop=True)

    selected: list[pd.Series] = []
    chosen_dates: list[pd.Timestamp] = []
    for _, row in pool.iterrows():
        d = pd.Timestamp(row["date"])
        if any(abs((d - prev).days) < min_gap_weeks * 7 for prev in chosen_dates):
            continue
        selected.append(row)
        chosen_dates.append(d)
        if len(selected) >= k:
            break

    return pd.DataFrame(selected).reset_index(drop=True), nc_p99


# ── 이후 수익률 계산 ────────────────────────────────────────────────────────────
def _compute_forward_returns(df: pd.DataFrame, similar: pd.DataFrame) -> pd.DataFrame:
    """각 유사 시점의 이후 4/13/26주 S&P500 단순 수익률 계산. 미래 미도래 시 NaN."""
    sp_series = df.set_index("date")["sp500"]
    sorted_dates = df["date"].sort_values().reset_index(drop=True)

    rows = []
    for _, row in similar.iterrows():
        t0 = pd.Timestamp(row["date"])
        p0 = sp_series.get(t0, np.nan)
        rec: dict = {
            "date": t0,
            "dist": round(float(row["dist"]), 3),
            "regime_label": row["regime_label"],
            "sp500_t0": p0,
        }
        for h in HORIZONS:
            target = t0 + pd.Timedelta(weeks=h)
            future = sorted_dates[sorted_dates >= target]
            if future.empty or np.isnan(p0):
                rec[f"ret_{h}w"] = np.nan
            else:
                p_fwd = sp_series.get(future.iloc[0], np.nan)
                rec[f"ret_{h}w"] = float((p_fwd / p0) - 1) if not np.isnan(p_fwd) else np.nan
        rows.append(rec)

    return pd.DataFrame(rows)


def _build_dist_figure(ret_df: pd.DataFrame) -> go.Figure:
    """이후 수익률 분포 — 롤리팝 차트, horizon별 3행 subplot."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f"이후 {h}주 수익률" for h in HORIZONS],
        vertical_spacing=0.14,
    )

    for row_i, h in enumerate(HORIZONS, start=1):
        col = f"ret_{h}w"
        valid = ret_df.dropna(subset=[col]).copy()

        if valid.empty:
            fig.add_annotation(
                text="데이터 부족 (미래 기간이 아직 도래하지 않음)",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                row=row_i, col=1,
            )
            continue

        x_vals = (valid[col] * 100).round(2).tolist()
        y_labels = [d.strftime("%Y-%m-%d") for d in valid["date"]]
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in x_vals]

        for x, y, c in zip(x_vals, y_labels, colors):
            fig.add_trace(
                go.Scatter(x=[0, x], y=[y, y], mode="lines",
                           line=dict(color=c, width=1.8),
                           showlegend=False, hoverinfo="skip"),
                row=row_i, col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_labels, mode="markers",
                marker=dict(size=11, color=colors, line=dict(color="white", width=1)),
                hovertemplate="%{y}<br>이후 " + f"{h}주: " + "%{x:.1f}%<extra></extra>",
                showlegend=False,
            ),
            row=row_i, col=1,
        )
        fig.add_vline(x=0, line_width=1.2, line_color="#6b7280", opacity=0.6, row=row_i, col=1)
        fig.update_xaxes(ticksuffix="%", row=row_i, col=1)

    fig.update_layout(height=520, margin=dict(t=60, l=110, r=30))
    return fig


def _dist_headline(ret_df: pd.DataFrame, h: int) -> tuple[str, str, str, str]:
    """지정 horizon의 (중앙값, 범위, 상승비율, N) 문자열 반환."""
    vals = ret_df[f"ret_{h}w"].dropna() * 100
    if vals.empty:
        return "—", "—", "—", "0"
    med = vals.median()
    mn, mx = vals.min(), vals.max()
    pct_up = (vals > 0).mean() * 100
    return f"{med:+.1f}%", f"{mn:.1f}%~{mx:+.1f}%", f"{pct_up:.0f}%", str(len(vals))


def _build_mini_chart(
    df: pd.DataFrame,
    t0: pd.Timestamp,
    ret_4w: float = np.nan,
    ret_13w: float = np.nan,
    ret_26w: float = np.nan,
    window_weeks: int = 26,
) -> go.Figure:
    """유사 시점 전후 S&P500 미니 차트 (T=0 기준 % 정규화, ±window_weeks주)."""
    start = t0 - pd.Timedelta(weeks=window_weeks)
    end_dt = t0 + pd.Timedelta(weeks=window_weeks)

    window = df[(df["date"] >= start) & (df["date"] <= end_dt)][["date", "sp500"]].copy()
    if window.empty:
        return go.Figure()

    p0_rows = df.loc[df["date"] == t0, "sp500"]
    if p0_rows.empty:
        return go.Figure()
    p0 = float(p0_rows.iloc[0])

    window["pct"] = (window["sp500"] / p0 - 1) * 100

    fig = go.Figure()
    fig.add_vrect(
        x0=start.isoformat(), x1=t0.isoformat(),
        fillcolor="#f3f4f6", opacity=0.55, layer="below", line_width=0,
    )
    fig.add_trace(go.Scatter(
        x=window["date"], y=window["pct"],
        mode="lines", line=dict(color="#2563eb", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>T=0 대비 %{y:+.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_width=0.8, line_color="#9ca3af")
    fig.add_vline(x=str(t0), line_width=1.5, line_color="#dc2626", line_dash="dash")

    for h, ret_val, color in [(4, ret_4w, "#16a34a"), (13, ret_13w, "#0ea5e9"), (26, ret_26w, "#7c3aed")]:
        if not np.isnan(ret_val):
            t_fwd = t0 + pd.Timedelta(weeks=h)
            fwd_rows = df[df["date"] >= t_fwd]
            if not fwd_rows.empty:
                act_date = fwd_rows["date"].iloc[0]
                pct_val = (float(fwd_rows["sp500"].iloc[0]) / p0 - 1) * 100
                fig.add_trace(go.Scatter(
                    x=[act_date], y=[pct_val], mode="markers",
                    marker=dict(size=9, color=color, line=dict(color="white", width=1.5)),
                    hovertemplate=f"T+{h}주: {ret_val * 100:+.1f}%<extra></extra>",
                    showlegend=False,
                ))
                fig.add_annotation(
                    x=act_date, y=pct_val, text=f"T+{h}w",
                    showarrow=False, yshift=13, font=dict(size=9, color=color),
                )

    fig.update_layout(
        height=230, showlegend=False,
        margin=dict(t=8, b=8, l=58, r=12),
        yaxis=dict(ticksuffix="%", zeroline=False),
        xaxis=dict(showgrid=False),
        hovermode="x unified", plot_bgcolor="white",
    )
    return fig


def _build_llm_query(
    current: pd.Series,
    similar: pd.DataFrame,
    ret_df: "pd.DataFrame | None",
) -> str:
    """Claude.ai 등 LLM 채팅창에 붙여넣을 문의 텍스트 생성. API 호출 없음."""
    cur_summary = (
        f"현재({pd.Timestamp(current['date']).strftime('%Y-%m-%d')}) — "
        f"국면: {current['regime_label']} | "
        f"이탈점수: {float(current['score']):.4f} | P(약세): {float(current['prob_bear']) * 100:.1f}% | "
        f"VIX: {float(current['vix']):.1f} | DXY변화: {float(current['dxy_chg']) * 100:.2f}% | "
        f"10Y금리변화: {float(current['y10_chg']):.3f}pp | HY스프레드: {float(current['credit_spread']):.2f}%"
    )

    sim_lines = []
    for rank, (_, row) in enumerate(similar.iterrows(), start=1):
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        ret_parts = []
        if ret_df is not None:
            match = ret_df[ret_df["date"] == pd.Timestamp(row["date"])]
            if not match.empty:
                rr = match.iloc[0]
                for h in [4, 13, 26]:
                    v = rr.get(f"ret_{h}w", np.nan)
                    ret_parts.append(
                        f"{h}주후 S&P500 {float(v) * 100:+.1f}%" if not np.isnan(float(v)) else f"{h}주후 N/A"
                    )
        ret_str = " / ".join(ret_parts) if ret_parts else "이후 수익률 데이터 없음"
        sim_lines.append(
            f"{rank}. {date_str} | 국면: {row['regime_label']} | "
            f"이탈점수: {float(row['score']):.4f} | P(약세): {float(row['prob_bear']) * 100:.1f}% | "
            f"VIX: {float(row['vix']):.1f} | DXY변화: {float(row['dxy_chg']) * 100:.2f}% | "
            f"10Y금리변화: {float(row['y10_chg']):.3f}pp | HY스프레드: {float(row['credit_spread']):.2f}% | "
            f"{ret_str}"
        )

    return (
        "지금과 가장 유사한 과거 시장상황을 찾아서 향후 주식시장을 예측해보는게 가능할지 검토중이야. "
        "아래 제시한 유사시점의 주식시장을 요약해주고 지금 주식시장에 투자자로서 시사점을 얻을수 있는지 의견 줘\n\n"
        f"[현재 시장 상태]\n{cur_summary}\n\n"
        f"[유사 시점 리스트 (유사도 상위 {len(similar)}개)]\n"
        + "\n".join(sim_lines)
    )


# ── 메인 UI 렌더 함수 ─────────────────────────────────────────────────────────
def render_similar_periods() -> None:
    """
    과거 유사 시점 탐색기 전체 UI.
    pages/3_forecast.py 의 메뉴에서도, pages/5_similar_periods.py 독립 페이지에서도 호출된다.
    """
    st.subheader("🔍 과거 유사 시점 탐색기")
    st.caption(
        "현재 국면·이탈 점수·P(약세)·거시 5지표와 가장 닮은 과거 시점을 찾아 맥락을 비교합니다.  "
        "**현재 상태 측정(coincident) · 예측 아님 · 투자권유 아님**"
    )

    st.warning(
        "⚠️ **이 도구의 한계 — 반드시 확인**\n\n"
        "- 극단 위기(2008 GFC·2020 COVID 등)는 유사 풀에서 **제외**됩니다 → "
        "아래 유사 시점 범위는 **비극단 환경만 반영**하며, 최악의 꼬리위험은 보여주지 않습니다.\n"
        "- '닮은 과거가 있다'는 것이 이후 결과도 같다는 의미가 아닙니다. "
        "**패턴이 아니라 불확실성의 범위를 보는 도구**입니다.",
        icon="⚠️",
    )

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    try:
        df, feat_cols = _build_similarity_dataset()
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    current = df.iloc[-1]  # 현재 = 최신 행

    # ── 현재 상태 KPI ─────────────────────────────────────────────────────────
    st.markdown("### 📍 현재 상태 (검색 기준점)")
    st.caption(f"기준일: **{pd.Timestamp(current['date']).strftime('%Y-%m-%d')}**  ·  이 벡터와 가장 가까운 과거 시점을 탐색합니다.")

    r1 = st.columns(4)
    r1[0].metric("이탈 점수", f"{current['score']:.4f}")
    r1[1].metric("이탈 점수 (표준화)", f"{current['std_score']:.2f}σ")
    r1[2].metric("P(약세) — raw filtered", f"{current['prob_bear'] * 100:.1f}%")
    r1[3].metric("국면", current["regime_label"])

    r2 = st.columns(5)
    r2[0].metric("DXY 13주 변화율", f"{current['dxy_chg'] * 100:.2f}%")
    r2[1].metric("10Y 금리 변화(pp)", f"{current['y10_chg']:.3f}")
    r2[2].metric("NFCI 금융여건", f"{current['financial_conditions']:.4f}")
    r2[3].metric("HY 신용스프레드", f"{current['credit_spread']:.2f}%")
    r2[4].metric("VIX", f"{current['vix']:.1f}")

    st.divider()

    # ── 검색 파라미터 ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ 검색 파라미터")
    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("유사 시점 수 (K)", min_value=3, max_value=10, value=6, step=1,
                      help="반환할 독립 유사 시점 개수", key="sp_k")
    with c2:
        min_gap = st.slider("에피소드 최소 간격 (주)", min_value=4, max_value=26, value=8, step=2,
                            help="선택 시점 간 최소 주 간격 — 같은 시장 에피소드 중복 방지",
                            key="sp_min_gap")

    similar, nc_p99 = _find_similar(df, feat_cols, k=k, min_gap_weeks=min_gap)

    # 이후 수익률 — 1단계 목록·2단계 분포·3단계 카드 공용 사전 계산
    ret_df: pd.DataFrame | None = (
        _compute_forward_returns(df, similar)
        if (not similar.empty and "sp500" in df.columns)
        else None
    )

    with st.expander("📋 유사 풀 제외 기준 상세", expanded=False):
        lines = [f"- **{s} ~ {e}**: {label}" for s, e, label in EXCLUDE_PERIODS]
        lines.append(f"- **이탈 점수 > {nc_p99:.4f}** (비위기 분포 99백분위): 극단 비위기 이상치 제외")
        lines.append("- 현재 행 자체(최신 주) 제외")
        st.markdown("\n".join(lines))
        total_excluded = sum(
            int(((df["date"] >= pd.Timestamp(s)) & (df["date"] <= pd.Timestamp(e))).sum())
            for s, e, _ in EXCLUDE_PERIODS
        )
        extreme_excl = int((df["score"] > nc_p99).sum())
        st.caption(
            f"전체 {len(df)}주 중 위기 구간 제외 ~{total_excluded}주, "
            f"극단 이상치 제외 {extreme_excl}주 → 유사 풀 약 {len(df) - total_excluded - extreme_excl}주"
        )

    st.divider()

    # ── 유사 시점 목록 ─────────────────────────────────────────────────────────
    st.markdown("### 🗂️ 유사 시점 목록")
    st.caption(
        "거리(유클리드): 7개 표준화 피처(이탈 점수·P(약세)·거시 5종)의 거리. "
        "값이 작을수록 현재와 닮은 환경입니다."
    )

    if similar.empty:
        st.warning("검색 조건에 맞는 유사 시점을 찾지 못했습니다. K 또는 최소 간격을 조정해 보세요.")
    else:
        current_row_disp = {
            "순위": "▶현재",
            "날짜": pd.Timestamp(current["date"]).strftime("%Y-%m-%d"),
            "거리": "—",
            "국면": current["regime_label"],
            "이탈 점수": round(float(current["score"]), 4),
            "P(약세)": f"{current['prob_bear'] * 100:.1f}%",
            "DXY 변화율": f"{current['dxy_chg'] * 100:.2f}%",
            "10Y 금리 변화": f"{current['y10_chg']:.3f}",
            "NFCI": f"{current['financial_conditions']:.4f}",
            "HY 스프레드": f"{current['credit_spread']:.2f}",
            "VIX": f"{current['vix']:.1f}",
        }
        sim_rows = []
        for rank, (_, row) in enumerate(similar.iterrows(), start=1):
            sim_rows.append({
                "순위": str(rank),
                "날짜": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "거리": f"{row['dist']:.3f}",
                "국면": row["regime_label"],
                "이탈 점수": round(float(row["score"]), 4),
                "P(약세)": f"{row['prob_bear'] * 100:.1f}%",
                "DXY 변화율": f"{row['dxy_chg'] * 100:.2f}%",
                "10Y 금리 변화": f"{row['y10_chg']:.3f}",
                "NFCI": f"{row['financial_conditions']:.4f}",
                "HY 스프레드": f"{row['credit_spread']:.2f}",
                "VIX": f"{row['vix']:.1f}",
            })

        table_df = pd.concat(
            [pd.DataFrame([current_row_disp]), pd.DataFrame(sim_rows)],
            ignore_index=True,
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        regime_counts = similar["regime_label"].value_counts().to_dict()
        regime_summary = "  ·  ".join(f"**{r}** {c}건" for r, c in regime_counts.items())
        st.caption(f"유사 시점 국면 분포: {regime_summary}")

    # ── 이후 수익률 분포 ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 이후 수익률 분포")
    st.caption(
        "각 유사 시점 이후 S&P500이 어떻게 움직였는지 **'분포'**로 표시합니다.  "
        "중앙값 하나가 아니라 **흩어짐(발산) 자체**가 핵심 정보입니다."
    )

    if ret_df is not None:
        recent_nan = ret_df[ret_df[[f"ret_{h}w" for h in HORIZONS]].isna().any(axis=1)]
        if not recent_nan.empty:
            st.warning(
                f"⚠️ 최근 유사 시점 {len(recent_nan)}개는 아직 해당 기간이 도래하지 않아 "
                "일부 horizon 수익률을 계산할 수 없습니다 (NaN).",
                icon="⚠️",
            )

        st.markdown(
            "<div style='background:#fef9c3;border:1px solid #fde047;border-radius:8px;"
            "padding:9px 12px;margin-bottom:10px;font-size:12.5px;color:#713f12;line-height:1.5;'>"
            "⚠️ <strong>극단 위기(GFC·COVID)는 유사 풀 제외 → 아래 범위는 비극단 환경만 반영</strong>합니다."
            "</div>",
            unsafe_allow_html=True,
        )

        h_cols = st.columns(3)
        for col_i, h in enumerate(HORIZONS):
            med, rng, pct_up, n = _dist_headline(ret_df, h)
            h_cols[col_i].markdown(
                f"**이후 {h}주** (N={n})  \n"
                f"범위: `{rng}`  \n"
                f"중앙값: `{med}`  ·  상승 `{pct_up}`",
            )

        st.plotly_chart(_build_dist_figure(ret_df), use_container_width=True, key="sp_dist_chart")

        with st.expander("📋 유사 시점별 수익률 상세", expanded=False):
            tbl_rows = []
            for _, row in ret_df.iterrows():
                tbl_rows.append({
                    "날짜": row["date"].strftime("%Y-%m-%d"),
                    "국면": row["regime_label"],
                    "거리": f"{row['dist']:.3f}",
                    "이후 4주(%)": f"{row['ret_4w']*100:.1f}" if not np.isnan(row["ret_4w"]) else "—",
                    "이후 13주(%)": f"{row['ret_13w']*100:.1f}" if not np.isnan(row["ret_13w"]) else "—",
                    "이후 26주(%)": f"{row['ret_26w']*100:.1f}" if not np.isnan(row["ret_26w"]) else "—",
                })
            st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

        st.caption("수익률 = 유사 시점 종가 대비 이후 N주 S&P500 단순 수익률.  초록=상승 · 빨강=하락.")

    elif not similar.empty:
        st.info("sp500 가격 데이터를 찾을 수 없어 수익률 분포를 계산할 수 없습니다.")

    # ── 맥락 카드 ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🃏 맥락 카드")
    st.caption(
        "각 유사 시점의 거시·국면 맥락을 현재와 나란히 비교합니다.  "
        "**앱은 당시 사건을 단정하지 않습니다** — 직접 찾아볼 날짜 단서만 제공합니다."
    )

    if similar.empty:
        st.info("유사 시점이 없어 맥락 카드를 표시할 수 없습니다.")
    else:
        sort_options = ["유사도순 (기본)", "날짜순"]
        if ret_df is not None:
            sort_options = ["유사도순 (기본)", "이후 26주 수익률순", "이후 13주 수익률순", "날짜순"]

        ctrl_c1, ctrl_c2 = st.columns(2)
        with ctrl_c1:
            sort_by = st.selectbox("정렬 기준", sort_options, key="sp_card_sort")
        with ctrl_c2:
            regime_options = ["전체"] + sorted(similar["regime_label"].unique().tolist())
            filter_regime = st.selectbox("국면 필터", regime_options, key="sp_card_filter")

        if ret_df is not None:
            card_data = similar.merge(
                ret_df[["date", "ret_4w", "ret_13w", "ret_26w"]], on="date", how="left"
            )
        else:
            card_data = similar.copy()
            for _rc in ["ret_4w", "ret_13w", "ret_26w"]:
                card_data[_rc] = np.nan

        if filter_regime != "전체":
            card_data = card_data[card_data["regime_label"] == filter_regime]

        if sort_by == "이후 26주 수익률순":
            card_data = card_data.sort_values("ret_26w", ascending=False, na_position="last")
        elif sort_by == "이후 13주 수익률순":
            card_data = card_data.sort_values("ret_13w", ascending=False, na_position="last")
        elif sort_by == "날짜순":
            card_data = card_data.sort_values("date")

        card_data = card_data.reset_index(drop=True)

        if card_data.empty:
            st.info(f"'{filter_regime}' 국면에 해당하는 유사 시점이 없습니다.")
        else:
            for card_idx, (_, row) in enumerate(card_data.iterrows()):
                t0 = pd.Timestamp(row["date"])
                date_str = t0.strftime("%Y-%m-%d")
                year_month_kr = f"{t0.year}년 {t0.month:02d}월"

                r4  = float(row["ret_4w"])  if not pd.isna(row["ret_4w"])  else np.nan
                r13 = float(row["ret_13w"]) if not pd.isna(row["ret_13w"]) else np.nan
                r26 = float(row["ret_26w"]) if not pd.isna(row["ret_26w"]) else np.nan

                header_label = (
                    f"**{card_idx + 1}위** &nbsp; {date_str} &nbsp;|&nbsp; "
                    f"{row['regime_label']} &nbsp;|&nbsp; 거리: {row['dist']:.3f}"
                )

                with st.expander(header_label, expanded=(card_idx == 0)):
                    col_left, col_right = st.columns([2, 3])

                    with col_left:
                        st.markdown("**지표 비교 — 현재 vs 유사 시점**")
                        cmp_rows = []
                        for col_name, label, fmt_fn in COMPARE_METRICS:
                            cur_v = current.get(col_name, np.nan)
                            sim_v = row.get(col_name, np.nan)
                            try:
                                cur_str = fmt_fn(float(cur_v)) if not pd.isna(cur_v) else "—"
                            except (TypeError, ValueError):
                                cur_str = "—"
                            try:
                                sim_str = fmt_fn(float(sim_v)) if not pd.isna(sim_v) else "—"
                            except (TypeError, ValueError):
                                sim_str = "—"
                            cmp_rows.append({"지표": label, "현재": cur_str, "유사 시점": sim_str})
                        st.dataframe(
                            pd.DataFrame(cmp_rows),
                            use_container_width=True, hide_index=True, height=268,
                        )

                        st.markdown("**이후 S&P500 수익률**")
                        ret_lines = []
                        for h, val in [(4, r4), (13, r13), (26, r26)]:
                            if not np.isnan(val):
                                arrow = "▲" if val >= 0 else "▼"
                                ret_lines.append(f"- {h}주후: **{arrow} {val * 100:+.1f}%**")
                            else:
                                ret_lines.append(f"- {h}주후: — *(미집계)*")
                        st.markdown("\n".join(ret_lines))

                    with col_right:
                        st.markdown("**S&P500 전후 52주 (T=0 정규화, 회색=과거 / 흰색=이후)**")
                        if "sp500" in df.columns:
                            mini_fig = _build_mini_chart(df, t0, ret_4w=r4, ret_13w=r13, ret_26w=r26)
                            st.plotly_chart(
                                mini_fig, use_container_width=True,
                                key=f"sp_mini_{card_idx}_{date_str}",
                            )
                        else:
                            st.caption("sp500 가격 데이터 없음")

                    st.markdown("---")
                    st.markdown(
                        f"🔍 **직접 알아보기 단서** — 앱은 당시 사건을 서술하지 않습니다. "
                        f"아래 날짜로 직접 검색해 맥락을 파악하세요.\n\n"
                        f"&emsp;📅 검색 기간: **{year_month_kr} 전후**\n\n"
                        f"&emsp;💡 확인 항목: S&P500 흐름, 금리·달러·신용스프레드·VIX가 당시 "
                        f"어떤 상황이었는지 외부에서 직접 확인",
                    )

    # ── LLM 문의 내용 생성 ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 LLM 문의 내용 생성")
    st.caption("유사 시점 수치를 정리한 문의 텍스트를 생성합니다. 복사 후 Claude.ai 등 LLM 채팅창에 붙여넣으세요.")

    if similar.empty:
        st.info("유사 시점이 없어 문의 내용을 생성할 수 없습니다.")
    else:
        query_c1, query_c2, _ = st.columns([1, 1, 4])
        with query_c1:
            build_clicked = st.button("📋 문의 내용 생성", key="sp_llm_build")
        with query_c2:
            if st.button("🗑️ 초기화", key="sp_llm_clear"):
                st.session_state.pop("sp_llm_query", None)
                st.rerun()

        if build_clicked:
            st.session_state["sp_llm_query"] = _build_llm_query(current, similar, ret_df)

        if "sp_llm_query" in st.session_state:
            st.markdown("아래 내용을 **전체 복사**해서 Claude.ai 등 LLM 채팅창에 붙여넣으세요.")
            st.text_area(
                label="LLM 문의 텍스트",
                value=st.session_state["sp_llm_query"],
                height=400,
                key="sp_llm_query_display",
                label_visibility="collapsed",
            )
            st.caption(
                f"기준일 {pd.Timestamp(current['date']).strftime('%Y-%m-%d')} · "
                f"유사 시점 K={len(similar)}  ·  "
                "슬라이더를 변경했다면 '🗑️ 초기화' 후 재생성하세요."
            )
