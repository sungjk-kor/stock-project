"""
1_data_view.py  —  데이터 수집 & 현황 대시보드

기능:
  1) 스마트 수집  : 요청 기간 중 파일에 없는 구간만 수집 (증분 업데이트)
  2) 데이터 현황표: 파일별 날짜 범위·행수·결측값 요약 → 클릭하면 시계열 차트
  3) 결측 구간    : 차트에서 붉은 음영으로 표시
  4) 최종 데이터셋 빌드
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_pipeline import (
    build_final_dataset,
    collect_and_build_dataset,
    collect_raw_data,
    collect_raw_data_incremental,
    get_raw_data_status,
)
from utils.preprocessing import PROJECT_ROOT, load_processed_dataset


# ── 상수 ──────────────────────────────────────────────────────────────────────

RAW_DIR    = PROJECT_ROOT / "data" / "raw"
ACCENT     = "#f97316"
# 금리·스프레드 컬럼은 우측 Y축으로 분리
RATE_COLS  = ("fed_funds", "treasury_2y", "treasury_10y", "term_spread_10y_2y",
               "credit_spread")

# 파일명 → 한글 설명
FILE_LABELS = {
    "market":    "◆ 시장 (주가·VIX)",
    "rates":     "▣ 금리 (국채·정책금리)",
    "macro":     "● 거시 (CPI·고용·생산)",
    "liquidity": "▤ 유동성 (M2·달러·금·원유)",
}

# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def _is_rate_col(col: str) -> bool:
    return col.startswith(RATE_COLS)


def _make_timeseries_chart(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    """
    선택한 컬럼들의 시계열 차트.
    금리 계열은 우측 Y축으로 분리.
    결측값(NaN) 구간은 오렌지색 반투명 음영으로 표시.
    """
    rate_cols    = [c for c in columns if _is_rate_col(c)]
    has_sec_y    = bool(rate_cols)
    fig          = make_subplots(specs=[[{"secondary_y": has_sec_y}]])

    colors = [
        "#2563eb", "#f97316", "#16a34a", "#7c3aed",
        "#0891b2", "#ca8a04", "#0f766e", "#4f46e5",
    ]

    for i, col in enumerate(columns):
        color       = colors[i % len(colors)]
        sec_y       = _is_rate_col(col) if has_sec_y else False
        na_count    = int(df[col].isna().sum())
        legend_name = f"{col}" + (f"  ⚠ 결측 {na_count}개" if na_count > 0 else "")

        # 메인 라인
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df[col],
                mode="lines",
                name=legend_name,
                line=dict(color=color, width=1.5),
                connectgaps=False,   # NaN 위치에서 선 끊김 — 시각적으로 갭 표시
            ),
            secondary_y=sec_y,
        )

        # ── 결측 구간 붉은 음영 ─────────────────────────────────────────────
        if na_count > 0:
            _add_nan_shading(fig, df, col)

    fig.update_layout(
        height=440,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(title_text="날짜")
    fig.update_yaxes(title_text="값", secondary_y=False)
    if has_sec_y:
        fig.update_yaxes(title_text="금리 / 스프레드 (%)", secondary_y=True)
    return fig


def _add_nan_shading(fig: go.Figure, df: pd.DataFrame, col: str) -> None:
    """
    NaN 구간(연속 결측)을 오렌지색 반투명 사각형으로 표시.
    단독 NaN 1개짜리도 세로선으로 표시.
    """
    nan_mask = df[col].isna()
    if not nan_mask.any():
        return

    dates      = df["date"].values
    in_gap     = False
    gap_start  = None
    added_once = False   # 범례 중복 방지

    for i, is_na in enumerate(nan_mask):
        if is_na and not in_gap:
            in_gap    = True
            gap_start = dates[i]
        elif not is_na and in_gap:
            in_gap   = False
            gap_end  = dates[i]
            fig.add_vrect(
                x0=pd.Timestamp(gap_start), x1=pd.Timestamp(gap_end),
                fillcolor=ACCENT, opacity=0.12,
                layer="below", line_width=0,
                annotation_text="결측" if not added_once else "",
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=ACCENT,
            )
            added_once = True

    # 끝까지 NaN이 이어지는 경우
    if in_gap:
        fig.add_vrect(
            x0=pd.Timestamp(gap_start), x1=pd.Timestamp(dates[-1]),
            fillcolor=ACCENT, opacity=0.12,
            layer="below", line_width=0,
        )


def _render_kpis(items: list[tuple[str, str]]) -> None:
    cards = "\n".join(
        f'<div class="data-kpi"><div class="data-kpi-label">{label}</div>'
        f'<div class="data-kpi-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="data-kpi-grid">{cards}</div>', unsafe_allow_html=True)


def _build_status_df(status: dict) -> pd.DataFrame:
    """
    get_raw_data_status() 결과를 Streamlit 표용 DataFrame으로 변환.
    """
    rows = []
    for name, info in status.items():
        label = FILE_LABELS.get(name, name)
        if not info.get("exists", False):
            rows.append({
                "파일": label,
                "시작일": "—",
                "종료일": "—",
                "행수": "—",
                "결측값": "—",
                "상태": "❌ 없음",
                "_name": name,
            })
        elif "error" in info:
            rows.append({
                "파일": label,
                "시작일": "오류",
                "종료일": "오류",
                "행수": "—",
                "결측값": "—",
                "상태": f"⚠ {info['error'][:30]}",
                "_name": name,
            })
        else:
            na = info.get("na_total", 0)
            rows.append({
                "파일": label,
                "시작일": str(info.get("start", "—")),
                "종료일": str(info.get("end",   "—")),
                "행수":   info.get("rows", 0),
                "결측값": na,
                "상태":   "✅ 정상" if na == 0 else f"⚠ 결측 {na}개",
                "_name": name,
            })
    return pd.DataFrame(rows)


# ── 페이지 레이아웃 ───────────────────────────────────────────────────────────

st.set_page_config(page_title="Data View", layout="wide")
st.markdown(
    f"""
    <style>
    .data-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px 0;
    }}
    .data-kpi {{
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }}
    .data-kpi-label {{
        color: #475569;
        font-size: 14px;
        line-height: 1.25;
        white-space: nowrap;
    }}
    .data-kpi-value {{
        color: #111827;
        font-size: 18px;
        font-weight: 700;
        line-height: 1.25;
        margin-top: 6px;
        overflow-wrap: anywhere;
    }}
    .data-note {{
        border-left: 4px solid {ACCENT};
        background: #fff7ed;
        padding: 12px 14px;
        border-radius: 8px;
        color: #334155;
        margin-bottom: 12px;
    }}
    @media (max-width: 900px) {{
        .data-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("데이터 수집 & 현황")
st.caption("시장, 금리, 거시, 유동성 데이터를 수집하고 최종 SARIMAX 입력 데이터셋을 관리합니다.")
st.markdown(
    '<div class="data-note">원천 데이터 수집과 최종 데이터셋 빌드를 한 화면에서 관리합니다. 결측 구간은 오렌지색 음영으로 표시됩니다.</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1: 수집 설정
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("데이터 수집 설정")

min_date = pd.Timestamp("1990-01-01").date()
max_date = pd.Timestamp.today().normalize().date()

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    start_date = st.date_input("시작일", value=pd.Timestamp("2000-01-01").date(),
                                min_value=min_date, max_value=max_date)
with c2:
    end_date = st.date_input("종료일", value=max_date,
                              min_value=min_date, max_value=max_date)
with c3:
    max_lag = st.number_input("Lag 주수", min_value=1, max_value=52, value=12)

if start_date > end_date:
    st.error("시작일이 종료일보다 늦습니다.")
    st.stop()

# ── 수집 버튼 3개 ─────────────────────────────────────────────────────────────

b1, b2, b3 = st.columns([1, 1, 1])

with b1:
    smart_btn = st.button(
        "🔍 스마트 수집 (증분)",
        type="primary",
        help="이미 있는 데이터는 건너뛰고, 없는 구간만 수집합니다.",
    )
with b2:
    full_btn = st.button(
        "♻️ 전체 재수집",
        type="secondary",
        help="기존 파일을 무시하고 처음부터 전부 다시 받습니다.",
    )
with b3:
    build_btn = st.button(
        "⚙️ 최종 데이터셋 빌드",
        type="primary",
        help="raw 파일로부터 final_dataset.csv를 만듭니다.",
    )

# ── 스마트 수집 실행 ──────────────────────────────────────────────────────────

if smart_btn:
    with st.spinner("기존 파일 확인 후 누락 구간 수집 중..."):
        results = collect_raw_data_incremental(
            start=str(start_date), end=str(end_date)
        )

    st.subheader("수집 결과")
    result_rows = []
    for name, info in results.items():
        action_map = {
            "skipped": "⏭ 건너뜀 (이미 보유)",
            "created": "✅ 신규 수집",
            "updated": "🔄 업데이트",
            "empty":   "⚠ 데이터 없음",
            "error":   f"❌ 오류: {info.get('error','')}",
        }
        action = action_map.get(info.get("action", ""), info.get("action", ""))
        coverage = info.get("coverage", (None, None))
        result_rows.append({
            "파일":      FILE_LABELS.get(name, name),
            "결과":      action,
            "추가 행수": info.get("rows_added", 0),
            "커버 기간": f"{coverage[0]} ~ {coverage[1]}" if coverage[0] else "—",
        })
    st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
    st.rerun()   # 수집 후 현황 테이블 갱신

# ── 전체 재수집 ───────────────────────────────────────────────────────────────

if full_btn:
    with st.spinner("전체 데이터 수집 중 (시간이 걸릴 수 있습니다)..."):
        datasets = collect_raw_data(start=str(start_date), end=str(end_date))
    saved = [FILE_LABELS.get(n, n) for n, df in datasets.items() if not df.empty]
    if saved:
        st.success(f"수집 완료: {', '.join(saved)}")
    else:
        st.warning("수집된 데이터 없음. 인터넷 연결과 FRED API 키를 확인하세요.")
    st.rerun()

# ── 최종 데이터셋 빌드 ────────────────────────────────────────────────────────

if build_btn:
    with st.spinner("주간 집계 → lag 변수 생성 → 저장 중..."):
        final_df = build_final_dataset(
            max_lag=int(max_lag), start=str(start_date), end=str(end_date)
        )
    if final_df.empty:
        st.warning("최종 데이터셋 생성 실패. 먼저 raw 데이터를 수집하세요.")
    else:
        st.success(f"✅ `data/processed/final_dataset.csv` 생성 완료: {len(final_df):,}행")

with st.expander("⚙️ 원클릭 전체 파이프라인 (수집 + 빌드 동시 실행)"):
    if st.button("전체 파이프라인 실행"):
        with st.spinner("수집 + 빌드 진행 중..."):
            final_df = collect_and_build_dataset(
                start=str(start_date), end=str(end_date), max_lag=int(max_lag)
            )
        if final_df.empty:
            st.warning("파이프라인 실패. 연결 및 API 키를 확인하세요.")
        else:
            st.success(f"파이프라인 완료: {len(final_df):,}행")


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2: 원시 데이터 현황
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("원시 데이터 현황")
st.caption("파일별 날짜 범위·결측 요약. 아래에서 파일을 선택하면 상세 차트가 표시됩니다.")

status = get_raw_data_status()
status_df = _build_status_df(status)

existing_count = sum(1 for info in status.values() if info.get("exists") and "error" not in info and info.get("rows", 0) > 0)
total_rows = sum(int(info.get("rows", 0)) for info in status.values() if info.get("exists") and "error" not in info)
total_na = sum(int(info.get("na_total", 0)) for info in status.values() if info.get("exists") and "error" not in info)
_render_kpis(
    [
        ("◆ 원천 파일", f"{existing_count} / {len(status)}"),
        ("▣ 총 행수", f"{total_rows:,}"),
        ("● 결측값", f"{total_na:,}"),
        ("▤ 저장 위치", "data/raw"),
    ]
)

# ── 현황 요약 테이블 ──────────────────────────────────────────────────────────

# 결측값 컬럼에 색상 적용
def _color_na(val):
    if isinstance(val, str) and "결측" in val:
        return "background-color: #fff7ed; color: #9a3412"
    if val == "❌ 없음":
        return "background-color: #f1f5f9; color: #475569"
    if val == "✅ 정상":
        return "background-color: #d1e7dd; color: #0f5132"
    return ""

display_cols = ["파일", "시작일", "종료일", "행수", "결측값", "상태"]
st.dataframe(
    status_df[display_cols].style.map(_color_na, subset=["상태"]),
    use_container_width=True,
    hide_index=True,
)

# ── 파일 선택 → 상세 뷰 ──────────────────────────────────────────────────────

available_files = [
    name for name, info in status.items()
    if info.get("exists") and "error" not in info and info.get("rows", 0) > 0
]

if not available_files:
    st.info("수집된 raw 파일이 없습니다. 위 **스마트 수집** 버튼을 눌러 데이터를 받으세요.")
else:
    st.markdown("#### 시계열 차트 뷰어")
    sel_name = st.radio(
        "파일 선택 (클릭하면 차트가 바뀜)",
        options=available_files,
        format_func=lambda x: FILE_LABELS.get(x, x),
        horizontal=True,
        key="raw_file_selector",
    )

    if sel_name and sel_name in status:
        info      = status[sel_name]
        raw_path  = info["path"]

        # raw CSV 로드
        try:
            raw_df = pd.read_csv(raw_path, parse_dates=["date"])
            raw_df = raw_df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
            st.stop()

        num_cols = [c for c in raw_df.columns if c != "date" and
                    pd.api.types.is_numeric_dtype(raw_df[c])]

        # ── 컬럼별 결측 상세 ─────────────────────────────────────────────────
        with st.expander("📌 컬럼별 결측값 상세", expanded=False):
            na_detail = pd.DataFrame({
                "컬럼":     num_cols,
                "결측 수":  [int(raw_df[c].isna().sum()) for c in num_cols],
                "결측률(%)": [round(raw_df[c].isna().mean() * 100, 2) for c in num_cols],
                "시작일":   [str(raw_df[raw_df[c].notna()]["date"].min().date())
                             if raw_df[c].notna().any() else "—" for c in num_cols],
                "종료일":   [str(raw_df[raw_df[c].notna()]["date"].max().date())
                             if raw_df[c].notna().any() else "—" for c in num_cols],
            })

            def _color_na_row(row):
                if row["결측 수"] > 0:
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            st.dataframe(
                na_detail.style.apply(_color_na_row, axis=1),
                use_container_width=True,
                hide_index=True,
            )

        # ── 컬럼 선택 ─────────────────────────────────────────────────────────
        default_cols = num_cols[:3] if len(num_cols) >= 3 else num_cols
        sel_cols = st.multiselect(
            "차트에 표시할 컬럼 선택 (최대 5개)",
            options=num_cols,
            default=default_cols,
            max_selections=5,
            key=f"col_sel_{sel_name}",
        )

        if sel_cols:
            # ── 시계열 차트 (결측 구간 붉은 음영) ─────────────────────────────
            na_cols_sel = [c for c in sel_cols if raw_df[c].isna().any()]
            if na_cols_sel:
                st.caption(
                    f"⚠ 결측값이 있는 컬럼: **{', '.join(na_cols_sel)}** — "
                    "차트에서 오렌지색 음영으로 표시됩니다."
                )

            fig = _make_timeseries_chart(raw_df, sel_cols)
            fig.update_layout(
                title=dict(
                    text=f"{FILE_LABELS.get(sel_name, sel_name)}  "
                         f"({raw_df['date'].min().date()} ~ {raw_df['date'].max().date()}, "
                         f"{len(raw_df):,}행)",
                    font=dict(size=13),
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # 원시 데이터 미리보기 (옵션)
            with st.expander("📄 원시 데이터 미리보기 (최근 50행)"):
                st.dataframe(raw_df[["date"] + sel_cols].tail(50),
                             use_container_width=True, hide_index=True)
        else:
            st.info("컬럼을 1개 이상 선택하면 차트가 표시됩니다.")


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3: 최종 처리 데이터셋
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("최종 처리 데이터셋 (final_dataset.csv)")

processed = load_processed_dataset()

if processed.empty:
    st.info("`data/processed/final_dataset.csv`가 없습니다. 위에서 데이터를 수집·빌드하세요.")
else:
    st.caption(f"행: {len(processed):,}  컬럼: {len(processed.columns):,}")
    _render_kpis(
        [
            ("◆ 행수", f"{len(processed):,}"),
            ("▣ 컬럼수", f"{len(processed.columns):,}"),
            ("● 시작일", str(pd.to_datetime(processed["date"]).min().date()) if "date" in processed.columns else "-"),
            ("▤ 종료일", str(pd.to_datetime(processed["date"]).max().date()) if "date" in processed.columns else "-"),
        ]
    )

    proc_num_cols = list(processed.select_dtypes("number").columns)

    # ── 금융여건 & 리스크 지표 ───────────────────────────────────────────────
    RISK_INDICATORS = [
        "financial_conditions", "nfci_leverage",
        "credit_spread", "vix", "term_spread_10y_2y",
    ]
    risk_avail = [c for c in RISK_INDICATORS if c in proc_num_cols]
    missing_risk = [c for c in RISK_INDICATORS if c not in proc_num_cols]

    if risk_avail:
        st.markdown("#### 금융여건 & 리스크 지표")
        risk_sel = st.multiselect(
            "리스크 지표 선택",
            options=risk_avail, default=risk_avail[:3],
            max_selections=5, key="proc_risk",
        )
        if risk_sel:
            st.plotly_chart(
                _make_timeseries_chart(processed, risk_sel),
                use_container_width=True,
            )
    if missing_risk:
        st.warning(
            f"아직 없는 지표: **{', '.join(missing_risk)}**  \n"
            "→ 원시 데이터 수집 후 **최종 데이터셋 빌드**를 실행하세요."
        )

    # ── 전체 컬럼 차트 ────────────────────────────────────────────────────────
    st.markdown("#### 전체 지표 차트")
    defaults = [c for c in ["sp500_log_return", "sp500", "vix", "financial_conditions"]
                if c in proc_num_cols]
    chart_cols = st.multiselect(
        "컬럼 선택 (최대 3개)",
        options=proc_num_cols,
        default=defaults[:3],
        max_selections=3,
        key="proc_chart",
    )
    if chart_cols:
        st.plotly_chart(
            _make_timeseries_chart(processed, chart_cols),
            use_container_width=True,
        )
    else:
        st.info("컬럼을 선택하면 차트가 표시됩니다.")

    # 데이터 테이블
    with st.expander("📄 최근 50행 데이터 미리보기"):
        st.dataframe(processed.tail(50), use_container_width=True)
