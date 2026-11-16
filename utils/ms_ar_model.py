# =============================================================================
# MS-AR 약세 확률 추정 엔진 (이탈 점수 교차검증용)
#
# `ms_ar_analysis.py`의 핵심 적합·정렬·일치도 분석 로직을 Streamlit 캐싱과
# 함께 재사용할 수 있도록 감싼 모듈입니다. 점수 계산 자체는 손대지 않고
# `utils.deviation_dashboard.DeviationDashboard`를 그대로 호출합니다.
#
# ★ 약세 = 낮은 평균 + 높은 분산 (국면 번호는 라벨 기반으로 식별, 수동 인덱싱 금지)
# ★ 전체 표본(위기 포함)으로 적합, search_reps + 시드 고정으로 재현성 확보
# =============================================================================

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="MS-AR 2국면 모델 적합 중... (최초 1회, 이후 캐시 사용)")
def fit_ms_ar_model(csv_path: str = "data/processed/final_dataset.csv", k_regimes: int = 2) -> dict:
    """2국면 MS-AR(1) 모델(switching_variance=True)을 적합하고 약세 국면을 식별한다.

    Returns:
        dict: prob_bear_series(date, prob_bear), 국면별 평균/분산/전이확률/기대지속기간,
              현재 P(약세), 수렴 여부 등을 담은 결과 묶음
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    returns = df["sp500_log_return"].dropna().values

    # AR(1): Y_t = c_s + phi_s * Y_{t-1} + sigma_s * eps_t
    endog = returns[1:]
    exog = returns[:-1]

    np.random.seed(42)
    model = MarkovRegression(endog, k_regimes=k_regimes, exog=exog, switching_variance=True)
    # statsmodels 0.15 이상은 search_reps 시작값 난수를 np.random.seed 로 고정하지 못하고
    # fit(rng=...) 로만 고정된다 (validation/outputs/검증_요약.md '원본 코드 문제' ①).
    # 0.14.x 에는 rng 인자가 없으므로 시그니처를 보고 분기한다 (0.14.x 동작은 그대로).
    if "rng" in inspect.signature(model.fit).parameters:
        result = model.fit(search_reps=20, disp=False, rng=np.random.default_rng(42))
    else:
        result = model.fit(search_reps=20, disp=False)

    # ★ 파라미터는 절대 수동 인덱싱하지 않고 param_names로 라벨링해 추출
    param_dict = dict(zip(model.param_names, result.params))
    means = np.array([param_dict[f"const[{i}]"] for i in range(k_regimes)])
    variances = np.array([param_dict[f"sigma2[{i}]"] for i in range(k_regimes)])

    # 약세 = 평균이 더 낮은 국면 (라벨 스위칭에 안전)
    bear_idx = int(np.argmin(means))
    bull_idx = 1 - bear_idx

    # statsmodels는 열-확률(column-stochastic): M[to, from]
    transition = result.regime_transition[:, :, 0]
    bear_to_bear = float(transition[bear_idx, bear_idx])
    bear_to_bull = float(transition[bull_idx, bear_idx])
    bull_to_bear = float(transition[bear_idx, bull_idx])
    bull_to_bull = float(transition[bull_idx, bull_idx])

    duration_bear = 1 / (1 - bear_to_bear) if bear_to_bear < 1 else float("inf")
    duration_bull = 1 / (1 - bull_to_bull) if bull_to_bull < 1 else float("inf")

    # filtered (실시간) P(약세) 시계열 — look-ahead 없음
    prob_bear_full = result.filtered_marginal_probabilities[:, bear_idx]
    df_prob = df[["date", "sp500_log_return"]].copy()
    df_prob["prob_bear"] = np.nan
    valid_idx = df_prob["sp500_log_return"].notna()
    valid_dates = df_prob.loc[valid_idx].index
    if len(valid_dates) > 0 and len(prob_bear_full) > 0:
        df_prob.loc[valid_dates[1:len(prob_bear_full) + 1], "prob_bear"] = prob_bear_full

    return {
        "prob_bear_series": df_prob[["date", "prob_bear"]],
        "bear_idx": bear_idx,
        "bull_idx": bull_idx,
        "mean_bear": float(means[bear_idx]),
        "mean_bull": float(means[bull_idx]),
        "var_bear": float(variances[bear_idx]),
        "var_bull": float(variances[bull_idx]),
        "var_ratio": float(variances[bear_idx] / variances[bull_idx]),
        "bear_to_bear": bear_to_bear,
        "bear_to_bull": bear_to_bull,
        "bull_to_bear": bull_to_bear,
        "bull_to_bull": bull_to_bull,
        "duration_bear_weeks": float(duration_bear),
        "duration_bull_weeks": float(duration_bull),
        "current_prob_bear": float(prob_bear_full[-1]) if len(prob_bear_full) else float("nan"),
        "current_date": str(df_prob["date"].iloc[-1].date()),
        "log_likelihood": float(result.llf),
        "converged": bool(result.mle_retvals.get("converged", False)),
    }


@st.cache_data(show_spinner=False)
def align_with_deviation_score(
    prob_bear_series: pd.DataFrame,
    csv_path: str = "data/processed/final_dataset.csv",
) -> pd.DataFrame:
    """이탈 점수(DeviationDashboard, 동결 표준화·센트로이드 그대로 재사용)와
    P(약세)를 날짜 기준으로 정렬한다. 위기구간에도 점수가 연속 존재하므로
    별도 재점수화는 불필요 — DeviationDashboard 결과를 그대로 병합한다.
    """
    from utils.deviation_dashboard import DeviationDashboard

    dd = DeviationDashboard(csv_path=csv_path)
    if not (dd.load_data() and dd.calculate_features() and dd.fit_clusters() and dd.calculate_scores()):
        raise RuntimeError("이탈 점수 계산에 실패했습니다 (DeviationDashboard).")

    df_score = dd.df_valid[["date", "score", "crisis"]].copy()
    merged = pd.merge(df_score, prob_bear_series, on="date", how="inner")
    merged = merged.dropna(subset=["score", "prob_bear"]).sort_values("date").reset_index(drop=True)
    return merged


@st.cache_data(show_spinner=False)
def compute_agreement(merged: pd.DataFrame, max_lag: int = 6) -> dict:
    """두 시계열(이탈 점수, P(약세))의 정량 일치도: 상관, 자카드, 리드-래그, 불일치 구간."""
    score = merged["score"].values
    prob = merged["prob_bear"].values

    correlation = float(np.corrcoef(score, prob)[0, 1])

    # 임계값: 이탈 점수는 비위기 90백분위, P(약세)는 0.5
    nc_scores = merged.loc[~merged["crisis"], "score"]
    score_threshold = float(nc_scores.quantile(0.90))
    prob_threshold = 0.5

    score_high = merged["score"] > score_threshold
    prob_high = merged["prob_bear"] > prob_threshold
    intersection = int((score_high & prob_high).sum())
    union = int((score_high | prob_high).sum())
    jaccard = intersection / union if union > 0 else float("nan")

    lag_results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            s, p = score[:lag], prob[-lag:]
        elif lag > 0:
            s, p = score[lag:], prob[:-lag]
        else:
            s, p = score, prob
        c = float(np.corrcoef(s, p)[0, 1]) if len(s) > 1 else float("nan")
        lag_results.append({"lag": lag, "correlation": c})

    valid_lags = [r for r in lag_results if not np.isnan(r["correlation"])]
    best_lag = max(valid_lags, key=lambda r: r["correlation"]) if valid_lags else {"lag": 0, "correlation": float("nan")}

    only_score_high = score_high & ~prob_high
    only_prob_high = ~score_high & prob_high

    parts = []
    a = merged.loc[only_score_high, ["date", "score", "prob_bear"]].copy()
    a["구분"] = "이탈 점수만 높음 (거시 스트레스↑·수익률 평온 → 위기 전조형)"
    b = merged.loc[only_prob_high, ["date", "score", "prob_bear"]].copy()
    b["구분"] = "P(약세)만 높음 (수익률 급변·거시는 평온 → 가격 충격형)"
    divergence = pd.concat([a, b], ignore_index=True).sort_values("date").reset_index(drop=True)

    return {
        "correlation": correlation,
        "jaccard": jaccard,
        "score_threshold": score_threshold,
        "prob_threshold": prob_threshold,
        "score_high_weeks": int(score_high.sum()),
        "prob_high_weeks": int(prob_high.sum()),
        "intersection": intersection,
        "union": union,
        "lag_results": lag_results,
        "best_lag": best_lag,
        "n_only_score_high": int(only_score_high.sum()),
        "n_only_prob_high": int(only_prob_high.sum()),
        "divergence": divergence,
    }


def compute_transition_probabilities(ms_ar: dict, horizons: tuple[int, ...] = (1, 4, 13)) -> pd.DataFrame:
    """필터링 P(약세) p_t 와 전이행렬 P 로 h주 후 약세확률·국면 전환확률 시계열을 만든다.

    정의는 validation/step5_evaluate.py (A안: MS-AR 전이행렬 × 필터링 국면확률)와 같다.
      행-확률 전이행렬  P = [[강→강, 강→약], [약→강, 약→약]]
      h주 후 약세확률    q_h  = (1-p_t)·P^h[강,약] + p_t·P^h[약,약]
      h주 후 전환확률    q'_h = (1-p_t)·(1-P^h[강,강]) + p_t·(1-P^h[약,약])
    전이행렬은 전체 표본 적합값으로 시간 불변이므로, 시점별 변동은 전부 p_t 에서 온다.
    p_t 는 filtered(실시간) 값이라 look-ahead 가 없다.

    Returns:
        DataFrame[date, prob_bear, regime, prob_bear_h{h}, prob_switch_h{h} ...]
        regime 은 p_t > 0.5 이면 "약세", 아니면 "강세" (검증과 같은 0.5 임계).
    """
    P = np.array(
        [
            [ms_ar["bull_to_bull"], ms_ar["bull_to_bear"]],
            [ms_ar["bear_to_bull"], ms_ar["bear_to_bear"]],
        ],
        dtype=float,
    )
    out = ms_ar["prob_bear_series"].dropna(subset=["prob_bear"]).copy()
    p = out["prob_bear"].to_numpy(dtype=float)
    out["regime"] = np.where(p > 0.5, "약세", "강세")
    for h in horizons:
        Ph = np.linalg.matrix_power(P, int(h))
        out[f"prob_bear_h{h}"] = (1.0 - p) * Ph[0, 1] + p * Ph[1, 1]
        out[f"prob_switch_h{h}"] = (1.0 - p) * (1.0 - Ph[0, 0]) + p * (1.0 - Ph[1, 1])
    return out.reset_index(drop=True)
