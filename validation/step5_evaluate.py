# =============================================================================
# validation/step5_evaluate.py — 5단계(3/4): 검증 지표 계산 (A안: MS-AR 전이행렬 × 필터링 국면확률)
#
# 입력
#   validation/data/final_dataset_20260605.csv  당시(2026-06-05 종료) 데이터 → θ0
#   validation/data/final_dataset_20261002.csv  확장(2026-10-02 종료) 데이터 → θT, 평가 입력
#   analysis/ms_ar_output.csv                    당시 발표 P(약세) (필터, θ0)
#   validation/outputs/walkforward_series.csv    걸어서 재적합(②) 시점별 P(약세)·전이확률 (step5_walkforward_params.py)
# 정의
#   · 사후 실제 국면 L_t = 1[θT 평활 P(약세) > 0.5]  (당시 코드 임계값 0.5, 마지막 8주 잠정)
#   · 버전 ① 당시 모델 고정: 06-05까지 당시 발표값, 06-12~ θ0 고정 필터. 전이행렬 P0 고정.
#   · 버전 ② 걸어서 재적합: 매주 t 까지 데이터로 재적합(수렴 실패·예외 주는 직전 수렴 파라미터 이월), 전이행렬 P_t.
#   · h주 후 약세확률 q = ([1-p_t, p_t]·P^h)[약세], 전환확률 q' = (1-p_t)(1-P^h[강,강]) + p_t(1-P^h[약,약])
#   · 기준선 ① 현재 국면 유지: q=p_t, q'=0.   기준선 ② 장기 평균: t까지 사후 라벨의 약세 비중 / h주 전환 빈도(확장 윈도우)
#   · log loss 는 확률을 [0.01, 0.99]로 자른 뒤 계산. Brier 는 자르지 않음.
# 산출: validation/outputs/ 국면확률_시계열.csv, 전환확률_예측vs실제.csv, 전환확률_지표.csv, 전환확률_캘리브레이션.csv,
#       국면전환_탐지지연.csv, 걸어서재적합_현황.csv, 최근구간_예측표.csv, step5_summary.json
# =============================================================================
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

OUT = ROOT / "validation" / "outputs"
EXT = ROOT / "validation" / "data" / "final_dataset_20261002.csv"
PRIOR = ROOT / "validation" / "data" / "final_dataset_20260605.csv"
PUB = ROOT / "analysis" / "ms_ar_output.csv"
WFS = OUT / "walkforward_series.csv"
WFR = OUT / "walkforward_refits.csv"
WFP = OUT / "walkforward_params.csv"

PRIOR_END = pd.Timestamp("2026-06-05")
WF_FIRST = pd.Timestamp("2011-08-05")     # 원본 환경에서 걸어서 재적합이 처음 수렴한 주 (2005~2011은 전부 예외·수렴실패)
EVAL_START = pd.Timestamp("2013-01-04")   # 매주 수렴 적합이 연속으로 성립하는 첫 주 (2011-08-12~2012-12-28은 단일 적합 75주 이월)
PROV_START = pd.Timestamp("2026-08-14")   # 마지막 8주 = 잠정 라벨
H_LIST = [1, 4, 13]
CLIP = (0.01, 0.99)
BLOCK, NBOOT = 13, 1000
VER = {"v1": "①당시고정", "v2": "②재적합"}

summary: dict = {"python": platform.python_version(), "statsmodels": __import__("statsmodels").__version__}
t_start = time.time()


# ── 보조 적합 ───────────────────────────────────────────────────────────────────
def spec(r: np.ndarray) -> MarkovRegression:
    return MarkovRegression(r[1:], k_regimes=2, exog=r[:-1], switching_variance=True)


def bridge_fit(csv: Path):
    df = pd.read_csv(csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    r = df["sp500_log_return"].dropna().values
    np.random.seed(42)
    res = spec(r).fit(search_reps=20, disp=False)
    pn = dict(zip(res.model.param_names, res.params))
    bear = int(np.argmin([pn["const[0]"], pn["const[1]"]]))
    return df, r, res, bear


def rowP(res, bear: int) -> np.ndarray:
    """행-확률 전이행렬, 순서 [강세, 약세]: [[강→강, 강→약], [약→강, 약→약]]"""
    T = res.regime_transition[:, :, 0]
    bull = 1 - bear
    return np.array([[T[bull, bull], T[bear, bull]], [T[bull, bear], T[bear, bear]]])


print("[1] 보조 적합 θ0(당시), θT(확장) + 원본 함수 대조")
from utils.ms_ar_model import fit_ms_ar_model  # noqa: E402

df0, r0, res0, bear0 = bridge_fit(PRIOR)
dfT, rT, resT, bearT = bridge_fit(EXT)
P0, PT = rowP(res0, bear0), rowP(resT, bearT)
f0 = fit_ms_ar_model(csv_path=str(PRIOR))
fT = fit_ms_ar_model(csv_path=str(EXT))
summary["bridge_check"] = {
    "theta0_llf_diff": abs(res0.llf - f0["log_likelihood"]), "theta0_bb_diff": abs(P0[1, 1] - f0["bear_to_bear"]),
    "theta0_ub_diff": abs(P0[0, 1] - f0["bull_to_bear"]), "theta0_p_last_diff": abs(res0.filtered_marginal_probabilities[-1, bear0] - f0["current_prob_bear"]),
    "thetaT_llf_diff": abs(resT.llf - fT["log_likelihood"]), "thetaT_bb_diff": abs(PT[1, 1] - fT["bear_to_bear"]),
    "thetaT_ub_diff": abs(PT[0, 1] - fT["bull_to_bear"]), "thetaT_converged": bool(fT["converged"]), "theta0_converged": bool(f0["converged"]),
}
summary["theta0"] = {"llf": res0.llf, "P": P0.tolist(), "mean_bear": f0["mean_bear"], "mean_bull": f0["mean_bull"], "var_ratio": f0["var_ratio"],
                     "params": dict(zip(res0.model.param_names, res0.params.tolist())), "bear_idx": bear0}
summary["thetaT"] = {"llf": resT.llf, "P": PT.tolist(), "mean_bear": fT["mean_bear"], "mean_bull": fT["mean_bull"], "var_ratio": fT["var_ratio"],
                     "params": dict(zip(resT.model.param_names, resT.params.tolist())), "bear_idx": bearT}
print(f"  θ0 llf {res0.llf:.4f} (원본 함수와 차 {summary['bridge_check']['theta0_llf_diff']:.1e}) | θT llf {resT.llf:.4f} (차 {summary['bridge_check']['thetaT_llf_diff']:.1e})")
print(f"  P0 [[강→강 {P0[0,0]:.5f}, 강→약 {P0[0,1]:.5f}], [약→강 {P0[1,0]:.5f}, 약→약 {P0[1,1]:.5f}]]")
print(f"  PT [[강→강 {PT[0,0]:.5f}, 강→약 {PT[0,1]:.5f}], [약→강 {PT[1,0]:.5f}, 약→약 {PT[1,1]:.5f}]]")

# ── 확률 시계열 ───────────────────────────────────────────────────────────────────
print("[2] 국면확률 시계열 구성")
dates = pd.DatetimeIndex(dfT["date"].iloc[1:].reset_index(drop=True))      # 2000-12-08 ~ 2026-10-02
ret = pd.Series(rT[1:], index=dates)
px = pd.Series(dfT["sp500"].iloc[1:].values, index=dates)
modT_ext = spec(rT)
f_th0 = pd.Series(modT_ext.filter(res0.params).filtered_marginal_probabilities[:, bear0], index=dates)
s_th0 = pd.Series(modT_ext.smooth(res0.params).smoothed_marginal_probabilities[:, bear0], index=dates)
f_thT = pd.Series(resT.filtered_marginal_probabilities[:, bearT], index=dates)
s_thT = pd.Series(resT.smoothed_marginal_probabilities[:, bearT], index=dates)
pub = pd.read_csv(PUB, parse_dates=["date"]).set_index("date")["prob_bear"].reindex(dates)
v1 = pub.where(dates <= PRIOR_END, f_th0)
wfs = pd.read_csv(WFS, parse_dates=["cutoff", "param_date"]).set_index("cutoff")
v2 = wfs["p_bear"].reindex(dates)
L = (s_thT > 0.5).astype(int)
L0 = (s_th0 > 0.5).astype(int)
ts = pd.DataFrame({
    "날짜": dates, "주간로그수익률": ret.values, "S&P500": px.values,
    "P약세_당시발표": pub.values, "P약세_θ0고정필터": f_th0.values, "P약세_①당시고정": v1.values,
    "P약세_②재적합": v2.values, "②이월여부": wfs["carried"].reindex(dates).values, "②파라미터기준일": wfs["param_date"].reindex(dates).values,
    "②파라미터경과주": wfs["param_age_weeks"].reindex(dates).values,
    "②강→강": wfs["bull_to_bull"].reindex(dates).values, "②강→약": wfs["bull_to_bear"].reindex(dates).values,
    "②약→강": wfs["bear_to_bull"].reindex(dates).values, "②약→약": wfs["bear_to_bear"].reindex(dates).values,
    "P약세_θT필터": f_thT.values, "P약세_θT평활": s_thT.values, "실제국면_θT": L.values,
    "P약세_θ0평활": s_th0.values, "실제국면_θ0": L0.values,
    "잠정": (dates >= PROV_START).astype(int), "최근구간": (dates >= PRIOR_END).astype(int),
})
ts["②가용구간"] = ts["P약세_②재적합"].notna().astype(int)
ts["평가구간"] = ((ts["날짜"] >= EVAL_START) & ts["P약세_②재적합"].notna()).astype(int)
ts.to_csv(OUT / "국면확률_시계열.csv", index=False)
summary["labels"] = {
    "n_weeks": int(len(ts)), "bear_weeks_thetaT": int(L.sum()), "bear_weeks_theta0": int(L0.sum()),
    "label_disagreement_weeks": int((L != L0).sum()), "provisional_weeks": int(ts["잠정"].sum()),
    "eval_weeks": int(ts["평가구간"].sum()), "eval_first": str(ts.loc[ts["평가구간"] == 1, "날짜"].min().date()),
    "recent_weeks": int(ts["최근구간"].sum()), "recent_bear_label_weeks": int(L[dates >= PRIOR_END].sum()),
    "recent_max_smoothed": float(s_thT[dates >= PRIOR_END].max()),
}
print(f"  라벨(θT 평활>0.5) 약세 {int(L.sum())}주 / {len(L)}주, θ0 라벨과 불일치 {int((L != L0).sum())}주 | 평가구간 {int(ts['평가구간'].sum())}주 | 최근 17주 중 약세 라벨 {int(L[dates >= PRIOR_END].sum())}주")


# ── 지표 함수 ──────────────────────────────────────────────────────────────────
def _clip(q):
    return np.clip(np.asarray(q, dtype=float), *CLIP)


def brier(q, y):
    return float(np.mean((np.asarray(q, float) - np.asarray(y, float)) ** 2))


def logloss(q, y):
    qc, y = _clip(q), np.asarray(y, float)
    return float(-np.mean(y * np.log(qc) + (1 - y) * np.log(1 - qc)))


def metrics_a(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = ~np.isnan(p)
    p, y = p[m], y[m]
    if len(p) == 0:
        return {"n": 0}
    pred = (p > 0.5).astype(int)
    return {"n": int(len(p)), "Brier": brier(p, y), "log_loss": logloss(p, y), "정확도": float((pred == y).mean()),
            "TP": int(((pred == 1) & (y == 1)).sum()), "FP": int(((pred == 1) & (y == 0)).sum()),
            "FN": int(((pred == 0) & (y == 1)).sum()), "TN": int(((pred == 0) & (y == 0)).sum()),
            "약세주수": int(y.sum())}


def block_boot_skill(bm, bb, block=BLOCK, n=NBOOT, seed=0):
    bm, bb = np.asarray(bm, float), np.asarray(bb, float)
    m = len(bm)
    if m < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(m / block))
    out = []
    for _ in range(n):
        starts = rng.integers(0, m - block + 1, size=nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:m]
        d = bb[idx].mean()
        out.append(1 - bm[idx].mean() / d if d > 0 else np.nan)
    out = np.array(out)
    return (float(np.nanpercentile(out, 5)), float(np.nanpercentile(out, 95)))


# ── (A) 국면 진단 ───────────────────────────────────────────────────────────────
print("[3] (A) 국면 진단 지표")
rowsA = []
masks = {"전체(2013-01~2026-10)": ts["평가구간"] == 1, "최근(2026-06-05~)": ts["최근구간"] == 1, "②가용(2011-08~2026-10)": ts["②가용구간"] == 1}
for vkey, col in [("v1", "P약세_①당시고정"), ("v2", "P약세_②재적합"), ("ref", "P약세_θT필터")]:
    for wname, m in masks.items():
        r = metrics_a(ts.loc[m, col], ts.loc[m, "실제국면_θT"])
        rowsA.append({"버전": VER.get(vkey, "참고:θT전체표본필터"), "구간": wname, **r})
dfA = pd.DataFrame(rowsA)
dfA.to_csv(OUT / "국면진단_지표.csv", index=False)
print(dfA.to_string(index=False))

# 전환 탐지 지연
sw_pos = np.where(np.diff(L.values) != 0)[0] + 1
lag_rows = []
for k, s in enumerate(sw_pos):
    new = int(L.values[s])
    nxt = sw_pos[k + 1] if k + 1 < len(sw_pos) else len(L)
    row = {"전환일": str(dates[s].date()), "새국면": "약세" if new == 1 else "강세", "지속주수": int(nxt - s),
           "잠정": int(dates[s] >= PROV_START)}
    for vkey, col in [("v1", "P약세_①당시고정"), ("v2", "P약세_②재적합")]:
        p = ts[col].values
        lag = None
        for i in range(s, nxt):
            if np.isnan(p[i]):
                continue
            if (p[i] > 0.5) == (new == 1):
                lag = i - s
                break
        row[f"{VER[vkey]}_탐지지연주"] = lag if lag is not None else ("해당없음" if np.isnan(p[s:nxt]).all() else "미탐지")
    lag_rows.append(row)
dfLag = pd.DataFrame(lag_rows)
dfLag.to_csv(OUT / "국면전환_탐지지연.csv", index=False)
for vkey, col in [("v1", "P약세_①당시고정"), ("v2", "P약세_②재적합")]:
    m = ts["평가구간"] == 1
    fa = m & (ts[col] > 0.5) & (ts["실제국면_θT"] == 0)
    runs = int(((fa.astype(int).diff() == 1) | (fa & (fa.shift(1, fill_value=False) == False))).sum())
    summary[f"false_alarm_{vkey}"] = {"weeks": int(fa.sum()), "episodes": runs}
print(f"  전환 {len(dfLag)}회 탐지지연 저장 | 거짓경보(p>0.5 & 라벨 강세) ①{summary['false_alarm_v1']} ②{summary['false_alarm_v2']}")

# ── (B) 전환확률 ───────────────────────────────────────────────────────────────
print("[4] (B) 전환확률 예측 vs 실제")
Lv = L.values.astype(float)
n = len(Lv)
cum_L = np.cumsum(Lv)
b2_level = cum_L / np.arange(1, n + 1)                     # t까지(포함) 약세 비중
rows = []
P2 = {"v1": P0}
for h in H_LIST:
    sw_h = (Lv[h:] != Lv[:-h]).astype(float)               # s=0..n-h-1: L[s+h]!=L[s]
    cum_sw = np.cumsum(sw_h)
    for vkey, col in [("v1", "P약세_①당시고정"), ("v2", "P약세_②재적합")]:
        pv = ts[col].values
        for i in range(n - h):
            t = dates[i]
            if np.isnan(pv[i]) or ts["②가용구간"].iat[i] != 1:
                continue
            p = float(pv[i])
            if vkey == "v1":
                P = P0
            else:
                P = np.array([[ts["②강→강"].iat[i], ts["②강→약"].iat[i]], [ts["②약→강"].iat[i], ts["②약→약"].iat[i]]])
            Ph = np.linalg.matrix_power(P, h)
            pi = np.array([1 - p, p])
            q = float((pi @ Ph)[1])
            q_sw = float(pi[0] * (1 - Ph[0, 0]) + pi[1] * (1 - Ph[1, 1]))
            y = float(Lv[i + h]); Lt = float(Lv[i]); y_sw = float(y != Lt)
            k = i - h   # 전환 빈도는 s+h<=t 인 s까지 → s<=i-h → 개수 i-h+1
            b2_sw = float(cum_sw[k] / (k + 1)) if k >= 0 else np.nan
            rows.append({
                "버전": VER[vkey], "h": h, "예측시점": t, "목표시점": dates[i + h], "p_t": p,
                "강→강^h": Ph[0, 0], "강→약^h": Ph[0, 1], "약→강^h": Ph[1, 0], "약→약^h": Ph[1, 1],
                "q_약세": q, "y_약세": y, "L_t": Lt, "y_전환": y_sw, "q_전환": q_sw,
                "B1_q_약세": p, "B2_q_약세": float(b2_level[i]), "B1_q_전환": 0.0, "B2_q_전환": b2_sw,
                "최근구간": int(t >= PRIOR_END), "목표잠정": int(dates[i + h] >= PROV_START), "이월의존구간": int(t < EVAL_START),
            })
dfB = pd.DataFrame(rows)
dfB.to_csv(OUT / "전환확률_예측vs실제.csv", index=False)
print(f"  예측 행 {len(dfB):,} (버전 2 × h 3)")

rowsM = []
for (ver, h), g in dfB.groupby(["버전", "h"], sort=False):
    for wname, m in {"전체(2013-01~2026-10)": (g["이월의존구간"] == 0).values, "최근(2026-06-05~)": (g["최근구간"] == 1).values, "②가용(2011-08~2026-10)": np.ones(len(g), bool)}.items():
        gg = g[m]
        if gg.empty:
            continue
        gg = gg.dropna(subset=["B2_q_전환"]) if wname.startswith("전체") else gg
        y, q, Lt = gg["y_약세"].values, gg["q_약세"].values, gg["L_t"].values
        ysw, qsw = gg["y_전환"].values, gg["q_전환"].values
        b1, b2 = gg["B1_q_약세"].values, gg["B2_q_약세"].values
        b1s, b2s = gg["B1_q_전환"].values, gg["B2_q_전환"].values
        rec = {"버전": ver, "h": h, "구간": wname, "n": int(len(gg)), "약세목표수": int(y.sum()), "전환발생수": int(ysw.sum()),
               "Brier_약세": brier(q, y), "Brier_약세_B1": brier(b1, y), "Brier_약세_B2": brier(b2, y),
               "LL_약세": logloss(q, y), "LL_약세_B1": logloss(b1, y), "LL_약세_B2": logloss(b2, y),
               "정확도_약세@0.5": float(((q > 0.5) == (y == 1)).mean()),
               "Brier_전환": brier(qsw, ysw), "Brier_전환_B1": brier(b1s, ysw), "Brier_전환_B2": brier(b2s, ysw),
               "LL_전환": logloss(qsw, ysw), "LL_전환_B1": logloss(b1s, ysw), "LL_전환_B2": logloss(b2s, ysw)}
        for key, (bm, bb) in {"약세_B1": ((q - y) ** 2, (b1 - y) ** 2), "약세_B2": ((q - y) ** 2, (b2 - y) ** 2),
                               "전환_B1": ((qsw - ysw) ** 2, (b1s - ysw) ** 2), "전환_B2": ((qsw - ysw) ** 2, (b2s - ysw) ** 2)}.items():
            d = bb.mean()
            rec[f"skill_{key}"] = float(1 - bm.mean() / d) if d > 0 else np.nan
            if wname.startswith("전체"):
                lo, hi = block_boot_skill(bm, bb)
                rec[f"skill_{key}_CI5"], rec[f"skill_{key}_CI95"] = lo, hi
        rowsM.append(rec)
dfM = pd.DataFrame(rowsM)
dfM.to_csv(OUT / "전환확률_지표.csv", index=False)
cols_show = ["버전", "h", "구간", "n", "전환발생수", "Brier_전환", "Brier_전환_B1", "Brier_전환_B2", "skill_전환_B1", "skill_전환_B2", "LL_전환", "LL_전환_B2", "Brier_약세", "Brier_약세_B2", "skill_약세_B2"]
print(dfM[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 캘리브레이션 (전체 구간)
edges = np.linspace(0, 1, 11)
rowsC = []
for (ver, h), g in dfB[dfB["이월의존구간"] == 0].groupby(["버전", "h"], sort=False):
    for kind, qc, yc in [("전환", "q_전환", "y_전환"), ("약세", "q_약세", "y_약세")]:
        q, y = g[qc].values, g[yc].values
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (q >= lo) & ((q < hi) if hi < 1 else (q <= hi))
            rowsC.append({"버전": ver, "h": h, "종류": kind, "구간하한": round(lo, 1), "구간상한": round(hi, 1), "n": int(m.sum()),
                          "평균예측확률": float(q[m].mean()) if m.any() else np.nan, "실제빈도": float(y[m].mean()) if m.any() else np.nan})
dfC = pd.DataFrame(rowsC)
dfC.to_csv(OUT / "전환확률_캘리브레이션.csv", index=False)

# 최근 구간 표
rec = dfB[dfB["최근구간"] == 1][["버전", "h", "예측시점", "목표시점", "p_t", "q_약세", "y_약세", "q_전환", "y_전환", "B2_q_전환", "목표잠정"]].copy()
rec.to_csv(OUT / "최근구간_예측표.csv", index=False)

# 걸어서 재적합 현황 (연도별)
wfr = pd.read_csv(WFR, parse_dates=["cutoff"])
wfr["year"] = wfr["cutoff"].dt.year
stat = wfr.groupby("year").apply(lambda g: pd.Series({
    "시점수": len(g), "예외": int(g["error"].notna().sum()),
    "수렴실패": int((g["error"].isna() & ~g["converged"].fillna(False).astype(bool)).sum()),
    "채택(수렴)": int((g["error"].isna() & g["converged"].fillna(False).astype(bool)).sum())}), include_groups=False).reset_index()
wfs2 = wfs.reset_index(); wfs2["year"] = wfs2["cutoff"].dt.year
stat = stat.merge(wfs2.groupby("year").agg(이월주수=("carried", "sum"), 최대경과주=("param_age_weeks", "max")).reset_index(), on="year", how="left")
stat.to_csv(OUT / "걸어서재적합_현황.csv", index=False)
summary["walkforward"] = {"cutoffs": int(len(wfr)), "errors": int(wfr["error"].notna().sum()),
                          "nonconverged": int((wfr["error"].isna() & ~wfr["converged"].fillna(False).astype(bool)).sum()),
                          "adopted": int((wfr["error"].isna() & wfr["converged"].fillna(False).astype(bool)).sum()),
                          "carried_weeks": int(wfs["carried"].fillna(0).sum()), "max_param_age_weeks": int(wfs["param_age_weeks"].max()),
                          "param_check": {k: float(v) for k, v in pd.read_csv(WFP)[["llf_diff", "p_diff"]].max().items()}}

# ── 손 검산 ────────────────────────────────────────────────────────────────────
last = dates[-1]
def mp(P, k):
    return np.linalg.matrix_power(P, k)
hand = {}
for vkey, col in [("v1", "P약세_①당시고정"), ("v2", "P약세_②재적합")]:
    p = float(ts.loc[ts["날짜"] == last, col].iloc[0])
    P = P0 if vkey == "v1" else np.array([[ts["②강→강"].iat[-1], ts["②강→약"].iat[-1]], [ts["②약→강"].iat[-1], ts["②약→약"].iat[-1]]])
    d = {"date": str(last.date()), "p_t": p, "P": P.tolist(), "P2": mp(P, 2).tolist(), "P4": mp(P, 4).tolist(), "P13": mp(P, 13).tolist()}
    for h in H_LIST:
        Ph = mp(P, h)
        d[f"q_약세_h{h}"] = float((1 - p) * Ph[0, 1] + p * Ph[1, 1])
        d[f"q_전환_h{h}"] = float((1 - p) * (1 - Ph[0, 0]) + p * (1 - Ph[1, 1]))
    hand[VER[vkey]] = d
summary["hand_check"] = hand
summary["seconds"] = round(time.time() - t_start, 1)
with open(OUT / "step5_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n[손검산] {hand[VER['v1']]['date']} ①: p_t={hand[VER['v1']]['p_t']:.5f}, q_약세 h1={hand[VER['v1']]['q_약세_h1']:.5f}, h4={hand[VER['v1']]['q_약세_h4']:.5f}, h13={hand[VER['v1']]['q_약세_h13']:.5f} | "
      f"②: p_t={hand[VER['v2']]['p_t']:.5f}, q_약세 h1={hand[VER['v2']]['q_약세_h1']:.5f}, h4={hand[VER['v2']]['q_약세_h4']:.5f}")
print(f"완료 {summary['seconds']}s → {OUT}")
