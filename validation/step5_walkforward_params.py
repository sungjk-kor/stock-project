# =============================================================================
# validation/step5_walkforward_params.py — 5단계(2/4): 걸어서 재적합 파라미터 확보 + 이월(carry-forward) 필터
#
# 배경: step5_walkforward_refit.py 에서 원본 함수(fit_ms_ar_model)를 매주 호출했더니
#   1,135회 중 270회 예외(statsmodels 'Steady-state probabilities could not be constructed' 등),
#   196회 수렴 실패(파라미터 퇴화)가 나왔다. 원본 코드·최적화 설정은 바꾸지 않는다.
#
# 처리(검증 측 규칙):
#   · 채택 = 예외 없이 끝나고 converged=True 인 주간 적합(669회).
#   · 채택 적합마다 '보조 적합'(원본과 같은 사양·같은 시드: np.random.seed(42), search_reps=20)을 다시 돌려
#     파라미터 벡터 8개를 얻는다. 로그우도가 원본 함수 기록과 일치하는지로 동일 적합임을 확인한다.
#   · 모든 주간 시점 t 에 대해, t 이하 가장 최근 채택 적합의 파라미터 θ_{t'} 를 t 까지의 수익률에
#     필터(.filter)만 적용해 P(약세_t) 를 구한다 (정보집합 ≤ t 유지, 파라미터만 t' 시점 것).
#     채택 시점 자체에서는 적합의 마지막 필터값과 같다(검증).
#
# 산출: validation/outputs/walkforward_params.csv  (채택 시점별 파라미터)
#       validation/outputs/walkforward_series.csv  (매주 시점별 P(약세)·전이확률·이월 정보)
# =============================================================================
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "validation" / "data" / "final_dataset_20261002.csv"
WF = ROOT / "validation" / "outputs" / "walkforward_refits.csv"
OUT_P = ROOT / "validation" / "outputs" / "walkforward_params.csv"
OUT_S = ROOT / "validation" / "outputs" / "walkforward_series.csv"
PARAM_NAMES = ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "x1[0]", "x1[1]", "sigma2[0]", "sigma2[1]"]


def _returns():
    import pandas as pd
    df = pd.read_csv(EXT, parse_dates=["date"], usecols=["date", "sp500_log_return"]).sort_values("date").reset_index(drop=True)
    return df


def _spec(r):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    return MarkovRegression(r[1:], k_regimes=2, exog=r[:-1], switching_variance=True)


def fit_chunk(cutoffs):
    warnings.filterwarnings("ignore")
    import numpy as np
    df = _returns()
    rows = []
    for c in cutoffs:
        r = df.loc[df["date"] <= c, "sp500_log_return"].dropna().values
        t0 = time.time()
        try:
            np.random.seed(42)
            res = _spec(r).fit(search_reps=20, disp=False)   # 원본 fit_ms_ar_model 과 동일 사양·시드
            pn = dict(zip(res.model.param_names, res.params))
            bear = int(np.argmin([pn["const[0]"], pn["const[1]"]]))
            row = {"cutoff": c, "llf": float(res.llf), "converged": bool(res.mle_retvals.get("converged", False)),
                   "bear_idx": bear, "last_filtered_p_bear": float(res.filtered_marginal_probabilities[-1, bear]),
                   "seconds": round(time.time() - t0, 2)}
            row.update({n: float(pn[n]) for n in PARAM_NAMES})
        except Exception as e:  # noqa: BLE001
            row = {"cutoff": c, "error": repr(e)[:160]}
        rows.append(row)
    return rows


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    wf = pd.read_csv(WF)
    good = wf[wf["error"].isna() & wf["converged"].fillna(False).astype(bool)].copy()
    cutoffs = sorted(good["cutoff"].tolist())
    print(f"채택(수렴) 시점 {len(cutoffs)} / 전체 {len(wf)}  ({cutoffs[0]} ~ {cutoffs[-1]})", flush=True)
    t0 = time.time()
    chunks = [cutoffs[i::8] for i in range(8)]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for part in ex.map(fit_chunk, chunks):
            rows.extend(part)
    par = pd.DataFrame(rows).sort_values("cutoff").reset_index(drop=True)
    par = par.merge(wf[["cutoff", "llf", "p_bear"]].rename(columns={"llf": "llf_원본함수", "p_bear": "p_bear_원본함수"}), on="cutoff", how="left")
    par["llf_diff"] = (par["llf"] - par["llf_원본함수"]).abs()
    par["p_diff"] = (par["last_filtered_p_bear"] - par["p_bear_원본함수"]).abs()
    par.to_csv(OUT_P, index=False)
    n_err = int(par["error"].notna().sum()) if "error" in par else 0
    print(f"보조 적합 완료 {len(par)}건, {time.time()-t0:.0f}s | 예외 {n_err} | 원본 함수와 llf 최대차 {par['llf_diff'].max():.2e}, P(약세) 최대차 {par['p_diff'].max():.2e}", flush=True)

    # ── 이월 필터: 매주 시점 t 마다 가장 최근 채택 파라미터로 필터 ──
    ok = par[par["error"].isna()] if "error" in par else par
    ok = ok[ok["converged"]].reset_index(drop=True)
    df = _returns()
    all_cut = [str(d.date()) for d in df["date"] if d >= pd.Timestamp(cutoffs[0])]
    good_dates = ok["cutoff"].tolist()
    series = []
    t1 = time.time()
    for c in all_cut:
        prior = [g for g in good_dates if g <= c]
        if not prior:
            series.append({"cutoff": c, "p_bear": np.nan}); continue
        g = prior[-1]
        prow = ok[ok["cutoff"] == g].iloc[0]
        params = np.array([prow[n] for n in PARAM_NAMES])
        bear = int(prow["bear_idx"]); bull = 1 - bear
        r = df.loc[df["date"] <= c, "sp500_log_return"].dropna().values
        res = _spec(r).filter(params)
        T = res.regime_transition[:, :, 0]   # 열-확률 M[to, from]
        series.append({
            "cutoff": c, "p_bear": float(res.filtered_marginal_probabilities[-1, bear]),
            "param_date": g, "carried": int(g != c), "param_age_weeks": int(round((pd.Timestamp(c) - pd.Timestamp(g)).days / 7)),
            "bull_to_bull": float(T[bull, bull]), "bull_to_bear": float(T[bear, bull]),
            "bear_to_bull": float(T[bull, bear]), "bear_to_bear": float(T[bear, bear]),
            "mean_bear": float(prow[f"const[{bear}]"]), "mean_bull": float(prow[f"const[{bull}]"]),
            "var_ratio": float(prow[f"sigma2[{bear}]"] / prow[f"sigma2[{bull}]"]),
        })
    ser = pd.DataFrame(series)
    chk = ser.merge(ok[["cutoff", "last_filtered_p_bear"]], on="cutoff", how="inner")
    print(f"이월 필터 완료 {len(ser)}주, {time.time()-t1:.0f}s | 채택 시점에서 적합값과 최대차 {(chk['p_bear']-chk['last_filtered_p_bear']).abs().max():.2e} | "
          f"이월 주 {int(ser['carried'].fillna(0).sum())}, 최대 경과 {int(ser['param_age_weeks'].max())}주", flush=True)
    ser.to_csv(OUT_S, index=False)
    print("saved", OUT_P, OUT_S, flush=True)
