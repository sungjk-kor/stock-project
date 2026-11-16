# =============================================================================
# validation/step5_walkforward_refit.py — 5단계(1/2): 걸어서(walk-forward) 매주 재적합
#
#   2005-01-07부터 2026-10-02까지 매주 금요일 t 마다, t 까지의 수익률만 담은 CSV를 만들어
#   원본 함수 utils.ms_ar_model.fit_ms_ar_model(csv_path) 를 그대로 호출하고,
#   그 시점의 필터 P(약세)(current_prob_bear)와 전이확률·평균·분산비·수렴 여부를 기록한다.
#   원본 코드 수정 없음. 임시 CSV는 시점별로 다른 이름(캐시 충돌 방지) → 적합 후 삭제.
#
# 산출: validation/outputs/walkforward_refits.csv
# 실행: .venv-stock/bin/python validation/step5_walkforward_refit.py   (8 프로세스 병렬)
# =============================================================================
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")   # 프로세스당 BLAS 스레드 1개 (병렬 8개와 충돌 방지)

import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "validation" / "data" / "final_dataset_20261002.csv"
OUT = ROOT / "validation" / "outputs" / "walkforward_refits.csv"
TMP = ROOT / "validation" / "tmp"
START = "2005-01-07"
N_WORKERS = 8


def run_chunk(cutoffs: list[str]) -> list[dict]:
    warnings.filterwarnings("ignore")
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import pandas as pd
    from utils.ms_ar_model import fit_ms_ar_model

    df = pd.read_csv(EXT, parse_dates=["date"], usecols=["date", "sp500_log_return"]).sort_values("date")
    pid = os.getpid()
    rows = []
    for c in cutoffs:
        sub = df[df["date"] <= c]
        tmp = TMP / f"wf_{pid}_{c}.csv"
        sub.to_csv(tmp, index=False)
        t0 = time.time()
        try:
            r = fit_ms_ar_model(csv_path=str(tmp))
            rows.append({
                "cutoff": c, "n_weeks": int(len(sub)), "p_bear": r["current_prob_bear"],
                "bear_to_bear": r["bear_to_bear"], "bear_to_bull": r["bear_to_bull"],
                "bull_to_bear": r["bull_to_bear"], "bull_to_bull": r["bull_to_bull"],
                "mean_bear": r["mean_bear"], "mean_bull": r["mean_bull"], "var_ratio": r["var_ratio"],
                "converged": bool(r["converged"]), "llf": r["log_likelihood"], "bear_idx": r["bear_idx"],
                "seconds": round(time.time() - t0, 2),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"cutoff": c, "n_weeks": int(len(sub)), "error": repr(e)[:200]})
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
    return rows


if __name__ == "__main__":
    import pandas as pd
    TMP.mkdir(parents=True, exist_ok=True)
    dates = pd.read_csv(EXT, parse_dates=["date"], usecols=["date"])["date"].sort_values()
    cutoffs = [str(d.date()) for d in dates if d >= pd.Timestamp(START)]
    chunks = [cutoffs[i::N_WORKERS] for i in range(N_WORKERS)]   # 라운드로빈 → 긴 표본이 고루 분산
    print(f"cutoffs: {len(cutoffs)} ({cutoffs[0]} ~ {cutoffs[-1]}), workers: {N_WORKERS}", flush=True)
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_chunk, ch) for ch in chunks]
        for i, f in enumerate(as_completed(futs), 1):
            results.extend(f.result())
            print(f"  chunk {i}/{N_WORKERS} done — {len(results)} fits, {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame(results).sort_values("cutoff").reset_index(drop=True)
    res.to_csv(OUT, index=False)
    n_err = int(res["error"].notna().sum()) if "error" in res else 0
    n_nc = int((~res["converged"].fillna(False).astype(bool)).sum()) if "converged" in res else 0
    print(f"saved {OUT}: {len(res)} rows | errors {n_err} | not converged {n_nc} | "
          f"mean {res['seconds'].mean():.2f}s/fit | total {time.time()-t0:.0f}s", flush=True)
