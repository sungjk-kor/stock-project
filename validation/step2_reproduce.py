# =============================================================================
# validation/step2_reproduce.py — 2단계: 당시 모델을 당시 데이터로 재현
#
# 원칙
#   · 기존 코드(utils/*, ms_ar_analysis.py)는 수정하지 않고 import만 한다.
#   · 커밋된 data/processed/final_dataset.csv(2026-02-27 종료본)는 건드리지 않는다.
#     build_final_dataset()가 내부에서 호출하는 save_processed_dataset를 이 스크립트
#     안에서만 no-op으로 바꿔 쓰고, 결과는 validation/data/ 아래에 따로 저장한다.
#
# 하는 일
#   1) raw CSV(2026-06-03까지) → 레포 파이프라인으로 2026-06-05 종료 데이터셋 재구성
#      → 커밋본(02-27 종료)과 앞 1,318주가 모든 컬럼에서 같은지 확인
#   2) fit_ms_ar_model(재구성 CSV) → P(약세) 1,332주를 analysis/ms_ar_output.csv(당시 산출)와 대조
#   3) DeviationDashboard(재구성 CSV) → 이탈 점수를 analysis/ms_ar_aligned.csv(당시 산출)와 대조
#   4) 환경(패키지 버전)과 대조 통계를 validation/outputs/step2_summary_<tag>.json 에 기록
#
# 실행: <venv>/bin/python validation/step2_reproduce.py <tag>
# =============================================================================
from __future__ import annotations

import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # DeviationDashboard.load_events 등이 상대경로("data/raw/events.csv")를 쓴다

import numpy as np
import pandas as pd

TAG = sys.argv[1] if len(sys.argv) > 1 else "env"
OUT_DATA = ROOT / "validation" / "data"
OUT = ROOT / "validation" / "outputs"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

REBUILT = OUT_DATA / "final_dataset_20260605.csv"
COMMITTED = ROOT / "data" / "processed" / "final_dataset.csv"
REF_MSAR = ROOT / "analysis" / "ms_ar_output.csv"      # 당시 P(약세) 산출 (2000-12-01 ~ 2026-06-05)
REF_ALIGNED = ROOT / "analysis" / "ms_ar_aligned.csv"  # 당시 이탈 점수 + P(약세) (2001-03-02 ~ 2026-06-05)

summary: dict = {"tag": TAG, "python": platform.python_version()}


def versions() -> dict:
    out = {}
    for name in ["numpy", "pandas", "scipy", "statsmodels", "sklearn", "streamlit"]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[name] = f"import 실패: {e}"
    return out


summary["versions"] = versions()
print(f"[환경 {TAG}] python {summary['python']} | " + ", ".join(f"{k} {v}" for k, v in summary["versions"].items()))

# ── 1) 2026-06-05 종료 데이터셋 재구성 ──────────────────────────────────────────
print("\n[1] raw CSV → final_dataset 재구성 (end=2026-06-05)")
import utils.data_pipeline as dp  # noqa: E402

_saved_fn = dp.save_processed_dataset
dp.save_processed_dataset = lambda df, path=None: None  # 커밋본 덮어쓰기 방지 (이 프로세스 안에서만)
try:
    t0 = time.time()
    raw = dp._load_saved_raw_datasets()
    rebuilt = dp.build_final_dataset(raw, max_lag=12, start="2000-01-01", end="2026-06-05")
    t_build = time.time() - t0
finally:
    dp.save_processed_dataset = _saved_fn

rebuilt.to_csv(REBUILT, index=False)
rebuilt = pd.read_csv(REBUILT, parse_dates=["date"])  # 커밋본과 같은 CSV 왕복을 거친 값으로 비교
committed = pd.read_csv(COMMITTED, parse_dates=["date"])
print(f"  재구성: {len(rebuilt):,}주 {rebuilt['date'].min().date()} ~ {rebuilt['date'].max().date()} ({t_build:.1f}s)")
print(f"  커밋본: {len(committed):,}주 {committed['date'].min().date()} ~ {committed['date'].max().date()}")

n = len(committed)
same_cols = list(rebuilt.columns) == list(committed.columns)
head = rebuilt.iloc[:n].reset_index(drop=True)
date_match = bool((head["date"].values == committed["date"].values).all()) if len(head) == n else False
col_report = {}
n_bad_cells = 0
for c in committed.columns:
    if c == "date" or c not in head.columns:
        continue
    a = head[c].to_numpy(dtype=float)
    b = committed[c].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    bad = int(np.sum(~(diff <= 1e-9)))  # NaN 불일치도 bad로 셈
    n_bad_cells += bad
    if bad:
        col_report[c] = {"mismatch_cells": bad, "max_abs_diff": float(np.nanmax(diff))}
summary["rebuild"] = {
    "rows": int(len(rebuilt)), "first": str(rebuilt["date"].min().date()), "last": str(rebuilt["date"].max().date()),
    "committed_rows": int(n), "same_columns": same_cols, "dates_match_first_n": date_match,
    "mismatch_cells_first_n": int(n_bad_cells), "mismatch_columns": col_report, "build_seconds": round(t_build, 1),
}
print(f"  컬럼 동일: {same_cols} | 앞 {n}주 날짜 동일: {date_match} | 값 불일치 셀: {n_bad_cells}")
if col_report:
    for c, v in list(col_report.items())[:10]:
        print(f"    - {c}: {v}")

# ── 2) MS-AR P(약세) 재현 ─────────────────────────────────────────────────────────
print("\n[2] fit_ms_ar_model(재구성 CSV) vs analysis/ms_ar_output.csv")
from utils.ms_ar_model import fit_ms_ar_model  # noqa: E402

t0 = time.time()
ms = fit_ms_ar_model(csv_path=str(REBUILT))
t_fit = time.time() - t0
params = {k: ms[k] for k in [
    "converged", "log_likelihood", "bear_idx", "mean_bear", "mean_bull", "var_bear", "var_bull", "var_ratio",
    "bear_to_bear", "bear_to_bull", "bull_to_bear", "bull_to_bull", "duration_bear_weeks", "duration_bull_weeks",
    "current_prob_bear", "current_date"]}
params["fit_seconds"] = round(t_fit, 1)
for k, v in params.items():
    print(f"  {k}: {v}")

new = ms["prob_bear_series"].rename(columns={"prob_bear": "prob_bear_new"})
ref = pd.read_csv(REF_MSAR, parse_dates=["date"]).rename(columns={"prob_bear": "prob_bear_ref"})
cmp = ref.merge(new, on="date", how="outer", indicator=True).sort_values("date").reset_index(drop=True)
cmp["abs_diff"] = (cmp["prob_bear_ref"] - cmp["prob_bear_new"]).abs()
both = cmp.dropna(subset=["prob_bear_ref", "prob_bear_new"])
msar_stats = {
    "ref_rows": int(len(ref)), "new_rows": int(len(new)),
    "rows_only_in_ref": int((cmp["_merge"] == "left_only").sum()), "rows_only_in_new": int((cmp["_merge"] == "right_only").sum()),
    "compared_rows": int(len(both)),
    "max_abs_diff": float(both["abs_diff"].max()), "mean_abs_diff": float(both["abs_diff"].mean()),
    "n_gt_1e-6": int((both["abs_diff"] > 1e-6).sum()), "n_gt_1e-3": int((both["abs_diff"] > 1e-3).sum()),
    "n_gt_1e-2": int((both["abs_diff"] > 1e-2).sum()),
    "date_of_max_diff": str(both.loc[both["abs_diff"].idxmax(), "date"].date()) if len(both) else None,
    "corr": float(np.corrcoef(both["prob_bear_ref"], both["prob_bear_new"])[0, 1]) if len(both) > 1 else None,
    "ref_last": {"date": str(ref["date"].iloc[-1].date()), "prob_bear": float(ref["prob_bear_ref"].iloc[-1])},
    "new_last": {"date": str(new["date"].iloc[-1].date()), "prob_bear": float(new["prob_bear_new"].iloc[-1])},
    "n_ref_gt_0.5": int((both["prob_bear_ref"] > 0.5).sum()), "n_new_gt_0.5": int((both["prob_bear_new"] > 0.5).sum()),
    "n_disagree_at_0.5": int(((both["prob_bear_ref"] > 0.5) != (both["prob_bear_new"] > 0.5)).sum()),
}
summary["msar"] = {"params": params, "compare": msar_stats}
cmp.drop(columns=["_merge"]).to_csv(OUT / f"step2_msar_compare_{TAG}.csv", index=False)
print(f"  비교 {msar_stats['compared_rows']}주 | 최대 |차이| {msar_stats['max_abs_diff']:.2e} ({msar_stats['date_of_max_diff']}) | "
      f"평균 |차이| {msar_stats['mean_abs_diff']:.2e} | >1e-6: {msar_stats['n_gt_1e-6']}주, >1e-3: {msar_stats['n_gt_1e-3']}주, >1e-2: {msar_stats['n_gt_1e-2']}주")
print(f"  0.5 기준 약세 판정 주수: 당시 {msar_stats['n_ref_gt_0.5']} vs 재현 {msar_stats['n_new_gt_0.5']} (불일치 {msar_stats['n_disagree_at_0.5']}주)")
print(f"  마지막 값: 당시 {msar_stats['ref_last']} | 재현 {msar_stats['new_last']}")

# ── 3) 이탈 점수 재현 ─────────────────────────────────────────────────────────────
print("\n[3] DeviationDashboard(재구성 CSV) vs analysis/ms_ar_aligned.csv (score)")
from utils.deviation_dashboard import DeviationDashboard  # noqa: E402

t0 = time.time()
dd = DeviationDashboard(csv_path=str(REBUILT))
ok = dd.load_data() and dd.calculate_features() and dd.fit_clusters() and dd.calculate_scores() and dd.calculate_stats()
t_dd = time.time() - t0
if not ok:
    raise SystemExit("이탈 점수 계산 실패")
score_new = dd.df_valid[["date", "score", "crisis", "p_peace"]].rename(columns={"score": "score_new"})
aligned = pd.read_csv(REF_ALIGNED, parse_dates=["date"]).rename(columns={"score": "score_ref", "prob_bear": "prob_bear_aligned_ref"})
c2 = aligned.merge(score_new, on="date", how="outer", indicator=True).sort_values("date").reset_index(drop=True)
c2["abs_diff"] = (c2["score_ref"] - c2["score_new"]).abs()
b2 = c2.dropna(subset=["score_ref", "score_new"])
dev_stats = {
    "ref_rows": int(len(aligned)), "new_rows": int(len(score_new)), "compared_rows": int(len(b2)),
    "rows_only_in_ref": int((c2["_merge"] == "left_only").sum()), "rows_only_in_new": int((c2["_merge"] == "right_only").sum()),
    "max_abs_diff": float(b2["abs_diff"].max()), "mean_abs_diff": float(b2["abs_diff"].mean()),
    "n_gt_1e-6": int((b2["abs_diff"] > 1e-6).sum()),
    "ref_last": {"date": str(aligned["date"].iloc[-1].date()), "score": float(aligned["score_ref"].iloc[-1])},
    "new_last": {"date": str(score_new["date"].iloc[-1].date()), "score": float(score_new["score_new"].iloc[-1])},
    "stats_new": {k: dd.stats[k] for k in ["current_score", "current_date", "current_pct", "current_peace", "max_score", "max_date"]},
    "peacetime_cluster": int(dd.peacetime_label), "danger_cluster": int(dd.danger_label),
    "seconds": round(t_dd, 1),
}
summary["deviation"] = dev_stats
c2.drop(columns=["_merge"]).to_csv(OUT / f"step2_deviation_compare_{TAG}.csv", index=False)
print(f"  비교 {dev_stats['compared_rows']}주 | 최대 |차이| {dev_stats['max_abs_diff']:.2e} | >1e-6: {dev_stats['n_gt_1e-6']}주")
print(f"  마지막 값: 당시 {dev_stats['ref_last']} | 재현 {dev_stats['new_last']} | stats {dev_stats['stats_new']}")

# ── 3b) 당시 산출물끼리의 정합성 (ms_ar_aligned.prob_bear == ms_ar_output.prob_bear ?) ──
chk = aligned.merge(ref, on="date", how="inner")
summary["ref_consistency_prob_bear"] = {
    "rows": int(len(chk)), "max_abs_diff": float((chk["prob_bear_aligned_ref"] - chk["prob_bear_ref"]).abs().max()),
}
print(f"\n[3b] 당시 산출물끼리: ms_ar_aligned.prob_bear vs ms_ar_output.prob_bear 최대 |차이| {summary['ref_consistency_prob_bear']['max_abs_diff']:.2e} ({len(chk)}주)")

with open(OUT / f"step2_summary_{TAG}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n저장: {OUT / f'step2_summary_{TAG}.json'}")
