# =============================================================================
# validation/step3_extend.py — 3단계: 최근 시장 데이터 확장 (2026-06-05 이후 ~ 2026-10-02)
#
# 원칙
#   · 커밋된 data/raw/*.csv 와 data/processed/final_dataset.csv 는 건드리지 않는다.
#     raw 사본을 validation/data/raw/ 에 만들고, 레포의 fetch 함수(같은 소스·같은 식별자)로
#     2026-05-01 이후를 받아 "커밋본 마지막 날짜 이후" 행만 붙인다.
#     겹치는 구간(2026-05-01 ~ 커밋본 마지막 날짜)은 값이 바뀌었는지(소급 수정)만 점검해 보고한다.
#   · 전처리는 레포의 build_final_dataset 을 그대로 호출한다 (RAW_DIR·save 함수만 이 프로세스 안에서 패치).
#   · 기존 코드 수정 없음.
#
# 산출
#   validation/data/raw/{market,rates,macro,liquidity,events,ism_pmi}.csv   (확장된 raw 사본)
#   validation/data/final_dataset_20261002.csv                            (확장 데이터셋)
#   validation/outputs/step3_overlap_revisions.csv                        (겹침 구간 소급수정 점검)
#   validation/outputs/step3_release_alignment.csv                        (거시지표 발표일 정렬 점검, ALFRED)
#   validation/outputs/step3_extension_window.csv                         (확장 구간 17주 주요 컬럼)
#   validation/outputs/step3_quality.json                                 (요약)
#
# 실행: <venv>/bin/python validation/step3_extend.py
# =============================================================================
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

CUTOFF_END = "2026-10-02"    # 마지막 완결 금요일 (오늘 2026-10-05 월요일)
FETCH_START = "2026-05-01"   # 겹침 구간 포함해서 받음 (소급수정 점검용)
PRIOR_END = "2026-06-05"     # 당시 데이터 종료 주

RAW_SRC = ROOT / "data" / "raw"
RAW_DST = ROOT / "validation" / "data" / "raw"
OUT_DATA = ROOT / "validation" / "data"
OUT = ROOT / "validation" / "outputs"
RAW_DST.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

EXT_PATH = OUT_DATA / "final_dataset_20261002.csv"
PRIOR_PATH = OUT_DATA / "final_dataset_20260605.csv"   # 2단계 재구성본

summary: dict = {"python": platform.python_version(), "cutoff_end": CUTOFF_END, "fetch_start": FETCH_START}
for name in ["numpy", "pandas", "yfinance", "fredapi"]:
    try:
        summary[f"v_{name}"] = __import__(name).__version__
    except Exception as e:  # noqa: BLE001
        summary[f"v_{name}"] = f"? ({e})"
print("[환경]", {k: v for k, v in summary.items() if k.startswith("v_") or k == "python"})

# ── 0) raw 사본 (항상 커밋본에서 새로 복사 → 멱등) ───────────────────────────────
for fn in ["market.csv", "rates.csv", "macro.csv", "liquidity.csv", "events.csv", "ism_pmi.csv"]:
    shutil.copy2(RAW_SRC / fn, RAW_DST / fn)

from utils.liquidity_data import fetch_liquidity_data  # noqa: E402
from utils.macro_data import fetch_macro_data  # noqa: E402
from utils.market_data import fetch_market_data  # noqa: E402
from utils.rates_data import fetch_rates_data  # noqa: E402
import utils.data_pipeline as dp  # noqa: E402


def _norm_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["date"])
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    df["date"] = d.dt.normalize()
    return df


def splice(name: str, new_df: pd.DataFrame) -> dict:
    """커밋본 raw(old)에 새 fetch(new)의 '커밋본 마지막 날짜 이후' 행만 붙인다. 겹침 구간은 비교만."""
    old = pd.read_csv(RAW_SRC / f"{name}.csv", parse_dates=["date"])
    old_max = old["date"].max()
    rep = {"file": name, "old_rows": int(len(old)), "old_max_date": str(old_max.date())}
    if new_df is None or new_df.empty:
        rep["status"] = "fetch EMPTY — 커밋본 그대로"
        return rep
    new = _norm_dates(new_df).sort_values("date").drop_duplicates("date", keep="last")
    rep["new_fetch_rows"] = int(len(new))
    rep["new_fetch_range"] = f"{new['date'].min().date()} ~ {new['date'].max().date()}"

    cols_old = [c for c in old.columns if c != "date"]
    missing_in_new = [c for c in cols_old if c not in new.columns]
    extra_in_new = [c for c in new.columns if c not in old.columns and c != "date"]
    rep["columns_missing_in_new_fetch"] = missing_in_new
    rep["columns_extra_in_new_fetch_ignored"] = extra_in_new

    # 겹침 구간 소급수정 점검 (양쪽 모두 값 있는 셀)
    ov = old.merge(new, on="date", how="inner", suffixes=("_old", "_new"))
    rev_rows = []
    for c in cols_old:
        if c not in new.columns:
            continue
        a = ov[f"{c}_old"].to_numpy(dtype=float)
        b = ov[f"{c}_new"].to_numpy(dtype=float)
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = np.abs(a - b)
        rel = diff / np.where(np.abs(a) > 0, np.abs(a), np.nan)
        changed = both & (diff > 1e-9)
        rev_rows.append({
            "file": name, "column": c, "overlap_cells": int(both.sum()),
            "changed_cells": int(changed.sum()),
            "max_abs_diff": float(np.nanmax(diff[both])) if both.any() else 0.0,
            "max_rel_diff": float(np.nanmax(rel[both])) if both.any() else 0.0,
            "filled_later_cells": int((np.isnan(a) & ~np.isnan(b)).sum()),   # 당시 NaN, 지금 값 있음
            "missing_now_cells": int((~np.isnan(a) & np.isnan(b)).sum()),    # 당시 값 있음, 지금 NaN
            "first_changed_date": str(ov.loc[changed, "date"].min().date()) if changed.any() else "",
        })
    rep["overlap_rows"] = int(len(ov))
    rep["revisions"] = rev_rows

    # 꼬리 붙이기 (커밋본 값·NaN 모두 보존, 마지막 날짜 이후만 추가)
    tail = new[new["date"] > old_max].copy()
    for c in missing_in_new:
        last52 = old[c].dropna().tail(52)
        if len(last52) and (last52 == 0).all():
            tail[c] = 0.0
            rep.setdefault("filled_zero_columns", []).append(c)
        else:
            tail[c] = np.nan
            rep.setdefault("WARNING_nan_columns", []).append(c)
    tail = tail[["date"] + cols_old]
    merged = pd.concat([old, tail], ignore_index=True).sort_values("date").reset_index(drop=True)
    merged.to_csv(RAW_DST / f"{name}.csv", index=False)
    rep["appended_rows"] = int(len(tail))
    rep["new_max_date"] = str(merged["date"].max().date())
    rep["last_obs_by_column"] = {c: str(merged.loc[merged[c].notna(), "date"].max().date()) for c in cols_old}
    return rep


print("\n[1] 레포 fetch 함수로 2026-05-01 이후 수집 → 커밋본 마지막 날짜 이후만 이어붙임")
t0 = time.time()
fetched = {}
for name, fn in [
    ("market", lambda: fetch_market_data(start=FETCH_START, end=None)),
    ("rates", lambda: fetch_rates_data(start=FETCH_START)),
    ("macro", lambda: fetch_macro_data(start=FETCH_START)),
    ("liquidity", lambda: fetch_liquidity_data(start=FETCH_START, end=None)),
]:
    try:
        fetched[name] = fn()
        print(f"  fetch {name}: {len(fetched[name])}행", end="")
        if len(fetched[name]):
            d = _norm_dates(fetched[name])["date"]
            print(f" ({d.min().date()} ~ {d.max().date()})")
        else:
            print()
    except Exception as e:  # noqa: BLE001
        fetched[name] = pd.DataFrame()
        print(f"  fetch {name}: 실패 — {type(e).__name__}: {str(e)[:150]}")
summary["fetch_seconds"] = round(time.time() - t0, 1)

splice_reports = [splice(name, fetched[name]) for name in ["market", "rates", "macro", "liquidity"]]
summary["splice"] = splice_reports
rev_df = pd.DataFrame([r for rep in splice_reports for r in rep.get("revisions", [])])
rev_df.to_csv(OUT / "step3_overlap_revisions.csv", index=False)
for rep in splice_reports:
    print(f"  {rep['file']}: 커밋본 ~{rep['old_max_date']} → 추가 {rep.get('appended_rows', 0)}행 → ~{rep.get('new_max_date', rep['old_max_date'])}"
          f" | 겹침 {rep.get('overlap_rows', 0)}행, 값 바뀐 컬럼: "
          + (", ".join(f"{r['column']}({r['changed_cells']}셀, 최대상대 {r['max_rel_diff']:.2e})" for r in rep.get("revisions", []) if r["changed_cells"]) or "없음")
          + (f" | 결측 경고 {rep['WARNING_nan_columns']}" if rep.get("WARNING_nan_columns") else "")
          + (f" | 0으로 채움 {rep['filled_zero_columns']}" if rep.get("filled_zero_columns") else ""))

# ── 2) 확장 데이터셋 생성 (레포 build_final_dataset 그대로) ──────────────────────
print(f"\n[2] build_final_dataset(start=2000-01-01, end={CUTOFF_END}) — RAW_DIR 을 사본 폴더로 패치")
_saved_fn, _saved_raw = dp.save_processed_dataset, dp.RAW_DIR
dp.save_processed_dataset = lambda df, path=None: None
dp.RAW_DIR = RAW_DST
try:
    raw_sets = dp._load_saved_raw_datasets()
    ext = dp.build_final_dataset(raw_sets, max_lag=12, start="2000-01-01", end=CUTOFF_END)
finally:
    dp.save_processed_dataset, dp.RAW_DIR = _saved_fn, _saved_raw
ext.to_csv(EXT_PATH, index=False)
ext = pd.read_csv(EXT_PATH, parse_dates=["date"])
prior = pd.read_csv(PRIOR_PATH, parse_dates=["date"])
print(f"  확장본: {len(ext):,}주 {ext['date'].min().date()} ~ {ext['date'].max().date()} | 2단계 재구성본: {len(prior):,}주 ~{prior['date'].max().date()}")

# 2a) 06-05 이전 행이 2단계 재구성본과 같은가 (06-05 행은 당시 수요일(06-03)까지만 수집돼 달라질 수 있음)
cmp = prior.merge(ext, on="date", how="left", suffixes=("_p", "_e"))
num_cols = [c for c in prior.columns if c != "date"]
per_date_mismatch = {}
for c in num_cols:
    a = cmp[f"{c}_p"].to_numpy(dtype=float); b = cmp[f"{c}_e"].to_numpy(dtype=float)
    bad = ~((np.abs(a - b) <= 1e-9) | (np.isnan(a) & np.isnan(b)))
    for d in cmp.loc[bad, "date"]:
        per_date_mismatch.setdefault(str(d.date()), []).append(c)
summary["prior_rows_identical_until"] = None
dates_bad = sorted(per_date_mismatch)
print(f"  2단계본과 값이 다른 날짜: {len(dates_bad)}개 → {dates_bad[:5]}{' ...' if len(dates_bad) > 5 else ''}")
if dates_bad:
    for d in dates_bad[:3]:
        cols = per_date_mismatch[d]
        base = [c for c in cols if "_lag_" not in c]
        print(f"    {d}: 다른 컬럼 {len(cols)}개 (lag 제외 {len(base)}개): {base[:14]}")
summary["prior_mismatch_dates"] = {d: [c for c in v if "_lag_" not in c] for d, v in per_date_mismatch.items()}
row0605 = {}
if PRIOR_END in per_date_mismatch:
    p = prior.set_index("date").loc[PRIOR_END]; e = ext.set_index("date").loc[PRIOR_END]
    for c in [c for c in per_date_mismatch[PRIOR_END] if "_lag_" not in c]:
        row0605[c] = {"당시(06-03까지 수집)": float(p[c]), "확장본(06-05 종가)": float(e[c])}
summary["row_20260605_changes"] = row0605

# 2b) 확장 구간 점검
win = ext[ext["date"] > PRIOR_END].reset_index(drop=True)
expected = pd.date_range(pd.Timestamp(PRIOR_END) + pd.Timedelta(weeks=1), CUTOFF_END, freq="W-FRI")
missing_weeks = [str(d.date()) for d in expected if d not in set(win["date"])]
print(f"\n[3] 확장 구간 ({expected[0].date()} ~ {expected[-1].date()}): 기대 {len(expected)}주, 실제 {len(win)}주, 빠진 주: {missing_weeks or '없음'}")
base_cols = [c for c in ext.columns if "_lag_" not in c and c != "date"]
na_counts = {c: int(win[c].isna().sum()) for c in base_cols if win[c].isna().any()}
print(f"  확장 구간 결측 셀(lag 제외): {na_counts or '없음'} (파이프라인 dropna 특성상 결측 행은 통째로 사라짐 → '빠진 주'로 확인)")

# 단위·범위: 직전 52주 대비 확장 구간 범위, 고유값 수(정체=ffill 월별)
prev52 = ext[(ext["date"] <= PRIOR_END)].tail(52)
range_rows = []
for c in base_cols:
    lo, hi = float(prev52[c].min()), float(prev52[c].max())
    wlo, whi = float(win[c].min()), float(win[c].max())
    span = (hi - lo) if hi > lo else (abs(hi) if hi != 0 else 1.0)
    flag = "범위 밖" if (wlo < lo - 0.5 * span or whi > hi + 0.5 * span) else ""
    range_rows.append({"column": c, "prev52_min": lo, "prev52_max": hi, "ext_min": wlo, "ext_max": whi,
                       "ext_distinct": int(win[c].nunique()), "flag": flag})
range_df = pd.DataFrame(range_rows)
summary["range_flags"] = range_df[range_df["flag"] != ""][["column", "prev52_min", "prev52_max", "ext_min", "ext_max"]].to_dict("records")
print(f"  직전 52주 범위를 크게 벗어난 컬럼(폭의 50% 초과): {[r['column'] for r in summary['range_flags']] or '없음'}")
stale = range_df[range_df["ext_distinct"] <= 5][["column", "ext_distinct"]].to_dict("records")
print(f"  17주 동안 고유값 5개 이하(월별·ffill 정체) 컬럼: {[(r['column'], r['ext_distinct']) for r in stale]}")
summary["stale_columns"] = stale

keep_cols = ["date", "sp500", "sp500_log_return", "nasdaq", "dow", "vix", "xli", "sox", "fed_funds", "treasury_2y", "treasury_10y",
             "term_spread_10y_2y", "cpi", "unemployment", "nonfarm_payroll", "m2", "financial_conditions", "nfci_leverage",
             "credit_spread", "dxy", "oil", "gold", "fomc", "cpi_release", "employment_release", "bigtech_earnings"]
win[[c for c in keep_cols if c in win.columns]].to_csv(OUT / "step3_extension_window.csv", index=False)
print("  확장 구간 S&P500 주간 로그수익률:")
print("   " + "  ".join(f"{d.date()}:{r*100:+.2f}%" for d, r in zip(win["date"], win["sp500_log_return"])))
ev_cols = ["fomc", "cpi_release", "employment_release", "bigtech_earnings"]
print(f"  이벤트 더미 합(확장 구간): {win[ev_cols].sum().to_dict()}  (events.csv 마지막 이벤트 2025-04-03 — 이후 추가된 이벤트 없음)")

# 원시 시계열 정체: 마지막 관측일과 그 뒤로 ffill 된 주 수
stale_rows = []
for rep in splice_reports:
    for c, last in rep.get("last_obs_by_column", {}).items():
        last_ts = pd.Timestamp(last)
        n_ffill = int(((ext["date"] > last_ts) & (ext["date"] <= CUTOFF_END)).sum())
        stale_rows.append({"file": rep["file"], "column": c, "last_obs": last, "weeks_ffilled_after_last_obs": n_ffill})
stale_df = pd.DataFrame(stale_rows)
summary["last_obs"] = stale_rows
print("  마지막 관측 이후 ffill 주 수 (3주 이상):")
for r in stale_rows:
    if r["weeks_ffilled_after_last_obs"] >= 3:
        print(f"    {r['file']}.{r['column']}: 마지막 관측 {r['last_obs']} → 이후 {r['weeks_ffilled_after_last_obs']}주 ffill")

# ── 4) 발표일 정렬 점검 (ALFRED) — 월별·주별 FRED 지표가 발표 전 주에 값이 들어가는지 ──
print("\n[4] 발표일 정렬 점검 (FRED ALFRED 첫 발표일 vs 데이터셋에 값이 처음 들어간 주)")
align_rows = []
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from fredapi import Fred
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    SERIES = {"cpi": "CPIAUCSL", "core_cpi": "CPILFESL", "unemployment": "UNRATE", "nonfarm_payroll": "PAYEMS",
              "retail_sales": "RSAFS", "industrial_production": "INDPRO", "m2": "M2SL", "fed_funds": "FEDFUNDS",
              "financial_conditions": "NFCI", "nfci_leverage": "NFCILEVERAGE"}
    for col, sid in SERIES.items():
        try:
            rel = fred.get_series_all_releases(sid, realtime_start="2026-03-01")
            rel["date"] = pd.to_datetime(rel["date"]); rel["realtime_start"] = pd.to_datetime(rel["realtime_start"])
            first = rel.groupby("date")["realtime_start"].min()
            first = first[first.index >= "2026-04-01"]
            for obs, rdate in first.items():
                # 데이터셋에서 이 관측값이 처음 등장하는 주: 관측일 이후 첫 금요일 (to_weekly last + ffill)
                first_week = (obs + pd.offsets.Week(weekday=4)) if obs.weekday() != 4 else obs
                release_week = (rdate + pd.offsets.Week(weekday=4)) if rdate.weekday() != 4 else rdate
                lookahead_weeks = int(max(0, (release_week - first_week).days // 7))
                if first_week > pd.Timestamp(CUTOFF_END):
                    continue
                align_rows.append({"column": col, "fred_id": sid, "observation_date": str(obs.date()),
                                   "first_release_date": str(rdate.date()), "first_week_in_dataset": str(first_week.date()),
                                   "lookahead_weeks": lookahead_weeks})
        except Exception as e:  # noqa: BLE001
            align_rows.append({"column": col, "fred_id": sid, "observation_date": "", "first_release_date": f"조회 실패: {str(e)[:80]}",
                               "first_week_in_dataset": "", "lookahead_weeks": None})
except Exception as e:  # noqa: BLE001
    print(f"  ALFRED 조회 불가: {e}")
align_df = pd.DataFrame(align_rows)
align_df.to_csv(OUT / "step3_release_alignment.csv", index=False)
if not align_df.empty and "lookahead_weeks" in align_df:
    ok = align_df.dropna(subset=["lookahead_weeks"])
    for col, g in ok.groupby("column", sort=False):
        print(f"  {col:22s} ({g['fred_id'].iloc[0]}): 관측 {len(g)}건, 발표 전에 값이 들어간 주 수 = " +
              ", ".join(f"{o[5:]}→{int(w)}주" for o, w in zip(g["observation_date"], g["lookahead_weeks"])))
summary["release_alignment_rows"] = int(len(align_df))

with open(OUT / "step3_quality.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n저장: {EXT_PATH}  |  {OUT / 'step3_quality.json'}")
