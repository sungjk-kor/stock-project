from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.event_data import build_weekly_event_features, create_events_template, load_events
from utils.liquidity_data import fetch_liquidity_data
from utils.macro_data import fetch_macro_data
from utils.market_data import fetch_market_data
from utils.preprocessing import (
    PROJECT_ROOT,
    add_lag_features,
    add_log_return,
    merge_on_date,
    save_processed_dataset,
    to_weekly,
)
from utils.rates_data import fetch_rates_data


RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ── 파일 커버리지 헬퍼 ──────────────────────────────────────────────────────────

def get_raw_data_status() -> dict[str, dict]:
    """
    각 raw 파일의 현황(날짜 범위, 행수, 결측값 수)을 반환.
    Streamlit 데이터 현황 테이블에서 사용.
    """
    status = {}
    for name in ["market", "rates", "macro", "liquidity"]:
        path = RAW_DIR / f"{name}.csv"
        if not path.exists():
            status[name] = {"exists": False, "path": path}
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            if df.empty or "date" not in df.columns:
                status[name] = {"exists": True, "path": path, "rows": 0}
                continue

            # 컬럼별 결측값 수
            na_by_col = df.isna().sum()
            na_by_col = na_by_col[na_by_col > 0].to_dict()

            status[name] = {
                "exists":    True,
                "path":      path,
                "start":     df["date"].min().date(),
                "end":       df["date"].max().date(),
                "rows":      len(df),
                "columns":   [c for c in df.columns if c != "date"],
                "na_total":  int(df.isna().sum().sum()),
                "na_by_col": na_by_col,
            }
        except Exception as e:
            status[name] = {"exists": True, "path": path, "error": str(e)}
    return status


def _get_coverage(csv_path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """기존 파일의 (최소 날짜, 최대 날짜)를 반환. 없으면 (None, None)."""
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], usecols=["date"])
        if df.empty:
            return None, None
        return df["date"].min(), df["date"].max()
    except Exception:
        return None, None


def _merge_incremental(csv_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    새 데이터를 기존 파일과 병합.
    같은 날짜가 있으면 새 데이터를 우선(덮어쓰기).
    """
    if new_df.empty:
        return new_df
    if not csv_path.exists():
        return new_df
    try:
        existing = pd.read_csv(csv_path, parse_dates=["date"])
        if existing.empty:
            return new_df
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        return merged.reset_index(drop=True)
    except Exception:
        return new_df


def collect_raw_data_incremental(
    start: str = "2000-01-01",
    end: str | None = None,
) -> dict[str, dict]:
    """
    증분 수집: 기존 파일의 날짜 커버리지를 확인하고
    요청 기간 중 없는 구간만 새로 수집한다.

    반환값: {파일명: {"action": "skipped"|"created"|"updated", "rows_added": int, ...}}
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    create_events_template()

    req_start = pd.Timestamp(start)
    req_end   = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    # 2주 여유 — 주간 데이터 경계 오차 방지
    UPDATE_BUFFER = pd.Timedelta(weeks=2)

    results: dict[str, dict] = {}

    for name in ["market", "rates", "macro", "liquidity"]:
        path = RAW_DIR / f"{name}.csv"
        ex_start, ex_end = _get_coverage(path)

        # ── 수집 필요 여부 판단 ──────────────────────────────────────────────
        needs_before = (ex_start is None) or (req_start < ex_start)
        needs_after  = (ex_end is None)   or (req_end > ex_end + UPDATE_BUFFER)

        if not needs_before and not needs_after:
            # 이미 커버됨 → 수집 불필요
            results[name] = {"action": "skipped", "rows_added": 0,
                             "coverage": (ex_start, ex_end)}
            continue

        # ── 수집할 구간 결정 ─────────────────────────────────────────────────
        if ex_start is None:
            # 파일 없음 → 전체 범위 수집
            fetch_start = str(req_start.date())
            fetch_end   = str(req_end.date())
        else:
            # 앞뒤 누락 구간 중 더 이른 시작~더 늦은 끝
            fetch_start = str(min(req_start, ex_start).date())
            fetch_end   = str(req_end.date())

        # ── 실제 수집 ────────────────────────────────────────────────────────
        try:
            if name == "market":
                new_df = fetch_market_data(start=fetch_start, end=fetch_end)
            elif name == "rates":
                new_df = fetch_rates_data(start=fetch_start)
            elif name == "macro":
                new_df = fetch_macro_data(start=fetch_start)
            elif name == "liquidity":
                new_df = fetch_liquidity_data(start=fetch_start, end=fetch_end)
            else:
                new_df = pd.DataFrame()

            if new_df.empty:
                results[name] = {"action": "empty", "rows_added": 0}
                continue

            # 기존 데이터와 병합
            merged = _merge_incremental(path, new_df)
            rows_added = len(merged) - (len(pd.read_csv(path)) if path.exists() else 0)
            merged.to_csv(path, index=False)

            action = "created" if ex_start is None else "updated"
            results[name] = {
                "action":     action,
                "rows_added": max(rows_added, 0),
                "coverage":   (merged["date"].min(), merged["date"].max()),
            }
        except Exception as e:
            results[name] = {"action": "error", "error": str(e)}

    return results


def collect_raw_data(start: str = "2000-01-01", end: str | None = None) -> dict[str, pd.DataFrame]:
    """Collect raw input data and save each group to data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    create_events_template()

    datasets = {
        "market": fetch_market_data(start=start, end=end),
        "rates": fetch_rates_data(start=start),
        "macro": fetch_macro_data(start=start),
        "liquidity": fetch_liquidity_data(start=start, end=end),
        "events": load_events(start=start, end=end),
    }

    for name, df in datasets.items():
        if name != "events" and not df.empty:
            df.to_csv(RAW_DIR / f"{name}.csv", index=False)

    return datasets


def build_final_dataset(
    raw_datasets: dict[str, pd.DataFrame] | None = None,
    max_lag: int = 12,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Build weekly SARIMAX input data and save data/processed/final_dataset.csv."""
    if raw_datasets is None:
        raw_datasets = _load_saved_raw_datasets()

    market = to_weekly(raw_datasets.get("market", pd.DataFrame()))
    rates = to_weekly(raw_datasets.get("rates", pd.DataFrame()))
    macro = to_weekly(raw_datasets.get("macro", pd.DataFrame()))
    liquidity = to_weekly(raw_datasets.get("liquidity", pd.DataFrame()))
    events = build_weekly_event_features(raw_datasets.get("events", pd.DataFrame()), start=start, end=end)

    market = _add_market_returns(market)
    liquidity = _add_price_returns(liquidity, ["dxy", "oil", "gold"])
    macro = _add_macro_changes(macro)

    merged = merge_on_date([market, rates, macro, liquidity, events])
    if merged.empty:
        return merged

    merged = _filter_by_date(merged, start=start, end=end)

    event_cols = ["fomc", "cpi_release", "employment_release", "bigtech_earnings"]
    for col in event_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    merged[event_cols] = merged[event_cols].fillna(0.0)

    feature_cols = [
        col
        for col in merged.select_dtypes("number").columns
        if col != "sp500_log_return"
    ]
    merged = add_lag_features(merged, feature_cols, max_lag=max_lag)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    save_processed_dataset(merged)
    return merged


def collect_and_build_dataset(start: str = "2000-01-01", end: str | None = None, max_lag: int = 12) -> pd.DataFrame:
    raw = collect_raw_data(start=start, end=end)
    return build_final_dataset(raw, max_lag=max_lag, start=start, end=end)


def _load_saved_raw_datasets() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for name in ["market", "rates", "macro", "liquidity", "events"]:
        path = RAW_DIR / f"{name}.csv"
        if path.exists():
            datasets[name] = pd.read_csv(path, parse_dates=["date"])
        else:
            datasets[name] = pd.DataFrame()
    return datasets


def _filter_by_date(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df

    filtered = df.copy()
    filtered["date"] = pd.to_datetime(filtered["date"])
    if start is not None:
        start_week = pd.to_datetime(start).to_period("W-FRI").end_time.normalize()
        filtered = filtered[filtered["date"] >= start_week]
    if end is not None:
        end_week = pd.to_datetime(end).to_period("W-FRI").end_time.normalize()
        filtered = filtered[filtered["date"] <= end_week]
    return filtered


def _add_market_returns(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return market

    result = market.copy()
    for col in ["sp500", "nasdaq", "dow", "vix", "xli", "sox"]:
        if col in result.columns:
            result = add_log_return(result, col, f"{col}_log_return")
    return result


def _add_price_returns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    for col in columns:
        if col in result.columns:
            result = add_log_return(result, col, f"{col}_log_return")
    return result


def _add_macro_changes(macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        return macro

    result = macro.copy()
    for col in result.select_dtypes("number").columns:
        result[f"{col}_change"] = result[col].diff()
    return result
