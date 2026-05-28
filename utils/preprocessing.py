from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


def load_processed_dataset() -> pd.DataFrame:
    return load_csv(PROCESSED_DATA_PATH)


def to_weekly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return df

    weekly = df.copy()
    weekly[date_col] = pd.to_datetime(weekly[date_col])
    weekly = weekly.sort_values(date_col).set_index(date_col)
    weekly = weekly.resample("W-FRI").last().ffill()
    weekly.index.name = date_col
    return weekly.reset_index()


def add_log_return(df: pd.DataFrame, price_col: str, output_col: str) -> pd.DataFrame:
    result = df.copy()
    result[output_col] = np.log(result[price_col]).diff()
    return result


def add_lag_features(df: pd.DataFrame, columns: list[str], max_lag: int = 12) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        for lag in range(1, max_lag + 1):
            result[f"{col}_lag_{lag}"] = result[col].shift(lag)
    return result


def merge_on_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    clean_frames = [frame for frame in frames if not frame.empty]
    if not clean_frames:
        return pd.DataFrame()

    merged = clean_frames[0]
    for frame in clean_frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    return merged.sort_values("date").ffill()


def save_processed_dataset(df: pd.DataFrame, path: str | Path = PROCESSED_DATA_PATH) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
