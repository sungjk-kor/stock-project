from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.preprocessing import PROJECT_ROOT


EVENT_TYPES = {
    "fomc": "FOMC meeting",
    "cpi_release": "CPI release",
    "employment_release": "Employment report release",
    "bigtech_earnings": "Big Tech earnings release",
}

EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "events.csv"


def create_events_template(path: str | Path = EVENTS_PATH) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        pd.DataFrame(
            columns=["date", "event_type", "intensity", "memo"],
            data=[
                ["2024-01-31", "fomc", 1.0, "Example: FOMC meeting"],
                ["2024-02-13", "cpi_release", 1.0, "Example: CPI release"],
                ["2024-02-02", "employment_release", 1.0, "Example: employment report release"],
                ["2024-01-30", "bigtech_earnings", 1.0, "Example: Big Tech earnings release"],
            ],
        ).to_csv(output_path, index=False)
    return output_path


def load_events(
    path: str | Path = EVENTS_PATH,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    event_path = Path(path)
    if not event_path.exists():
        create_events_template(event_path)

    events = pd.read_csv(event_path, parse_dates=["date"])
    if events.empty:
        return events

    events["event_type"] = events["event_type"].str.strip().str.lower()
    events["intensity"] = pd.to_numeric(events["intensity"], errors="coerce").fillna(1.0)
    events = filter_events_by_date(events, start=start, end=end)
    return events


def filter_events_by_date(
    events: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if events.empty:
        return events

    filtered = events.copy()
    filtered["date"] = pd.to_datetime(filtered["date"])
    if start is not None:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start)]
    if end is not None:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end)]
    return filtered


def build_weekly_event_features(
    events: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    week_index = _weekly_date_index(start=start, end=end)
    if events.empty:
        return _empty_weekly_event_features(week_index)

    weekly = events.copy()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly = filter_events_by_date(weekly, start=start, end=end)
    if weekly.empty:
        return _empty_weekly_event_features(week_index)

    weekly["week"] = weekly["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

    features = (
        weekly.pivot_table(
            index="week",
            columns="event_type",
            values="intensity",
            aggfunc="sum",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
        .rename(columns={"week": "date"})
    )

    for event_type in EVENT_TYPES:
        if event_type not in features.columns:
            features[event_type] = 0.0

    features = features[["date", *EVENT_TYPES.keys()]]
    if week_index.empty:
        return features

    base = pd.DataFrame({"date": week_index})
    features = base.merge(features, on="date", how="left")
    features[list(EVENT_TYPES.keys())] = features[list(EVENT_TYPES.keys())].fillna(0.0)
    return features


def _weekly_date_index(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    if start is None or end is None:
        return pd.DatetimeIndex([])

    start_week = pd.to_datetime(start).to_period("W-FRI").end_time.normalize()
    end_week = pd.to_datetime(end).to_period("W-FRI").end_time.normalize()
    return pd.date_range(start=start_week, end=end_week, freq="W-FRI")


def _empty_weekly_event_features(week_index: pd.DatetimeIndex) -> pd.DataFrame:
    if week_index.empty:
        return pd.DataFrame()

    features = pd.DataFrame({"date": week_index})
    for event_type in EVENT_TYPES:
        features[event_type] = 0.0
    return features
