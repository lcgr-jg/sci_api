"""Date-range filtering helpers for dashboard views."""

from __future__ import annotations

import pandas as pd

DATE_RANGE_OPTIONS = {
    "full": "Full history",
    "3m": "Last 3 months",
    "6m": "Last 6 months",
    "12m": "Last 12 months",
    "ytd": "Year to date",
}


def filter_by_date_range(
    df: pd.DataFrame,
    date_col: str,
    preset: str,
) -> pd.DataFrame:
    """Filter dataframe to a preset window ending at the latest observation."""
    if df.empty or preset == "full":
        return df

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    if out.empty:
        return out

    end = out[date_col].max()
    if preset == "ytd":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif preset == "3m":
        start = end - pd.DateOffset(months=3)
    elif preset == "6m":
        start = end - pd.DateOffset(months=6)
    elif preset == "12m":
        start = end - pd.DateOffset(months=12)
    else:
        return out

    return out[out[date_col] >= start].reset_index(drop=True)


def date_range_label(preset: str) -> str:
    return DATE_RANGE_OPTIONS.get(preset, preset)
