"""Load 2020–2024 historical data from the SCI Excel archive export."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.transforms import (
    REFUELING_ITEM_LABELS,
    WEEKLY_BREAKDOWNS,
    WEEKLY_RUNS_ITEMS,
    _normalize_consumption_product,
)

DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "SCI China Oil & Chemical Industry Data - Ownership (2020-2024).xlsx"
)


def excel_archive_enabled() -> bool:
    """Opt-in via SCI_INCLUDE_EXCEL_ARCHIVE=1 — fills 2020 gap not available from API."""
    return os.getenv("SCI_INCLUDE_EXCEL_ARCHIVE", "").strip().lower() in ("1", "true", "yes")


def maybe_merge_excel_archive(frames: dict, path: Path | None = None) -> dict:
    if not excel_archive_enabled():
        return frames
    return merge_excel_archive(frames, path=path)


def _is_date_column(col) -> bool:
    if isinstance(col, (pd.Timestamp, datetime)):
        return True
    ts = pd.to_datetime(col, errors="coerce")
    return pd.notna(ts) and getattr(ts, "year", 0) >= 2000


def load_excel_weekly(path: Path = DEFAULT_ARCHIVE) -> pd.DataFrame:
    """Weekly summary sheet → same long schema as parse_weekly_summary."""
    df = pd.read_excel(path, sheet_name="Weekly summary", header=1)
    id_cols = ["Breakdown", "Item", "Unit"]
    date_cols = [c for c in df.columns if c not in id_cols and _is_date_column(c)]

    week_dates = {c: pd.to_datetime(c).normalize() for c in date_cols}

    body = df[df["Breakdown"].isin(WEEKLY_BREAKDOWNS)]
    body = body[body["Item"].isin(WEEKLY_RUNS_ITEMS)]

    long = body.melt(
        id_vars=id_cols,
        value_vars=list(week_dates),
        var_name="week_label",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["week_start"] = long["week_label"].map(week_dates)
    long = long.dropna(subset=["week_start", "value"])
    long["metric"] = long["Item"].map(WEEKLY_RUNS_ITEMS)
    long = long.rename(columns={"Breakdown": "breakdown", "Unit": "unit"})
    long["breakdown"] = long["breakdown"].replace({"SOE": "SOEs"})

    return (
        long[["week_start", "breakdown", "metric", "value", "unit"]]
        .sort_values(["metric", "breakdown", "week_start"])
        .reset_index(drop=True)
    )


def load_excel_profit_model(path: Path = DEFAULT_ARCHIVE) -> pd.DataFrame:
    """Profit model sheet → same long schema as parse_profit_model (2023+ in archive)."""
    raw = pd.read_excel(path, sheet_name="Profit model", header=None)
    rows: list[dict] = []
    i = 0
    while i < len(raw):
        cell = str(raw.iloc[i, 0])
        match = re.match(r"^\d+\.\s+(.+)$", cell)
        if not match:
            i += 1
            continue

        refinery_type = match.group(1).strip()
        header_row = i + 1
        if header_row >= len(raw) or str(raw.iloc[header_row, 0]).strip() != "Indicator":
            i += 1
            continue

        date_col_idx: list[int] = []
        for j in range(3, raw.shape[1]):
            if pd.notna(pd.to_datetime(raw.iloc[header_row, j], errors="coerce")):
                date_col_idx.append(j)

        data_start = header_row + 1
        data_end = len(raw)
        for k in range(data_start, len(raw)):
            if re.match(r"^\d+\.", str(raw.iloc[k, 0])):
                data_end = k
                break

        for row_idx in range(data_start, data_end):
            section = raw.iloc[row_idx, 0]
            factor = raw.iloc[row_idx, 1]
            unit = raw.iloc[row_idx, 2]
            if pd.isna(section) or pd.isna(factor):
                continue
            for j in date_col_idx:
                week_start = pd.to_datetime(raw.iloc[header_row, j]).normalize()
                value = pd.to_numeric(raw.iloc[row_idx, j], errors="coerce")
                if pd.notna(value):
                    rows.append(
                        {
                            "week_start": week_start,
                            "refinery_type": refinery_type,
                            "section": str(section).strip(),
                            "factor": str(factor).strip(),
                            "unit": str(unit).strip() if pd.notna(unit) else "",
                            "value": value,
                        }
                    )
        i = data_end

    if not rows:
        return pd.DataFrame(
            columns=["week_start", "refinery_type", "section", "factor", "unit", "value"]
        )
    return pd.DataFrame(rows).sort_values(
        ["week_start", "refinery_type", "section", "factor"]
    ).reset_index(drop=True)


def load_excel_shandong_stocks(path: Path = DEFAULT_ARCHIVE) -> pd.DataFrame:
    """Shandong stocks sheet → same schema as parse_shandong_stocks."""
    df = pd.read_excel(path, sheet_name="Stocks", header=1)
    df = df.rename(columns={"Date": "week_start", "Region": "region"})
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce").dt.normalize()

    products = ["Crude oil", "Gasoline", "Diesel", "Naphtha"]
    long = df.melt(
        id_vars=["week_start", "region"],
        value_vars=[c for c in products if c in df.columns],
        var_name="product",
        value_name="value_kt",
    )
    long["value_kt"] = pd.to_numeric(long["value_kt"], errors="coerce")
    long["product"] = long["product"].str.lower().str.replace(" ", "_")
    return long.dropna(subset=["value_kt"]).sort_values(["week_start", "region", "product"])


def _melt_monthly_sheet(
    df: pd.DataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    date_cols = [c for c in df.columns if c not in id_cols and _is_date_column(c)]
    long = df.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="month_col",
        value_name="consumption_kt",
    )
    long["consumption_kt"] = pd.to_numeric(long["consumption_kt"], errors="coerce")
    long["month_start"] = pd.to_datetime(long["month_col"], errors="coerce")
    long = long.dropna(subset=["consumption_kt", "month_start"])
    long["year"] = long["month_start"].dt.year
    long["month"] = long["month_start"].dt.month
    return long


def load_excel_consumption(path: Path = DEFAULT_ARCHIVE) -> pd.DataFrame:
    """Consumption sheet with monthly datetime columns."""
    df = pd.read_excel(path, sheet_name="Consumption", header=1)
    id_cols = ["Product", "Region", "Province"]
    long = _melt_monthly_sheet(df, id_cols)
    long["product"] = long["Product"].map(_normalize_consumption_product)
    return (
        long.rename(columns={"Region": "region", "Province": "province"})[
            ["product", "region", "province", "year", "month", "consumption_kt"]
        ]
        .sort_values(["product", "year", "month", "province"])
        .reset_index(drop=True)
    )


def load_excel_kerosene_refueling(path: Path = DEFAULT_ARCHIVE) -> pd.DataFrame:
    """Kerosene refueling sheet with monthly datetime columns."""
    df = pd.read_excel(path, sheet_name="Kerosene refueling volume", header=1)
    id_cols = ["Item", "Province"]
    long = _melt_monthly_sheet(df, id_cols)
    long["product"] = long["Item"].map(REFUELING_ITEM_LABELS)
    return (
        long.rename(columns={"Province": "province"})[
            ["product", "province", "year", "month", "consumption_kt"]
        ]
        .dropna(subset=["product"])
        .sort_values(["product", "year", "month", "province"])
        .reset_index(drop=True)
    )


def merge_excel_archive(
    frames: dict,
    path: Path | None = None,
) -> dict:
    """
    Prepend Excel archive history and dedupe (API/cache wins on overlapping dates).

    Extends weekly runs/products back to ~2020; consumption/refueling monthly back to 2020;
    Shandong stocks to 2020; profit model to 2023 (Excel limit).
    """
    path = path or DEFAULT_ARCHIVE
    if not path.exists():
        return frames

    from src.transforms import (
        build_consumption_national,
        build_kerosene_refueling_national,
        build_margins_wide,
        build_products_stocks_wide,
        build_runs_wide,
        merge_consumption,
        merge_kerosene_refueling,
        merge_profit_models,
        merge_shandong_stocks,
        merge_weekly_summaries,
        shandong_stocks_totals,
    )

    weekly = merge_weekly_summaries([load_excel_weekly(path), frames["weekly"]])
    margins = merge_profit_models([load_excel_profit_model(path), frames["margins"]])
    stocks = merge_shandong_stocks([load_excel_shandong_stocks(path), frames["shandong_stocks"]])
    consumption = merge_consumption([load_excel_consumption(path), frames["consumption"]])
    refueling = merge_kerosene_refueling(
        [load_excel_kerosene_refueling(path), frames["kerosene_refueling"]]
    )

    frames["weekly"] = weekly
    frames["runs_wide"] = build_runs_wide(weekly)
    frames["products_stocks_wide"] = build_products_stocks_wide(weekly)
    frames["shandong_stocks"] = stocks
    frames["shandong_stocks_total"] = shandong_stocks_totals(stocks)
    frames["consumption"] = consumption
    frames["consumption_national"] = build_consumption_national(consumption)
    frames["kerosene_refueling"] = refueling
    frames["kerosene_refueling_national"] = build_kerosene_refueling_national(refueling)
    frames["margins"] = margins
    frames["margins_wide"] = build_margins_wide(margins)

    meta = frames["meta"].copy()
    meta["excel_archive"] = str(path.name)
    frames["meta"] = meta
    return frames
