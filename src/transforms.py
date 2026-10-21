"""Normalize SCI datapack sheets into tidy time series."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# Weekly summary rows we care about for crude/runs and product stocks.
WEEKLY_RUNS_ITEMS = {
    "Feedstock primary processing volume (kt)": "processing_kt",
    "Feedstock primary processing unit operating rate": "operating_rate",
    "Gasoline output (kt)": "gasoline_output_kt",
    "Diesel output (kt)": "diesel_output_kt",
    "Kerosene output (kt)": "kerosene_output_kt",
    "Gasoline commercial stocks (kt)": "gasoline_stocks_kt",
    "Diesel commercial stocks (kt)": "diesel_stocks_kt",
}

WEEKLY_BREAKDOWNS = {"National", "SOE", "Independents"}

CONSUMPTION_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _week_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if re.match(r"^\d+(st|nd|rd|th) week$", str(c))]


def _melt_week_columns(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    week_cols = _week_columns(df.columns)
    if not week_cols:
        return pd.DataFrame(columns=id_cols + ["week_label", "week_start", "value"])

    long = df.melt(id_vars=id_cols, value_vars=week_cols, var_name="week_label", value_name="value")
    long = long.dropna(subset=["value"])
    return long


def parse_weekly_summary(sheet: dict) -> pd.DataFrame:
    """
    Wide weekly summary → long format.

    Row 0 holds ISO week-start dates; subsequent rows are metrics by breakdown
    (National / SOE / Independents).
    """
    raw = pd.DataFrame(sheet.get("DataList", []))
    if raw.empty:
        return pd.DataFrame(
            columns=["week_start", "breakdown", "metric", "value", "unit", "pub_date"]
        )

    week_cols = _week_columns(raw.columns)
    date_row = raw.iloc[0]
    week_dates = {}
    for col in week_cols:
        ts = pd.to_datetime(date_row[col], errors="coerce")
        week_dates[col] = ts.normalize() if pd.notna(ts) else pd.NaT

    body = raw.iloc[1:].copy()
    body = body[body["Breakdown"].isin(WEEKLY_BREAKDOWNS)]
    body = body[body["Item"].isin(WEEKLY_RUNS_ITEMS)]

    long = _melt_week_columns(body, ["Breakdown", "Item", "Unit"])
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


def parse_shandong_stocks(sheet: dict) -> pd.DataFrame:
    """Shandong independent refinery stocks by city and product."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=["week_start", "region", "product", "value_kt"])

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


def _parse_ownership_monthly(
    sheet: dict,
    value_col: str,
    pub_year: int | None = None,
) -> pd.DataFrame:
    """Monthly province × ownership sheet (processing volume or operating rate)."""
    df = pd.DataFrame(sheet.get("DataList", []))
    base_cols = ["month_start", "unit", "province", "ownership", value_col]
    if df.empty:
        return pd.DataFrame(columns=base_cols)

    month_cols = [c for c in df.columns if c not in ("Unit", "Province", "Ownership")]
    long = df.melt(
        id_vars=["Unit", "Province", "Ownership"],
        value_vars=month_cols,
        var_name="month_label",
        value_name=value_col,
    )
    # Historical ownership snapshots sometimes use placeholders like "/".
    # Coerce to numeric so only valid monthly values are kept.
    long[value_col] = pd.to_numeric(long[value_col], errors="coerce")
    long = long.dropna(subset=[value_col])

    year = pub_year or pd.Timestamp.today().year
    long["month_start"] = pd.to_datetime(
        long["month_label"].astype(str) + f" {year}",
        format="%b %Y",
        errors="coerce",
    )
    long = long.dropna(subset=["month_start"])

    return (
        long.rename(
            columns={
                "Unit": "unit",
                "Province": "province",
                "Ownership": "ownership",
            }
        )[base_cols]
        .sort_values(["month_start", "province", "ownership"])
        .reset_index(drop=True)
    )


def parse_processing_ownership(sheet: dict, pub_year: int | None = None) -> pd.DataFrame:
    """Monthly feedstock processing by province and ownership."""
    return _parse_ownership_monthly(sheet, "processing_kt", pub_year=pub_year)


def parse_operating_rate_ownership(sheet: dict, pub_year: int | None = None) -> pd.DataFrame:
    """Monthly unit operating rate by province and ownership (0–1 fraction)."""
    return _parse_ownership_monthly(sheet, "operating_rate", pub_year=pub_year)


REGIONAL_OUTPUT_PRODUCTS = ["Gasoline", "Diesel", "Kerosene", "Naphtha", "VGO", "Residue"]


def _melt_provincial_months(
    df: pd.DataFrame,
    id_cols: list[str],
    value_col: str,
    calendar_year: int,
) -> pd.DataFrame:
    month_cols = [c for c in CONSUMPTION_MONTHS if c in df.columns]
    long = df.melt(
        id_vars=id_cols,
        value_vars=month_cols,
        var_name="month_label",
        value_name=value_col,
    )
    long[value_col] = pd.to_numeric(long[value_col], errors="coerce")
    long = long.dropna(subset=[value_col])
    long["month"] = long["month_label"].map(
        {m: i + 1 for i, m in enumerate(CONSUMPTION_MONTHS)}
    )
    long["year"] = calendar_year
    long["month_start"] = pd.to_datetime(
        {"year": long["year"], "month": long["month"], "day": 1}
    )
    return long


def parse_provincial_processing(sheet: dict, calendar_year: int) -> pd.DataFrame:
    """Provincial monthly crude processing (kt/month) — total by province."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=["province", "year", "month", "month_start", "processing_kt"]
        )

    df = df[
        (df["Unit"] == "FEEDSTOCK PRIMARY PROCESSING UNITS") & (df["Group"] == "China")
    ]
    long = _melt_provincial_months(
        df, ["Province"], "processing_kt", calendar_year
    )
    return (
        long.rename(columns={"Province": "province"})[
            ["province", "year", "month", "month_start", "processing_kt"]
        ]
        .sort_values(["month_start", "province"])
        .reset_index(drop=True)
    )


def parse_provincial_product_output(sheet: dict, calendar_year: int) -> pd.DataFrame:
    """Provincial monthly oil product output (kt/month)."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=["product", "province", "year", "month", "month_start", "output_kt"]
        )

    df = df[df["Data breakdown"] == "China"]
    df = df[df["Product"].isin(REGIONAL_OUTPUT_PRODUCTS)]
    long = _melt_provincial_months(
        df, ["Product", "Province"], "output_kt", calendar_year
    )
    return (
        long.rename(columns={"Product": "product", "Province": "province"})[
            ["product", "province", "year", "month", "month_start", "output_kt"]
        ]
        .sort_values(["product", "month_start", "province"])
        .reset_index(drop=True)
    )


def merge_provincial_processing(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_provincial_processing({"DataList": []}, calendar_year=2000)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["province", "year", "month"])
        .drop_duplicates(subset=["province", "year", "month"], keep="last")
        .reset_index(drop=True)
    )


def merge_provincial_product_output(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_provincial_product_output({"DataList": []}, calendar_year=2000)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["product", "province", "year", "month"])
        .drop_duplicates(subset=["product", "province", "year", "month"], keep="last")
        .reset_index(drop=True)
    )


# Equity-owner columns on SCI "15.Crude output". "Others" oilfield is not attributed
# to these four in the source totals, so it is kept as its own ownership bucket.
CRUDE_EQUITY_OWNERS = ("PetroChina", "Sinopec", "CNOOC", "Yanchang")

# Stable oilfield → operator map (validated against 2026 SCI monthly totals).
OILFIELD_OWNERSHIP: dict[str, str] = {
    "Dagang Oilfield": "PetroChina",
    "Huabei Oilfield": "PetroChina",
    "Jidong Oilfield": "PetroChina",
    "Liaohe Oilfield": "PetroChina",
    "Jilin Oilfield": "PetroChina",
    "Daqing Oilfield": "PetroChina",
    "Changqing Oilfield": "PetroChina",
    "Yumen Oilfield": "PetroChina",
    "Qinghai Oilfield": "PetroChina",
    "Tuha Oilfield": "PetroChina",
    "Xinjiang Oilfield": "PetroChina",
    "Tarim Oilfield": "PetroChina",
    "Xinan Oilfield": "PetroChina",
    "Baise Oilfield": "PetroChina",
    "Shanghai Oilfield": "Sinopec",
    "Jiangsu Oilfield": "Sinopec",
    "Shengli Oilfield": "Sinopec",
    "Zhongyuan Oilfield": "Sinopec",
    "Henan Oilfield": "Sinopec",
    "Jianghan Oilfield": "Sinopec",
    "Tahe Oilfield": "Sinopec",
    "Offshore crude": "CNOOC",
    "Yanchang Oilfield": "Yanchang",
    "Others": "Others",
}


def parse_crude_output(sheet: dict) -> pd.DataFrame:
    """
    Monthly crude production by province, equity owner, and oilfield (kt).

    Returns long rows with metric_type in {province_total, equity_owner, field}.
    National rollup row Province=='Total' is dropped (rebuild from provinces if needed).
    """
    cols = [
        "month_start",
        "province",
        "metric_type",
        "name",
        "ownership",
        "production_kt",
    ]
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["month_start"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["month_start"])
    df = df[df["Province"].astype(str) != "Total"]

    field_cols = [
        c
        for c in df.columns
        if c not in ("Date", "Province", "Total", *CRUDE_EQUITY_OWNERS, "month_start")
    ]

    rows: list[dict] = []
    for _, raw in df.iterrows():
        province = str(raw["Province"])
        month_start = raw["month_start"]
        total = pd.to_numeric(raw.get("Total"), errors="coerce")
        if pd.notna(total):
            rows.append(
                {
                    "month_start": month_start,
                    "province": province,
                    "metric_type": "province_total",
                    "name": province,
                    "ownership": None,
                    "production_kt": float(total),
                }
            )

        for owner in CRUDE_EQUITY_OWNERS:
            val = pd.to_numeric(raw.get(owner), errors="coerce")
            if pd.isna(val) or val == 0:
                continue
            rows.append(
                {
                    "month_start": month_start,
                    "province": province,
                    "metric_type": "equity_owner",
                    "name": owner,
                    "ownership": owner,
                    "production_kt": float(val),
                }
            )

        for field in field_cols:
            val = pd.to_numeric(raw.get(field), errors="coerce")
            if pd.isna(val) or val == 0:
                continue
            owner = OILFIELD_OWNERSHIP.get(field, "Others")
            rows.append(
                {
                    "month_start": month_start,
                    "province": province,
                    "metric_type": "field",
                    "name": field,
                    "ownership": owner,
                    "production_kt": float(val),
                }
            )

        # Residual "Others" oilfield is not in the four equity-owner columns.
        others_val = pd.to_numeric(raw.get("Others"), errors="coerce")
        if pd.notna(others_val) and others_val != 0:
            # equity_owner row for Others (field row already emitted above)
            rows.append(
                {
                    "month_start": month_start,
                    "province": province,
                    "metric_type": "equity_owner",
                    "name": "Others",
                    "ownership": "Others",
                    "production_kt": float(others_val),
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    # Drop duplicate equity_owner Others if field loop + residual both added it once each
    # (field loop does not add equity_owner; residual does — no dup). Keep stable order.
    return (
        out[cols]
        .drop_duplicates(
            subset=["month_start", "province", "metric_type", "name", "ownership"],
            keep="last",
        )
        .sort_values(["month_start", "province", "metric_type", "ownership", "name"])
        .reset_index(drop=True)
    )


def merge_crude_output(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_crude_output({"DataList": []})
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["month_start", "province", "metric_type", "ownership", "name"])
        .drop_duplicates(
            subset=["month_start", "province", "metric_type", "name", "ownership"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def build_runs_wide(weekly: pd.DataFrame) -> pd.DataFrame:
    """Pivot weekly runs metrics to one row per week_start × breakdown."""
    runs = weekly[weekly["metric"].isin(["processing_kt", "operating_rate"])]
    return (
        runs.pivot_table(
            index=["week_start", "breakdown"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["week_start", "breakdown"])
    )


def build_products_stocks_wide(weekly: pd.DataFrame) -> pd.DataFrame:
    """National weekly product output and commercial stocks."""
    metrics = [
        "gasoline_output_kt",
        "diesel_output_kt",
        "kerosene_output_kt",
        "gasoline_stocks_kt",
        "diesel_stocks_kt",
    ]
    subset = weekly[(weekly["breakdown"] == "National") & (weekly["metric"].isin(metrics))]
    return (
        subset.pivot_table(
            index="week_start",
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("week_start")
    )


def shandong_stocks_totals(stocks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Shandong independent city stocks to province total by week."""
    return (
        stocks.groupby(["week_start", "product"], as_index=False)["value_kt"]
        .sum()
        .sort_values(["week_start", "product"])
    )


def _dedupe_time_series(df: pd.DataFrame, subset: list[str], time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    out = out.sort_values(time_col).drop_duplicates(subset=subset, keep="last")
    return out.reset_index(drop=True)


def merge_weekly_summaries(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["week_start", "breakdown", "metric", "value", "unit"])
    merged = pd.concat(parts, ignore_index=True)
    return _dedupe_time_series(merged, ["week_start", "breakdown", "metric"], "week_start")


def merge_shandong_stocks(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["week_start", "region", "product", "value_kt"])
    merged = pd.concat(parts, ignore_index=True)
    return _dedupe_time_series(merged, ["week_start", "region", "product"], "week_start")


def parse_balance_monthly(sheet: dict) -> pd.DataFrame:
    """Monthly national balances — processing, utilisation, product stocks."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "month_start",
                "crude_processing_kt",
                "cdu_operating_rate",
                "gasoline_stocks_kt",
                "diesel_stocks_kt",
            ]
        )

    df["month_start"] = pd.to_datetime(df["Date"], errors="coerce")
    # Keep monthly rows only (exclude annual totals like 2025.0 / 2026E).
    df = df[df["Date"].astype(str).str.contains("-", na=False)]
    df = df[df["month_start"].notna()]

    col_map = {
        "Crude oil processing volume": "crude_processing_kt",
        "CDU operating rate": "cdu_operating_rate",
        "Gasoline ending stocks": "gasoline_stocks_kt",
        "Diesel ending stocks": "diesel_stocks_kt",
    }
    keep = ["month_start"] + [c for c in col_map if c in df.columns]
    out = df[keep].rename(columns=col_map)
    return out.sort_values("month_start").reset_index(drop=True)


def merge_balance_monthly(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_balance_monthly({"DataList": []})
    merged = pd.concat(parts, ignore_index=True)
    return _dedupe_time_series(merged, ["month_start"], "month_start")


def parse_profit_model(sheet: dict) -> pd.DataFrame:
    """Weekly refinery profit model — costs, prices, margins by refinery type."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "week_start",
                "refinery_type",
                "section",
                "factor",
                "unit",
                "value",
            ]
        )

    df["week_start"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.drop(columns=["Date"], errors="ignore")

    long = df.melt(id_vars=["week_start"], var_name="column", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value", "week_start"])

    parsed = long["column"].str.split(" - ", n=3, expand=True)
    parsed.columns = ["refinery_type", "section", "factor", "unit"]
    out = pd.concat([long[["week_start", "value"]], parsed], axis=1)
    return out.sort_values(["week_start", "refinery_type", "section", "factor"]).reset_index(
        drop=True
    )


def merge_profit_models(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_profit_model({"DataList": []})
    merged = pd.concat(parts, ignore_index=True)
    return _dedupe_time_series(
        merged,
        ["week_start", "refinery_type", "section", "factor", "unit"],
        "week_start",
    )


def build_margins_wide(margins: pd.DataFrame) -> pd.DataFrame:
    """Pivot key margin metrics for dashboard overlays."""
    if margins.empty:
        return pd.DataFrame(columns=["week_start"])

    profit_factors = margins["factor"].isin(["Net profit", "Profit per ton"])
    focus = margins[
        (margins["section"] == "Profit")
        & (
            margins["factor"].str.contains("margin", case=False, na=False)
            | profit_factors
        )
    ].copy()

    focus["metric_key"] = (
        focus["refinery_type"]
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "", regex=False)
        + "_"
        + focus["factor"]
        .str.replace(" (%)", "_pct", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )

    wide = focus.pivot_table(
        index="week_start",
        columns="metric_key",
        values="value",
        aggfunc="first",
    ).reset_index()

    return wide.sort_values("week_start")


CONSUMPTION_PRODUCT_ORDER = ["Gasoline", "Diesel", "Kerosene", "Naphtha", "VGO", "Residue"]

PETCHEM_PRODUCT_ORDER = ["PX", "Benzene", "Toluene", "Xylene", "Propylene", "Butadiene", "Ethylene"]

WEEKLY_CHEMICAL_OUTPUT_BREAKDOWN = "Chemical product output"
WEEKLY_CHEMICAL_RATE_BREAKDOWN = "Chemical unit operating rate"

WEEKLY_CHEMICAL_OUTPUT_ITEMS = {
    "PX output (kt)": "PX",
    "Benzene output (kt)": "Benzene",
    "Toluene output (kt)": "Toluene",
    "Xylene output (kt)": "Xylene",
    "Propylene output (kt)": "Propylene",
    "Butadiene output (kt)": "Butadiene",
}

WEEKLY_CHEMICAL_RATE_ITEMS = {
    "Benzene operating rate": "Benzene",
    "Toluene operating rate": "Toluene",
    "Xylene operating rate": "Xylene",
    "Propylene operating rate": "Propylene",
    "Butadiene operating rate": "Butadiene",
}

MONTHLY_CHEMICAL_OUTPUT_ITEMS = {
    "Benzene (kt)": "Benzene",
    "Toluene (kt)": "Toluene",
    "Xylene (kt)": "Xylene",
    "PX (kt)": "PX",
    "Ethylene (kt)": "Ethylene",
    "Propylene (kt)": "Propylene",
    "Butadiene (kt)": "Butadiene",
}

MONTHLY_CHEMICAL_RATE_ITEMS = {
    "Ethylene cracker": "Ethylene",
    "PX unit": "PX",
}

MONTHLY_INVENTORY_ITEMS = {
    "Benzene stocks at ports (kt)": ("Benzene", "port"),
    "Toluene stocks at ports (kt)": ("Toluene", "port"),
    "Xylene stocks at ports (kt)": ("Xylene", "port"),
    "Independent refinery naphtha stocks (kt)": ("Naphtha", "refinery"),
}

JET_FUEL_PRODUCT_ORDER = [
    "Jet fuel (total)",
    "Jet fuel (international)",
    "Jet fuel (domestic)",
]

REFUELING_ITEM_LABELS = {
    "International flight": "Jet fuel (international)",
    "Domestic flight": "Jet fuel (domestic)",
}


def _normalize_consumption_product(name: str) -> str:
    aliases = {"Residual oil": "Residue", "Residue": "Residue"}
    return aliases.get(name, name)


def parse_consumption(sheet: dict, calendar_year: int) -> pd.DataFrame:
    """
    Provincial monthly product consumption (kt/month).

    Each API snapshot stores one calendar year in the Jan–Dec columns.
    """
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=["product", "region", "province", "year", "month", "consumption_kt"]
        )

    month_cols = [c for c in CONSUMPTION_MONTHS if c in df.columns]
    long = df.melt(
        id_vars=["Product", "Region", "Province"],
        value_vars=month_cols,
        var_name="month_label",
        value_name="consumption_kt",
    )
    long["consumption_kt"] = pd.to_numeric(long["consumption_kt"], errors="coerce")
    long = long.dropna(subset=["consumption_kt"])
    long["month"] = long["month_label"].map(
        {m: i + 1 for i, m in enumerate(CONSUMPTION_MONTHS)}
    )
    long["year"] = calendar_year
    long["product"] = long["Product"].map(_normalize_consumption_product)

    return (
        long.rename(columns={"Region": "region", "Province": "province"})[
            ["product", "region", "province", "year", "month", "consumption_kt"]
        ]
        .sort_values(["product", "year", "month", "province"])
        .reset_index(drop=True)
    )


def merge_consumption(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_consumption({"DataList": []}, calendar_year=2000)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["product", "province", "year", "month"])
        .drop_duplicates(subset=["product", "province", "year", "month"], keep="last")
        .reset_index(drop=True)
    )


def build_consumption_national(consumption: pd.DataFrame) -> pd.DataFrame:
    """Sum provinces to national consumption by product × year × month."""
    if consumption.empty:
        return pd.DataFrame(columns=["product", "year", "month", "consumption_kt"])
    return (
        consumption.groupby(["product", "year", "month"], as_index=False)["consumption_kt"]
        .sum()
        .sort_values(["product", "year", "month"])
    )


def parse_kerosene_refueling(sheet: dict, calendar_year: int) -> pd.DataFrame:
    """
    Provincial kerosene refueling volume (kt/month) — proxy for jet fuel uplift.

    Split by international vs domestic flights at province level.
    """
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=["product", "province", "year", "month", "consumption_kt"]
        )

    month_cols = [c for c in CONSUMPTION_MONTHS if c in df.columns]
    long = df.melt(
        id_vars=["Item", "Province"],
        value_vars=month_cols,
        var_name="month_label",
        value_name="consumption_kt",
    )
    long["consumption_kt"] = pd.to_numeric(long["consumption_kt"], errors="coerce")
    long = long.dropna(subset=["consumption_kt"])
    long["month"] = long["month_label"].map(
        {m: i + 1 for i, m in enumerate(CONSUMPTION_MONTHS)}
    )
    long["year"] = calendar_year
    long["product"] = long["Item"].map(REFUELING_ITEM_LABELS)

    return (
        long.rename(columns={"Province": "province"})[
            ["product", "province", "year", "month", "consumption_kt"]
        ]
        .sort_values(["product", "year", "month", "province"])
        .reset_index(drop=True)
    )


def merge_kerosene_refueling(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_kerosene_refueling({"DataList": []}, calendar_year=2000)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["product", "province", "year", "month"])
        .drop_duplicates(subset=["product", "province", "year", "month"], keep="last")
        .reset_index(drop=True)
    )


def build_kerosene_refueling_national(refueling: pd.DataFrame) -> pd.DataFrame:
    """National jet fuel refueling by segment plus total (intl + domestic)."""
    if refueling.empty:
        return pd.DataFrame(columns=["product", "year", "month", "consumption_kt"])

    by_segment = (
        refueling.groupby(["product", "year", "month"], as_index=False)["consumption_kt"]
        .sum()
        .sort_values(["product", "year", "month"])
    )
    total = (
        refueling.groupby(["year", "month"], as_index=False)["consumption_kt"]
        .sum()
        .assign(product="Jet fuel (total)")
    )
    return pd.concat([total, by_segment], ignore_index=True).sort_values(
        ["product", "year", "month"]
    )


def _parse_weekly_chemical_rows(
    sheet: dict,
    *,
    breakdown: str,
    item_map: dict[str, str],
    metric: str,
) -> pd.DataFrame:
    """Extract one metric family from the weekly summary chemical sections."""
    raw = pd.DataFrame(sheet.get("DataList", []))
    base_cols = ["week_start", "product", "metric", "value", "unit"]
    if raw.empty:
        return pd.DataFrame(columns=base_cols)

    week_cols = _week_columns(raw.columns)
    date_row = raw.iloc[0]
    week_dates = {}
    for col in week_cols:
        ts = pd.to_datetime(date_row[col], errors="coerce")
        week_dates[col] = ts.normalize() if pd.notna(ts) else pd.NaT

    body = raw.iloc[1:].copy()
    body = body[(body["Breakdown"] == breakdown) & (body["Item"].isin(item_map))]
    if body.empty:
        return pd.DataFrame(columns=base_cols)

    long = _melt_week_columns(body, ["Breakdown", "Item", "Unit"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["week_start"] = long["week_label"].map(week_dates)
    long = long.dropna(subset=["week_start", "value"])
    long["product"] = long["Item"].map(item_map)
    long["metric"] = metric
    long = long.rename(columns={"Unit": "unit"})

    return (
        long[["week_start", "product", "metric", "value", "unit"]]
        .sort_values(["product", "metric", "week_start"])
        .reset_index(drop=True)
    )


def parse_weekly_chemical(sheet: dict) -> pd.DataFrame:
    """
    National weekly petchem output (kt) and unit operating rates (0–1 fraction).

    Sourced from the Chemical product output / Chemical unit operating rate
    sections of the weekly summary sheet.
    """
    output = _parse_weekly_chemical_rows(
        sheet,
        breakdown=WEEKLY_CHEMICAL_OUTPUT_BREAKDOWN,
        item_map=WEEKLY_CHEMICAL_OUTPUT_ITEMS,
        metric="output_kt",
    )
    rates = _parse_weekly_chemical_rows(
        sheet,
        breakdown=WEEKLY_CHEMICAL_RATE_BREAKDOWN,
        item_map=WEEKLY_CHEMICAL_RATE_ITEMS,
        metric="operating_rate",
    )
    parts = [df for df in (output, rates) if not df.empty]
    if not parts:
        return pd.DataFrame(columns=["week_start", "product", "metric", "value", "unit"])
    return pd.concat(parts, ignore_index=True).sort_values(
        ["product", "metric", "week_start"]
    )


def merge_weekly_chemical(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["week_start", "product", "metric", "value", "unit"])
    merged = pd.concat(parts, ignore_index=True)
    return _dedupe_time_series(merged, ["week_start", "product", "metric"], "week_start")


def _melt_monthly_summary_items(
    df: pd.DataFrame,
    item_map: dict[str, str],
    *,
    calendar_year: int,
    metric: str,
    value_col: str = "value",
) -> pd.DataFrame:
    """Melt Jan–Dec columns for selected monthly-summary Item rows."""
    if df.empty:
        return pd.DataFrame(columns=["product", "year", "month", "month_start", "metric", value_col])

    month_cols = [c for c in CONSUMPTION_MONTHS if c in df.columns]
    rows = df[df["Item"].isin(item_map)]
    if rows.empty:
        return pd.DataFrame(columns=["product", "year", "month", "month_start", "metric", value_col])

    long = rows.melt(
        id_vars=["Item"],
        value_vars=month_cols,
        var_name="month_label",
        value_name=value_col,
    )
    long[value_col] = pd.to_numeric(long[value_col], errors="coerce")
    long = long.dropna(subset=[value_col])
    long["month"] = long["month_label"].map(
        {m: i + 1 for i, m in enumerate(CONSUMPTION_MONTHS)}
    )
    long["year"] = calendar_year
    long["month_start"] = pd.to_datetime({"year": long["year"], "month": long["month"], "day": 1})
    long["product"] = long["Item"].map(item_map)
    long["metric"] = metric
    return long[["product", "year", "month", "month_start", "metric", value_col]].sort_values(
        ["product", "metric", "month_start"]
    )


def parse_monthly_petchem(sheet: dict, calendar_year: int) -> pd.DataFrame:
    """National monthly petchem output and unit operating rates."""
    df = pd.DataFrame(sheet.get("DataList", []))
    output = _melt_monthly_summary_items(
        df,
        MONTHLY_CHEMICAL_OUTPUT_ITEMS,
        calendar_year=calendar_year,
        metric="output_kt",
        value_col="value",
    )
    rates = _melt_monthly_summary_items(
        df,
        MONTHLY_CHEMICAL_RATE_ITEMS,
        calendar_year=calendar_year,
        metric="operating_rate",
        value_col="value",
    )
    parts = [part for part in (output, rates) if not part.empty]
    if not parts:
        return pd.DataFrame(
            columns=["product", "year", "month", "month_start", "metric", "value"]
        )
    return pd.concat(parts, ignore_index=True).sort_values(
        ["product", "metric", "month_start"]
    )


def parse_monthly_inventories(sheet: dict, calendar_year: int) -> pd.DataFrame:
    """Port BTX stocks and independent-refinery naphtha stocks (kt, monthly)."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=["product", "inventory_type", "year", "month", "month_start", "value_kt"]
        )

    month_cols = [c for c in CONSUMPTION_MONTHS if c in df.columns]
    rows = df[df["Item"].isin(MONTHLY_INVENTORY_ITEMS)]
    if rows.empty:
        return pd.DataFrame(
            columns=["product", "inventory_type", "year", "month", "month_start", "value_kt"]
        )

    long = rows.melt(
        id_vars=["Item"],
        value_vars=month_cols,
        var_name="month_label",
        value_name="value_kt",
    )
    long["value_kt"] = pd.to_numeric(long["value_kt"], errors="coerce")
    long = long.dropna(subset=["value_kt"])
    long["month"] = long["month_label"].map(
        {m: i + 1 for i, m in enumerate(CONSUMPTION_MONTHS)}
    )
    long["year"] = calendar_year
    long["month_start"] = pd.to_datetime({"year": long["year"], "month": long["month"], "day": 1})
    mapped = long["Item"].map(MONTHLY_INVENTORY_ITEMS)
    long["product"] = mapped.map(lambda x: x[0])
    long["inventory_type"] = mapped.map(lambda x: x[1])

    return (
        long[["product", "inventory_type", "year", "month", "month_start", "value_kt"]]
        .sort_values(["inventory_type", "product", "month_start"])
        .reset_index(drop=True)
    )


def merge_monthly_petchem(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_monthly_petchem({"DataList": []}, calendar_year=2000)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["product", "metric", "year", "month"])
        .drop_duplicates(subset=["product", "metric", "year", "month"], keep="last")
        .reset_index(drop=True)
    )


def merge_monthly_inventories(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_monthly_inventories({"DataList": []}, calendar_year=2000)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["inventory_type", "product", "year", "month"])
        .drop_duplicates(
            subset=["product", "inventory_type", "year", "month"],
            keep="last",
        )
        .reset_index(drop=True)
    )


# Refined product export shipping (sheet 40/41) and quotas (sheet 45).
EXPORT_CARGO_LABELS = {
    "Gasoline": "Gasoline",
    "Kerosene": "Jet",
    "Diesel": "Gasoil/Diesel",
}
EXPORT_QUOTA_PRODUCTS = {"Gasoline", "Diesel", "Kerosene"}
EXPORT_TABLE_PRODUCTS = ["Gasoline", "Jet", "Gasoil/Diesel", "FO"]


def _normalize_export_cargo(cargo: str) -> str | None:
    return EXPORT_CARGO_LABELS.get(str(cargo).strip())


def parse_refined_export_shipping(sheet: dict) -> pd.DataFrame:
    """Cargo-level refined oil export shipments (sheet 40 — actual)."""
    df = pd.DataFrame(sheet.get("DataList", []))
    cols = [
        "province",
        "departure_port",
        "ship_name",
        "departure_date",
        "arrival_date",
        "destination",
        "volume_kt",
        "cargo",
        "product",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    out = df.rename(
        columns={
            "Province": "province",
            "Departure port": "departure_port",
            "Ship name": "ship_name",
            "Departure date": "departure_date",
            "Actual arrival date": "arrival_date",
            "Destination": "destination",
            "Volume (kt)": "volume_kt",
            "Cargo": "cargo",
        }
    )
    out["departure_date"] = pd.to_datetime(out["departure_date"], errors="coerce").dt.normalize()
    out["arrival_date"] = pd.to_datetime(out["arrival_date"], errors="coerce").dt.normalize()
    out["volume_kt"] = pd.to_numeric(out["volume_kt"], errors="coerce")
    out = out.dropna(subset=["departure_date", "volume_kt", "cargo"])
    out["product"] = out["cargo"].map(_normalize_export_cargo)
    out = out.dropna(subset=["product"])
    out["month_start"] = out["departure_date"].values.astype("datetime64[M]")
    return out[cols + ["month_start"]].sort_values(["departure_date", "province", "destination"])


def parse_refined_export_forecast(sheet: dict) -> pd.DataFrame:
    """Forward refined oil export cargoes (sheet 41 — forecast)."""
    df = pd.DataFrame(sheet.get("DataList", []))
    cols = [
        "province",
        "departure_port",
        "ship_name",
        "departure_date",
        "arrival_date",
        "destination",
        "volume_kt",
        "cargo",
        "product",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    out = df.rename(
        columns={
            "Province": "province",
            "Departure port": "departure_port",
            "Ship name": "ship_name",
            "Departure date": "departure_date",
            "Estimated arrival date": "arrival_date",
            "Destination": "destination",
            "Volume (kt)": "volume_kt",
            "Cargo": "cargo",
        }
    )
    out["departure_date"] = pd.to_datetime(out["departure_date"], errors="coerce").dt.normalize()
    out["arrival_date"] = pd.to_datetime(out["arrival_date"], errors="coerce").dt.normalize()
    out["volume_kt"] = pd.to_numeric(out["volume_kt"], errors="coerce")
    out = out.dropna(subset=["departure_date", "volume_kt", "cargo"])
    out["product"] = out["cargo"].map(_normalize_export_cargo)
    out = out.dropna(subset=["product"])
    out["month_start"] = out["departure_date"].values.astype("datetime64[M]")
    return out[cols + ["month_start"]].sort_values(["departure_date", "province", "destination"])


def merge_refined_export_shipping(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_refined_export_shipping({"DataList": []})
    merged = pd.concat(parts, ignore_index=True)
    key = ["ship_name", "departure_date", "destination", "volume_kt", "cargo", "province"]
    return (
        merged.sort_values("departure_date")
        .drop_duplicates(subset=key, keep="last")
        .reset_index(drop=True)
    )


def _parse_sci_month(values: pd.Series) -> pd.Series:
    """
    Parse SCI sheet Month values.

    Older datapacks use ISO dates (YYYY-MM-DD). From ~2026-09-04 the API
    switched some sheets to Excel serial day numbers (e.g. 46023 → 2026-01-01).
    """
    iso = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    serial = pd.to_numeric(values, errors="coerce")
    from_serial = pd.to_datetime(
        serial, unit="D", origin="1899-12-30", errors="coerce"
    )
    return iso.fillna(from_serial)


def parse_export_quotas_monthly(sheet: dict) -> pd.DataFrame:
    """Monthly national export volumes by product from sheet 45."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=["product", "year", "month", "month_start", "export_kt"])

    rows = df[df["Product"].isin(EXPORT_QUOTA_PRODUCTS)].copy()
    rows = rows[rows["Month"].astype(str).str.lower() != "total"]
    rows["Month"] = _parse_sci_month(rows["Month"])
    rows = rows[rows["Month"].notna()]
    rows["export_kt"] = pd.to_numeric(rows["Export"], errors="coerce")
    rows = rows.dropna(subset=["export_kt"])
    rows["year"] = rows["Month"].dt.year
    rows["month"] = rows["Month"].dt.month
    rows["month_start"] = rows["Month"].dt.to_period("M").dt.to_timestamp()
    rows["product"] = rows["Product"].map(_normalize_export_cargo)

    return (
        rows[["product", "year", "month", "month_start", "export_kt"]]
        .sort_values(["product", "month_start"])
        .reset_index(drop=True)
    )


def parse_export_quotas_summary(sheet: dict) -> pd.DataFrame:
    """Quota batches, annual ceiling, and utilization from sheet 45."""
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "quota_year",
                "batch_1_kt",
                "batch_2_kt",
                "batch_3_kt",
                "total_quota_kt",
                "ytd_export_kt",
                "utilization",
            ]
        )

    lookup = df.set_index("Product")["Export"].to_dict()

    def _num(key: str) -> float | None:
        val = lookup.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)

    quota_year = None
    total_quota_kt = None
    for key in lookup:
        m = re.search(r"Total Quota - (\d{4})", str(key))
        if m:
            quota_year = int(m.group(1))
            total_quota_kt = _num(str(key))
            break

    return pd.DataFrame(
        [
            {
                "quota_year": quota_year,
                "batch_1_kt": _num("Export Quota - 1st batch"),
                "batch_2_kt": _num("Export Quota - 2nd batch"),
                "batch_3_kt": _num("Export Quota - 3rd batch"),
                "total_quota_kt": total_quota_kt,
                "ytd_export_kt": _num("Gasoline, diesel and kerosene total export"),
                "utilization": _num("Utilization"),
            }
        ]
    )


def merge_export_quotas_monthly(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return parse_export_quotas_monthly({"DataList": []})
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values(["product", "year", "month"])
        .drop_duplicates(subset=["product", "year", "month"], keep="last")
        .reset_index(drop=True)
    )


def build_export_monthly_wide(
    monthly_long: pd.DataFrame,
    *,
    value_col: str = "volume_kt",
) -> pd.DataFrame:
    """
    Pivot long monthly exports to a Kpler-style wide table.

    Columns: Gasoline, Jet, Gasoil/Diesel, FO (FO is always 0 — not in SCI shipping).
    Index: month labels YYYY-MM.
    """
    if monthly_long.empty:
        return pd.DataFrame(columns=["month"] + EXPORT_TABLE_PRODUCTS)

    agg = (
        monthly_long.groupby(["month_start", "product"], as_index=False)[value_col]
        .sum()
    )
    wide = agg.pivot(index="month_start", columns="product", values=value_col).fillna(0)
    for col in EXPORT_TABLE_PRODUCTS:
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide[EXPORT_TABLE_PRODUCTS]
    wide["month"] = wide.index.strftime("%Y-%m")
    return wide.reset_index(drop=True)[["month"] + EXPORT_TABLE_PRODUCTS]


def build_export_monthly_from_shipping(shipping: pd.DataFrame) -> pd.DataFrame:
    """Monthly export totals from cargo-level shipping actuals."""
    if shipping.empty:
        return build_export_monthly_wide(pd.DataFrame())
    return build_export_monthly_wide(
        shipping[["month_start", "product", "volume_kt"]],
        value_col="volume_kt",
    )


def build_export_monthly_from_quotas(quotas_monthly: pd.DataFrame) -> pd.DataFrame:
    """Monthly export totals from national quota utilization sheet."""
    if quotas_monthly.empty:
        return build_export_monthly_wide(pd.DataFrame())
    return build_export_monthly_wide(
        quotas_monthly[["month_start", "product", "export_kt"]].rename(
            columns={"export_kt": "volume_kt"}
        ),
        value_col="volume_kt",
    )


# ---------------------------------------------------------------------------
# Crude port shipping (sheets 36–39) and independent crude import quotas (44).
# ---------------------------------------------------------------------------

_CRUDE_MONTH_COLS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

CRUDE_SHIPPING_COLS = [
    "trade",
    "ship_name",
    "ship_grade",
    "origin_country",
    "departure_date",
    "province",
    "discharge_port",
    "arrival_date",
    "discharge_date",
    "volume_kt",
    "crude_grade",
    "month_start",
]


def parse_crude_port_shipping(
    sheet: dict,
    *,
    trade: str,
    is_forecast: bool = False,
) -> pd.DataFrame:
    """
    Cargo-level crude arrivals at Chinese ports.

    trade: ``foreign`` | ``domestic``
    Arrival date is actual (sheet 36/38) or estimated (sheet 37/39).
    """
    empty = pd.DataFrame(columns=CRUDE_SHIPPING_COLS)
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return empty

    arrival_src = "Estimated arrival date" if is_forecast else "Actual arrival date"
    # API headers include awkward double-spaces / trailing spaces — match flexibly.
    colmap: dict[str, str] = {}
    for c in df.columns:
        cl = str(c).strip().lower().replace("  ", " ")
        if cl == "ship name":
            colmap[c] = "ship_name"
        elif cl == "ship grade":
            colmap[c] = "ship_grade"
        elif cl == "origin country":
            colmap[c] = "origin_country"
        elif cl.startswith("departure date"):
            colmap[c] = "departure_date"
        elif "province" in cl and "discharge" in cl:
            colmap[c] = "province"
        elif cl == "port of discharge":
            colmap[c] = "discharge_port"
        elif cl == arrival_src.strip().lower():
            colmap[c] = "arrival_date"
        elif cl == "discharge date":
            colmap[c] = "discharge_date"
        elif cl.startswith("volume"):
            colmap[c] = "volume_kt"
        elif cl == "crude grade":
            colmap[c] = "crude_grade"

    out = df.rename(columns=colmap)
    for col in ("departure_date", "arrival_date", "discharge_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
        else:
            out[col] = pd.NaT

    if "volume_kt" not in out.columns:
        return empty
    out["volume_kt"] = pd.to_numeric(out["volume_kt"], errors="coerce")
    out = out.dropna(subset=["arrival_date", "volume_kt"])
    out["trade"] = trade
    if "origin_country" not in out.columns:
        out["origin_country"] = None
    for col in ("ship_name", "ship_grade", "province", "discharge_port", "crude_grade"):
        if col not in out.columns:
            out[col] = None
    out["month_start"] = out["arrival_date"].values.astype("datetime64[M]")
    return (
        out[CRUDE_SHIPPING_COLS]
        .sort_values(["arrival_date", "province", "discharge_port"])
        .reset_index(drop=True)
    )


def merge_crude_port_shipping(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=CRUDE_SHIPPING_COLS)
    merged = pd.concat(parts, ignore_index=True)
    key = [
        "trade",
        "ship_name",
        "arrival_date",
        "discharge_port",
        "volume_kt",
        "crude_grade",
        "origin_country",
    ]
    return (
        merged.sort_values("arrival_date")
        .drop_duplicates(subset=key, keep="last")
        .reset_index(drop=True)
    )


def parse_crude_import_quotas(
    sheet: dict,
    *,
    quota_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Independent-refinery crude import quotas (sheet 44).

    Returns (summary_one_row, monthly_long).
    """
    summary_cols = [
        "quota_year",
        "ceiling_kt",
        "batch_1_kt",
        "batch_2_kt",
        "batch_3_kt",
        "batch_4_kt",
        "batch_5_kt",
        "annual_quota_kt",
        "total_imports_kt",
        "utilization",
        "remarks",
    ]
    monthly_cols = ["quota_year", "month", "month_start", "import_kt"]
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=monthly_cols)

    row = df.iloc[0]
    topic = str(sheet.get("Topic") or "")
    year = quota_year
    if year is None:
        m = re.search(r"(20\d{2})", topic)
        year = int(m.group(1)) if m else pd.Timestamp.today().year

    def _num(col: str) -> float | None:
        if col not in row.index:
            return None
        val = pd.to_numeric(row[col], errors="coerce")
        return float(val) if pd.notna(val) else None

    summary = pd.DataFrame(
        [
            {
                "quota_year": year,
                "ceiling_kt": _num("Ceiling quotas for independent refineries"),
                "batch_1_kt": _num("1st batch of crude import quota"),
                "batch_2_kt": _num("2nd batch of crude import quota"),
                "batch_3_kt": _num("3rd batch of crude import quota"),
                "batch_4_kt": _num("4th batch of crude import quota"),
                "batch_5_kt": _num("5th batch of crude import quota"),
                "annual_quota_kt": _num("Annual total quota"),
                "total_imports_kt": _num("Total imports"),
                "utilization": _num("Utilization"),
                "remarks": row.get("Remarks"),
            }
        ]
    )

    monthly_rows = []
    for i, label in enumerate(_CRUDE_MONTH_COLS, start=1):
        val = _num(label)
        if val is None:
            continue
        monthly_rows.append(
            {
                "quota_year": year,
                "month": i,
                "month_start": pd.Timestamp(year=year, month=i, day=1),
                "import_kt": val,
            }
        )
    monthly = pd.DataFrame(monthly_rows, columns=monthly_cols)
    return summary, monthly


# ---------------------------------------------------------------------------
# Shandong independent processing by oil grade (sheet 16).
# ---------------------------------------------------------------------------

_OIL_GRADE_META_COLS = {
    "Month",
    "City",
    "Total primary feedstock processing volume",
    "Secondary feedstock processing volume",
}

_OIL_GRADE_DOMESTIC = {
    "Shengli crude oil",
    "Marine oil",
    "Other domestic crude oil",
}

_OIL_GRADE_SECONDARY = {
    "M100",
    "Straight-run 180CST",
    "380CST",
    "Straight-run 280CST",
    "Cutback asphalt",
    "Chinese-made residue",
    "Chinese-made VGO",
}

_COUNTRY_ALIASES = {
    "u.s.": "United States",
    "us": "United States",
    "u.s": "United States",
    "columbia": "Colombia",
    "saudi arabia": "Saudi Arabia",
    "uae": "UAE",
    "uk": "United Kingdom",
    "equatorial guinea": "Equatorial Guinea",
    "south africa": "South Africa",
}


def _oil_grade_bucket(grade: str) -> str:
    g = grade.strip()
    if g in _OIL_GRADE_DOMESTIC:
        return "Domestic"
    if g in _OIL_GRADE_SECONDARY or g.startswith("Straight-run") or g.endswith("CST"):
        return "Secondary / residue"
    if g == "Other imported crude":
        return "Other imported"
    # "Country-Grade" or "Country -Grade"
    parts = re.split(r"\s*-\s*", g, maxsplit=1)
    if len(parts) >= 1 and parts[0]:
        key = parts[0].strip().lower()
        return _COUNTRY_ALIASES.get(key, parts[0].strip())
    return "Other"


def parse_processing_by_oil_grade(sheet: dict) -> pd.DataFrame:
    """
    Shandong independent refinery feedstock by city and oil grade (sheet 16).

    Long rows: month_start, city, grade, origin_bucket, volume_kt.
    Also emits synthetic rows grade='_total_primary' / '_total_secondary' for totals.
    """
    cols = ["month_start", "city", "grade", "origin_bucket", "volume_kt"]
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Normalize awkward column names (trailing spaces / encoding).
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "Month" not in df.columns or "City" not in df.columns:
        return pd.DataFrame(columns=cols)

    df["month_start"] = pd.to_datetime(df["Month"], errors="coerce").dt.normalize()
    df["city"] = df["City"].astype(str).str.strip()
    df = df.dropna(subset=["month_start", "city"])

    grade_cols = [c for c in df.columns if c not in ("Month", "City", "month_start", "city")]
    rows: list[dict] = []
    for _, raw in df.iterrows():
        month_start = raw["month_start"]
        city = raw["city"]
        for col in grade_cols:
            val = pd.to_numeric(raw.get(col), errors="coerce")
            if pd.isna(val) or val == 0:
                continue
            if col == "Total primary feedstock processing volume":
                grade, bucket = "_total_primary", "Total"
            elif col == "Secondary feedstock processing volume":
                grade, bucket = "_total_secondary", "Total"
            else:
                grade, bucket = col, _oil_grade_bucket(col)
            rows.append(
                {
                    "month_start": month_start,
                    "city": city,
                    "grade": grade,
                    "origin_bucket": bucket,
                    "volume_kt": float(val),
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(rows)[cols]
        .sort_values(["month_start", "city", "origin_bucket", "grade"])
        .reset_index(drop=True)
    )


def merge_processing_by_oil_grade(parts: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["month_start", "city", "grade", "origin_bucket", "volume_kt"]
    if not parts:
        return pd.DataFrame(columns=cols)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values("month_start")
        .drop_duplicates(subset=["month_start", "city", "grade"], keep="last")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Vehicle sales / possession (sheets 26–30) — mobility vs gasoline/diesel demand
# ---------------------------------------------------------------------------

# Truck segments that matter for diesel (micro trucks are mostly gasoline).
DIESEL_TRUCK_TYPES = ("Heavy truck", "Medium truck", "Light truck", "Pickup truck")

_PASSENGER_POWERTRAIN_COLS = {
    "Gasoline": "gasoline",
    "Diesel": "diesel",
    "BEV": "bev",
    "Plug-in hybrid": "phev",
    "Non plug-in hybrid": "hev",
    "Extended-range electric": "erev",
    "Non-plug-in extended-range electric": "non_plugin_erev",
    "CNG": "cng",
    "Others": "others",
    "Total": "total",
}

_COMMERCIAL_FUEL_COLS = {
    "Diesel": "diesel",
    "Gasoline": "gasoline",
    "CNG": "cng",
    "LNG": "lng",
    "New energy": "new_energy",
    "Others": "others",
    "Total": "total",
}


def _sheet_calendar_year(sheet: dict, fallback: int | None = None) -> int | None:
    """Prefer year embedded in Topic (e.g. '2026 China Vehicle Sales')."""
    topic = str(sheet.get("Topic") or "")
    m = re.search(r"(20\d{2})", topic)
    if m:
        return int(m.group(1))
    return fallback


def parse_vehicle_sales_simple(sheet: dict, calendar_year: int | None = None) -> pd.DataFrame:
    """
    Sheet 26 — province × month sales by broad vehicle type (ICE gasoline/diesel/NEV/LNG).

    Long rows: month_start, province, vehicle_type, units.
    """
    cols = ["month_start", "province", "vehicle_type", "units"]
    year = _sheet_calendar_year(sheet, calendar_year)
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty or year is None:
        return pd.DataFrame(columns=cols)

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "Province" not in df.columns or "Vehicle type" not in df.columns:
        return pd.DataFrame(columns=cols)

    month_cols = [c for c in CONSUMPTION_MONTHS if c in df.columns]
    long = df.melt(
        id_vars=["Province", "Vehicle type"],
        value_vars=month_cols,
        var_name="month_label",
        value_name="units",
    )
    long["units"] = pd.to_numeric(long["units"], errors="coerce")
    long = long.dropna(subset=["units"])
    long["month"] = long["month_label"].map({m: i + 1 for i, m in enumerate(CONSUMPTION_MONTHS)})
    long["month_start"] = pd.to_datetime(
        dict(year=year, month=long["month"], day=1)
    )
    long["province"] = long["Province"].astype(str).str.strip()
    long["vehicle_type"] = long["Vehicle type"].astype(str).str.strip()
    return (
        long[cols]
        .sort_values(["month_start", "province", "vehicle_type"])
        .reset_index(drop=True)
    )


def parse_vehicle_sales_passenger(sheet: dict) -> pd.DataFrame:
    """Sheet 28 — passenger sales by powertrain (province × month)."""
    value_cols = list(_PASSENGER_POWERTRAIN_COLS.values())
    cols = ["month_start", "province"] + value_cols
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    # "Breakdown" holds province names in this sheet.
    geo_col = "Breakdown" if "Breakdown" in df.columns else "Province"
    if "Date" not in df.columns or geo_col not in df.columns:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame()
    out["month_start"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    out["province"] = df[geo_col].astype(str).str.strip()
    for src, dst in _PASSENGER_POWERTRAIN_COLS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.NA
    out = out.dropna(subset=["month_start", "province"])
    return out[cols].sort_values(["month_start", "province"]).reset_index(drop=True)


def parse_vehicle_sales_commercial(sheet: dict) -> pd.DataFrame:
    """Sheet 27 — commercial sales by vehicle type × fuel (province × month)."""
    value_cols = list(_COMMERCIAL_FUEL_COLS.values())
    cols = ["month_start", "province", "vehicle_type"] + value_cols
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "Date" not in df.columns or "Province" not in df.columns or "Vehicle type" not in df.columns:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame()
    out["month_start"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    out["province"] = df["Province"].astype(str).str.strip()
    out["vehicle_type"] = df["Vehicle type"].astype(str).str.strip()
    for src, dst in _COMMERCIAL_FUEL_COLS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.NA
    out = out.dropna(subset=["month_start", "province", "vehicle_type"])
    return (
        out[cols]
        .sort_values(["month_start", "province", "vehicle_type"])
        .reset_index(drop=True)
    )


def parse_vehicle_possession_passenger(sheet: dict) -> pd.DataFrame:
    """
    Sheet 30 — passenger fleet possession by powertrain.

    Annual snapshot (Date often a year float like 2025.0).
    """
    value_cols = list(_PASSENGER_POWERTRAIN_COLS.values())
    cols = ["year", "province"] + value_cols
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    geo_col = "Breakdown" if "Breakdown" in df.columns else "Province"
    if "Date" not in df.columns or geo_col not in df.columns:
        return pd.DataFrame(columns=cols)

    raw_date = pd.to_numeric(df["Date"], errors="coerce")
    # Year float (2025.0) or a timestamp year.
    years = raw_date.round().astype("Int64")
    # If values look like unix/ordinal nonsense, try datetime year.
    if years.dropna().empty or (years.dropna() < 1990).any() or (years.dropna() > 2100).any():
        years = pd.to_datetime(df["Date"], errors="coerce").dt.year.astype("Int64")

    out = pd.DataFrame()
    out["year"] = years
    out["province"] = df[geo_col].astype(str).str.strip()
    for src, dst in _PASSENGER_POWERTRAIN_COLS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.NA
    out = out.dropna(subset=["year", "province"])
    out["year"] = out["year"].astype(int)
    return out[cols].sort_values(["year", "province"]).reset_index(drop=True)


def parse_vehicle_possession_commercial(sheet: dict) -> pd.DataFrame:
    """Sheet 29 — commercial fleet possession by vehicle type × fuel (monthly)."""
    value_cols = list(_COMMERCIAL_FUEL_COLS.values())
    cols = ["month_start", "province", "vehicle_type"] + value_cols
    df = pd.DataFrame(sheet.get("DataList", []))
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "Date" not in df.columns or "Province" not in df.columns or "Vehicle type" not in df.columns:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame()
    out["month_start"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    out["province"] = df["Province"].astype(str).str.strip()
    out["vehicle_type"] = df["Vehicle type"].astype(str).str.strip()
    for src, dst in _COMMERCIAL_FUEL_COLS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.NA
    out = out.dropna(subset=["month_start", "province", "vehicle_type"])
    return (
        out[cols]
        .sort_values(["month_start", "province", "vehicle_type"])
        .reset_index(drop=True)
    )


def merge_vehicle_sales_simple(parts: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["month_start", "province", "vehicle_type", "units"]
    if not parts:
        return pd.DataFrame(columns=cols)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values("month_start")
        .drop_duplicates(subset=["month_start", "province", "vehicle_type"], keep="last")
        .reset_index(drop=True)
    )


def merge_vehicle_sales_passenger(parts: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["month_start", "province"] + list(_PASSENGER_POWERTRAIN_COLS.values())
    if not parts:
        return pd.DataFrame(columns=cols)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values("month_start")
        .drop_duplicates(subset=["month_start", "province"], keep="last")
        .reset_index(drop=True)
    )


def merge_vehicle_sales_commercial(parts: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["month_start", "province", "vehicle_type"] + list(_COMMERCIAL_FUEL_COLS.values())
    if not parts:
        return pd.DataFrame(columns=cols)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values("month_start")
        .drop_duplicates(subset=["month_start", "province", "vehicle_type"], keep="last")
        .reset_index(drop=True)
    )


def merge_vehicle_possession_passenger(parts: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["year", "province"] + list(_PASSENGER_POWERTRAIN_COLS.values())
    if not parts:
        return pd.DataFrame(columns=cols)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values("year")
        .drop_duplicates(subset=["year", "province"], keep="last")
        .reset_index(drop=True)
    )


def merge_vehicle_possession_commercial(parts: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["month_start", "province", "vehicle_type"] + list(_COMMERCIAL_FUEL_COLS.values())
    if not parts:
        return pd.DataFrame(columns=cols)
    merged = pd.concat(parts, ignore_index=True)
    return (
        merged.sort_values("month_start")
        .drop_duplicates(subset=["month_start", "province", "vehicle_type"], keep="last")
        .reset_index(drop=True)
    )


def package_to_frames(package: dict) -> dict[str, pd.DataFrame]:
    """Extract dashboard-ready frames from a full API package payload."""
    pub_date = pd.to_datetime(package.get("PubDate"), errors="coerce")
    pub_year = pub_date.year if pd.notna(pub_date) else None

    sheets = {s["SheetName"]: s for s in package.get("SheetList", [])}

    def _find(*parts: str) -> dict:
        parts_l = [p.lower() for p in parts]
        for name, sheet in sheets.items():
            if all(p in name.lower() for p in parts_l):
                return sheet
        raise KeyError(f"Sheet not found: {parts}")

    weekly = parse_weekly_summary(_find("weekly", "summary"))
    stocks = parse_shandong_stocks(_find("stocks"))
    processing = parse_processing_ownership(
        _find("processing", "ownership"),
        pub_year=pub_year,
    )
    operating_rate = parse_operating_rate_ownership(
        _find("operating", "ownership"),
        pub_year=pub_year,
    )
    crude_output = parse_crude_output(_find("crude", "output"))
    ws = _find("weekly", "summary")
    ms = _find("monthly", "summary")
    petchem_weekly = parse_weekly_chemical(ws)
    petchem_monthly = parse_monthly_petchem(ms, calendar_year=pub_year or pd.Timestamp.today().year)
    petchem_inventories = parse_monthly_inventories(
        ms, calendar_year=pub_year or pd.Timestamp.today().year
    )

    meta = pd.DataFrame(
        [
            {
                "package_name": package.get("DataPackageName"),
                "pub_date": pub_date,
                "fetched_at": pd.Timestamp.utcnow(),
            }
        ]
    )

    return {
        "meta": meta,
        "weekly": weekly,
        "runs_wide": build_runs_wide(weekly),
        "products_stocks_wide": build_products_stocks_wide(weekly),
        "shandong_stocks": stocks,
        "shandong_stocks_total": shandong_stocks_totals(stocks),
        "processing_ownership": processing,
        "operating_rate_ownership": operating_rate,
        "crude_output": crude_output,
        "petchem_weekly": petchem_weekly,
        "petchem_monthly": petchem_monthly,
        "petchem_inventories": petchem_inventories,
    }
