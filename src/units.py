"""Volume unit conversions for SCI China data (metric tons ↔ kb/d)."""

from __future__ import annotations

import pandas as pd

# Barrels per metric ton by product (desk conversion table).
# Light condensate / crude fallback when product is unknown.
DEFAULT_BARRELS_PER_MT = 7.33

BARRELS_PER_MT_BY_PRODUCT: dict[str, float] = {
    "propane": 12.4,
    "butane": 11.55,
    "naphtha": 8.9,
    "gasoline": 8.33,
    "kerosene": 7.88,
    "gasoil": 7.45,
    "diesel": 7.45,
    "vgo": 6.70,
    "fuel_oil": 6.33,
    "residue": 6.33,
    "crude": DEFAULT_BARRELS_PER_MT,
    "crude_oil": DEFAULT_BARRELS_PER_MT,
    "jet_fuel": 7.88,
    "benzene": 8.50,
    "toluene": 8.30,
    "xylene": 8.20,
    "px": 8.50,
    "ethylene": 11.00,
    "propylene": 11.50,
    "butadiene": 11.00,
}

# Dashboard column / metric names → product key for conversion.
METRIC_TO_PRODUCT: dict[str, str | None] = {
    "processing_kt": "crude",
    "gasoline_output_kt": "gasoline",
    "diesel_output_kt": "gasoil",
    "kerosene_output_kt": "kerosene",
    "gasoline_stocks_kt": "gasoline",
    "diesel_stocks_kt": "gasoil",
}

CONSUMPTION_PRODUCT_MAP: dict[str, str] = {
    "Gasoline": "gasoline",
    "Diesel": "gasoil",
    "Kerosene": "kerosene",
    "Naphtha": "naphtha",
    "VGO": "vgo",
    "Residue": "fuel_oil",
    "Jet fuel (total)": "kerosene",
    "Jet fuel (international)": "kerosene",
    "Jet fuel (domestic)": "kerosene",
}

SHANDONG_PRODUCT_MAP: dict[str, str] = {
    "crude_oil": "crude",
    "gasoline": "gasoline",
    "diesel": "gasoil",
    "naphtha": "naphtha",
}

PETCHEM_PRODUCT_MAP: dict[str, str] = {
    "PX": "px",
    "Benzene": "benzene",
    "Toluene": "toluene",
    "Xylene": "xylene",
    "Propylene": "propylene",
    "Butadiene": "butadiene",
    "Ethylene": "ethylene",
    "Naphtha": "naphtha",
}


def normalize_product_key(product: str | None) -> str | None:
    if product is None:
        return None
    key = product.lower().strip().replace(" ", "_").replace("-", "_")
    return key


def barrels_per_mt(product: str | None = None) -> float:
    """Return barrels/mt for a product; crude/unknown uses light condensate factor."""
    key = normalize_product_key(product)
    if not key:
        return DEFAULT_BARRELS_PER_MT
    return BARRELS_PER_MT_BY_PRODUCT.get(key, DEFAULT_BARRELS_PER_MT)


def product_for_metric(metric: str | None) -> str | None:
    if not metric:
        return None
    if metric in METRIC_TO_PRODUCT:
        return METRIC_TO_PRODUCT[metric]
    for token, prod in METRIC_TO_PRODUCT.items():
        if token in metric:
            return prod
    return None


def weekly_kt_to_kbd(weekly_kt: pd.Series, product: str | None = None) -> pd.Series:
    """Convert weekly throughput (kt/week) to average kb/d."""
    factor = barrels_per_mt(product)
    return weekly_kt / 7.0 * factor


def monthly_kt_to_kbd(
    monthly_kt: pd.Series,
    month_starts: pd.Series,
    product: str | None = None,
) -> pd.Series:
    """Convert monthly throughput (kt/month) to average kb/d."""
    factor = barrels_per_mt(product)
    days = pd.to_datetime(month_starts, errors="coerce").dt.days_in_month
    return monthly_kt / days * factor


def volume_label(unit: str) -> str:
    return "kb/d" if unit == "kbd" else "kt"


def convert_volume_series(
    values: pd.Series,
    unit: str,
    *,
    freq: str = "weekly",
    dates: pd.Series | None = None,
    product: str | None = None,
    metric: str | None = None,
) -> pd.Series:
    """Convert kt series to display unit. Operating rates pass through unchanged."""
    if unit != "kbd":
        return values
    prod = product or product_for_metric(metric)
    if freq == "weekly":
        return weekly_kt_to_kbd(values, prod)
    if freq == "monthly" and dates is not None:
        return monthly_kt_to_kbd(values, pd.to_datetime(dates), prod)
    return values
