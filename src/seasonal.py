"""Seasonal month-of-year charts (year-over-year lines)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.transforms import CONSUMPTION_PRODUCT_ORDER, JET_FUEL_PRODUCT_ORDER
from src.units import (
    CONSUMPTION_PRODUCT_MAP,
    barrels_per_mt,
    convert_volume_series,
    volume_label,
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Muted historical years; current year is highlighted separately.
SEASONAL_YEAR_COLORS: dict[int, str] = {
    2021: "#6b7280",
    2022: "#8b949e",
    2023: "#58a6ff",
    2024: "#3fb950",
    2025: "#bc8cff",
    2026: "#f85149",
    2027: "#ffa657",
}

CURRENT_YEAR_COLOR = "#ff4444"
CURRENT_YEAR_WIDTH = 3.0
HIST_YEAR_WIDTH = 1.6


def weekly_to_monthly_mean(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    unit: str,
) -> pd.DataFrame:
    """
    Collapse weekly observations to one point per calendar month (mean weekly level).

    Using the within-month mean of weekly readings keeps partial months (e.g. current
    year) representative without summing fractional weeks twice.
    """
    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month

    monthly = out.groupby(["year", "month"], as_index=False)[value_col].mean()
    monthly["value"] = convert_volume_series(
        monthly[value_col], unit, freq="weekly", metric=value_col
    )
    return monthly.sort_values(["year", "month"])


def build_runs_seasonal(runs_wide: pd.DataFrame, breakdown: str, unit: str) -> pd.DataFrame:
    sub = runs_wide[runs_wide["breakdown"] == breakdown]
    return weekly_to_monthly_mean(sub, "week_start", "processing_kt", unit)


def build_product_seasonal(products_wide: pd.DataFrame, product_col: str, unit: str) -> pd.DataFrame:
    if product_col not in products_wide.columns:
        return pd.DataFrame(columns=["year", "month", "value"])
    return weekly_to_monthly_mean(products_wide, "week_start", product_col, unit)


def fig_seasonal_lines(
    monthly: pd.DataFrame,
    title: str,
    unit: str,
    *,
    current_year: int | None = None,
) -> go.Figure:
    """Year-over-year lines on a Jan–Dec axis (style matches desk seasonal charts)."""
    fig = go.Figure()
    if monthly.empty:
        return fig.update_layout(title=title)

    if current_year is None:
        current_year = int(monthly["year"].max())

    ylab = volume_label(unit)
    years = sorted(monthly["year"].unique())

    for yr in years:
        sub = monthly[monthly["year"] == yr].sort_values("month")
        is_current = yr == current_year
        color = CURRENT_YEAR_COLOR if is_current else SEASONAL_YEAR_COLORS.get(yr, "#8b949e")

        fig.add_trace(
            go.Scatter(
                x=sub["month"],
                y=sub["value"],
                name=str(yr),
                mode="lines+markers",
                line=dict(color=color, width=CURRENT_YEAR_WIDTH if is_current else HIST_YEAR_WIDTH),
                marker=dict(size=6 if is_current else 4, color=color),
                customdata=[MONTH_LABELS[int(m) - 1] for m in sub["month"]],
                hovertemplate=f"{yr}<br>%{{customdata}}<br>%{{y:,.1f}} {ylab}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=MONTH_LABELS,
            title="",
        ),
        yaxis_title=ylab,
        hovermode="x unified",
        legend=dict(title="Year", orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def apply_consumption_unit(consumption_national: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Add display `value` column from monthly national consumption (kt/month)."""
    if consumption_national.empty:
        return pd.DataFrame(columns=["product", "year", "month", "value"])

    out = consumption_national.copy()
    out["month_start"] = pd.to_datetime(
        {"year": out["year"], "month": out["month"], "day": 1}
    )
    if unit == "kbd":
        prod_keys = out["product"].map(lambda p: CONSUMPTION_PRODUCT_MAP.get(p, p))
        days = out["month_start"].dt.days_in_month
        out["value"] = out["consumption_kt"] / days * prod_keys.map(barrels_per_mt)
    else:
        out["value"] = out["consumption_kt"]
    return out


def consumption_by_product(consumption_national: pd.DataFrame, unit: str) -> dict[str, pd.DataFrame]:
    """Split national consumption into per-product seasonal frames."""
    df = apply_consumption_unit(consumption_national, unit)
    products = [p for p in CONSUMPTION_PRODUCT_ORDER if p in df["product"].unique()]
    extras = sorted(set(df["product"].unique()) - set(products))
    products.extend(extras)

    out: dict[str, pd.DataFrame] = {}
    for product in products:
        sub = df[df["product"] == product][["year", "month", "value"]].copy()
        if not sub.empty:
            out[product] = sub.sort_values(["year", "month"])
    return out


def _add_seasonal_traces(
    fig: go.Figure,
    monthly: pd.DataFrame,
    *,
    ylab: str,
    legend_added: set[str],
    row: int,
    col: int,
    current_year: int,
) -> None:
    years = sorted(monthly["year"].unique())
    for yr in years:
        sub = monthly[monthly["year"] == yr].sort_values("month")
        is_current = yr == current_year
        color = CURRENT_YEAR_COLOR if is_current else SEASONAL_YEAR_COLORS.get(int(yr), "#8b949e")
        key = str(yr)
        fig.add_trace(
            go.Scatter(
                x=sub["month"],
                y=sub["value"],
                name=key,
                legendgroup=key,
                showlegend=key not in legend_added,
                mode="lines+markers",
                line=dict(color=color, width=CURRENT_YEAR_WIDTH if is_current else HIST_YEAR_WIDTH),
                marker=dict(size=5 if is_current else 3, color=color),
                customdata=[MONTH_LABELS[int(m) - 1] for m in sub["month"]],
                hovertemplate=f"{yr}<br>%{{customdata}}<br>%{{y:,.1f}} {ylab}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        legend_added.add(key)


def fig_consumption_seasonal_grid(
    consumption_national: pd.DataFrame,
    unit: str,
    kerosene_refueling_national: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Small-multiples grid: one panel per product, year-over-year seasonal lines.

    Includes product consumption plus jet fuel refueling (international, domestic, total).
    """
    by_product = consumption_by_product(consumption_national, unit)

    if kerosene_refueling_national is not None and not kerosene_refueling_national.empty:
        jet = apply_consumption_unit(kerosene_refueling_national, unit)
        for product in JET_FUEL_PRODUCT_ORDER:
            sub = jet[jet["product"] == product][["year", "month", "value"]]
            if not sub.empty:
                by_product[product] = sub.sort_values(["year", "month"])

    products = [p for p in CONSUMPTION_PRODUCT_ORDER if p in by_product]
    products.extend(p for p in JET_FUEL_PRODUCT_ORDER if p in by_product)
    if not products:
        return go.Figure().update_layout(title="No consumption data available")

    frames_to_combine = [consumption_national]
    if kerosene_refueling_national is not None and not kerosene_refueling_national.empty:
        frames_to_combine.append(kerosene_refueling_national)
    combined = pd.concat(frames_to_combine, ignore_index=True)
    current_year = int(combined["year"].max()) if not combined.empty else pd.Timestamp.today().year
    ylab = volume_label(unit)
    n = len(products)
    ncols = 2
    nrows = (n + 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=products,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    legend_added: set[str] = set()
    for i, product in enumerate(products):
        r, c = i // ncols + 1, i % ncols + 1
        _add_seasonal_traces(
            fig,
            by_product[product],
            ylab=ylab,
            legend_added=legend_added,
            row=r,
            col=c,
            current_year=current_year,
        )

    for idx in range(1, nrows * ncols + 1):
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=MONTH_LABELS,
            row=(idx - 1) // ncols + 1,
            col=(idx - 1) % ncols + 1,
        )
        fig.update_yaxes(
            title_text=ylab if (idx - 1) % ncols == 0 else "",
            row=(idx - 1) // ncols + 1,
            col=(idx - 1) % ncols + 1,
        )

    fig.update_layout(
        title="Seasonality by year — national consumption & jet fuel refueling",
        height=320 * nrows,
        legend=dict(title="Year", orientation="h", yanchor="top", y=-0.08, x=0),
        hovermode="x unified",
    )
    return fig
