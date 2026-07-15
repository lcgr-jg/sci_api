"""Petchem production, operating rates, and inventory charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.filters import filter_by_date_range
from src.seasonal import (
    CURRENT_YEAR_COLOR,
    CURRENT_YEAR_WIDTH,
    HIST_YEAR_WIDTH,
    MONTH_LABELS,
    SEASONAL_YEAR_COLORS,
    weekly_to_monthly_mean,
)
from src.transforms import PETCHEM_PRODUCT_ORDER
from src.units import PETCHEM_PRODUCT_MAP, convert_volume_series, volume_label


def _petchem_product_key(product: str) -> str:
    return PETCHEM_PRODUCT_MAP.get(product, product.lower())


def _weekly_output(petchem_weekly: pd.DataFrame, product: str) -> pd.DataFrame:
    if petchem_weekly.empty:
        return pd.DataFrame(columns=["week_start", "value"])
    sub = petchem_weekly[
        (petchem_weekly["product"] == product) & (petchem_weekly["metric"] == "output_kt")
    ]
    return sub[["week_start", "value"]].sort_values("week_start")


def _weekly_rate(petchem_weekly: pd.DataFrame, product: str) -> pd.DataFrame:
    if petchem_weekly.empty:
        return pd.DataFrame(columns=["week_start", "value"])
    sub = petchem_weekly[
        (petchem_weekly["product"] == product) & (petchem_weekly["metric"] == "operating_rate")
    ]
    return sub[["week_start", "value"]].sort_values("week_start")


def _monthly_output(petchem_monthly: pd.DataFrame, product: str) -> pd.DataFrame:
    if petchem_monthly.empty:
        return pd.DataFrame(columns=["month_start", "value"])
    sub = petchem_monthly[
        (petchem_monthly["product"] == product) & (petchem_monthly["metric"] == "output_kt")
    ]
    return sub[["month_start", "value"]].sort_values("month_start")


def _monthly_rate(petchem_monthly: pd.DataFrame, product: str) -> pd.DataFrame:
    if petchem_monthly.empty:
        return pd.DataFrame(columns=["month_start", "value"])
    sub = petchem_monthly[
        (petchem_monthly["product"] == product) & (petchem_monthly["metric"] == "operating_rate")
    ]
    return sub[["month_start", "value"]].sort_values("month_start")


def build_petchem_seasonal_frame(
    petchem_weekly: pd.DataFrame,
    petchem_monthly: pd.DataFrame,
    product: str,
    unit: str,
) -> pd.DataFrame:
    """
    Monthly seasonal frame for one product.

    Prefer weekly output collapsed to monthly mean; fall back to monthly summary
    (needed for Ethylene, which is not in the weekly chemical output block).
    """
    weekly = _weekly_output(petchem_weekly, product)
    if not weekly.empty:
        return weekly_to_monthly_mean(
            weekly.rename(columns={"value": "output_kt"}),
            "week_start",
            "output_kt",
            unit,
        ).rename(columns={"value": "value"})

    monthly = _monthly_output(petchem_monthly, product)
    if monthly.empty:
        return pd.DataFrame(columns=["year", "month", "value"])

    out = monthly.copy()
    out["month_start"] = pd.to_datetime(out["month_start"], errors="coerce")
    out["year"] = out["month_start"].dt.year
    out["month"] = out["month_start"].dt.month
    out["value"] = convert_volume_series(
        out["value"],
        unit,
        freq="monthly",
        dates=out["month_start"],
        product=_petchem_product_key(product),
    )
    return out[["year", "month", "value"]].sort_values(["year", "month"])


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


def fig_petchem_seasonal_grid(
    petchem_weekly: pd.DataFrame,
    petchem_monthly: pd.DataFrame,
    unit: str,
) -> go.Figure:
    """Small-multiples seasonal grid for the seven core petchem products."""
    by_product = {
        product: build_petchem_seasonal_frame(petchem_weekly, petchem_monthly, product, unit)
        for product in PETCHEM_PRODUCT_ORDER
    }
    products = [p for p in PETCHEM_PRODUCT_ORDER if not by_product[p].empty]
    if not products:
        return go.Figure().update_layout(title="No petchem output data available")

    combined = pd.concat(
        [df.assign(product=p) for p, df in by_product.items() if not df.empty],
        ignore_index=True,
    )
    current_year = int(combined["year"].max()) if not combined.empty else pd.Timestamp.today().year
    ylab = volume_label(unit)
    ncols = 2
    nrows = (len(products) + 1) // ncols

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
        title="Seasonality by year — national petchem output",
        height=320 * nrows,
        legend=dict(title="Year", orientation="h", yanchor="top", y=-0.08, x=0),
        hovermode="x unified",
    )
    return fig


def fig_petchem_timeseries(
    petchem_weekly: pd.DataFrame,
    petchem_monthly: pd.DataFrame,
    product: str,
    unit: str,
    date_range: str,
) -> go.Figure:
    """Weekly output vs operating rate for the selected petchem product."""
    prod_key = _petchem_product_key(product)
    ylab = volume_label(unit)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    weekly_out = filter_by_date_range(_weekly_output(petchem_weekly, product), "week_start", date_range)
    if not weekly_out.empty:
        y_out = convert_volume_series(
            weekly_out["value"],
            unit,
            freq="weekly",
            product=prod_key,
        )
        fig.add_trace(
            go.Scatter(
                x=weekly_out["week_start"],
                y=y_out,
                name=f"{product} output",
                mode="lines",
                line=dict(color="#58a6ff", width=2),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:,.1f}} {ylab}<extra>output</extra>",
            ),
            secondary_y=False,
        )
    else:
        monthly_out = filter_by_date_range(
            _monthly_output(petchem_monthly, product),
            "month_start",
            date_range,
        )
        if not monthly_out.empty:
            y_out = convert_volume_series(
                monthly_out["value"],
                unit,
                freq="monthly",
                dates=monthly_out["month_start"],
                product=prod_key,
            )
            fig.add_trace(
                go.Scatter(
                    x=monthly_out["month_start"],
                    y=y_out,
                    name=f"{product} output (monthly)",
                    mode="lines+markers",
                    line=dict(color="#58a6ff", width=2),
                    hovertemplate=f"%{{x|%Y-%m}}<br>%{{y:,.1f}} {ylab}<extra>output</extra>",
                ),
                secondary_y=False,
            )

    weekly_rate = filter_by_date_range(_weekly_rate(petchem_weekly, product), "week_start", date_range)
    if not weekly_rate.empty:
        fig.add_trace(
            go.Scatter(
                x=weekly_rate["week_start"],
                y=weekly_rate["value"] * 100,
                name=f"{product} operating rate",
                mode="lines",
                line=dict(color="#e3b341", width=2, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>rate</extra>",
            ),
            secondary_y=True,
        )
    else:
        monthly_rate = filter_by_date_range(
            _monthly_rate(petchem_monthly, product),
            "month_start",
            date_range,
        )
        if not monthly_rate.empty:
            fig.add_trace(
                go.Scatter(
                    x=monthly_rate["month_start"],
                    y=monthly_rate["value"] * 100,
                    name=f"{product} operating rate (monthly)",
                    mode="lines+markers",
                    line=dict(color="#e3b341", width=2, dash="dot"),
                    hovertemplate="%{x|%Y-%m}<br>%{y:.1f}%<extra>rate</extra>",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=f"{product} — output ({ylab}) vs unit operating rate",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=f"Output ({ylab})", secondary_y=False)
    fig.update_yaxes(title_text="Operating rate (%)", secondary_y=True)
    return fig


def fig_petchem_inventories(
    petchem_inventories: pd.DataFrame,
    date_range: str,
    inventory_view: str = "all",
) -> go.Figure:
    """
    Monthly inventory trends.

    ``inventory_view``: ``port`` | ``refinery`` | ``all``
    """
    inv = filter_by_date_range(petchem_inventories, "month_start", date_range)
    if inv.empty:
        return go.Figure().update_layout(title="No petchem inventory data available")

    port_colors = {
        "Benzene": "#58a6ff",
        "Toluene": "#3fb950",
        "Xylene": "#bc8cff",
    }

    if inventory_view == "all":
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Port stocks — aromatics (kt)", "Independent refinery naphtha (kt)"),
            vertical_spacing=0.10,
        )
        port = inv[inv["inventory_type"] == "port"]
        for product, color in port_colors.items():
            sub = port[port["product"] == product].sort_values("month_start")
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["month_start"],
                    y=sub["value_kt"],
                    name=product,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    hovertemplate="%{x|%Y-%m}<br>%{y:,.0f} kt<extra>" + product + "</extra>",
                ),
                row=1,
                col=1,
            )

        naphtha = inv[inv["inventory_type"] == "refinery"].sort_values("month_start")
        if not naphtha.empty:
            fig.add_trace(
                go.Scatter(
                    x=naphtha["month_start"],
                    y=naphtha["value_kt"],
                    name="Naphtha",
                    mode="lines+markers",
                    line=dict(color="#ffa657", width=2),
                    hovertemplate="%{x|%Y-%m}<br>%{y:,.0f} kt<extra>naphtha</extra>",
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="kt", row=1, col=1)
        fig.update_yaxes(title_text="kt", row=2, col=1)
        fig.update_layout(title="Petchem inventories", hovermode="x unified", height=620)
        return fig

    fig = go.Figure()
    if inventory_view == "port":
        port = inv[inv["inventory_type"] == "port"]
        for product, color in port_colors.items():
            sub = port[port["product"] == product].sort_values("month_start")
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["month_start"],
                    y=sub["value_kt"],
                    name=product,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                )
            )
        title = "Port stocks — Benzene, Toluene, Xylene (kt)"
    else:
        naphtha = inv[inv["inventory_type"] == "refinery"].sort_values("month_start")
        if not naphtha.empty:
            fig.add_trace(
                go.Scatter(
                    x=naphtha["month_start"],
                    y=naphtha["value_kt"],
                    name="Naphtha",
                    mode="lines+markers",
                    line=dict(color="#ffa657", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(255,166,87,0.12)",
                )
            )
        title = "Independent refinery naphtha stocks (kt)"

    fig.update_layout(title=title, xaxis_title="Month", yaxis_title="kt", hovermode="x unified")
    return fig
