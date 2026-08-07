"""
China Runs & Stocks Dashboard
=============================
Run:  python app.py
      python app.py --refresh --history

Tabs
----
0  Crude Runs     – weekly/monthly processing & operating rates
1  Processing & Operating rate – provincial CDU volume/rate by owner vs refinery map
2  Products       – gasoline/diesel output + national commercial stocks + weekly change table
3  Shandong       – independent refinery city stocks
4  Economics      – refinery margins vs runs
5  Seasonal         – year-over-year crude runs & product output by month
6  Consumption      – national demand seasonality, MoM/YoY, mobility vs gasoline/diesel
7  Petchem          – chemical output seasonality, weekly trends, inventories + MoM/YoY table
8  Monthly Detail – provincial processing by ownership
9  Regional       – geographic runs, output & consumption comparison
10 Exports        – refined product shipping, quotas, forward plan tracker & monthly export table
11 Crude Imports  – foreign/domestic crude shipping, fleet/grades & independent import quotas
12 Oil Grade      – Shandong independent feedstock slate by grade / origin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from dash import dash_table
from plotly.subplots import make_subplots

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fetch_data import fetch_and_cache, load_cached_frames
from src.transforms import (
    CONSUMPTION_PRODUCT_ORDER,
    JET_FUEL_PRODUCT_ORDER,
    PETCHEM_PRODUCT_ORDER,
    REGIONAL_OUTPUT_PRODUCTS,
)
from src.regional import (
    aggregate_consumption,
    aggregate_output,
    aggregate_runs,
    fig_regional_balance_gap,
    fig_regional_compare_bars,
    fig_regional_map,
    fig_regional_timeseries,
    fig_top_provinces,
    province_region_map,
)
from src.filters import DATE_RANGE_OPTIONS, filter_by_date_range
from src.infra_kmz import load_infra_sites
from src.ownership_geo import (
    SCI_OWNERSHIP_ORDER,
    fig_ownership_heatmap,
    fig_ownership_infra_map,
    fig_ownership_province_bars,
    fig_ownership_timeseries,
)
from src.seasonal import (
    build_product_seasonal,
    build_runs_seasonal,
    fig_consumption_seasonal_grid,
    fig_seasonal_lines,
)
from src.petchem import (
    fig_petchem_inventories,
    fig_petchem_seasonal_grid,
    fig_petchem_timeseries,
)
from src.exports import (
    export_forecast_summary_records,
    export_forecast_table_records,
    export_monthly_table_records,
    fig_export_destinations,
    fig_export_forecast_plan,
    fig_export_monthly_stacked,
    fig_export_provinces,
    fig_export_quota_summary,
    fig_export_quotas_monthly,
    fig_export_sankey,
)
from src.transforms import EXPORT_TABLE_PRODUCTS
from src.units import convert_volume_series, volume_label
from src.change_tables import (
    change_table_columns,
    consumption_change_records,
    fig_products_delta_bars,
    petchem_change_records,
    products_scorecard_records,
    products_scorecard_style,
)
from src.crude_shipping import (
    fig_crude_discharge_lag_trend,
    fig_crude_fleet_mix,
    fig_crude_fleet_mix_trend,
    fig_crude_grade_mix_trend,
    fig_crude_grades,
    fig_crude_monthly_arrivals,
    fig_crude_origin_mix_trend,
    fig_crude_origins,
    fig_crude_province_mix_trend,
    fig_crude_provinces,
    fig_crude_quota_summary,
    fig_crude_quotas_monthly,
    fig_crude_transit_trend,
    origin_transit_compare_columns,
    origin_transit_compare_records,
    origin_transit_delta_style,
)
from src.oil_grade import (
    fig_oil_grade_city_bars,
    fig_oil_grade_origin_mix,
    fig_oil_grade_top_grades,
    fig_oil_grade_totals,
)
from src.mobility import (
    fig_diesel_vs_trucks,
    fig_gasoline_vs_nev,
    fig_passenger_possession,
    fig_passenger_sales_mix,
    fig_truck_possession_mix,
    fig_truck_sales_mix,
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

DARK_TPL = go.layout.Template(
    layout=dict(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", size=11),
        xaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
        legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1),
        colorway=["#58a6ff", "#3fb950", "#f85149", "#e3b341", "#bc8cff", "#ffa657"],
    )
)

BREAKDOWN_COLORS = {
    "National": "#58a6ff",
    "SOEs": "#3fb950",
    "Independents": "#f85149",
}

PRODUCT_COLORS = {
    "gasoline_output_kt": "#58a6ff",
    "diesel_output_kt": "#f85149",
    "kerosene_output_kt": "#bc8cff",
    "gasoline_stocks_kt": "#39d353",
    "diesel_stocks_kt": "#ffa657",
}

MARGIN_COLORS = {
    "gross": "#e3b341",
    "net": "#bc8cff",
    "contribution": "#ffa657",
}


def _apply_dark(
    fig: go.Figure,
    *,
    height: int | None = None,
    margin: dict | None = None,
) -> go.Figure:
    m = dict(l=50, r=60, t=50, b=40)
    if margin:
        m.update(margin)
    # Keep taller top margin when charts reserve space for an external-title + top legend.
    existing_t = fig.layout.margin.t if fig.layout.margin else None
    if existing_t is not None and existing_t > m["t"]:
        m["t"] = existing_t
    fig.update_layout(template=DARK_TPL, margin=m)
    if height:
        fig.update_layout(height=height)
    return fig


def _dropdown(**kwargs) -> dbc.Select:
    """Bootstrap select styled for the dark dashboard theme."""
    props = {
        "className": "dashboard-dropdown bg-dark text-light border-secondary",
        "style": {
            "backgroundColor": "#21262d",
            "color": "#e6edf3",
            "borderColor": "#30363d",
        },
    }
    props.update(kwargs)
    # dcc.Dropdown-only props that dbc.Select does not support
    props.pop("searchable", None)
    props.pop("optionHeight", None)
    props.pop("maxHeight", None)
    return dbc.Select(**props)


# Shared dark DataTable chrome (matches Exports monthly table).
_TABLE_STYLE_HEADER = {
    "backgroundColor": "#21262d",
    "color": "#e6edf3",
    "fontWeight": "600",
    "border": "1px solid #30363d",
}
_TABLE_STYLE_CELL = {
    "backgroundColor": "#161b22",
    "color": "#e6edf3",
    "border": "1px solid #30363d",
    "textAlign": "right",
    "padding": "6px 10px",
    "fontSize": "0.85rem",
}


def _dark_datatable(**kwargs) -> dash_table.DataTable:
    props = {
        "style_table": {"overflowX": "auto"},
        "style_header": _TABLE_STYLE_HEADER,
        "style_cell": _TABLE_STYLE_CELL,
        "style_as_list_view": True,
        "page_size": 20,
    }
    props.update(kwargs)
    return dash_table.DataTable(**props)


def _vol(
    values: pd.Series,
    unit: str,
    freq: str = "weekly",
    dates: pd.Series | None = None,
    metric: str | None = None,
) -> pd.Series:
    return convert_volume_series(values, unit, freq=freq, dates=dates, metric=metric)


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------


def _load_data(refresh: bool, history: bool) -> dict[str, pd.DataFrame]:
    if refresh:
        fetch_and_cache(include_history=history)
    frames = load_cached_frames()
    if frames is None or "margins_wide" not in frames or "consumption_national" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "kerosene_refueling_national" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "operating_rate_ownership" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "provincial_processing" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "petchem_weekly" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "export_shipping" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "crude_shipping" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "processing_oil_grade" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None or "vehicle_sales_passenger" not in frames:
        fetch_and_cache(include_history=history)
        frames = load_cached_frames()
    if frames is None:
        raise RuntimeError("No cached data and API fetch failed.")
    return frames


def _history_span(frames: dict[str, pd.DataFrame]) -> str:
    w = frames.get("weekly")
    if w is None or w.empty:
        return ""
    return f"{w['week_start'].min():%Y-%m-%d} to {w['week_start'].max():%Y-%m-%d}"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def fig_runs_volume(runs: pd.DataFrame, unit: str, date_range: str) -> go.Figure:
    runs = filter_by_date_range(runs, "week_start", date_range)
    ylab = volume_label(unit)
    fig = go.Figure()
    for bd, color in BREAKDOWN_COLORS.items():
        sub = runs[runs["breakdown"] == bd].sort_values("week_start")
        if sub.empty:
            continue
        y = _vol(sub["processing_kt"], unit, metric="processing_kt")
        fig.add_trace(
            go.Scatter(
                x=sub["week_start"],
                y=y,
                name=bd,
                mode="lines+markers",
                line=dict(color=color, width=2),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:,.1f}} {ylab}<extra>{bd}</extra>",
            )
        )
    fig.update_layout(
        title=f"Weekly feedstock processing ({ylab})",
        xaxis_title="Week",
        yaxis_title=ylab,
        hovermode="x unified",
    )
    return _apply_dark(fig)


def fig_runs_rate(runs: pd.DataFrame, date_range: str) -> go.Figure:
    runs = filter_by_date_range(runs, "week_start", date_range)
    fig = go.Figure()
    for bd, color in BREAKDOWN_COLORS.items():
        sub = runs[runs["breakdown"] == bd].sort_values("week_start")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["week_start"],
                y=sub["operating_rate"] * 100,
                name=bd,
                mode="lines+markers",
                line=dict(color=color, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>" + bd + "</extra>",
            )
        )
    fig.update_layout(title="CDU operating rate (%)", xaxis_title="Week", yaxis_title="%", hovermode="x unified")
    return _apply_dark(fig)


def fig_balance_monthly(balance: pd.DataFrame, unit: str, date_range: str) -> go.Figure:
    bal = filter_by_date_range(balance, "month_start", date_range)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "crude_processing_kt" in bal.columns:
        y = _vol(bal["crude_processing_kt"], unit, freq="monthly", dates=bal["month_start"], metric="processing_kt")
        fig.add_trace(
            go.Bar(x=bal["month_start"], y=y, name=f"Processing ({volume_label(unit)})", opacity=0.75),
            secondary_y=False,
        )
    if "cdu_operating_rate" in bal.columns:
        fig.add_trace(
            go.Scatter(
                x=bal["month_start"],
                y=bal["cdu_operating_rate"] * 100,
                name="CDU rate (%)",
                mode="lines+markers",
                line=dict(color="#e3b341", width=2),
            ),
            secondary_y=True,
        )
    fig.update_layout(title="Monthly national balance (SCI)", hovermode="x unified")
    fig.update_yaxes(title_text=volume_label(unit), secondary_y=False)
    fig.update_yaxes(title_text="Operating rate (%)", secondary_y=True)
    return _apply_dark(fig)


def fig_soe_indep_spread(runs: pd.DataFrame, date_range: str) -> go.Figure:
    runs = filter_by_date_range(runs, "week_start", date_range)
    soe = runs[runs["breakdown"] == "SOEs"].set_index("week_start")["operating_rate"]
    ind = runs[runs["breakdown"] == "Independents"].set_index("week_start")["operating_rate"]
    spread = (soe - ind).dropna() * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=spread.index,
            y=spread.values,
            marker_color=["#3fb950" if v >= 0 else "#f85149" for v in spread.values],
            name="SOE - Independent (pp)",
        )
    )
    fig.update_layout(title="SOE vs Independent operating rate spread", xaxis_title="Week", yaxis_title="pp")
    return _apply_dark(fig)


def fig_products(products: pd.DataFrame, unit: str, date_range: str) -> go.Figure:
    products = filter_by_date_range(products, "week_start", date_range)
    ylab = volume_label(unit)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df = products.sort_values("week_start")

    for col, color in [
        ("gasoline_output_kt", PRODUCT_COLORS["gasoline_output_kt"]),
        ("diesel_output_kt", PRODUCT_COLORS["diesel_output_kt"]),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["week_start"],
                    y=_vol(df[col], unit, metric=col),
                    name=col.replace("_kt", "").replace("_", " ").title(),
                    mode="lines",
                    line=dict(color=color, width=2),
                ),
                secondary_y=False,
            )

    for col, color in [
        ("gasoline_stocks_kt", PRODUCT_COLORS["gasoline_stocks_kt"]),
        ("diesel_stocks_kt", PRODUCT_COLORS["diesel_stocks_kt"]),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["week_start"],
                    y=df[col],
                    name=col.replace("_kt", "").replace("_", " ").title(),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=f"National output ({ylab}, solid) vs commercial stocks (kt, dotted)",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=f"Weekly output ({ylab})", secondary_y=False)
    fig.update_yaxes(title_text="Commercial stocks (kt)", secondary_y=True)
    return _apply_dark(fig)


def fig_shandong_totals(totals: pd.DataFrame, product: str, date_range: str) -> go.Figure:
    sub = filter_by_date_range(totals[totals["product"] == product], "week_start", date_range)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["week_start"],
            y=sub["value_kt"],
            mode="lines+markers",
            line=dict(width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.12)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} kt<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Shandong independents — total {product.replace('_', ' ')} stocks (kt)",
        xaxis_title="Week",
        yaxis_title="kt",
    )
    return _apply_dark(fig)


def fig_shandong_heatmap(stocks: pd.DataFrame, product: str, date_range: str) -> go.Figure:
    sub = filter_by_date_range(stocks[stocks["product"] == product], "week_start", date_range)
    if sub.empty:
        return _apply_dark(
            go.Figure().update_layout(title="No Shandong stock data"),
            height=420,
        )

    # Monthly mean reduces ~270 weekly columns to ~50 months; easier to read.
    sub = sub.copy()
    sub["month_start"] = sub["week_start"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        sub.groupby(["region", "month_start"], as_index=False)["value_kt"]
        .mean()
    )
    pivot = monthly.pivot(index="region", columns="month_start", values="value_kt")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    z_raw = pivot.values.astype(float)
    z_max = float(np.nanmax(z_raw)) if np.any(~np.isnan(z_raw)) else 1.0
    z_max = max(z_max, 1.0)

    # Plotly renders NaN as the plot background (looks like a black gap). Use -1 → gray.
    z_color = np.where(np.isnan(z_raw), -1.0, z_raw)
    colorscale = [
        [0.0, "#30363d"],
        [0.001, "#dbeafe"],
        [1.0, "#1d4ed8"],
    ]

    custom_hover = np.empty(z_raw.shape, dtype=object)
    for i, region in enumerate(pivot.index):
        for j, month in enumerate(pivot.columns):
            val = z_raw[i, j]
            if np.isnan(val):
                custom_hover[i, j] = f"{region}<br>{pd.Timestamp(month):%Y-%m}<br>No data"
            else:
                custom_hover[i, j] = f"{region}<br>{pd.Timestamp(month):%Y-%m}<br>{val:,.0f} kt"

    fig = go.Figure(
        data=go.Heatmap(
            z=z_color,
            x=[pd.Timestamp(c).strftime("%Y-%m") for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=colorscale,
            zmin=-1,
            zmax=z_max,
            customdata=custom_hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(title="kt"),
        )
    )
    fig.update_layout(
        title=f"Shandong {product.replace('_', ' ')} stocks by city (kt, monthly avg)",
        xaxis=dict(tickangle=-45, type="category"),
    )
    return _apply_dark(fig, height=420)


def fig_monthly_processing(proc: pd.DataFrame, ownership: str, unit: str, date_range: str) -> go.Figure:
    sub = proc[
        (proc["ownership"] == ownership)
        & (proc["unit"] == "FEEDSTOCK PRIMARY PROCESSING UNITS")
    ]
    sub = filter_by_date_range(sub, "month_start", date_range)
    top = sub.groupby("province")["processing_kt"].sum().sort_values(ascending=False).head(8).index.tolist()
    sub = sub[sub["province"].isin(top)].sort_values("month_start")
    ylab = volume_label(unit)

    fig = go.Figure()
    for prov in top:
        p = sub[sub["province"] == prov]
        fig.add_trace(
            go.Scatter(
                x=p["month_start"],
                y=_vol(p["processing_kt"], unit, freq="monthly", dates=p["month_start"], metric="processing_kt"),
                name=prov,
                mode="lines+markers",
            )
        )
    fig.update_layout(
        title=f"Monthly processing by province — {ownership} ({ylab})",
        xaxis_title="Month",
        yaxis_title=ylab,
        hovermode="x unified",
    )
    return _apply_dark(fig)


def _margin_column(
    margins_wide: pd.DataFrame,
    refinery_type: str,
    metric: str,
    display_mode: str = "pct",
) -> str | None:
    prefix = refinery_type.replace(" ", "_").replace("-", "")
    if metric == "contribution":
        suffix = "contribution_margin"
    elif metric == "gross":
        suffix = "gross_profit_margin_pct" if display_mode == "pct" else "profit_per_ton"
    elif metric == "net":
        suffix = "net_profit_margin_pct" if display_mode == "pct" else "net_profit"
    else:
        suffix = ""
    # Fuzzy match because pivot keys vary slightly
    for c in margins_wide.columns:
        if c == "week_start":
            continue
        if prefix.lower() not in c.lower():
            continue
        cl = c.lower()
        if suffix == "gross_profit_margin_pct" and "gross_profit_margin" in cl:
            return c
        if suffix == "net_profit_margin_pct" and "net_profit_margin" in cl:
            return c
        if suffix == "profit_per_ton" and "profit_per_ton" in cl:
            return c
        if suffix == "net_profit" and "net_profit" in cl and "margin" not in cl:
            return c
        if suffix == "contribution_margin" and "contribution_margin" in cl:
            return c
    return None


def _margin_axis_label(margin_metric: str, display_mode: str) -> str:
    if margin_metric == "contribution":
        return "RMB"
    if margin_metric in ("gross", "net") and display_mode == "pct":
        return "%"
    if margin_metric == "gross":
        return "RMB/mt"
    return "RMB"


def _scale_margin_values(values: pd.Series, margin_metric: str, display_mode: str) -> pd.Series:
    if margin_metric in ("gross", "net") and display_mode == "pct":
        return values * 100
    return values


def fig_economics_overlay(
    runs: pd.DataFrame,
    margins_wide: pd.DataFrame,
    unit: str,
    date_range: str,
    refinery_type: str,
    margin_metric: str,
    runs_breakdown: str,
    display_mode: str = "pct",
) -> go.Figure:
    runs = filter_by_date_range(runs, "week_start", date_range)
    margins_wide = filter_by_date_range(margins_wide, "week_start", date_range)
    ylab = volume_label(unit)

    run_sub = runs[runs["breakdown"] == runs_breakdown].sort_values("week_start")
    margin_col = _margin_column(margins_wide, refinery_type, margin_metric, display_mode)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not run_sub.empty:
        fig.add_trace(
            go.Scatter(
                x=run_sub["week_start"],
                y=_vol(run_sub["processing_kt"], unit, metric="processing_kt"),
                name=f"{runs_breakdown} runs",
                line=dict(color="#58a6ff", width=2),
                mode="lines",
            ),
            secondary_y=False,
        )

    if margin_col and margin_col in margins_wide.columns:
        m = margins_wide.sort_values("week_start")
        y_margin = _scale_margin_values(m[margin_col], margin_metric, display_mode)
        fig.add_trace(
            go.Scatter(
                x=m["week_start"],
                y=y_margin,
                name=f"{refinery_type} {margin_metric} margin",
                line=dict(color=MARGIN_COLORS.get(margin_metric, "#e3b341"), width=2, dash="dot"),
                mode="lines",
            ),
            secondary_y=True,
        )

    fig.update_layout(title="Refinery runs vs modelled margins", hovermode="x unified")
    fig.update_yaxes(title_text=f"Processing ({ylab})", secondary_y=False)
    y2 = _margin_axis_label(margin_metric, display_mode)
    fig.update_yaxes(title_text=f"Margin ({y2})", secondary_y=True)
    return _apply_dark(fig)


def fig_economics_scatter(
    runs: pd.DataFrame,
    margins_wide: pd.DataFrame,
    date_range: str,
    refinery_type: str,
    margin_metric: str,
    runs_breakdown: str,
    display_mode: str = "pct",
) -> go.Figure:
    runs = filter_by_date_range(runs, "week_start", date_range)
    margins_wide = filter_by_date_range(margins_wide, "week_start", date_range)
    margin_col = _margin_column(margins_wide, refinery_type, margin_metric, display_mode)

    run_sub = runs[runs["breakdown"] == runs_breakdown][["week_start", "processing_kt"]]
    if margin_col is None or run_sub.empty:
        return _apply_dark(go.Figure().update_layout(title="Insufficient margin data"))

    merged = run_sub.merge(margins_wide[["week_start", margin_col]], on="week_start", how="inner")
    x = _scale_margin_values(merged[margin_col], margin_metric, display_mode)
    xlab = _margin_axis_label(margin_metric, display_mode)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_vol(merged["processing_kt"], "kbd", metric="processing_kt"),
            mode="markers",
            marker=dict(size=8, color="#58a6ff", opacity=0.75),
            text=[d.strftime("%Y-%m-%d") for d in merged["week_start"]],
            hovertemplate="Week: %{text}<br>Margin: %{x:.1f}<br>Runs: %{y:,.0f} kb/d<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{runs_breakdown} runs (kb/d) vs {refinery_type} {margin_metric} margin",
        xaxis_title=f"Margin ({xlab})",
        yaxis_title="Runs (kb/d)",
    )
    return _apply_dark(fig)


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="China Runs & Stocks",
    suppress_callback_exceptions=True,
)
server = app.server

FRAMES: dict[str, pd.DataFrame] = {}
INFRA_SITES = pd.DataFrame()
META = pd.Series(dtype=object)
PUB_LABEL = ""
HISTORY_SPAN = ""


def _init_frames(refresh: bool, history: bool) -> None:
    global FRAMES, INFRA_SITES, META, PUB_LABEL, HISTORY_SPAN
    FRAMES = _load_data(refresh, history)
    INFRA_SITES = load_infra_sites(_HERE / "data" / "China oil infra.kmz")
    META = FRAMES["meta"].iloc[0]
    PUB_LABEL = pd.Timestamp(META["pub_date"]).strftime("%Y-%m-%d") if pd.notna(META["pub_date"]) else "unknown"
    HISTORY_SPAN = _history_span(FRAMES)


def _control_bar() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                _dropdown(
                    id="unit-select",
                    options=[
                        {"label": "Metric tons (kt)", "value": "kt"},
                        {"label": "Barrels per day (kb/d)", "value": "kbd"},
                    ],
                    value="kbd",
                ),
                width=2,
            ),
            dbc.Col(
                _dropdown(
                    id="date-range-select",
                    options=[{"label": v, "value": k} for k, v in DATE_RANGE_OPTIONS.items()],
                    value="6m",
                ),
                width=2,
            ),
            dbc.Col(
                html.Small(f"Weekly history: {HISTORY_SPAN}", className="text-muted"),
                width=4,
                className="pt-2",
            ),
        ],
        className="mb-2 mt-2 g-2",
    )


def _po_province_options() -> list[dict[str, str]]:
    proc = FRAMES.get("processing_ownership", pd.DataFrame())
    if proc.empty:
        return [{"label": "Shandong", "value": "Shandong"}]
    cdu = proc[proc["unit"] == "FEEDSTOCK PRIMARY PROCESSING UNITS"]
    provinces = sorted(cdu["province"].dropna().unique())
    return [{"label": p, "value": p} for p in provinces]


def _oil_grade_city_options() -> list[dict[str, str]]:
    df = FRAMES.get("processing_oil_grade", pd.DataFrame())
    opts = [{"label": "All cities", "value": "All"}]
    if df is None or df.empty or "city" not in df.columns:
        return opts
    cities = sorted(df["city"].dropna().unique())
    return opts + [{"label": c, "value": c} for c in cities]


def _build_layout() -> dbc.Container:
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("SCI China — Runs & Stocks", className="mt-3 mb-0"),
                                html.P(
                                    f"Source: {META['package_name']}  ·  Published {PUB_LABEL}",
                                    className="text-muted small",
                                ),
                            ]
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        dbc.Button("Refresh from API", id="btn-refresh", color="primary", className="mt-3 float-end"),
                        width=3,
                    ),
                ]
            ),
            dbc.Row([dbc.Col(html.Div(id="refresh-status"), width=12)]),
            _control_bar(),
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Crude Runs",
                        children=[
                            dcc.Graph(id="runs-volume-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="runs-rate-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="balance-monthly-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="spread-chart", config={"displayModeBar": True}),
                        ],
                    ),
                    dbc.Tab(
                        label="Processing & Operating rate",
                        children=[
                            html.P(
                                "Refinery footprints from China oil infra.kmz overlaid with "
                                "SCI monthly CDU processing volume or operating rate by province "
                                "and ownership. All KMZ layers shown (pipelines, terminals, etc.).",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="po-metric",
                                            options=[
                                                {"label": "Processing volume", "value": "processing"},
                                                {"label": "Operating rate", "value": "operating_rate"},
                                            ],
                                            value="processing",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="po-ownership",
                                            options=[{"label": "All owners", "value": "All"}]
                                            + [
                                                {"label": o, "value": o}
                                                for o in SCI_OWNERSHIP_ORDER
                                            ],
                                            value="All",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="po-province",
                                            options=_po_province_options(),
                                            value=_po_province_options()[0]["value"]
                                            if _po_province_options()
                                            else "Shandong",
                                        ),
                                        width=2,
                                    ),
                                ],
                                className="mt-2 g-2",
                            ),
                            dcc.Graph(id="po-map-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="po-heatmap-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="po-province-bars-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="po-timeseries-chart", config={"displayModeBar": True}),
                        ],
                    ),
                    dbc.Tab(
                        label="Products & Stocks",
                        children=[
                            dcc.Graph(id="products-chart", config={"displayModeBar": True}),
                            html.H5("Weekly scorecard", className="mt-3 mb-1"),
                            html.P(
                                "Latest week at a glance. Apparent demand ≈ output − inventory build "
                                "(ex-exports). Days cover = stocks ÷ daily production "
                                "(stocks × 7 / weekly output) — not vs demand. "
                                "Kerosene has no national commercial stocks in SCI.",
                                className="text-muted small mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="products-stock-unit",
                                            options=[
                                                {"label": "Stocks: mb", "value": "mb"},
                                                {"label": "Stocks: kt", "value": "kt"},
                                            ],
                                            value="mb",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="products-flow-unit",
                                            options=[
                                                {"label": "Flows: kb/d", "value": "kbd"},
                                                {"label": "Flows: kt", "value": "kt"},
                                            ],
                                            value="kbd",
                                        ),
                                        width=2,
                                    ),
                                ],
                                className="g-2 mb-2",
                            ),
                            _dark_datatable(
                                id="products-change-table",
                                columns=[{"name": "product", "id": "product"}],
                                data=[],
                                style_cell_conditional=[
                                    {"if": {"column_id": "product"}, "textAlign": "left"}
                                ],
                                style_data_conditional=products_scorecard_style(),
                                page_size=10,
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="products-inv-delta-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="products-output-delta-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-2 mt-2",
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Shandong Stocks",
                        children=[
                            html.P(
                                "Heatmap shows monthly average stocks by city. "
                                "Dark gray cells = no data reported for that city/month (not zero). "
                                "Hover for values.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="shandong-product",
                                            options=[
                                                {"label": "Crude oil", "value": "crude_oil"},
                                                {"label": "Gasoline", "value": "gasoline"},
                                                {"label": "Diesel", "value": "diesel"},
                                                {"label": "Naphtha", "value": "naphtha"},
                                            ],
                                            value="crude_oil",
                                        ),
                                        width=3,
                                    ),
                                ],
                                className="mt-2",
                            ),
                            dcc.Graph(id="shandong-total-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="shandong-heatmap-chart", config={"displayModeBar": True}),
                        ],
                    ),
                    dbc.Tab(
                        label="Economics",
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="econ-refinery-type",
                                            options=[
                                                {"label": "Fuel-oriented refineries", "value": "Fuel-oriented refineries"},
                                                {
                                                    "label": "Oil Refining and Chemical Complexes",
                                                    "value": "Oil Refining and Chemical Complexes",
                                                },
                                                {
                                                    "label": "Oil Refining and Aromatics Complexes",
                                                    "value": "Oil Refining and Aromatics Complexes",
                                                },
                                            ],
                                            value="Fuel-oriented refineries",
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="econ-margin-metric",
                                            options=[
                                                {"label": "Gross profit margin", "value": "gross"},
                                                {"label": "Net profit margin", "value": "net"},
                                                {"label": "Contribution margin (RMB)", "value": "contribution"},
                                            ],
                                            value="gross",
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="econ-display-mode",
                                            options=[
                                                {"label": "Margin (%)", "value": "pct"},
                                                {"label": "RMB value", "value": "money"},
                                            ],
                                            value="pct",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="econ-runs-breakdown",
                                            options=[
                                                {"label": "Independents", "value": "Independents"},
                                                {"label": "National", "value": "National"},
                                                {"label": "SOEs", "value": "SOEs"},
                                            ],
                                            value="Independents",
                                        ),
                                        width=3,
                                    ),
                                ],
                                className="mt-2 g-2",
                            ),
                            dcc.Graph(id="econ-overlay-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="econ-scatter-chart", config={"displayModeBar": True}),
                        ],
                    ),
                    dbc.Tab(
                        label="Seasonal",
                        children=[
                            html.P(
                                "Monthly year-over-year comparison (mean weekly level within each month). "
                                "Current year highlighted in red.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="seasonal-runs-breakdown",
                                            options=[
                                                {"label": "National", "value": "National"},
                                                {"label": "SOEs", "value": "SOEs"},
                                                {"label": "Independents", "value": "Independents"},
                                            ],
                                            value="National",
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="seasonal-product",
                                            options=[
                                                {"label": "Gasoline output", "value": "gasoline_output_kt"},
                                                {"label": "Diesel output", "value": "diesel_output_kt"},
                                                {"label": "Kerosene output", "value": "kerosene_output_kt"},
                                            ],
                                            value="gasoline_output_kt",
                                        ),
                                        width=3,
                                    ),
                                ],
                                className="g-2 mb-2",
                            ),
                            dcc.Graph(id="seasonal-runs-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="seasonal-product-chart", config={"displayModeBar": True}),
                        ],
                    ),
                    dbc.Tab(
                        label="Consumption",
                        children=[
                            html.P(
                                "Each line is one calendar year (Jan–Dec). "
                                "Bold red is the latest year. Jet fuel panels use SCI kerosene "
                                "refueling volume (international + domestic flights).",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dcc.Graph(id="consumption-seasonal-grid", config={"displayModeBar": True}),
                            html.H5("Monthly demand change (MoM / YoY)", className="mt-3 mb-1"),
                            html.P(
                                "SCI national consumption products plus jet refueling. "
                                "Levels follow the global unit toggle (kt / kb/d).",
                                className="text-muted small mb-2",
                            ),
                            _dark_datatable(
                                id="consumption-change-table",
                                columns=change_table_columns("kbd"),
                                data=[],
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ("product", "month")
                                ],
                            ),
                            html.H5("Mobility vs fuel demand", className="mt-4 mb-1"),
                            html.P(
                                "SCI vehicle sales & possession only (no external source). "
                                "Passenger powertrains overlay gasoline demand; "
                                "heavy/medium/light/pickup trucks overlay diesel. "
                                "Sales shares move first; possession mix shows the slower "
                                "fleet shift that actually drives fuel demand.",
                                className="text-muted small mb-1",
                            ),
                            html.P(
                                [
                                    html.Strong("Powertrain terms: "),
                                    "BEV = battery electric (no engine); ",
                                    "PHEV = plug-in hybrid (grid-chargeable + engine); ",
                                    "EREV = extended-range EV (drives electric, engine mainly as generator); ",
                                    "HEV = non-plug-in hybrid (engine + battery, cannot charge from grid). ",
                                    "NEV in SCI broadly covers BEV + PHEV + EREV.",
                                ],
                                className="text-muted small mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Gasoline demand vs passenger NEV / ICE sales share",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-gasoline-nev",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Passenger sales mix by powertrain",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-passenger-mix",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Diesel demand vs truck diesel / NEV sales share",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-diesel-trucks",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Truck sales mix (H/M/L/pickup)",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-truck-sales-mix",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="g-2 mt-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Truck fleet mix by fuel (possession)",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-truck-possession",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Passenger fleet mix by powertrain (possession, annual)",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-passenger-possession",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="g-2 mt-1",
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Petchem",
                        children=[
                            html.P(
                                "National petchem output seasonality (weekly → monthly mean). "
                                "Time series pairs weekly output with unit operating rate where "
                                "available; PX and Ethylene use monthly rates. Inventories show "
                                "port BTX stocks and independent-refinery naphtha.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dcc.Graph(id="petchem-seasonal-grid", config={"displayModeBar": True}),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="petchem-product",
                                            options=[
                                                {"label": p, "value": p}
                                                for p in PETCHEM_PRODUCT_ORDER
                                            ],
                                            value="PX",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="petchem-inventory-view",
                                            options=[
                                                {"label": "All inventories", "value": "all"},
                                                {"label": "Port stocks (BTX)", "value": "port"},
                                                {
                                                    "label": "Refinery naphtha",
                                                    "value": "refinery",
                                                },
                                            ],
                                            value="all",
                                        ),
                                        width=3,
                                    ),
                                ],
                                className="mt-2 g-2",
                            ),
                            dcc.Graph(id="petchem-timeseries-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="petchem-inventories-chart", config={"displayModeBar": True}),
                            html.H5("Monthly petchem output change (MoM / YoY)", className="mt-3 mb-1"),
                            html.P(
                                "SCI monthly chemical output. Levels follow the global unit toggle.",
                                className="text-muted small mb-2",
                            ),
                            _dark_datatable(
                                id="petchem-change-table",
                                columns=change_table_columns("kbd"),
                                data=[],
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ("product", "month")
                                ],
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Regional",
                        children=[
                            html.P(
                                "Compare provincial/regional feedstock processing (runs), "
                                "oil product output, and consumption. "
                                "Responds to the global unit toggle (kt vs kb/d). "
                                "Map uses province bubbles sized and colored by the selected metric.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="regional-product",
                                            options=[
                                                {"label": p, "value": p} for p in REGIONAL_OUTPUT_PRODUCTS
                                            ],
                                            value="Gasoline",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="regional-geo",
                                            options=[
                                                {"label": "SCI region (7)", "value": "region"},
                                                {"label": "Province", "value": "province"},
                                            ],
                                            value="region",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="regional-map-metric",
                                            options=[
                                                {"label": "Map: Processing (runs)", "value": "processing"},
                                                {"label": "Map: Product output", "value": "output"},
                                                {"label": "Map: Consumption", "value": "consumption"},
                                                {"label": "Map: Output − consumption", "value": "balance"},
                                            ],
                                            value="processing",
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Trends chart",
                                                    className="text-muted small mb-1",
                                                ),
                                                dcc.RadioItems(
                                                    id="regional-balance-view",
                                                    options=[
                                                        {"label": " By region", "value": "regional"},
                                                        {"label": " National total", "value": "national"},
                                                    ],
                                                    value="regional",
                                                    inline=True,
                                                    inputStyle={"marginRight": "6px", "marginLeft": "12px"},
                                                    labelStyle={"color": "#e6edf3", "fontSize": "0.9rem"},
                                                ),
                                            ]
                                        ),
                                        width=3,
                                        className="pt-1",
                                    ),
                                ],
                                className="mt-2 g-2",
                            ),
                            dcc.Graph(id="regional-map-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="regional-compare-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="regional-ts-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="regional-balance-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="regional-top-provinces-chart", config={"displayModeBar": True}),
                        ],
                    ),
                    dbc.Tab(
                        label="Exports",
                        children=[
                            html.P(
                                "Refined product exports: cargo-level shipping actuals (sheet 40), "
                                "forward cargoes (sheet 41), and national export quotas (sheet 45). "
                                "Kerosene maps to Jet; diesel to Gasoil/Diesel. FO is not reported "
                                "in SCI refined shipping — shown as zero. Shandong may be under-counted "
                                "if product moves by road before coastal loading.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="export-product",
                                            options=[
                                                {"label": "All products", "value": "All"},
                                                {"label": "Gasoline", "value": "Gasoline"},
                                                {"label": "Jet (kerosene)", "value": "Jet"},
                                                {"label": "Gasoil/Diesel", "value": "Gasoil/Diesel"},
                                            ],
                                            value="All",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="export-period",
                                            options=[
                                                {"label": "Latest month", "value": "latest"},
                                                {"label": "Full selected period", "value": "full"},
                                            ],
                                            value="latest",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="export-monthly-source",
                                            options=[
                                                {
                                                    "label": "Monthly table: shipping actuals",
                                                    "value": "shipping",
                                                },
                                                {
                                                    "label": "Monthly table: quota reported",
                                                    "value": "quotas",
                                                },
                                            ],
                                            value="shipping",
                                        ),
                                        width=3,
                                    ),
                                ],
                                className="mt-2 g-2",
                            ),
                            dcc.Graph(id="export-monthly-chart", config={"displayModeBar": True}),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="export-destinations-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="export-provinces-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            dcc.Graph(id="export-sankey-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="export-quotas-monthly-chart", config={"displayModeBar": True}),
                            dcc.Graph(id="export-quota-summary-chart", config={"displayModeBar": True}),
                            html.H5("Forward export plan tracker", className="mt-3 mb-1"),
                            html.P(
                                "Sheet 41 weekly cargo forecasts stand in for SCI company monthly "
                                "export plans (not published in the API datapack). Compare planned "
                                "loadings vs MTD actuals and the same month last year.",
                                className="text-muted small mb-2",
                            ),
                            dcc.Graph(id="export-forecast-plan-chart", config={"displayModeBar": True}),
                            _dark_datatable(
                                id="export-forecast-summary-table",
                                columns=[
                                    {"name": "month", "id": "month"},
                                    {"name": "product", "id": "product"},
                                    {
                                        "name": "planned_kt",
                                        "id": "planned_kt",
                                        "type": "numeric",
                                        "format": {"specifier": ",.1f"},
                                    },
                                    {
                                        "name": "mtd_actual_kt",
                                        "id": "mtd_actual_kt",
                                        "type": "numeric",
                                        "format": {"specifier": ",.1f"},
                                    },
                                    {
                                        "name": "remaining_vs_plan_kt",
                                        "id": "remaining_vs_plan_kt",
                                        "type": "numeric",
                                        "format": {"specifier": ",.1f"},
                                    },
                                    {
                                        "name": "same_month_ly_kt",
                                        "id": "same_month_ly_kt",
                                        "type": "numeric",
                                        "format": {"specifier": ",.1f"},
                                    },
                                ],
                                data=[],
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ("month", "product")
                                ],
                                page_size=10,
                            ),
                            html.H6("Plan-month cargo detail", className="mt-3 mb-2"),
                            _dark_datatable(
                                id="export-forecast-cargo-table",
                                columns=[
                                    {"name": "departure", "id": "departure"},
                                    {"name": "product", "id": "product"},
                                    {
                                        "name": "volume_kt",
                                        "id": "volume_kt",
                                        "type": "numeric",
                                        "format": {"specifier": ",.1f"},
                                    },
                                    {"name": "province", "id": "province"},
                                    {"name": "port", "id": "port"},
                                    {"name": "destination", "id": "destination"},
                                    {"name": "ship", "id": "ship"},
                                ],
                                data=[],
                                style_cell_conditional=[
                                    {
                                        "if": {"column_id": c},
                                        "textAlign": "left",
                                    }
                                    for c in (
                                        "departure",
                                        "product",
                                        "province",
                                        "port",
                                        "destination",
                                        "ship",
                                    )
                                ],
                                page_size=12,
                            ),
                            html.H5("Monthly exports (kt)", className="mt-3 mb-2"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Download CSV",
                                            id="export-download-btn",
                                            color="secondary",
                                            size="sm",
                                            className="mb-2",
                                        ),
                                        width="auto",
                                    ),
                                ]
                            ),
                            dash_table.DataTable(
                                id="export-monthly-table",
                                columns=[
                                    {"name": "month", "id": "month"},
                                    *[
                                        {
                                            "name": col,
                                            "id": col,
                                            "type": "numeric",
                                            "format": {"specifier": ",.0f"},
                                        }
                                        for col in EXPORT_TABLE_PRODUCTS
                                    ],
                                ],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_header=_TABLE_STYLE_HEADER,
                                style_cell=_TABLE_STYLE_CELL,
                                style_cell_conditional=[
                                    {"if": {"column_id": "month"}, "textAlign": "left"}
                                ],
                                page_size=15,
                            ),
                            dcc.Download(id="export-monthly-download"),
                        ],
                    ),
                    dbc.Tab(
                        label="Crude Imports",
                        children=[
                            html.P(
                                "Port crude arrivals: foreign trade (sheet 36/37) and domestic "
                                "coastal (38/39), plus independent-refinery crude import quotas "
                                "(sheet 44). Forecast cargoes overlay the plan month.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="crude-trade",
                                            options=[
                                                {"label": "All trades", "value": "all"},
                                                {"label": "Foreign", "value": "foreign"},
                                                {"label": "Domestic", "value": "domestic"},
                                            ],
                                            value="foreign",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        _dropdown(
                                            id="crude-period",
                                            options=[
                                                {"label": "Latest month", "value": "latest"},
                                                {
                                                    "label": "Full selected period",
                                                    "value": "full",
                                                },
                                            ],
                                            value="latest",
                                        ),
                                        width=2,
                                    ),
                                ],
                                className="mt-2 g-2",
                            ),
                            dcc.Graph(
                                id="crude-monthly-chart",
                                config={"displayModeBar": True},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="crude-origins-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="crude-provinces-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="crude-fleet-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="crude-grades-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            html.H5("Mix over time", className="mt-3 mb-1"),
                            html.P(
                                "Monthly share of arrivals (100% stacked). Dashed line = "
                                "top-3 category concentration. Top 8 categories + Other; "
                                "follows trade + date-range filters.",
                                className="text-muted small mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Fleet mix",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="crude-fleet-mix-trend",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Origin-country mix",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="crude-origin-mix-trend",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Discharge-province mix",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="crude-province-mix-trend",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Crude-grade mix",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="crude-grade-mix-trend",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            html.H5("Voyage logistics", className="mt-3 mb-1"),
                            html.P(
                                "Transit = arrival - origin departure. Discharge lag = "
                                "discharge - arrival (port wait proxy). Extreme transit "
                                "outliers are dropped. Red delta vs LY = slower; green = faster.",
                                className="text-muted small mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Voyage transit days (median + p25-p75)",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="crude-transit-trend",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Discharge lag (median + p25-p75)",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="crude-discharge-lag-trend",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            html.H6(
                                "Origin transit — latest month vs same month LY",
                                className="mt-3 mb-2 text-center",
                            ),
                            _dark_datatable(
                                id="crude-origin-transit-table",
                                columns=origin_transit_compare_columns(),
                                data=[],
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ("month", "origin")
                                ],
                                style_data_conditional=origin_transit_delta_style(),
                                page_size=12,
                            ),
                            html.H5(
                                "Independent crude import quotas",
                                className="mt-3 mb-1",
                            ),
                            html.P(
                                "Quota sheet covers independent refineries only "
                                "(not SOE imports). Ceiling can exceed issued annual quota.",
                                className="text-muted small mb-2",
                            ),
                            dcc.Graph(
                                id="crude-quotas-monthly-chart",
                                config={"displayModeBar": True},
                            ),
                            dcc.Graph(
                                id="crude-quota-summary-chart",
                                config={"displayModeBar": True},
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Oil Grade",
                        children=[
                            html.P(
                                "Shandong independent refinery crude processing by oil grade "
                                "(sheet 16). Origin buckets group grades by country / domestic / "
                                "secondary residue feedstock.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        _dropdown(
                                            id="oil-grade-city",
                                            options=_oil_grade_city_options(),
                                            value="All",
                                        ),
                                        width=3,
                                    ),
                                ],
                                className="mt-2 g-2",
                            ),
                            dcc.Graph(
                                id="oil-grade-totals-chart",
                                config={"displayModeBar": True},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="oil-grade-city-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="oil-grade-top-chart",
                                            config={"displayModeBar": True},
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            html.H6(
                                "Origin-bucket mix over time",
                                className="mt-3 mb-1 text-center text-light",
                            ),
                            html.P(
                                "Monthly share of grade volumes (ex totals). Dashed line = "
                                "top-3 origin concentration.",
                                className="text-muted small mb-2 text-center",
                            ),
                            dcc.Graph(
                                id="oil-grade-mix-chart",
                                config={"displayModeBar": True},
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Monthly Detail",
                        children=[
                            dbc.Row(
                                dbc.Col(
                                    _dropdown(
                                        id="ownership-select",
                                        options=[
                                            {"label": o, "value": o}
                                            for o in sorted(FRAMES["processing_ownership"]["ownership"].unique())
                                        ],
                                        value="Independent refinery",
                                    ),
                                    width=4,
                                    className="mt-2",
                                )
                            ),
                            dcc.Graph(id="monthly-processing-chart", config={"displayModeBar": True}),
                        ],
                    ),
                ],
                className="mt-2",
            ),
        ],
        fluid=True,
    )


_init_frames(refresh=False, history=True)
app.layout = _build_layout()


@app.callback(
    Output("runs-volume-chart", "figure"),
    Output("runs-rate-chart", "figure"),
    Output("balance-monthly-chart", "figure"),
    Output("spread-chart", "figure"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
)
def update_runs_charts(unit, date_range):
    return (
        fig_runs_volume(FRAMES["runs_wide"], unit, date_range),
        fig_runs_rate(FRAMES["runs_wide"], date_range),
        fig_balance_monthly(FRAMES.get("balance_monthly", pd.DataFrame()), unit, date_range),
        fig_soe_indep_spread(FRAMES["runs_wide"], date_range),
    )


@app.callback(
    Output("products-chart", "figure"),
    Output("products-change-table", "columns"),
    Output("products-change-table", "data"),
    Output("products-inv-delta-chart", "figure"),
    Output("products-output-delta-chart", "figure"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("products-stock-unit", "value"),
    Input("products-flow-unit", "value"),
)
def update_products(unit, date_range, stock_unit, flow_unit):
    wide = FRAMES.get("products_stocks_wide", pd.DataFrame())
    stock_unit = stock_unit or "mb"
    flow_unit = flow_unit or "kbd"
    records, columns = products_scorecard_records(
        wide,
        stock_unit=stock_unit,
        flow_unit=flow_unit,
    )
    return (
        fig_products(wide, unit, date_range),
        columns,
        records,
        _apply_dark(
            fig_products_delta_bars(
                wide, kind="stocks", unit=stock_unit, date_range=date_range
            )
        ),
        _apply_dark(
            fig_products_delta_bars(
                wide, kind="output", unit=flow_unit, date_range=date_range
            )
        ),
    )


@app.callback(
    Output("shandong-total-chart", "figure"),
    Output("shandong-heatmap-chart", "figure"),
    Input("shandong-product", "value"),
    Input("date-range-select", "value"),
)
def update_shandong(product, date_range):
    return (
        fig_shandong_totals(FRAMES["shandong_stocks_total"], product, date_range),
        fig_shandong_heatmap(FRAMES["shandong_stocks"], product, date_range),
    )


@app.callback(
    Output("econ-overlay-chart", "figure"),
    Output("econ-scatter-chart", "figure"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("econ-refinery-type", "value"),
    Input("econ-margin-metric", "value"),
    Input("econ-display-mode", "value"),
    Input("econ-runs-breakdown", "value"),
)
def update_economics(unit, date_range, refinery_type, margin_metric, display_mode, runs_breakdown):
    margins = FRAMES.get("margins_wide", pd.DataFrame())
    mode = display_mode if margin_metric in ("gross", "net") else "pct"
    return (
        fig_economics_overlay(
            FRAMES["runs_wide"],
            margins,
            unit,
            date_range,
            refinery_type,
            margin_metric,
            runs_breakdown,
            mode,
        ),
        fig_economics_scatter(
            FRAMES["runs_wide"],
            margins,
            date_range,
            refinery_type,
            margin_metric,
            runs_breakdown,
            mode,
        ),
    )


@app.callback(
    Output("seasonal-runs-chart", "figure"),
    Output("seasonal-product-chart", "figure"),
    Input("unit-select", "value"),
    Input("seasonal-runs-breakdown", "value"),
    Input("seasonal-product", "value"),
)
def update_seasonal(unit, runs_breakdown, product_col):
    runs_monthly = build_runs_seasonal(FRAMES["runs_wide"], runs_breakdown, unit)
    prod_monthly = build_product_seasonal(FRAMES["products_stocks_wide"], product_col, unit)

    product_label = product_col.replace("_output_kt", "").replace("_", " ").title()
    runs_fig = _apply_dark(
        fig_seasonal_lines(
            runs_monthly,
            f"Feedstock processing — {runs_breakdown}",
            unit,
        )
    )
    prod_fig = _apply_dark(
        fig_seasonal_lines(
            prod_monthly,
            f"{product_label} output — National",
            unit,
        )
    )
    return runs_fig, prod_fig


@app.callback(
    Output("consumption-seasonal-grid", "figure"),
    Output("consumption-change-table", "columns"),
    Output("consumption-change-table", "data"),
    Input("unit-select", "value"),
)
def update_consumption_seasonal(unit):
    cons = FRAMES.get("consumption_national", pd.DataFrame())
    jet = FRAMES.get("kerosene_refueling_national", pd.DataFrame())
    n_panels = len(CONSUMPTION_PRODUCT_ORDER) + len(JET_FUEL_PRODUCT_ORDER)
    nrows = (n_panels + 1) // 2
    table_unit = unit if unit in ("kt", "kbd") else "kbd"
    return (
        _apply_dark(
            fig_consumption_seasonal_grid(cons, unit, jet),
            height=320 * nrows,
        ),
        change_table_columns(table_unit),
        consumption_change_records(cons, jet, unit=table_unit),
    )


@app.callback(
    Output("mobility-gasoline-nev", "figure"),
    Output("mobility-passenger-mix", "figure"),
    Output("mobility-diesel-trucks", "figure"),
    Output("mobility-truck-sales-mix", "figure"),
    Output("mobility-truck-possession", "figure"),
    Output("mobility-passenger-possession", "figure"),
    Input("unit-select", "value"),
)
def update_mobility_charts(unit):
    cons = FRAMES.get("consumption_national", pd.DataFrame())
    pax_sales = FRAMES.get("vehicle_sales_passenger", pd.DataFrame())
    com_sales = FRAMES.get("vehicle_sales_commercial", pd.DataFrame())
    com_poss = FRAMES.get("vehicle_possession_commercial", pd.DataFrame())
    pax_poss = FRAMES.get("vehicle_possession_passenger", pd.DataFrame())
    return (
        _apply_dark(fig_gasoline_vs_nev(cons, pax_sales, unit=unit), height=360),
        _apply_dark(fig_passenger_sales_mix(pax_sales), height=360),
        _apply_dark(
            fig_diesel_vs_trucks(cons, com_sales, com_poss, unit=unit), height=360
        ),
        _apply_dark(fig_truck_sales_mix(com_sales), height=360),
        _apply_dark(fig_truck_possession_mix(com_poss), height=360),
        _apply_dark(fig_passenger_possession(pax_poss), height=360),
    )


@app.callback(
    Output("petchem-seasonal-grid", "figure"),
    Input("unit-select", "value"),
)
def update_petchem_seasonal(unit):
    weekly = FRAMES.get("petchem_weekly", pd.DataFrame())
    monthly = FRAMES.get("petchem_monthly", pd.DataFrame())
    n_panels = len(PETCHEM_PRODUCT_ORDER)
    nrows = (n_panels + 1) // 2
    return _apply_dark(
        fig_petchem_seasonal_grid(weekly, monthly, unit),
        height=320 * nrows,
    )


@app.callback(
    Output("petchem-timeseries-chart", "figure"),
    Output("petchem-inventories-chart", "figure"),
    Output("petchem-change-table", "columns"),
    Output("petchem-change-table", "data"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("petchem-product", "value"),
    Input("petchem-inventory-view", "value"),
)
def update_petchem(unit, date_range, product, inventory_view):
    weekly = FRAMES.get("petchem_weekly", pd.DataFrame())
    monthly = FRAMES.get("petchem_monthly", pd.DataFrame())
    inventories = FRAMES.get("petchem_inventories", pd.DataFrame())
    table_unit = unit if unit in ("kt", "kbd") else "kbd"
    return (
        _apply_dark(
            fig_petchem_timeseries(weekly, monthly, product or "PX", unit, date_range)
        ),
        _apply_dark(
            fig_petchem_inventories(
                inventories,
                date_range,
                inventory_view=inventory_view or "all",
            ),
            height=620 if (inventory_view or "all") == "all" else None,
        ),
        change_table_columns(table_unit),
        petchem_change_records(monthly, unit=table_unit),
    )


@app.callback(
    Output("regional-map-chart", "figure"),
    Output("regional-compare-chart", "figure"),
    Output("regional-ts-chart", "figure"),
    Output("regional-balance-chart", "figure"),
    Output("regional-top-provinces-chart", "figure"),
    Input("unit-select", "value"),
    Input("regional-product", "value"),
    Input("regional-geo", "value"),
    Input("regional-map-metric", "value"),
    Input("regional-balance-view", "value"),
    Input("date-range-select", "value"),
)
def update_regional(unit, product, geo, map_metric, timeseries_view, date_range):
    mapping = province_region_map(FRAMES.get("consumption", pd.DataFrame()))
    proc = FRAMES.get("provincial_processing", pd.DataFrame())
    out = FRAMES.get("provincial_output", pd.DataFrame())
    cons = FRAMES.get("consumption", pd.DataFrame())

    runs = aggregate_runs(proc, mapping, geo)
    output = aggregate_output(out, mapping, product, geo)
    consumption = aggregate_consumption(cons, product, geo)

    return (
        _apply_dark(
            fig_regional_map(
                proc,
                out,
                cons,
                mapping,
                product=product,
                metric=map_metric,
                date_range=date_range,
                unit=unit,
            ),
            height=520,
        ),
        _apply_dark(
            fig_regional_compare_bars(
                runs,
                output,
                consumption,
                product=product,
                geo=geo,
                date_range=date_range,
                unit=unit,
            )
        ),
        _apply_dark(
            fig_regional_timeseries(
                runs,
                output,
                consumption,
                product=product,
                geo=geo,
                date_range=date_range,
                unit=unit,
                timeseries_view=timeseries_view or "regional",
            )
        ),
        _apply_dark(
            fig_regional_balance_gap(
                runs,
                output,
                consumption,
                product=product,
                geo=geo,
                date_range=date_range,
                unit=unit,
            )
        ),
        _apply_dark(
            fig_top_provinces(
                proc,
                out,
                cons,
                mapping,
                product=product,
                date_range=date_range,
                unit=unit,
            )
        ),
    )


@app.callback(
    Output("po-map-chart", "figure"),
    Output("po-heatmap-chart", "figure"),
    Output("po-province-bars-chart", "figure"),
    Output("po-timeseries-chart", "figure"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("po-metric", "value"),
    Input("po-ownership", "value"),
    Input("po-province", "value"),
)
def update_processing_operating(unit, date_range, metric, ownership, province):
    proc = FRAMES.get("processing_ownership", pd.DataFrame())
    rates = FRAMES.get("operating_rate_ownership", pd.DataFrame())
    return (
        _apply_dark(
            fig_ownership_infra_map(
                proc,
                rates,
                INFRA_SITES,
                metric=metric or "processing",
                ownership=ownership or "All",
                date_range=date_range,
                unit=unit,
            ),
            height=560,
        ),
        _apply_dark(
            fig_ownership_heatmap(
                proc,
                rates,
                metric=metric or "processing",
                date_range=date_range,
                unit=unit,
            )
        ),
        _apply_dark(
            fig_ownership_province_bars(
                proc,
                rates,
                province=province or "Shandong",
                date_range=date_range,
                unit=unit,
            )
        ),
        _apply_dark(
            fig_ownership_timeseries(
                proc,
                rates,
                province=province or "Shandong",
                ownership=ownership or "All",
                date_range=date_range,
                unit=unit,
            )
        ),
    )


@app.callback(
    Output("monthly-processing-chart", "figure"),
    Input("ownership-select", "value"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
)
def update_monthly(ownership, unit, date_range):
    return fig_monthly_processing(FRAMES["processing_ownership"], ownership, unit, date_range)


@app.callback(
    Output("export-monthly-chart", "figure"),
    Output("export-destinations-chart", "figure"),
    Output("export-provinces-chart", "figure"),
    Output("export-sankey-chart", "figure"),
    Output("export-quotas-monthly-chart", "figure"),
    Output("export-quota-summary-chart", "figure"),
    Output("export-forecast-plan-chart", "figure"),
    Output("export-forecast-summary-table", "data"),
    Output("export-forecast-cargo-table", "data"),
    Output("export-monthly-table", "data"),
    Input("date-range-select", "value"),
    Input("export-product", "value"),
    Input("export-period", "value"),
    Input("export-monthly-source", "value"),
)
def update_exports(date_range, product, period, monthly_source):
    shipping = FRAMES.get("export_shipping", pd.DataFrame())
    forecast = FRAMES.get("export_forecast", pd.DataFrame())
    quotas_monthly = FRAMES.get("export_quotas_monthly", pd.DataFrame())
    quotas_summary = FRAMES.get("export_quotas_summary", pd.DataFrame())

    product = product or "All"
    period = period or "latest"
    monthly_source = monthly_source or "shipping"

    return (
        _apply_dark(
            fig_export_monthly_stacked(
                shipping,
                forecast,
                product=product,
                date_range=date_range,
            )
        ),
        _apply_dark(
            fig_export_destinations(
                shipping,
                product=product,
                date_range=date_range,
                period=period,
            )
        ),
        _apply_dark(
            fig_export_provinces(
                shipping,
                product=product,
                date_range=date_range,
                period=period,
            )
        ),
        _apply_dark(
            fig_export_sankey(
                shipping,
                product=product,
                date_range=date_range,
                period=period,
            ),
            height=520,
        ),
        _apply_dark(fig_export_quotas_monthly(quotas_monthly)),
        _apply_dark(fig_export_quota_summary(quotas_summary, quotas_monthly)),
        _apply_dark(fig_export_forecast_plan(forecast, shipping, product=product)),
        export_forecast_summary_records(forecast, shipping, product=product),
        export_forecast_table_records(forecast, product=product),
        export_monthly_table_records(
            shipping,
            quotas_monthly,
            source=monthly_source,
        ),
    )


@app.callback(
    Output("export-monthly-download", "data"),
    Input("export-download-btn", "n_clicks"),
    Input("export-monthly-source", "value"),
    prevent_initial_call=True,
)
def download_export_monthly(n_clicks, monthly_source):
    from src.exports import export_monthly_table_df

    df = export_monthly_table_df(
        FRAMES.get("export_shipping", pd.DataFrame()),
        FRAMES.get("export_quotas_monthly", pd.DataFrame()),
        source=monthly_source or "shipping",
    )
    return dcc.send_data_frame(df.to_csv, "china_exports_monthly.csv", index=False)


@app.callback(
    Output("crude-monthly-chart", "figure"),
    Output("crude-origins-chart", "figure"),
    Output("crude-provinces-chart", "figure"),
    Output("crude-fleet-chart", "figure"),
    Output("crude-grades-chart", "figure"),
    Output("crude-fleet-mix-trend", "figure"),
    Output("crude-origin-mix-trend", "figure"),
    Output("crude-province-mix-trend", "figure"),
    Output("crude-grade-mix-trend", "figure"),
    Output("crude-transit-trend", "figure"),
    Output("crude-discharge-lag-trend", "figure"),
    Output("crude-origin-transit-table", "data"),
    Output("crude-quotas-monthly-chart", "figure"),
    Output("crude-quota-summary-chart", "figure"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("crude-trade", "value"),
    Input("crude-period", "value"),
)
def update_crude_imports(unit, date_range, trade, period):
    shipping = FRAMES.get("crude_shipping", pd.DataFrame())
    forecast = FRAMES.get("crude_forecast", pd.DataFrame())
    quotas_monthly = FRAMES.get("crude_quotas_monthly", pd.DataFrame())
    quotas_summary = FRAMES.get("crude_quotas_summary", pd.DataFrame())
    trade = trade or "foreign"
    period = period or "latest"
    unit = unit if unit in ("kt", "kbd") else "kt"

    return (
        _apply_dark(
            fig_crude_monthly_arrivals(
                shipping,
                forecast,
                trade=trade,
                date_range=date_range,
                unit=unit,
            )
        ),
        _apply_dark(
            fig_crude_origins(
                shipping, trade=trade, date_range=date_range, period=period
            )
        ),
        _apply_dark(
            fig_crude_provinces(
                shipping, trade=trade, date_range=date_range, period=period
            )
        ),
        _apply_dark(
            fig_crude_fleet_mix(
                shipping, trade=trade, date_range=date_range, period=period
            )
        ),
        _apply_dark(
            fig_crude_grades(
                shipping, trade=trade, date_range=date_range, period=period
            )
        ),
        _apply_dark(
            fig_crude_fleet_mix_trend(
                shipping, trade=trade, date_range=date_range
            ),
            height=480,
        ),
        _apply_dark(
            fig_crude_origin_mix_trend(
                shipping, trade=trade, date_range=date_range
            ),
            height=480,
        ),
        _apply_dark(
            fig_crude_province_mix_trend(
                shipping, trade=trade, date_range=date_range
            ),
            height=480,
        ),
        _apply_dark(
            fig_crude_grade_mix_trend(
                shipping, trade=trade, date_range=date_range
            ),
            height=480,
        ),
        _apply_dark(
            fig_crude_transit_trend(
                shipping, trade=trade, date_range=date_range
            ),
            height=480,
        ),
        _apply_dark(
            fig_crude_discharge_lag_trend(
                shipping, trade=trade, date_range=date_range
            ),
            height=480,
        ),
        origin_transit_compare_records(shipping, trade=trade),
        _apply_dark(fig_crude_quotas_monthly(quotas_monthly)),
        _apply_dark(fig_crude_quota_summary(quotas_summary, quotas_monthly)),
    )


@app.callback(
    Output("oil-grade-totals-chart", "figure"),
    Output("oil-grade-city-chart", "figure"),
    Output("oil-grade-top-chart", "figure"),
    Output("oil-grade-mix-chart", "figure"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("oil-grade-city", "value"),
)
def update_oil_grade(unit, date_range, city):
    df = FRAMES.get("processing_oil_grade", pd.DataFrame())
    city = city or "All"
    unit = unit if unit in ("kt", "kbd") else "kt"
    return (
        _apply_dark(
            fig_oil_grade_totals(df, city=city, date_range=date_range, unit=unit)
        ),
        _apply_dark(
            fig_oil_grade_city_bars(df, date_range=date_range, unit=unit)
        ),
        _apply_dark(
            fig_oil_grade_top_grades(
                df, city=city, date_range=date_range, unit=unit
            )
        ),
        _apply_dark(
            fig_oil_grade_origin_mix(df, city=city, date_range=date_range),
            height=480,
        ),
    )


@app.callback(
    Output("refresh-status", "children"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_data(n_clicks):
    if not n_clicks:
        return ""
    try:
        out = fetch_and_cache(include_history=True)
        return dbc.Alert(
            f"Refreshed with history merge. Restart app to reload. Cache: {out}",
            color="success",
        )
    except Exception as exc:
        return dbc.Alert(f"Refresh failed: {exc}", color="danger")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--refresh", action="store_true", help="Fetch from API on startup.")
    parser.add_argument("--history", action="store_true", help="Merge historical snapshots on refresh.")
    parser.add_argument("--latest-only", action="store_true")
    args = parser.parse_args()

    history = not args.latest_only
    if args.refresh:
        fetch_and_cache(include_history=history)

    _init_frames(refresh=False, history=history)
    app.layout = _build_layout()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
