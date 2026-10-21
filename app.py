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
4  Economics      – refinery margins vs runs + WoW profit drivers / waterfall
5  Seasonal         – year-over-year crude runs & product output by month
6  Consumption      – national demand seasonality, MoM/YoY, mobility vs gasoline/diesel
7  Petchem          – chemical output seasonality, weekly trends, inventories + MoM/YoY table
8  Monthly Detail – provincial processing by ownership
9  Regional       – geographic runs, output & consumption comparison
10 Exports        – refined product shipping, quotas, forward forecast tracker & monthly export table
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
    export_monthly_pivot_columns,
    export_monthly_pivot_records,
    export_monthly_table_records,
    fig_export_destinations,
    fig_export_forecast_plan,
    fig_export_forecast_sankey,
    fig_export_monthly_stacked,
    fig_export_provinces,
    fig_export_quota_summary,
    fig_export_quotas_monthly,
    fig_export_sankey,
)
from src.transforms import EXPORT_TABLE_PRODUCTS
from src.units import (
    DEFAULT_BARRELS_PER_MT,
    SHANDONG_PRODUCT_MAP,
    convert_volume_series,
    volume_label,
)
from src.change_tables import (
    change_table_columns,
    consumption_change_records,
    fig_products_delta_bars,
    petchem_change_records,
    products_scorecard_records,
    products_scorecard_style,
    runs_scorecard_records,
    runs_scorecard_style,
)
from src.economics import (
    fig_profit_waterfall,
    profit_driver_scorecard,
    profit_driver_scorecard_style,
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
    fig_heavy_truck_possession_mix,
    fig_heavy_truck_sales_mix,
    fig_passenger_possession,
    fig_passenger_sales_mix,
    fig_passenger_sales_vs_possession,
    fig_provincial_nev_bars,
    fig_provincial_nev_map,
    fig_truck_possession_mix,
    fig_truck_sales_mix,
    fig_truck_sales_vs_possession,
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
    product: str | None = None,
) -> pd.Series:
    return convert_volume_series(
        values, unit, freq=freq, dates=dates, metric=metric, product=product
    )


def _inventory_unit(flow_unit: str) -> str:
    """Stocks are levels, not rates: kb/d mode → mb; metric mode → kt."""
    return "mb" if (flow_unit or "kbd") == "kbd" else "kt"


SHANDONG_STOCK_PRODUCTS = ("crude_oil", "gasoline", "diesel", "naphtha")


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
    flow_unit = unit or "kbd"
    # Stocks are inventory levels — never kb/d. Barrel flows → mb; metric → kt.
    stock_unit = _inventory_unit(flow_unit)
    flow_lbl = volume_label(flow_unit)
    stock_lbl = volume_label(stock_unit)
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
                    y=_vol(df[col], flow_unit, metric=col),
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
                    y=_vol(df[col], stock_unit, metric=col),
                    name=col.replace("_kt", "").replace("_", " ").title(),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=(
            f"National output ({flow_lbl}, solid) vs commercial stocks "
            f"({stock_lbl}, dotted)"
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=f"Weekly output ({flow_lbl})", secondary_y=False)
    fig.update_yaxes(title_text=f"Commercial stocks ({stock_lbl})", secondary_y=True)
    return _apply_dark(fig)


def fig_shandong_totals(
    totals: pd.DataFrame,
    product: str,
    date_range: str,
    unit: str = "kbd",
) -> go.Figure:
    stock_unit = _inventory_unit(unit)
    ylab = volume_label(stock_unit)
    prod_key = SHANDONG_PRODUCT_MAP.get(product, product)
    sub = filter_by_date_range(totals[totals["product"] == product], "week_start", date_range)
    y = (
        _vol(sub["value_kt"], stock_unit, product=prod_key)
        if not sub.empty
        else sub.get("value_kt", pd.Series(dtype=float))
    )
    hover_fmt = "%{y:,.2f}" if stock_unit == "mb" else "%{y:,.0f}"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["week_start"] if not sub.empty else [],
            y=y,
            mode="lines+markers",
            line=dict(width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.12)",
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{hover_fmt} {ylab}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"Shandong independents — total {product.replace('_', ' ')} "
            f"stocks ({ylab})"
        ),
        xaxis_title="Week",
        yaxis_title=ylab,
    )
    return _apply_dark(fig)


def fig_shandong_heatmap(
    stocks: pd.DataFrame,
    product: str,
    date_range: str,
    unit: str = "kbd",
) -> go.Figure:
    stock_unit = _inventory_unit(unit)
    ylab = volume_label(stock_unit)
    prod_key = SHANDONG_PRODUCT_MAP.get(product, product)
    sub = filter_by_date_range(stocks[stocks["product"] == product], "week_start", date_range)
    if sub.empty:
        return _apply_dark(
            go.Figure().update_layout(title=f"No Shandong {product.replace('_', ' ')} stock data"),
            height=420,
        )

    # Monthly mean reduces ~270 weekly columns to ~50 months; easier to read.
    sub = sub.copy()
    sub["month_start"] = sub["week_start"].dt.to_period("M").dt.to_timestamp()
    sub["value_display"] = _vol(sub["value_kt"], stock_unit, product=prod_key)
    monthly = (
        sub.groupby(["region", "month_start"], as_index=False)["value_display"]
        .mean()
    )
    pivot = monthly.pivot(index="region", columns="month_start", values="value_display")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    z_raw = pivot.values.astype(float)
    z_max = float(np.nanmax(z_raw)) if np.any(~np.isnan(z_raw)) else 1.0
    z_max = max(z_max, 1e-6)

    # Plotly renders NaN as the plot background (looks like a black gap). Use -1 → gray.
    # For mb scales, sentinel must sit below data; use a small negative vs data max.
    z_sentinel = -1.0 if stock_unit == "kt" else -max(z_max * 0.05, 0.01)
    z_color = np.where(np.isnan(z_raw), z_sentinel, z_raw)
    colorscale = [
        [0.0, "#30363d"],
        [0.001, "#dbeafe"],
        [1.0, "#1d4ed8"],
    ]

    val_fmt = ",.2f" if stock_unit == "mb" else ",.0f"
    custom_hover = np.empty(z_raw.shape, dtype=object)
    for i, region in enumerate(pivot.index):
        for j, month in enumerate(pivot.columns):
            val = z_raw[i, j]
            if np.isnan(val):
                custom_hover[i, j] = f"{region}<br>{pd.Timestamp(month):%Y-%m}<br>No data"
            else:
                custom_hover[i, j] = (
                    f"{region}<br>{pd.Timestamp(month):%Y-%m}<br>{val:{val_fmt}} {ylab}"
                )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_color,
            x=[pd.Timestamp(c).strftime("%Y-%m") for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=colorscale,
            zmin=z_sentinel,
            zmax=z_max,
            customdata=custom_hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(title=ylab),
        )
    )
    fig.update_layout(
        title=(
            f"Shandong {product.replace('_', ' ')} stocks by city "
            f"({ylab}, monthly avg)"
        ),
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


def _margin_factor_lookup(
    margins: pd.DataFrame,
    refinery_type: str,
    factor: str,
) -> pd.Series:
    """Week-indexed values for one SCI profit-model factor."""
    if margins.empty:
        return pd.Series(dtype=float)
    sub = margins[
        (margins["refinery_type"] == refinery_type) & (margins["factor"] == factor)
    ][["week_start", "value"]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    return (
        sub.sort_values("week_start")
        .drop_duplicates("week_start", keep="last")
        .set_index("week_start")["value"]
        .astype(float)
    )


def _margin_axis_label(display_mode: str) -> str:
    return {
        "pct": "%",
        "money": "RMB",
        "rmb_mt": "RMB/mt",
        "usd_bbl": "USD/bbl",
    }.get(display_mode, display_mode)


def _margin_series(
    margins: pd.DataFrame,
    *,
    refinery_type: str,
    margin_metric: str,
    display_mode: str,
) -> tuple[pd.DataFrame, str]:
    """
    Build week_start × margin value for Economics charts.

    display_mode:
      pct     – gross/net % (contribution falls back to absolute RMB)
      money   – absolute RMB (gross keeps legacy Profit-per-ton series)
      rmb_mt  – unit margin in RMB/mt
      usd_bbl – unit margin converted via USD/RMB FX and crude bbl/mt
    """
    empty = pd.DataFrame(columns=["week_start", "margin"])
    label = _margin_axis_label(display_mode)

    ppt = _margin_factor_lookup(margins, refinery_type, "Profit per ton")
    net = _margin_factor_lookup(margins, refinery_type, "Net profit")
    contrib = _margin_factor_lookup(margins, refinery_type, "Contribution margin")
    fx = _margin_factor_lookup(margins, refinery_type, "USD to RMB Exchange Rate")
    gross_pct = _margin_factor_lookup(margins, refinery_type, "Gross profit margin (%)")
    net_pct = _margin_factor_lookup(margins, refinery_type, "Net profit margin (%)")

    # Throughput implied by SCI: Net profit / Profit per ton (ppt treated as net RMB/mt).
    throughput = (net / ppt).where((ppt != 0) & ppt.notna() & net.notna())

    def _frame(series: pd.Series) -> pd.DataFrame:
        s = series.dropna()
        if s.empty:
            return empty
        return (
            s.rename("margin")
            .rename_axis("week_start")
            .reset_index()
            .sort_values("week_start")
        )

    def _to_usd_bbl(rmb_mt: pd.Series) -> pd.Series:
        aligned = pd.concat([rmb_mt.rename("rmb"), fx.rename("fx")], axis=1).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        return aligned["rmb"] / aligned["fx"] / DEFAULT_BARRELS_PER_MT

    if display_mode == "pct":
        if margin_metric == "gross" and not gross_pct.empty:
            return _frame(gross_pct * 100.0), "%"
        if margin_metric == "net" and not net_pct.empty:
            return _frame(net_pct * 100.0), "%"
        # Contribution has no % series — show absolute RMB.
        if not contrib.empty:
            return _frame(contrib), "RMB"
        return empty, label

    if display_mode == "money":
        # Preserve prior behaviour: gross→profit/ton, net→net RMB, contrib→contrib RMB.
        if margin_metric == "gross" and not ppt.empty:
            return _frame(ppt), "RMB/mt"
        if margin_metric == "net" and not net.empty:
            return _frame(net), "RMB"
        if not contrib.empty:
            return _frame(contrib), "RMB"
        return empty, label

    # Unit margins (RMB/mt), then optionally USD/bbl.
    if margin_metric == "net":
        rmb_mt = ppt
    elif margin_metric in ("gross", "contribution"):
        rmb_mt = (contrib / throughput).where(throughput.notna() & (throughput != 0))
    else:
        rmb_mt = pd.Series(dtype=float)

    if rmb_mt.dropna().empty:
        return empty, label

    if display_mode == "rmb_mt":
        return _frame(rmb_mt), "RMB/mt"
    if display_mode == "usd_bbl":
        return _frame(_to_usd_bbl(rmb_mt)), "USD/bbl"
    return empty, label


def fig_economics_overlay(
    runs: pd.DataFrame,
    margins: pd.DataFrame,
    unit: str,
    date_range: str,
    refinery_type: str,
    margin_metric: str,
    runs_breakdown: str,
    display_mode: str = "pct",
) -> go.Figure:
    runs = filter_by_date_range(runs, "week_start", date_range)
    ylab = volume_label(unit)

    run_sub = runs[runs["breakdown"] == runs_breakdown].sort_values("week_start")
    margin_df, y2 = _margin_series(
        margins,
        refinery_type=refinery_type,
        margin_metric=margin_metric,
        display_mode=display_mode,
    )
    margin_df = filter_by_date_range(margin_df, "week_start", date_range)

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

    if not margin_df.empty:
        fig.add_trace(
            go.Scatter(
                x=margin_df["week_start"],
                y=margin_df["margin"],
                name=f"{refinery_type} {margin_metric} ({y2})",
                line=dict(color=MARGIN_COLORS.get(margin_metric, "#e3b341"), width=2, dash="dot"),
                mode="lines",
            ),
            secondary_y=True,
        )

    fig.update_layout(title="Refinery runs vs modelled margins", hovermode="x unified")
    fig.update_yaxes(title_text=f"Processing ({ylab})", secondary_y=False)
    fig.update_yaxes(title_text=f"Margin ({y2})", secondary_y=True)
    return _apply_dark(fig)


def fig_economics_scatter(
    runs: pd.DataFrame,
    margins: pd.DataFrame,
    date_range: str,
    refinery_type: str,
    margin_metric: str,
    runs_breakdown: str,
    display_mode: str = "pct",
) -> go.Figure:
    runs = filter_by_date_range(runs, "week_start", date_range)
    margin_df, xlab = _margin_series(
        margins,
        refinery_type=refinery_type,
        margin_metric=margin_metric,
        display_mode=display_mode,
    )
    margin_df = filter_by_date_range(margin_df, "week_start", date_range)

    run_sub = runs[runs["breakdown"] == runs_breakdown][["week_start", "processing_kt"]]
    if margin_df.empty or run_sub.empty:
        return _apply_dark(go.Figure().update_layout(title="Insufficient margin data"))

    merged = run_sub.merge(margin_df, on="week_start", how="inner")
    if merged.empty:
        return _apply_dark(go.Figure().update_layout(title="Insufficient margin data"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["margin"],
            y=_vol(merged["processing_kt"], "kbd", metric="processing_kt"),
            mode="markers",
            marker=dict(size=8, color="#58a6ff", opacity=0.75),
            text=[d.strftime("%Y-%m-%d") for d in merged["week_start"]],
            hovertemplate=(
                f"Week: %{{text}}<br>Margin: %{{x:.2f}} {xlab}"
                "<br>Runs: %{y:,.0f} kb/d<extra></extra>"
            ),
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
                            html.H5("Weekly scorecard", className="mt-3 mb-1"),
                            html.P(
                                id="runs-scorecard-subtitle",
                                children=(
                                    "Latest SCI weekly summary. Processing follows the global "
                                    "unit toggle; operating rates in %. National is the total."
                                ),
                                className="text-muted small mb-2",
                            ),
                            html.Div(
                                _dark_datatable(
                                    id="runs-scorecard-table",
                                    columns=[{"name": "segment", "id": "segment"}],
                                    data=[],
                                    style_cell_conditional=[
                                        {
                                            "if": {"column_id": "segment"},
                                            "textAlign": "left",
                                        }
                                    ],
                                    style_data_conditional=runs_scorecard_style(),
                                    page_size=5,
                                ),
                                className="mb-4",
                            ),
                            dcc.Graph(
                                id="runs-volume-chart",
                                config={"displayModeBar": True},
                                className="mt-2",
                            ),
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
                                "Kerosene has no national commercial stocks in SCI. "
                                "Stocks use mb when flows are kb/d, else kt.",
                                className="text-muted small mb-2",
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
                                "One total + city heatmap per product. "
                                "Heatmaps use monthly average stocks by city. "
                                "Dark gray cells = no data reported for that city/month (not zero). "
                                "Stocks use mb when the global unit is kb/d, else kt.",
                                className="text-muted small mt-2 mb-1",
                            ),
                            *[
                                item
                                for product in SHANDONG_STOCK_PRODUCTS
                                for item in (
                                    html.H5(
                                        product.replace("_", " ").title(),
                                        className="mt-3 mb-1",
                                    ),
                                    dcc.Graph(
                                        id=f"shandong-total-{product}",
                                        config={"displayModeBar": True},
                                    ),
                                    dcc.Graph(
                                        id=f"shandong-heatmap-{product}",
                                        config={"displayModeBar": True},
                                    ),
                                )
                            ],
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
                                                {"label": "Contribution margin", "value": "contribution"},
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
                                                {"label": "RMB/mt", "value": "rmb_mt"},
                                                {"label": "USD/bbl", "value": "usd_bbl"},
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
                            html.H5("Profit drivers (WoW)", className="mt-4 mb-1"),
                            html.P(
                                id="econ-driver-subtitle",
                                children=(
                                    "SCI profit-model bridge for the selected refinery type. "
                                    "Sales − feedstock = contribution; contribution − taxes/opex = net."
                                ),
                                className="text-muted small mb-2",
                            ),
                            dcc.Graph(
                                id="econ-waterfall-chart",
                                config={"displayModeBar": True},
                                className="mb-3",
                            ),
                            html.H6("P&L", className="mt-2 mb-1"),
                            html.Div(
                                _dark_datatable(
                                    id="econ-driver-pnl-table",
                                    columns=[{"name": "driver", "id": "driver"}],
                                    data=[],
                                    style_cell_conditional=[
                                        {"if": {"column_id": c}, "textAlign": "left"}
                                        for c in ("driver", "unit", "note")
                                    ],
                                    style_data_conditional=profit_driver_scorecard_style(),
                                    page_size=16,
                                ),
                                className="mb-3",
                            ),
                            html.H6("Prices & crude markers", className="mt-2 mb-1"),
                            html.P(
                                "All product prices in the selected model (RMB/mt), plus Brent/WTI. "
                                "Sorted by largest absolute WoW move.",
                                className="text-muted small mb-2",
                            ),
                            html.Div(
                                _dark_datatable(
                                    id="econ-driver-price-table",
                                    columns=[{"name": "driver", "id": "driver"}],
                                    data=[],
                                    style_cell_conditional=[
                                        {"if": {"column_id": c}, "textAlign": "left"}
                                        for c in ("driver", "unit", "note")
                                    ],
                                    style_data_conditional=profit_driver_scorecard_style(),
                                    page_size=25,
                                ),
                                className="mb-3",
                            ),
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
                            html.H5("Mobility depth", className="mt-4 mb-1"),
                            html.P(
                                "Sales vs possession shows the electrification lag. "
                                "Heavy trucks isolate the main diesel fleet. "
                                "Provincial map uses the latest passenger sales month.",
                                className="text-muted small mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Passenger: NEV sales vs fleet share (lag)",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-pax-sales-vs-fleet",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Trucks: sales vs fleet fuel shares",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-truck-sales-vs-fleet",
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
                                                "Heavy truck sales mix",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-heavy-sales-mix",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Heavy truck fleet mix",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-heavy-fleet-mix",
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
                                                "Provincial passenger NEV sales share (map)",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-provincial-nev-map",
                                                config={"displayModeBar": True},
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Top provinces by passenger NEV sales share",
                                                className="mb-1 text-light text-center",
                                            ),
                                            dcc.Graph(
                                                id="mobility-provincial-nev-bars",
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
                            html.H5("Exports", className="mt-3 mb-1"),
                            html.P(
                                "Latest-year monthly totals by product. Forward months without "
                                "shipping actuals use sheet-41 forecast cargoes (unconfirmed). "
                                "Follows the global unit toggle.",
                                className="text-muted small mb-2",
                            ),
                            _dark_datatable(
                                id="export-monthly-pivot-table",
                                columns=[{"name": "Exports (kt)", "id": "product"}],
                                data=[],
                                style_cell_conditional=[
                                    {"if": {"column_id": "product"}, "textAlign": "left"}
                                ],
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": '{product} = "Total"'},
                                        "fontWeight": "700",
                                    }
                                ],
                                page_size=10,
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
                            html.H5("Forward export forecast tracker", className="mt-3 mb-1"),
                            html.P(
                                "Sheet 41 weekly cargo forecasts (not the SCI company monthly export "
                                "plan, which is not in the API datapack). Compare forecasted loadings "
                                "vs MTD actuals and the same month last year.",
                                className="text-muted small mb-2",
                            ),
                            dcc.Graph(id="export-forecast-plan-chart", config={"displayModeBar": True}),
                            _dark_datatable(
                                id="export-forecast-summary-table",
                                columns=[
                                    {"name": "month", "id": "month"},
                                    {"name": "product", "id": "product"},
                                    {
                                        "name": "forecasted",
                                        "id": "forecasted",
                                        "type": "numeric",
                                        "format": {"specifier": ",.1f"},
                                    },
                                    {
                                        "name": "mtd_actual",
                                        "id": "mtd_actual",
                                        "type": "numeric",
                                        "format": {"specifier": ",.1f"},
                                    },
                                    {
                                        "name": "same_month_ly",
                                        "id": "same_month_ly",
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
                                        "name": "volume",
                                        "id": "volume",
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
                            dcc.Graph(
                                id="export-forecast-sankey-chart",
                                config={"displayModeBar": True},
                            ),
                            html.H5("Monthly exports", className="mt-3 mb-2"),
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
    Output("runs-scorecard-table", "columns"),
    Output("runs-scorecard-table", "data"),
    Output("runs-scorecard-subtitle", "children"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
)
def update_runs_charts(unit, date_range):
    unit = unit or "kbd"
    records, columns, week_lbl = runs_scorecard_records(
        FRAMES.get("runs_wide", pd.DataFrame()),
        unit=unit,
    )
    week_bit = f"Week ending {week_lbl}. " if week_lbl else ""
    subtitle = (
        f"{week_bit}Latest SCI weekly summary. Processing follows the global "
        "unit toggle; operating rates in %. National is the total."
    )
    return (
        fig_runs_volume(FRAMES["runs_wide"], unit, date_range),
        fig_runs_rate(FRAMES["runs_wide"], date_range),
        fig_balance_monthly(FRAMES.get("balance_monthly", pd.DataFrame()), unit, date_range),
        fig_soe_indep_spread(FRAMES["runs_wide"], date_range),
        columns,
        records,
        subtitle,
    )


@app.callback(
    Output("products-chart", "figure"),
    Output("products-change-table", "columns"),
    Output("products-change-table", "data"),
    Output("products-inv-delta-chart", "figure"),
    Output("products-output-delta-chart", "figure"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
)
def update_products(unit, date_range):
    wide = FRAMES.get("products_stocks_wide", pd.DataFrame())
    flow_unit = unit or "kbd"
    stock_unit = _inventory_unit(flow_unit)
    records, columns = products_scorecard_records(
        wide,
        stock_unit=stock_unit,
        flow_unit=flow_unit,
    )
    return (
        fig_products(wide, flow_unit, date_range),
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
    *[
        Output(f"shandong-total-{product}", "figure")
        for product in SHANDONG_STOCK_PRODUCTS
    ],
    *[
        Output(f"shandong-heatmap-{product}", "figure")
        for product in SHANDONG_STOCK_PRODUCTS
    ],
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
)
def update_shandong(unit, date_range):
    totals = FRAMES.get("shandong_stocks_total", pd.DataFrame())
    stocks = FRAMES.get("shandong_stocks", pd.DataFrame())
    unit = unit or "kbd"
    total_figs = [
        fig_shandong_totals(totals, product, date_range, unit=unit)
        for product in SHANDONG_STOCK_PRODUCTS
    ]
    heat_figs = [
        fig_shandong_heatmap(stocks, product, date_range, unit=unit)
        for product in SHANDONG_STOCK_PRODUCTS
    ]
    return (*total_figs, *heat_figs)


@app.callback(
    Output("econ-overlay-chart", "figure"),
    Output("econ-scatter-chart", "figure"),
    Output("econ-waterfall-chart", "figure"),
    Output("econ-driver-pnl-table", "columns"),
    Output("econ-driver-pnl-table", "data"),
    Output("econ-driver-price-table", "columns"),
    Output("econ-driver-price-table", "data"),
    Output("econ-driver-subtitle", "children"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("econ-refinery-type", "value"),
    Input("econ-margin-metric", "value"),
    Input("econ-display-mode", "value"),
    Input("econ-runs-breakdown", "value"),
)
def update_economics(unit, date_range, refinery_type, margin_metric, display_mode, runs_breakdown):
    margins = FRAMES.get("margins", pd.DataFrame())
    mode = display_mode or "pct"
    metric = margin_metric or "gross"
    refinery = refinery_type or "Fuel-oriented refineries"
    breakdown = runs_breakdown or "Independents"
    pnl_recs, pnl_cols, price_recs, price_cols, subtitle = profit_driver_scorecard(
        margins, refinery_type=refinery
    )
    return (
        fig_economics_overlay(
            FRAMES["runs_wide"],
            margins,
            unit,
            date_range,
            refinery,
            metric,
            breakdown,
            mode,
        ),
        fig_economics_scatter(
            FRAMES["runs_wide"],
            margins,
            date_range,
            refinery,
            metric,
            breakdown,
            mode,
        ),
        _apply_dark(fig_profit_waterfall(margins, refinery_type=refinery)),
        pnl_cols,
        pnl_recs,
        price_cols,
        price_recs,
        subtitle,
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
    Output("mobility-pax-sales-vs-fleet", "figure"),
    Output("mobility-truck-sales-vs-fleet", "figure"),
    Output("mobility-heavy-sales-mix", "figure"),
    Output("mobility-heavy-fleet-mix", "figure"),
    Output("mobility-provincial-nev-map", "figure"),
    Output("mobility-provincial-nev-bars", "figure"),
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
        _apply_dark(
            fig_passenger_sales_vs_possession(pax_sales, pax_poss), height=360
        ),
        _apply_dark(
            fig_truck_sales_vs_possession(com_sales, com_poss), height=360
        ),
        _apply_dark(fig_heavy_truck_sales_mix(com_sales), height=360),
        _apply_dark(fig_heavy_truck_possession_mix(com_poss), height=360),
        _apply_dark(fig_provincial_nev_map(pax_sales), height=420),
        _apply_dark(fig_provincial_nev_bars(pax_sales), height=420),
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
    Output("export-forecast-sankey-chart", "figure"),
    Output("export-monthly-pivot-table", "columns"),
    Output("export-monthly-pivot-table", "data"),
    Output("export-monthly-table", "data"),
    Input("unit-select", "value"),
    Input("date-range-select", "value"),
    Input("export-product", "value"),
    Input("export-period", "value"),
    Input("export-monthly-source", "value"),
)
def update_exports(unit, date_range, product, period, monthly_source):
    shipping = FRAMES.get("export_shipping", pd.DataFrame())
    forecast = FRAMES.get("export_forecast", pd.DataFrame())
    quotas_monthly = FRAMES.get("export_quotas_monthly", pd.DataFrame())
    quotas_summary = FRAMES.get("export_quotas_summary", pd.DataFrame())

    unit = unit or "kt"
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
                unit=unit,
            )
        ),
        _apply_dark(
            fig_export_destinations(
                shipping,
                product=product,
                date_range=date_range,
                period=period,
                unit=unit,
            )
        ),
        _apply_dark(
            fig_export_provinces(
                shipping,
                product=product,
                date_range=date_range,
                period=period,
                unit=unit,
            )
        ),
        _apply_dark(
            fig_export_sankey(
                shipping,
                product=product,
                date_range=date_range,
                period=period,
                unit=unit,
            ),
            height=520,
        ),
        _apply_dark(fig_export_quotas_monthly(quotas_monthly, unit=unit)),
        _apply_dark(fig_export_quota_summary(quotas_summary, quotas_monthly, unit=unit)),
        _apply_dark(
            fig_export_forecast_plan(forecast, shipping, product=product, unit=unit)
        ),
        export_forecast_summary_records(
            forecast, shipping, product=product, unit=unit
        ),
        export_forecast_table_records(forecast, product=product, unit=unit),
        _apply_dark(
            fig_export_forecast_sankey(forecast, product=product, unit=unit),
            height=520,
        ),
        export_monthly_pivot_columns(
            shipping,
            quotas_monthly,
            forecast,
            source=monthly_source,
            unit=unit,
        ),
        export_monthly_pivot_records(
            shipping,
            quotas_monthly,
            forecast,
            source=monthly_source,
            unit=unit,
        ),
        export_monthly_table_records(
            shipping,
            quotas_monthly,
            source=monthly_source,
            unit=unit,
        ),
    )


@app.callback(
    Output("export-monthly-download", "data"),
    Input("export-download-btn", "n_clicks"),
    Input("export-monthly-source", "value"),
    Input("unit-select", "value"),
    prevent_initial_call=True,
)
def download_export_monthly(n_clicks, monthly_source, unit):
    from src.exports import export_monthly_table_df

    df = export_monthly_table_df(
        FRAMES.get("export_shipping", pd.DataFrame()),
        FRAMES.get("export_quotas_monthly", pd.DataFrame()),
        source=monthly_source or "shipping",
        unit=unit or "kt",
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
