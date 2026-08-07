"""Vehicle sales / possession overlays vs gasoline and diesel demand (SCI sheets 26–30)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.crude_shipping import _apply_top_legend_domain, _empty_fig, _top_legend_layout
from src.transforms import DIESEL_TRUCK_TYPES
from src.units import convert_volume_series, volume_label

_PASSENGER_STACK = [
    ("gasoline", "Gasoline ICE", "#58a6ff"),
    ("hev", "HEV (non-plugin)", "#79c0ff"),
    ("phev", "PHEV", "#bc8cff"),
    ("erev", "EREV", "#d2a8ff"),
    ("bev", "BEV", "#3fb950"),
    ("diesel", "Diesel", "#8b949e"),
]

_TRUCK_STACK = [
    ("diesel", "Diesel", "#f85149"),
    ("lng", "LNG", "#e3b341"),
    ("new_energy", "New energy", "#3fb950"),
    ("gasoline", "Gasoline", "#58a6ff"),
]


def _national_passenger_sales(sales: pd.DataFrame) -> pd.DataFrame:
    if sales.empty:
        return sales
    num = [
        c
        for c in (
            "gasoline",
            "diesel",
            "bev",
            "phev",
            "hev",
            "erev",
            "non_plugin_erev",
            "cng",
            "others",
            "total",
        )
        if c in sales.columns
    ]
    out = sales.groupby("month_start", as_index=False)[num].sum(min_count=1)
    out["nev"] = (
        out.get("bev", 0).fillna(0)
        + out.get("phev", 0).fillna(0)
        + out.get("erev", 0).fillna(0)
        + out.get("non_plugin_erev", 0).fillna(0)
    )
    tot = out["total"].replace(0, pd.NA)
    out["nev_share"] = out["nev"] / tot * 100.0
    return out.sort_values("month_start")


def _national_truck_sales(commercial: pd.DataFrame) -> pd.DataFrame:
    if commercial.empty:
        return commercial
    sub = commercial[commercial["vehicle_type"].isin(DIESEL_TRUCK_TYPES)]
    if sub.empty:
        return sub
    num = [c for c in ("diesel", "gasoline", "cng", "lng", "new_energy", "others", "total") if c in sub.columns]
    out = sub.groupby("month_start", as_index=False)[num].sum(min_count=1)
    tot = out["total"].replace(0, pd.NA)
    out["diesel_share"] = out["diesel"] / tot * 100.0
    out["nev_share"] = out["new_energy"] / tot * 100.0
    return out.sort_values("month_start")


def _national_truck_possession(possession: pd.DataFrame) -> pd.DataFrame:
    if possession.empty:
        return possession
    sub = possession[possession["vehicle_type"].isin(DIESEL_TRUCK_TYPES)]
    if sub.empty:
        return sub
    num = [c for c in ("diesel", "gasoline", "cng", "lng", "new_energy", "others", "total") if c in sub.columns]
    out = sub.groupby("month_start", as_index=False)[num].sum(min_count=1)
    tot = out["total"].replace(0, pd.NA)
    out["diesel_share"] = out["diesel"] / tot * 100.0
    out["nev_share"] = out["new_energy"] / tot * 100.0
    return out.sort_values("month_start")


def _national_passenger_possession(possession: pd.DataFrame) -> pd.DataFrame:
    if possession.empty:
        return possession
    num = [
        c
        for c in (
            "gasoline",
            "diesel",
            "bev",
            "phev",
            "hev",
            "erev",
            "non_plugin_erev",
            "cng",
            "others",
            "total",
        )
        if c in possession.columns
    ]
    out = possession.groupby("year", as_index=False)[num].sum(min_count=1)
    out["nev"] = (
        out.get("bev", 0).fillna(0)
        + out.get("phev", 0).fillna(0)
        + out.get("erev", 0).fillna(0)
        + out.get("non_plugin_erev", 0).fillna(0)
    )
    tot = out["total"].replace(0, pd.NA)
    out["nev_share"] = out["nev"] / tot * 100.0
    out["month_start"] = pd.to_datetime(dict(year=out["year"], month=12, day=31))
    return out.sort_values("year")


def _national_fuel_demand(consumption_national: pd.DataFrame, product: str) -> pd.DataFrame:
    if consumption_national.empty:
        return pd.DataFrame(columns=["month_start", "consumption_kt"])
    sub = consumption_national[consumption_national["product"] == product].copy()
    if sub.empty:
        return pd.DataFrame(columns=["month_start", "consumption_kt"])
    sub["month_start"] = pd.to_datetime(
        dict(year=sub["year"], month=sub["month"], day=1)
    )
    return (
        sub.groupby("month_start", as_index=False)["consumption_kt"]
        .sum()
        .sort_values("month_start")
    )


def fig_gasoline_vs_nev(
    consumption_national: pd.DataFrame,
    passenger_sales: pd.DataFrame,
    *,
    unit: str,
) -> go.Figure:
    """Gasoline demand (left) vs passenger NEV / gasoline ICE sales share (right)."""
    demand = _national_fuel_demand(consumption_national, "Gasoline")
    sales = _national_passenger_sales(passenger_sales)
    if demand.empty and sales.empty:
        return _empty_fig("No gasoline / passenger vehicle data")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ylab = volume_label(unit)

    if not demand.empty:
        y = convert_volume_series(
            demand["consumption_kt"],
            unit,
            freq="monthly",
            dates=demand["month_start"],
            product="gasoline",
        )
        fig.add_trace(
            go.Scatter(
                x=demand["month_start"],
                y=y,
                name=f"Gasoline demand ({ylab})",
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2.5),
                marker=dict(size=5),
            ),
            secondary_y=False,
        )

    if not sales.empty:
        ice_share = sales["gasoline"] / sales["total"].replace(0, pd.NA) * 100.0
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=sales["nev_share"],
                name="Passenger NEV sales share %",
                mode="lines+markers",
                line=dict(color="#3fb950", width=2),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=ice_share,
                name="Passenger gasoline ICE sales share %",
                mode="lines+markers",
                line=dict(color="#79c0ff", width=2, dash="dot"),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

    fig.update_layout(**_top_legend_layout())
    fig.update_yaxes(title_text=ylab, secondary_y=False)
    fig.update_yaxes(title_text="Sales share %", secondary_y=True, range=[0, 100])
    return _apply_top_legend_domain(fig)


def fig_passenger_sales_mix(passenger_sales: pd.DataFrame) -> go.Figure:
    """National passenger sales mix by powertrain (stacked %)."""
    sales = _national_passenger_sales(passenger_sales)
    if sales.empty:
        return _empty_fig("No passenger vehicle sales")

    fig = go.Figure()
    for col, label, color in _PASSENGER_STACK:
        if col not in sales.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=sales[col].fillna(0),
                name=label,
                mode="lines",
                stackgroup="one",
                groupnorm="percent",
                line=dict(width=0.5, color=color),
                fillcolor=color,
            )
        )
    fig.update_layout(
        **_top_legend_layout(
            yaxis_title="Share of passenger sales %",
            yaxis=dict(range=[0, 100]),
        )
    )
    return _apply_top_legend_domain(fig)


def fig_diesel_vs_trucks(
    consumption_national: pd.DataFrame,
    commercial_sales: pd.DataFrame,
    commercial_possession: pd.DataFrame,
    *,
    unit: str,
) -> go.Figure:
    """Diesel demand (left) vs truck diesel / NEV sales shares (right)."""
    demand = _national_fuel_demand(consumption_national, "Diesel")
    sales = _national_truck_sales(commercial_sales)
    # Possession used in dedicated fleet chart; kept in signature for future overlays.
    _ = commercial_possession
    if demand.empty and sales.empty:
        return _empty_fig("No diesel / truck vehicle data")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ylab = volume_label(unit)

    if not demand.empty:
        y = convert_volume_series(
            demand["consumption_kt"],
            unit,
            freq="monthly",
            dates=demand["month_start"],
            product="gasoil",
        )
        fig.add_trace(
            go.Scatter(
                x=demand["month_start"],
                y=y,
                name=f"Diesel demand ({ylab})",
                mode="lines+markers",
                line=dict(color="#f85149", width=2.5),
                marker=dict(size=5),
            ),
            secondary_y=False,
        )

    if not sales.empty:
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=sales["diesel_share"],
                name="Truck diesel sales share %",
                mode="lines+markers",
                line=dict(color="#ffa657", width=2),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=sales["nev_share"],
                name="Truck NEV sales share %",
                mode="lines+markers",
                line=dict(color="#3fb950", width=2, dash="dot"),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

    fig.update_layout(**_top_legend_layout())
    fig.update_yaxes(title_text=ylab, secondary_y=False)
    fig.update_yaxes(title_text="Truck sales share %", secondary_y=True, range=[0, 100])
    return _apply_top_legend_domain(fig)


def fig_truck_sales_mix(commercial_sales: pd.DataFrame) -> go.Figure:
    """Heavy/medium/light/pickup truck sales mix by fuel (stacked %)."""
    sales = _national_truck_sales(commercial_sales)
    if sales.empty:
        return _empty_fig("No truck sales data")

    stack_cols = [c for c, _, _ in _TRUCK_STACK if c in sales.columns]
    base = sales[stack_cols].fillna(0).sum(axis=1).replace(0, pd.NA)
    fig = go.Figure()
    for col, label, color in _TRUCK_STACK:
        if col not in sales.columns:
            continue
        share = sales[col].fillna(0) / base * 100.0
        fig.add_trace(
            go.Bar(
                x=sales["month_start"],
                y=share,
                name=label,
                marker_color=color,
            )
        )
    fig.update_layout(
        **_top_legend_layout(
            barmode="stack",
            yaxis_title="Share of truck sales %",
            yaxis=dict(range=[0, 100]),
        )
    )
    return _apply_top_legend_domain(fig)


def fig_truck_possession_mix(commercial_possession: pd.DataFrame) -> go.Figure:
    """Truck fleet possession mix by fuel (stacked %) — stock turnover vs sales mix."""
    fleet = _national_truck_possession(commercial_possession)
    if fleet.empty:
        return _empty_fig("No truck possession data")

    stack_cols = [c for c, _, _ in _TRUCK_STACK if c in fleet.columns]
    base = fleet[stack_cols].fillna(0).sum(axis=1).replace(0, pd.NA)
    fig = go.Figure()
    for col, label, color in _TRUCK_STACK:
        if col not in fleet.columns:
            continue
        share = fleet[col].fillna(0) / base * 100.0
        fig.add_trace(
            go.Scatter(
                x=fleet["month_start"],
                y=share,
                name=label,
                mode="lines",
                stackgroup="one",
                groupnorm="percent",
                line=dict(width=0.5, color=color),
                fillcolor=color,
            )
        )
    fig.update_layout(
        **_top_legend_layout(
            yaxis_title="Share of truck fleet %",
            yaxis=dict(range=[0, 100]),
        )
    )
    return _apply_top_legend_domain(fig)


def fig_passenger_possession(passenger_possession: pd.DataFrame) -> go.Figure:
    """Annual passenger fleet mix by powertrain (stacked %) — stock vs sales shift."""
    fleet = _national_passenger_possession(passenger_possession)
    if fleet.empty:
        return _empty_fig("No passenger possession data")

    stack_cols = [c for c, _, _ in _PASSENGER_STACK if c in fleet.columns]
    base = fleet[stack_cols].fillna(0).sum(axis=1).replace(0, pd.NA)
    x = fleet["year"].astype(str)
    fig = go.Figure()
    for col, label, color in _PASSENGER_STACK:
        if col not in fleet.columns:
            continue
        share = fleet[col].fillna(0) / base * 100.0
        fig.add_trace(
            go.Bar(
                x=x,
                y=share,
                name=label,
                marker_color=color,
            )
        )
    fig.update_layout(
        **_top_legend_layout(
            barmode="stack",
            yaxis_title="Share of passenger fleet %",
            yaxis=dict(range=[0, 100]),
            xaxis_title="Year",
        )
    )
    return _apply_top_legend_domain(fig)
