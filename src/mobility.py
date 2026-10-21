"""Vehicle sales / possession overlays vs gasoline and diesel demand (SCI sheets 26–30)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.china_geo import PROVINCE_CENTROIDS
from src.crude_shipping import _apply_top_legend_domain, _empty_fig, _top_legend_layout
from src.transforms import DIESEL_TRUCK_TYPES
from src.units import convert_volume_series, volume_label

HEAVY_TRUCK_TYPES = ("Heavy truck",)

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


def _national_truck_sales(
    commercial: pd.DataFrame,
    vehicle_types: tuple[str, ...] = DIESEL_TRUCK_TYPES,
) -> pd.DataFrame:
    if commercial.empty:
        return commercial
    sub = commercial[commercial["vehicle_type"].isin(vehicle_types)]
    if sub.empty:
        return sub
    num = [c for c in ("diesel", "gasoline", "cng", "lng", "new_energy", "others", "total") if c in sub.columns]
    out = sub.groupby("month_start", as_index=False)[num].sum(min_count=1)
    tot = out["total"].replace(0, pd.NA)
    out["diesel_share"] = out["diesel"] / tot * 100.0
    out["nev_share"] = out["new_energy"] / tot * 100.0
    out["lng_share"] = out["lng"] / tot * 100.0 if "lng" in out.columns else pd.NA
    return out.sort_values("month_start")


def _national_truck_possession(
    possession: pd.DataFrame,
    vehicle_types: tuple[str, ...] = DIESEL_TRUCK_TYPES,
) -> pd.DataFrame:
    if possession.empty:
        return possession
    sub = possession[possession["vehicle_type"].isin(vehicle_types)]
    if sub.empty:
        return sub
    num = [c for c in ("diesel", "gasoline", "cng", "lng", "new_energy", "others", "total") if c in sub.columns]
    out = sub.groupby("month_start", as_index=False)[num].sum(min_count=1)
    tot = out["total"].replace(0, pd.NA)
    out["diesel_share"] = out["diesel"] / tot * 100.0
    out["nev_share"] = out["new_energy"] / tot * 100.0
    out["lng_share"] = out["lng"] / tot * 100.0 if "lng" in out.columns else pd.NA
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


# ---------------------------------------------------------------------------
# Mobility depth — sales vs stock lag, heavy-truck focus, provincial NEV
# ---------------------------------------------------------------------------


def fig_passenger_sales_vs_possession(
    passenger_sales: pd.DataFrame,
    passenger_possession: pd.DataFrame,
) -> go.Figure:
    """Passenger NEV share: monthly sales vs annual fleet (step) — electrification lag."""
    sales = _national_passenger_sales(passenger_sales)
    fleet = _national_passenger_possession(passenger_possession)
    if sales.empty and fleet.empty:
        return _empty_fig("No passenger sales / possession data")

    fig = go.Figure()
    if not sales.empty:
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=sales["nev_share"],
                name="NEV sales share %",
                mode="lines+markers",
                line=dict(color="#3fb950", width=2.5),
                marker=dict(size=5),
            )
        )
        ice = sales["gasoline"] / sales["total"].replace(0, pd.NA) * 100.0
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=ice,
                name="Gasoline ICE sales share %",
                mode="lines",
                line=dict(color="#58a6ff", width=1.5, dash="dot"),
            )
        )
    if not fleet.empty:
        fig.add_trace(
            go.Scatter(
                x=fleet["month_start"],
                y=fleet["nev_share"],
                name="NEV fleet share % (year-end)",
                mode="lines+markers",
                line=dict(color="#e3b341", width=2.5, shape="hv"),
                marker=dict(size=8, symbol="diamond"),
            )
        )
        gas_fleet = fleet["gasoline"] / fleet["total"].replace(0, pd.NA) * 100.0
        fig.add_trace(
            go.Scatter(
                x=fleet["month_start"],
                y=gas_fleet,
                name="Gasoline ICE fleet share % (year-end)",
                mode="lines+markers",
                line=dict(color="#79c0ff", width=2, shape="hv", dash="dash"),
                marker=dict(size=7, symbol="diamond"),
            )
        )
    fig.update_layout(
        **_top_legend_layout(
            yaxis_title="Share %",
            yaxis=dict(range=[0, 100]),
        )
    )
    return _apply_top_legend_domain(fig)


def fig_truck_sales_vs_possession(
    commercial_sales: pd.DataFrame,
    commercial_possession: pd.DataFrame,
) -> go.Figure:
    """Truck diesel / NEV / LNG: sales shares vs fleet shares on the same axis."""
    sales = _national_truck_sales(commercial_sales)
    fleet = _national_truck_possession(commercial_possession)
    if sales.empty and fleet.empty:
        return _empty_fig("No truck sales / possession data")

    fig = go.Figure()
    if not sales.empty:
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=sales["diesel_share"],
                name="Diesel sales %",
                mode="lines+markers",
                line=dict(color="#f85149", width=2),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sales["month_start"],
                y=sales["nev_share"],
                name="NEV sales %",
                mode="lines+markers",
                line=dict(color="#3fb950", width=2),
                marker=dict(size=4),
            )
        )
        if "lng_share" in sales.columns:
            fig.add_trace(
                go.Scatter(
                    x=sales["month_start"],
                    y=sales["lng_share"],
                    name="LNG sales %",
                    mode="lines",
                    line=dict(color="#e3b341", width=1.5, dash="dot"),
                )
            )
    if not fleet.empty:
        fig.add_trace(
            go.Scatter(
                x=fleet["month_start"],
                y=fleet["diesel_share"],
                name="Diesel fleet %",
                mode="lines+markers",
                line=dict(color="#ffa657", width=2.5, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fleet["month_start"],
                y=fleet["nev_share"],
                name="NEV fleet %",
                mode="lines+markers",
                line=dict(color="#7ee787", width=2.5, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            )
        )
    fig.update_layout(
        **_top_legend_layout(
            yaxis_title="Share of trucks %",
            yaxis=dict(range=[0, 100]),
        )
    )
    return _apply_top_legend_domain(fig)


def _truck_fuel_stack_fig(
    frame: pd.DataFrame,
    *,
    empty_msg: str,
    y_title: str,
) -> go.Figure:
    if frame.empty:
        return _empty_fig(empty_msg)
    stack_cols = [c for c, _, _ in _TRUCK_STACK if c in frame.columns]
    base = frame[stack_cols].fillna(0).sum(axis=1).replace(0, pd.NA)
    fig = go.Figure()
    for col, label, color in _TRUCK_STACK:
        if col not in frame.columns:
            continue
        share = frame[col].fillna(0) / base * 100.0
        fig.add_trace(
            go.Bar(
                x=frame["month_start"],
                y=share,
                name=label,
                marker_color=color,
            )
        )
    fig.update_layout(
        **_top_legend_layout(
            barmode="stack",
            yaxis_title=y_title,
            yaxis=dict(range=[0, 100]),
        )
    )
    return _apply_top_legend_domain(fig)


def fig_heavy_truck_sales_mix(commercial_sales: pd.DataFrame) -> go.Figure:
    """Heavy truck only — sales fuel mix (diesel / LNG / NEV substitution)."""
    return _truck_fuel_stack_fig(
        _national_truck_sales(commercial_sales, HEAVY_TRUCK_TYPES),
        empty_msg="No heavy truck sales data",
        y_title="Share of heavy truck sales %",
    )


def fig_heavy_truck_possession_mix(commercial_possession: pd.DataFrame) -> go.Figure:
    """Heavy truck only — fleet fuel mix."""
    return _truck_fuel_stack_fig(
        _national_truck_possession(commercial_possession, HEAVY_TRUCK_TYPES),
        empty_msg="No heavy truck possession data",
        y_title="Share of heavy truck fleet %",
    )


def _provincial_passenger_nev_latest(passenger_sales: pd.DataFrame) -> pd.DataFrame:
    """Latest-month passenger NEV share by province."""
    if passenger_sales.empty:
        return pd.DataFrame(columns=["province", "month_start", "nev_share", "total", "nev"])
    latest = passenger_sales["month_start"].max()
    sub = passenger_sales[passenger_sales["month_start"] == latest].copy()
    sub["nev"] = (
        sub.get("bev", 0).fillna(0)
        + sub.get("phev", 0).fillna(0)
        + sub.get("erev", 0).fillna(0)
        + sub.get("non_plugin_erev", 0).fillna(0)
    )
    g = sub.groupby("province", as_index=False).agg(total=("total", "sum"), nev=("nev", "sum"))
    g["nev_share"] = g["nev"] / g["total"].replace(0, pd.NA) * 100.0
    g["month_start"] = latest
    return g.dropna(subset=["nev_share"]).sort_values("nev_share", ascending=False)


def fig_provincial_nev_map(passenger_sales: pd.DataFrame) -> go.Figure:
    """Latest-month passenger NEV sales share by province (bubble map)."""
    data = _provincial_passenger_nev_latest(passenger_sales)
    if data.empty:
        return _empty_fig("No provincial passenger NEV data")

    month_label = pd.Timestamp(data["month_start"].iloc[0]).strftime("%b %Y")
    lats, lons, sizes, colors, hover = [], [], [], [], []
    for _, row in data.iterrows():
        prov = str(row["province"])
        if prov not in PROVINCE_CENTROIDS:
            continue
        lat, lon = PROVINCE_CENTROIDS[prov]
        share = float(row["nev_share"])
        lats.append(lat)
        lons.append(lon)
        sizes.append(max(11.0, 10.0 + share * 0.35))
        colors.append(share)
        hover.append(
            f"<b>{prov}</b><br>NEV sales share: {share:.1f}%<br>"
            f"NEV units: {row['nev']:,.0f} / total {row['total']:,.0f}"
        )

    if not lats:
        return _empty_fig("No mappable provinces for NEV share")

    fig = go.Figure(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=hover,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="YlGn",
                cmin=0,
                cmax=100,
                colorbar=dict(title="NEV %"),
                line=dict(width=0.8, color="#30363d"),
                opacity=0.9,
            ),
        )
    )
    fig.update_geos(
        scope="asia",
        center=dict(lat=35.0, lon=105.0),
        projection_scale=2.8,
        showland=True,
        landcolor="#21262d",
        showocean=True,
        oceancolor="#0d1117",
        showcountries=True,
        countrycolor="#30363d",
        showframe=False,
        bgcolor="#161b22",
    )
    fig.update_layout(
        **_top_legend_layout(
            margin=dict(l=10, r=10, t=24, b=10),
            annotations=[
                dict(
                    text=f"Passenger NEV sales share — {month_label}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.02,
                    showarrow=False,
                    font=dict(size=11, color="#8b949e"),
                )
            ],
        )
    )
    return fig


def fig_provincial_nev_bars(passenger_sales: pd.DataFrame, *, top_n: int = 15) -> go.Figure:
    """Top provinces by passenger NEV sales share (latest month)."""
    data = _provincial_passenger_nev_latest(passenger_sales)
    if data.empty:
        return _empty_fig("No provincial passenger NEV data")

    month_label = pd.Timestamp(data["month_start"].iloc[0]).strftime("%b %Y")
    top = data.head(top_n).sort_values("nev_share")
    fig = go.Figure(
        go.Bar(
            x=top["nev_share"],
            y=top["province"],
            orientation="h",
            marker_color="#3fb950",
            text=[f"{v:.0f}%" for v in top["nev_share"]],
            textposition="outside",
            hovertemplate="%{y}<br>NEV share: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_top_legend_layout(
            xaxis_title=f"Passenger NEV sales share % ({month_label})",
            xaxis=dict(range=[0, 105]),
            yaxis_title=None,
            margin=dict(l=90, r=40, t=24, b=40),
        )
    )
    return _apply_top_legend_domain(fig)
