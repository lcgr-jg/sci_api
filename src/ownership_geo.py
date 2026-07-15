"""Provincial processing & operating rate vs refinery geography."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.china_geo import PROVINCE_CENTROIDS
from src.filters import date_range_label, filter_by_date_range
from src.infra_kmz import LAYER_COLORS, load_infra_sites
from src.units import monthly_kt_to_kbd, volume_label

CDU_UNIT = "FEEDSTOCK PRIMARY PROCESSING UNITS"

SCI_OWNERSHIP_ORDER = ["Sinopec", "PetroChina", "CNOOC", "Independent refinery"]

OWNERSHIP_COLORS = {
    "Sinopec": "#3fb950",
    "PetroChina": "#58a6ff",
    "CNOOC": "#bc8cff",
    "Independent refinery": "#f85149",
}


def _cdu_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["unit"] == CDU_UNIT].copy()


def _filter_ownership(df: pd.DataFrame, ownership: str) -> pd.DataFrame:
    if df.empty or ownership == "All":
        return df
    return df[df["ownership"] == ownership]


def _merge_volume_rate(
    processing: pd.DataFrame,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["month_start", "unit", "province", "ownership"]
    if processing.empty or rates.empty:
        return pd.DataFrame(columns=keys + ["processing_kt", "operating_rate"])
    return processing.merge(rates, on=keys, how="inner")


def province_period_processing(
    processing: pd.DataFrame,
    *,
    ownership: str,
    date_range: str,
    unit: str,
) -> pd.DataFrame:
    """Period summary of CDU processing by province."""
    df = _filter_ownership(_cdu_filter(processing), ownership)
    df = filter_by_date_range(df, "month_start", date_range)
    if df.empty:
        return pd.DataFrame(columns=["province", "value"])

    monthly = (
        df.groupby(["month_start", "province"], as_index=False)["processing_kt"]
        .sum()
        .rename(columns={"processing_kt": "value"})
    )
    if unit == "kbd":
        monthly["value"] = monthly_kt_to_kbd(monthly["value"], monthly["month_start"], "crude")
        out = monthly.groupby("province", as_index=False)["value"].mean()
    else:
        out = monthly.groupby("province", as_index=False)["value"].sum()
    return out.sort_values("value", ascending=False)


def province_period_operating_rate(
    processing: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    ownership: str,
    date_range: str,
) -> pd.DataFrame:
    """Volume-weighted CDU operating rate by province over the selected period."""
    proc = _filter_ownership(_cdu_filter(processing), ownership)
    rate = _filter_ownership(_cdu_filter(rates), ownership)
    proc = filter_by_date_range(proc, "month_start", date_range)
    rate = filter_by_date_range(rate, "month_start", date_range)
    merged = _merge_volume_rate(proc, rate)
    if merged.empty:
        return pd.DataFrame(columns=["province", "value"])

    merged["weighted"] = merged["operating_rate"] * merged["processing_kt"]
    monthly = merged.groupby(["month_start", "province"], as_index=False).agg(
        weighted=("weighted", "sum"),
        volume=("processing_kt", "sum"),
    )
    monthly["value"] = monthly["weighted"] / monthly["volume"].replace(0, np.nan)
    monthly = monthly.dropna(subset=["value"])
    out = monthly.groupby("province", as_index=False)["value"].mean()
    return out.sort_values("value", ascending=False)


def ownership_province_matrix(
    processing: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    metric: str,
    date_range: str,
    unit: str,
) -> pd.DataFrame:
    """Pivot table: province × ownership for heatmap."""
    proc = _cdu_filter(processing)
    rate = _cdu_filter(rates)
    proc = filter_by_date_range(proc, "month_start", date_range)
    rate = filter_by_date_range(rate, "month_start", date_range)
    if proc.empty:
        return pd.DataFrame()

    if metric == "processing":
        if unit == "kbd":
            proc = proc.copy()
            proc["processing_kt"] = monthly_kt_to_kbd(
                proc["processing_kt"], proc["month_start"], "crude"
            )
            agg = proc.groupby(["province", "ownership"], as_index=False)["processing_kt"].mean()
        else:
            agg = proc.groupby(["province", "ownership"], as_index=False)["processing_kt"].sum()
        val_col = "processing_kt"
    else:
        merged = _merge_volume_rate(proc, rate)
        if merged.empty:
            return pd.DataFrame()
        merged["weighted"] = merged["operating_rate"] * merged["processing_kt"]
        agg = merged.groupby(["province", "ownership"], as_index=False).agg(
            weighted=("weighted", "sum"),
            volume=("processing_kt", "sum"),
        )
        agg["operating_rate"] = agg["weighted"] / agg["volume"].replace(0, np.nan)
        val_col = "operating_rate"

    pivot = agg.pivot(index="province", columns="ownership", values=val_col)
    for col in SCI_OWNERSHIP_ORDER:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot[SCI_OWNERSHIP_ORDER].sort_index()


def _china_geo_layout(fig: go.Figure, *, title: str, height: int = 560) -> go.Figure:
    fig.update_geos(
        scope="asia",
        center=dict(lat=35.0, lon=105.0),
        projection_scale=3.8,
        showcountries=True,
        countrycolor="#30363d",
        showland=True,
        landcolor="#21262d",
        showocean=True,
        oceancolor="#161b22",
        showlakes=True,
        lakecolor="#161b22",
        bgcolor="#161b22",
        lataxis=dict(range=[17, 54]),
        lonaxis=dict(range=[73, 136]),
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(bgcolor="#161b22"),
    )
    return fig


def _add_infra_traces(fig: go.Figure, infra: pd.DataFrame) -> None:
    """Draw KMZ polygons/lines grouped by folder — all layers visible."""
    if infra.empty:
        return

    drawn_groups: set[str] = set()
    for kmz_group, group in infra.groupby("kmz_group"):
        lats: list[float | None] = []
        lons: list[float | None] = []
        for coords in group["coords"]:
            for lat, lon in coords:
                lats.append(lat)
                lons.append(lon)
            lats.append(None)
            lons.append(None)

        color = LAYER_COLORS.get(kmz_group, "#768390")
        show_legend = kmz_group not in drawn_groups
        drawn_groups.add(kmz_group)

        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=1.2, color=color),
                name=kmz_group,
                legendgroup=kmz_group,
                showlegend=show_legend,
                hoverinfo="skip",
                opacity=0.55 if kmz_group not in {"Pipeline", "Terminals", "Oilfield"} else 0.35,
            )
        )

    # Refinery centroids for hover detail.
    refineries = infra[infra["is_refinery"]].drop_duplicates(subset=["name", "kmz_group"])
    if not refineries.empty:
        hover = [
            f"<b>{row['name']}</b><br>{row['kmz_group']}"
            + (f"<br>{row['province']}" if pd.notna(row["province"]) else "")
            for _, row in refineries.iterrows()
        ]
        fig.add_trace(
            go.Scattergeo(
                lat=refineries["lat"],
                lon=refineries["lon"],
                mode="markers",
                marker=dict(size=5, color="#e6edf3", line=dict(width=0.5, color="#30363d")),
                text=hover,
                hoverinfo="text",
                name="Refinery sites",
                showlegend=True,
            )
        )


def fig_ownership_infra_map(
    processing: pd.DataFrame,
    rates: pd.DataFrame,
    infra: pd.DataFrame | None = None,
    *,
    metric: str,
    ownership: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """
    Map overlay: KMZ refinery footprints + province bubbles from SCI data.

    Bubble size/color reflects processing volume or operating rate for the filter.
    """
    if infra is None:
        infra = load_infra_sites()

    if metric == "processing":
        data = province_period_processing(
            processing, ownership=ownership, date_range=date_range, unit=unit
        )
        ylab = volume_label(unit)
        metric_label = "CDU processing"
        colorscale = "Blues"
        cmin = 0
    else:
        data = province_period_operating_rate(
            processing, rates, ownership=ownership, date_range=date_range
        )
        ylab = "Operating rate (%)"
        metric_label = "CDU operating rate"
        colorscale = "RdYlGn"
        cmin = 0

    fig = go.Figure()
    _add_infra_traces(fig, infra)

    if data.empty:
        return _china_geo_layout(
            fig,
            title=f"China refinery map — no SCI data ({date_range_label(date_range)})",
        )

    max_val = float(data["value"].max()) or 1.0
    lats, lons, sizes, colors, hover = [], [], [], [], []

    for _, row in data.iterrows():
        prov = row["province"]
        if prov not in PROVINCE_CENTROIDS:
            continue
        lat, lon = PROVINCE_CENTROIDS[prov]
        val = float(row["value"])
        display_val = val * 100 if metric == "operating_rate" else val
        lats.append(lat)
        lons.append(lon)
        sizes.append(max(14.0, 10.0 + 38.0 * (abs(val) / max_val) ** 0.5))
        colors.append(display_val)
        hover.append(
            f"<b>{prov}</b><br>{metric_label}: {display_val:,.1f} {ylab.replace('(%)', '').strip()}"
            + ("%" if metric == "operating_rate" else f" {ylab}")
        )

    if lats:
        cmax = max(colors) if colors else 100
        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                text=hover,
                hoverinfo="text",
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title=ylab),
                    line=dict(width=1.2, color="#e6edf3"),
                    opacity=0.92,
                ),
                name="SCI province data",
            )
        )

    owner_note = ownership if ownership != "All" else "all owners"
    period = date_range_label(date_range)
    return _china_geo_layout(
        fig,
        title=f"Refinery locations × SCI {metric_label} — {owner_note} ({period})",
    )


def fig_ownership_heatmap(
    processing: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    metric: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """Province × ownership matrix for processing volume or operating rate."""
    pivot = ownership_province_matrix(
        processing, rates, metric=metric, date_range=date_range, unit=unit
    )
    if pivot.empty:
        return go.Figure().update_layout(title="No data for heatmap")

    z = pivot.values
    if metric == "operating_rate":
        z = z * 100
        z_title = "Operating rate (%)"
        colorscale = "RdYlGn"
    else:
        z_title = volume_label(unit)
        colorscale = "Blues"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            hovertemplate="Province: %{y}<br>Owner: %{x}<br>Value: %{z:.1f}<extra></extra>",
            colorbar=dict(title=z_title),
        )
    )
    fig.update_layout(
        title=f"Province × ownership — {metric.replace('_', ' ')} ({date_range_label(date_range)})",
        xaxis_title="Ownership",
        yaxis_title="Province",
        height=max(420, 22 * len(pivot)),
    )
    return fig


def fig_ownership_province_bars(
    processing: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    province: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """Compare owners within one province — processing bars + operating rate line."""
    proc = filter_by_date_range(_cdu_filter(processing), "month_start", date_range)
    rate = filter_by_date_range(_cdu_filter(rates), "month_start", date_range)
    proc = proc[proc["province"] == province]
    rate = rate[rate["province"] == province]
    if proc.empty:
        return go.Figure().update_layout(title=f"No data for {province}")

    if unit == "kbd":
        proc = proc.copy()
        proc["processing_kt"] = monthly_kt_to_kbd(
            proc["processing_kt"], proc["month_start"], "crude"
        )
        vol = proc.groupby("ownership", as_index=False)["processing_kt"].mean()
        ylab = volume_label(unit)
    else:
        vol = proc.groupby("ownership", as_index=False)["processing_kt"].sum()
        ylab = volume_label(unit)

    merged = _merge_volume_rate(proc, rate)
    if not merged.empty:
        merged["weighted"] = merged["operating_rate"] * merged["processing_kt"]
        rate_agg = merged.groupby("ownership", as_index=False).agg(
            weighted=("weighted", "sum"),
            volume=("processing_kt", "sum"),
        )
        rate_agg["operating_rate"] = rate_agg["weighted"] / rate_agg["volume"].replace(0, np.nan)
    else:
        rate_agg = pd.DataFrame(columns=["ownership", "operating_rate"])

    owners = [o for o in SCI_OWNERSHIP_ORDER if o in vol["ownership"].values]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=owners,
            y=[float(vol.set_index("ownership").loc[o, "processing_kt"]) if o in vol["ownership"].values else 0 for o in owners],
            name="Processing",
            marker_color=[OWNERSHIP_COLORS.get(o, "#58a6ff") for o in owners],
        ),
        secondary_y=False,
    )
    if not rate_agg.empty:
        fig.add_trace(
            go.Scatter(
                x=owners,
                y=[
                    float(rate_agg.set_index("ownership").loc[o, "operating_rate"] * 100)
                    if o in rate_agg["ownership"].values
                    else None
                    for o in owners
                ],
                name="Operating rate",
                mode="lines+markers",
                line=dict(color="#e3b341", width=2),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"{province} — processing & operating rate by owner ({date_range_label(date_range)})",
        xaxis_title="Ownership",
        barmode="group",
    )
    fig.update_yaxes(title_text=ylab, secondary_y=False)
    fig.update_yaxes(title_text="Operating rate (%)", secondary_y=True, range=[0, 110])
    return fig


def fig_ownership_timeseries(
    processing: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    province: str,
    ownership: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """Monthly trends for one province (optionally filtered to one owner)."""
    proc = _filter_ownership(_cdu_filter(processing), ownership)
    rate = _filter_ownership(_cdu_filter(rates), ownership)
    proc = filter_by_date_range(proc, "month_start", date_range)
    rate = filter_by_date_range(rate, "month_start", date_range)
    proc = proc[proc["province"] == province]
    rate = rate[rate["province"] == province]
    if proc.empty:
        return go.Figure().update_layout(title=f"No time series for {province}")

    ylab = volume_label(unit)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if ownership == "All":
        for owner in SCI_OWNERSHIP_ORDER:
            sub = proc[proc["ownership"] == owner].sort_values("month_start")
            if sub.empty:
                continue
            y = sub["processing_kt"]
            if unit == "kbd":
                y = monthly_kt_to_kbd(y, sub["month_start"], "crude")
            fig.add_trace(
                go.Scatter(
                    x=sub["month_start"],
                    y=y,
                    name=f"{owner} vol",
                    mode="lines+markers",
                    line=dict(color=OWNERSHIP_COLORS.get(owner, "#58a6ff")),
                ),
                secondary_y=False,
            )
        merged = _merge_volume_rate(proc, rate)
        if not merged.empty:
            merged["weighted"] = merged["operating_rate"] * merged["processing_kt"]
            monthly_rate = merged.groupby("month_start", as_index=False).agg(
                weighted=("weighted", "sum"),
                volume=("processing_kt", "sum"),
            )
            monthly_rate["rate"] = monthly_rate["weighted"] / monthly_rate["volume"].replace(0, np.nan)
            fig.add_trace(
                go.Scatter(
                    x=monthly_rate["month_start"],
                    y=monthly_rate["rate"] * 100,
                    name="Weighted avg rate",
                    mode="lines",
                    line=dict(color="#e3b341", width=2, dash="dot"),
                ),
                secondary_y=True,
            )
    else:
        sub = proc.sort_values("month_start")
        y = sub["processing_kt"]
        if unit == "kbd":
            y = monthly_kt_to_kbd(y, sub["month_start"], "crude")
        fig.add_trace(
            go.Scatter(
                x=sub["month_start"],
                y=y,
                name="Processing",
                mode="lines+markers",
                line=dict(color=OWNERSHIP_COLORS.get(ownership, "#58a6ff")),
            ),
            secondary_y=False,
        )
        rsub = rate.sort_values("month_start")
        if not rsub.empty:
            fig.add_trace(
                go.Scatter(
                    x=rsub["month_start"],
                    y=rsub["operating_rate"] * 100,
                    name="Operating rate",
                    mode="lines+markers",
                    line=dict(color="#e3b341", width=2),
                ),
                secondary_y=True,
            )

    owner_note = ownership if ownership != "All" else "all owners"
    fig.update_layout(
        title=f"{province} — monthly CDU processing & rate ({owner_note})",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=ylab, secondary_y=False)
    fig.update_yaxes(title_text="Operating rate (%)", secondary_y=True, range=[0, 110])
    return fig
