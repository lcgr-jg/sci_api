"""Regional / provincial analytics — runs, output, and consumption."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.filters import filter_by_date_range
from src.units import CONSUMPTION_PRODUCT_MAP, monthly_kt_to_kbd, volume_label
from src.china_geo import PROVINCE_CENTROIDS

REGION_LABELS = {
    "E-China": "East China",
    "N-China": "North China",
    "NE-China": "Northeast",
    "NW-China": "Northwest",
    "S-China": "South China",
    "SW-China": "Southwest",
    "Central China": "Central China",
}

REGION_COLORS = {
    "East China": "#58a6ff",
    "North China": "#3fb950",
    "Northeast": "#bc8cff",
    "Northwest": "#ffa657",
    "South China": "#f85149",
    "Southwest": "#e3b341",
    "Central China": "#79c0ff",
}


def province_region_map(consumption: pd.DataFrame) -> pd.DataFrame:
    """Province → SCI region lookup from consumption sheet."""
    if consumption.empty:
        return pd.DataFrame(columns=["province", "region"])
    return (
        consumption[["province", "region"]]
        .drop_duplicates()
        .sort_values("province")
        .reset_index(drop=True)
    )


def _label_region(region: str) -> str:
    return REGION_LABELS.get(region, region)


def attach_region(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if df.empty or mapping.empty:
        out = df.copy()
        out["region"] = pd.NA
        return out
    return df.merge(mapping, on="province", how="left")


def aggregate_runs(
    processing: pd.DataFrame,
    mapping: pd.DataFrame,
    geo: str,
) -> pd.DataFrame:
    df = attach_region(processing, mapping)
    df = df.dropna(subset=["processing_kt"])
    if geo == "province":
        group = ["month_start", "province"]
        label_col = "province"
    else:
        group = ["month_start", "region"]
        label_col = "region"
        df = df.dropna(subset=["region"])
    out = df.groupby(group, as_index=False)["processing_kt"].sum()
    out = out.rename(columns={"processing_kt": "value", label_col: "geo"})
    if geo == "region":
        out["geo"] = out["geo"].map(_label_region)
    return out.sort_values(["month_start", "geo"])


def aggregate_output(
    output: pd.DataFrame,
    mapping: pd.DataFrame,
    product: str,
    geo: str,
) -> pd.DataFrame:
    df = output[output["product"] == product].copy()
    df = attach_region(df, mapping)
    if geo == "province":
        group = ["month_start", "province"]
        label_col = "province"
    else:
        group = ["month_start", "region"]
        label_col = "region"
        df = df.dropna(subset=["region"])
    out = df.groupby(group, as_index=False)["output_kt"].sum()
    out = out.rename(columns={"output_kt": "value", label_col: "geo"})
    if geo == "region":
        out["geo"] = out["geo"].map(_label_region)
    return out.sort_values(["month_start", "geo"])


def aggregate_consumption(
    consumption: pd.DataFrame,
    product: str,
    geo: str,
) -> pd.DataFrame:
    df = consumption[consumption["product"] == product].copy()
    if df.empty:
        return pd.DataFrame(columns=["month_start", "geo", "value"])
    df["month_start"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    if geo == "province":
        group = ["month_start", "province"]
        label_col = "province"
    else:
        group = ["month_start", "region"]
        label_col = "region"
    out = df.groupby(group, as_index=False)["consumption_kt"].sum()
    out = out.rename(columns={"consumption_kt": "value", label_col: "geo"})
    if geo == "region":
        out["geo"] = out["geo"].map(_label_region)
    return out.sort_values(["month_start", "geo"])


def _product_key(product: str, *, for_runs: bool = False) -> str:
    if for_runs:
        return "crude"
    return CONSUMPTION_PRODUCT_MAP.get(product, product.lower())


def apply_display_unit(
    df: pd.DataFrame,
    unit: str,
    product_key: str,
) -> pd.DataFrame:
    """Convert monthly kt values to kb/d when requested."""
    if unit != "kbd" or df.empty:
        return df
    out = df.copy()
    out["value"] = monthly_kt_to_kbd(out["value"], out["month_start"], product_key)
    return out


def _period_total(df: pd.DataFrame) -> pd.DataFrame:
    """Sum over the filtered window (total kt in period)."""
    if df.empty:
        return df
    return (
        df.groupby("geo", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
    )


def _period_aggregate(
    df: pd.DataFrame,
    unit: str,
    product_key: str,
    date_range: str,
) -> pd.DataFrame:
    """
    Period summary by geo.

    kt → sum over window; kb/d → mean daily rate (avg of monthly kb/d in window).
    """
    filtered = filter_by_date_range(df, "month_start", date_range)
    if filtered.empty:
        return filtered
    if unit == "kbd":
        converted = apply_display_unit(filtered, unit, product_key)
        return (
            converted.groupby("geo", as_index=False)["value"]
            .mean()
            .sort_values("value", ascending=False)
        )
    return _period_total(filtered)


def _period_label(unit: str) -> str:
    if unit == "kbd":
        return "avg kb/d (mean over period)"
    return "kt (sum over period)"


def _national_monthly_total(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum all geographies to a single national series by month."""
    if frame.empty:
        return frame
    out = frame.groupby("month_start", as_index=False)["value"].sum()
    out["geo"] = "China (national)"
    return out.sort_values("month_start")


def _national_monthly_balance(
    output: pd.DataFrame,
    consumption: pd.DataFrame,
) -> pd.DataFrame:
    """National output minus consumption by month (sum geos first, then subtract)."""
    if output.empty and consumption.empty:
        return pd.DataFrame(columns=["month_start", "geo", "value"])
    out_n = output.groupby("month_start", as_index=False)["value"].sum()
    cons_n = consumption.groupby("month_start", as_index=False)["value"].sum()
    merged = out_n.rename(columns={"value": "output"}).merge(
        cons_n.rename(columns={"value": "consumption"}),
        on="month_start",
        how="outer",
    ).fillna(0)
    merged["value"] = merged["output"] - merged["consumption"]
    merged["geo"] = "China (national)"
    return merged[["month_start", "geo", "value"]].sort_values("month_start")


def fig_regional_timeseries(
    runs: pd.DataFrame,
    output: pd.DataFrame,
    consumption: pd.DataFrame,
    *,
    product: str,
    geo: str,
    date_range: str,
    unit: str = "kt",
    timeseries_view: str = "regional",
) -> go.Figure:
    """Four-panel lines: processing, output, consumption, and output minus consumption."""
    ylab = volume_label(unit)
    national = timeseries_view == "national"
    runs = apply_display_unit(
        filter_by_date_range(runs, "month_start", date_range), unit, _product_key(product, for_runs=True)
    )
    output = apply_display_unit(
        filter_by_date_range(output, "month_start", date_range), unit, _product_key(product)
    )
    consumption = apply_display_unit(
        filter_by_date_range(consumption, "month_start", date_range), unit, _product_key(product)
    )

    balance = (
        output.rename(columns={"value": "output"})
        .merge(
            consumption.rename(columns={"value": "consumption"}),
            on=["month_start", "geo"],
            how="outer",
        )
        .fillna(0)
    )
    balance["value"] = balance["output"] - balance["consumption"]
    balance = balance[["month_start", "geo", "value"]]

    if national:
        runs = _national_monthly_total(runs)
        output = _national_monthly_total(output)
        consumption = _national_monthly_total(consumption)
        balance = _national_monthly_balance(output, consumption)

    title_suffix = " — national total" if national else ""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"Feedstock processing{title_suffix} ({ylab})",
            f"{product} output{title_suffix} ({ylab})",
            f"{product} consumption{title_suffix} ({ylab})",
            f"{product} balance — output minus consumption{title_suffix} ({ylab})",
        ),
    )

    national_color = "#58a6ff"
    for row, frame, ylab_short in [
        (1, runs, "Processing"),
        (2, output, "Output"),
        (3, consumption, "Consumption"),
        (4, balance, "Output − Consumption"),
    ]:
        if frame.empty:
            continue
        if national:
            sub = frame.sort_values("month_start")
            fig.add_trace(
                go.Scatter(
                    x=sub["month_start"],
                    y=sub["value"],
                    name="China (national)",
                    legendgroup="national",
                    showlegend=(row == 1),
                    mode="lines+markers",
                    line=dict(color=national_color, width=3),
                    marker=dict(size=5),
                    hovertemplate=(
                        f"China (national)<br>%{{x|%Y-%m}}<br>%{{y:,.1f}} {ylab}"
                        f"<extra>{ylab_short}</extra>"
                    ),
                ),
                row=row,
                col=1,
            )
            continue
        for g in sorted(frame["geo"].unique()):
            sub = frame[frame["geo"] == g].sort_values("month_start")
            color = REGION_COLORS.get(g, "#8b949e")
            fig.add_trace(
                go.Scatter(
                    x=sub["month_start"],
                    y=sub["value"],
                    name=g,
                    legendgroup=g,
                    showlegend=(row == 1),
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f"{g}<br>%{{x|%Y-%m}}<br>%{{y:,.1f}} {ylab}<extra>{ylab_short}</extra>",
                ),
                row=row,
                col=1,
            )

    if not balance.empty:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#484f58",
            line_width=1,
            row=4,
            col=1,
        )

    geo_label = "National" if national else ("Province" if geo == "province" else "Region")
    chart_title = f"{'National' if national else 'Regional'} trends — {product} ({geo_label})"
    fig.update_layout(
        title=chart_title,
        hovermode="x unified",
        height=960,
        legend=dict(title=geo_label, orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text=ylab)
    return fig


def fig_regional_compare_bars(
    runs: pd.DataFrame,
    output: pd.DataFrame,
    consumption: pd.DataFrame,
    *,
    product: str,
    geo: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """Grouped bars: processing vs output vs consumption in the date window."""
    ylab = volume_label(unit)
    pk = _product_key(product)
    runs = _period_aggregate(runs, unit, _product_key(product, for_runs=True), date_range)
    output = _period_aggregate(output, unit, pk, date_range)
    consumption = _period_aggregate(consumption, unit, pk, date_range)

    def _geo_set(frame: pd.DataFrame) -> set[str]:
        return set(frame["geo"].unique()) if not frame.empty else set()

    geos = sorted(_geo_set(runs) | _geo_set(output) | _geo_set(consumption))
    if not geos:
        return go.Figure().update_layout(title="No regional data in range")

    def _align(frame: pd.DataFrame) -> list[float]:
        lookup = frame.set_index("geo")["value"] if not frame.empty else {}
        return [float(lookup.get(g, 0)) for g in geos]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Processing",
            x=geos,
            y=_align(runs),
            marker_color="#58a6ff",
            hovertemplate=f"%{{x}}<br>Processing: %{{y:,.1f}} {ylab}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name=f"{product} output",
            x=geos,
            y=_align(output),
            marker_color="#3fb950",
            hovertemplate=f"%{{x}}<br>Output: %{{y:,.1f}} {ylab}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name=f"{product} consumption",
            x=geos,
            y=_align(consumption),
            marker_color="#f85149",
            hovertemplate=f"%{{x}}<br>Consumption: %{{y:,.1f}} {ylab}<extra></extra>",
        )
    )

    geo_label = "Province" if geo == "province" else "Region"
    fig.update_layout(
        title=f"Processing vs output vs consumption — {product} ({_period_label(unit)})",
        barmode="group",
        xaxis_title=geo_label,
        yaxis_title=ylab,
        hovermode="x unified",
    )
    return fig


def fig_regional_balance_gap(
    runs: pd.DataFrame,
    output: pd.DataFrame,
    consumption: pd.DataFrame,
    *,
    product: str,
    geo: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """
    Output minus consumption by region (period total).

    Positive = net product surplus in region; negative = deficit (imports from elsewhere).
    """
    ylab = volume_label(unit)
    pk = _product_key(product)
    output = _period_aggregate(output, unit, pk, date_range)
    consumption = _period_aggregate(consumption, unit, pk, date_range)

    if output.empty and consumption.empty:
        return go.Figure().update_layout(title="No data for balance view")

    merged = output.rename(columns={"value": "output"}).merge(
        consumption.rename(columns={"value": "consumption"}),
        on="geo",
        how="outer",
    ).fillna(0)
    merged["gap"] = merged["output"] - merged["consumption"]
    merged = merged.sort_values("gap")

    colors = ["#f85149" if v < 0 else "#3fb950" for v in merged["gap"]]
    fig = go.Figure(
        go.Bar(
            x=merged["gap"],
            y=merged["geo"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+,.0f}" for v in merged["gap"]],
            textposition="outside",
            hovertemplate=f"%{{y}}<br>Output − Consumption: %{{x:,.1f}} {ylab}<extra></extra>",
        )
    )
    geo_label = "Province" if geo == "province" else "Region"
    fig.update_layout(
        title=f"{product} regional balance — output minus consumption ({_period_label(unit)})",
        xaxis_title=f"Output − Consumption ({ylab})",
        yaxis_title=geo_label,
        height=max(360, 28 * len(merged)),
    )
    return fig


def _provincial_map_values(
    processing: pd.DataFrame,
    output: pd.DataFrame,
    consumption: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    product: str,
    metric: str,
    date_range: str,
    unit: str,
) -> pd.DataFrame:
    """Build province-level values for map coloring."""
    pk = _product_key(product)
    runs = aggregate_runs(processing, mapping, "province")
    out = aggregate_output(output, mapping, product, "province")
    cons = aggregate_consumption(consumption, product, "province")

    if metric == "processing":
        frame = _period_aggregate(runs, unit, _product_key(product, for_runs=True), date_range)
        label = "Processing"
    elif metric == "output":
        frame = _period_aggregate(out, unit, pk, date_range)
        label = f"{product} output"
    elif metric == "consumption":
        frame = _period_aggregate(cons, unit, pk, date_range)
        label = f"{product} consumption"
    else:
        out_p = _period_aggregate(out, unit, pk, date_range).rename(columns={"value": "output"})
        cons_p = _period_aggregate(cons, unit, pk, date_range).rename(columns={"value": "consumption"})
        merged = out_p.merge(cons_p, on="geo", how="outer").fillna(0)
        merged["value"] = merged["output"] - merged["consumption"]
        frame = merged[["geo", "value"]]
        label = f"{product} output − consumption"

    frame = frame.rename(columns={"geo": "province"})
    if not mapping.empty:
        frame = frame.merge(mapping, on="province", how="left")
        frame["region_label"] = frame["region"].map(_label_region)
    frame["metric_label"] = label
    return frame


def fig_regional_map(
    processing: pd.DataFrame,
    output: pd.DataFrame,
    consumption: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    product: str,
    metric: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """
    China map — bubble size and color by province-level metric.

    Uses geographic centroids; bubble area scales with magnitude.
    """
    ylab = volume_label(unit)
    data = _provincial_map_values(
        processing,
        output,
        consumption,
        mapping,
        product=product,
        metric=metric,
        date_range=date_range,
        unit=unit,
    )
    if data.empty:
        return go.Figure().update_layout(title="No provincial data for map")

    lats, lons, sizes, colors, hover = [], [], [], [], []
    max_val = float(data["value"].abs().max()) or 1.0

    for _, row in data.iterrows():
        prov = row["province"]
        if prov not in PROVINCE_CENTROIDS:
            continue
        lat, lon = PROVINCE_CENTROIDS[prov]
        val = float(row["value"])
        lats.append(lat)
        lons.append(lon)
        sizes.append(max(12.0, 8.0 + 34.0 * (abs(val) / max_val) ** 0.5))
        colors.append(val)
        region = row.get("region_label", row.get("region", ""))
        hover.append(
            f"<b>{prov}</b><br>{region}<br>{row['metric_label']}: {val:,.1f} {ylab}"
        )

    if not lats:
        return go.Figure().update_layout(title="No mappable provinces (missing coordinates)")

    colorscale = "RdYlGn" if metric == "balance" else "Blues"

    fig = go.Figure(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=hover,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale=colorscale,
                cmin=-max_val if metric == "balance" else 0,
                cmax=max_val,
                cmid=0 if metric == "balance" else None,
                colorbar=dict(title=ylab),
                line=dict(width=0.8, color="#30363d"),
                opacity=0.88,
            ),
        )
    )
    metric_titles = {
        "processing": "Feedstock processing",
        "output": f"{product} output",
        "consumption": f"{product} consumption",
        "balance": f"{product} output − consumption",
    }
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
        title=f"China map — {metric_titles.get(metric, metric)} ({_period_label(unit)})",
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(bgcolor="#161b22"),
    )
    return fig


def fig_top_provinces(
    processing: pd.DataFrame,
    output: pd.DataFrame,
    consumption: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    unit: str = "kt",
    top_n: int = 12,
) -> go.Figure:
    """Top provinces by processing with output/consumption for context."""
    ylab = volume_label(unit)
    pk = _product_key(product)
    runs = aggregate_runs(processing, mapping, "province")
    out = aggregate_output(output, mapping, product, "province")
    cons = aggregate_consumption(consumption, product, "province")

    runs = _period_aggregate(runs, unit, _product_key(product, for_runs=True), date_range)
    out = _period_aggregate(out, unit, pk, date_range)
    cons = _period_aggregate(cons, unit, pk, date_range)

    if runs.empty:
        return go.Figure().update_layout(title="No provincial processing data")

    top = runs.nlargest(top_n, "value")["geo"].tolist()
    geos = list(reversed(top))

    def _vals(frame: pd.DataFrame) -> list[float]:
        lookup = frame.set_index("geo")["value"] if not frame.empty else {}
        return [float(lookup.get(g, 0)) for g in geos]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Processing", y=geos, x=_vals(runs), orientation="h", marker_color="#58a6ff")
    )
    fig.add_trace(
        go.Bar(name=f"{product} output", y=geos, x=_vals(out), orientation="h", marker_color="#3fb950")
    )
    fig.add_trace(
        go.Bar(
            name=f"{product} consumption",
            y=geos,
            x=_vals(cons),
            orientation="h",
            marker_color="#f85149",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} provinces — processing vs {product} output & consumption",
        barmode="group",
        xaxis_title=f"{ylab} ({_period_label(unit)})",
        height=max(400, 26 * len(geos)),
    )
    return fig
