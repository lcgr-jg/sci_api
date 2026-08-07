"""Shandong independent refinery processing by oil grade (sheet 16)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.crude_shipping import _apply_top_legend_domain, _top_legend_layout
from src.filters import filter_by_date_range
from src.units import convert_volume_series, volume_label

_BUCKET_COLORS = {
    "Domestic": "#3fb950",
    "Secondary / residue": "#e3b341",
    "Other imported": "#8b949e",
    "Brazil": "#58a6ff",
    "Angola": "#f85149",
    "Saudi Arabia": "#bc8cff",
    "Iraq": "#ffa657",
    "UAE": "#79c0ff",
    "Malaysia": "#7ee787",
    "United States": "#d2a8ff",
    "Canada": "#ff7b72",
    "Oman": "#a5d6ff",
}

_PALETTE = [
    "#58a6ff",
    "#3fb950",
    "#f85149",
    "#e3b341",
    "#bc8cff",
    "#ffa657",
    "#79c0ff",
    "#7ee787",
    "#8b949e",
]


def _grade_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop synthetic total rows used for KPI totals."""
    if df.empty:
        return df
    return df[~df["grade"].astype(str).str.startswith("_total")]


def _filter_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    if df.empty or city in (None, "", "All"):
        return df
    return df[df["city"] == city]


def fig_oil_grade_totals(
    oil_grade: pd.DataFrame,
    *,
    city: str,
    date_range: str,
    unit: str,
) -> go.Figure:
    """Monthly primary vs secondary feedstock totals."""
    sub = filter_by_date_range(_filter_city(oil_grade, city), "month_start", date_range)
    if sub.empty:
        return go.Figure().update_layout(title="No oil-grade processing data")

    primary = (
        sub[sub["grade"] == "_total_primary"]
        .groupby("month_start", as_index=False)["volume_kt"]
        .sum()
        .sort_values("month_start")
    )
    secondary = (
        sub[sub["grade"] == "_total_secondary"]
        .groupby("month_start", as_index=False)["volume_kt"]
        .sum()
        .sort_values("month_start")
    )
    ylab = volume_label(unit)
    fig = go.Figure()
    if not primary.empty:
        y = convert_volume_series(
            primary["volume_kt"],
            unit,
            freq="monthly",
            dates=primary["month_start"],
            product="crude",
        )
        fig.add_trace(
            go.Bar(
                x=primary["month_start"],
                y=y,
                name="Primary feedstock",
                marker_color="#58a6ff",
            )
        )
    if not secondary.empty:
        y = convert_volume_series(
            secondary["volume_kt"],
            unit,
            freq="monthly",
            dates=secondary["month_start"],
            product="crude",
        )
        fig.add_trace(
            go.Bar(
                x=secondary["month_start"],
                y=y,
                name="Secondary feedstock",
                marker_color="#e3b341",
            )
        )
    city_lbl = city if city not in (None, "", "All") else "all cities"
    fig.update_layout(
        title=f"Shandong independent feedstock — {city_lbl} ({ylab})",
        barmode="group",
        xaxis_title="Month",
        yaxis_title=ylab,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def fig_oil_grade_city_bars(
    oil_grade: pd.DataFrame,
    *,
    date_range: str,
    unit: str,
) -> go.Figure:
    """Latest month (in filter) primary processing by city."""
    sub = filter_by_date_range(oil_grade, "month_start", date_range)
    prim = sub[sub["grade"] == "_total_primary"]
    if prim.empty:
        return go.Figure().update_layout(title="No city totals")
    latest = prim["month_start"].max()
    cur = (
        prim[prim["month_start"] == latest]
        .groupby("city", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=True)
    )
    y = convert_volume_series(cur["volume_kt"], unit, product="crude")
    ylab = volume_label(unit)
    fig = go.Figure(
        go.Bar(
            x=y,
            y=cur["city"],
            orientation="h",
            marker_color="#3fb950",
            text=[f"{v:,.0f}" for v in y],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Primary processing by city — {pd.Timestamp(latest):%Y-%m} ({ylab})",
        xaxis_title=ylab,
        yaxis_title="",
    )
    return fig


def fig_oil_grade_origin_mix(
    oil_grade: pd.DataFrame,
    *,
    city: str,
    date_range: str,
    top_n: int = 8,
) -> go.Figure:
    """100% stacked monthly mix by origin bucket (top N + Other)."""
    sub = _grade_rows(
        filter_by_date_range(_filter_city(oil_grade, city), "month_start", date_range)
    )
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(**_top_legend_layout())
        return _apply_top_legend_domain(fig)

    totals = (
        sub.groupby("origin_bucket", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=False)
    )
    keep = totals["origin_bucket"].head(top_n).tolist()
    plot = sub.copy()
    plot["bucket"] = plot["origin_bucket"].where(
        plot["origin_bucket"].isin(keep), "Other"
    )
    monthly = (
        plot.groupby(["month_start", "bucket"], as_index=False)["volume_kt"]
        .sum()
        .sort_values("month_start")
    )
    month_tot = monthly.groupby("month_start")["volume_kt"].transform("sum")
    monthly["share_pct"] = (monthly["volume_kt"] / month_tot * 100.0).where(month_tot > 0)

    order = (
        monthly.groupby("bucket")["volume_kt"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if "Other" in order:
        order = [b for b in order if b != "Other"] + ["Other"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i, bucket in enumerate(order):
        s = monthly[monthly["bucket"] == bucket].sort_values("month_start")
        color = _BUCKET_COLORS.get(bucket, _PALETTE[i % len(_PALETTE)])
        fig.add_trace(
            go.Scatter(
                x=s["month_start"],
                y=s["share_pct"],
                name=bucket,
                mode="lines",
                stackgroup="mix",
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate="%{x|%Y-%m}<br>%{fullData.name}: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    # Top-3 concentration among named buckets.
    named = monthly[monthly["bucket"] != "Other"]
    if not named.empty:
        top3 = named.groupby("month_start")["share_pct"].apply(
            lambda s: s.nlargest(min(3, len(s))).sum()
        )
        fig.add_trace(
            go.Scatter(
                x=top3.index,
                y=top3.values,
                name="Top-3 share",
                mode="lines+markers",
                line=dict(color="#e6edf3", width=2, dash="dash"),
                marker=dict(size=5),
                hovertemplate="%{x|%Y-%m}<br>Top-3: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(**_top_legend_layout())
    fig.update_yaxes(title_text="Share (%)", range=[0, 100], secondary_y=False)
    fig.update_yaxes(
        title_text="Top-3 (%)", range=[0, 100], showgrid=False, secondary_y=True
    )
    fig.update_xaxes(title_text="Month")
    return _apply_top_legend_domain(fig)


def fig_oil_grade_top_grades(
    oil_grade: pd.DataFrame,
    *,
    city: str,
    date_range: str,
    unit: str,
    top_n: int = 15,
) -> go.Figure:
    """Top individual grades in the latest month of the filter window."""
    sub = _grade_rows(
        filter_by_date_range(_filter_city(oil_grade, city), "month_start", date_range)
    )
    if sub.empty:
        return go.Figure().update_layout(title="No grade detail")
    latest = sub["month_start"].max()
    cur = (
        sub[sub["month_start"] == latest]
        .groupby(["grade", "origin_bucket"], as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=True)
        .tail(top_n)
    )
    y = convert_volume_series(cur["volume_kt"], unit, product="crude")
    ylab = volume_label(unit)
    colors = [
        _BUCKET_COLORS.get(b, "#58a6ff") for b in cur["origin_bucket"]
    ]
    fig = go.Figure(
        go.Bar(
            x=y,
            y=cur["grade"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:,.0f}" for v in y],
            textposition="outside",
        )
    )
    city_lbl = city if city not in (None, "", "All") else "all cities"
    fig.update_layout(
        title=f"Top grades — {city_lbl}, {pd.Timestamp(latest):%Y-%m} ({ylab})",
        xaxis_title=ylab,
        yaxis_title="",
        margin=dict(l=160, r=40, t=50, b=40),
    )
    return fig
