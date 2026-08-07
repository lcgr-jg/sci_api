"""Crude port shipping (foreign/domestic) and independent import quota charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.filters import filter_by_date_range
from src.units import convert_volume_series, volume_label

TRADE_COLORS = {
    "foreign": "#58a6ff",
    "domestic": "#3fb950",
}

SHIP_GRADE_COLORS = {
    "VLCC": "#58a6ff",
    "AFRA": "#bc8cff",
    "Suezmax": "#e3b341",
    "LR1": "#3fb950",
    "LR2": "#ffa657",
    "MR": "#f85149",
}


def _filter_trade(shipping: pd.DataFrame, trade: str) -> pd.DataFrame:
    if shipping.empty or trade == "all":
        return shipping
    return shipping[shipping["trade"] == trade]


def fig_crude_monthly_arrivals(
    shipping: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
    unit: str,
) -> go.Figure:
    """Monthly arrival volumes by trade, with forecast overlay for the plan month."""
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    ylab = volume_label(unit)
    fig = go.Figure()

    if sub.empty:
        return fig.update_layout(title="No crude shipping actuals for selection")

    if trade == "all":
        monthly = (
            sub.groupby(["month_start", "trade"], as_index=False)["volume_kt"]
            .sum()
            .sort_values("month_start")
        )
        for tr in ("foreign", "domestic"):
            s = monthly[monthly["trade"] == tr]
            if s.empty:
                continue
            y = convert_volume_series(
                s["volume_kt"], unit, freq="monthly", dates=s["month_start"], product="crude"
            )
            fig.add_trace(
                go.Bar(
                    x=s["month_start"],
                    y=y,
                    name=tr.title(),
                    marker_color=TRADE_COLORS[tr],
                )
            )
        fig.update_layout(barmode="stack", title=f"Monthly crude arrivals by trade ({ylab})")
    else:
        monthly = (
            sub.groupby("month_start", as_index=False)["volume_kt"]
            .sum()
            .sort_values("month_start")
        )
        y = convert_volume_series(
            monthly["volume_kt"],
            unit,
            freq="monthly",
            dates=monthly["month_start"],
            product="crude",
        )
        fig.add_trace(
            go.Bar(
                x=monthly["month_start"],
                y=y,
                name=trade.title(),
                marker_color=TRADE_COLORS.get(trade, "#58a6ff"),
            )
        )
        fig.update_layout(title=f"Monthly {trade} crude arrivals ({ylab})")

    fc = _filter_trade(forecast, trade)
    if not fc.empty:
        fc_m = (
            fc.groupby("month_start", as_index=False)["volume_kt"]
            .sum()
            .sort_values("month_start")
        )
        yf = convert_volume_series(
            fc_m["volume_kt"],
            unit,
            freq="monthly",
            dates=fc_m["month_start"],
            product="crude",
        )
        fig.add_trace(
            go.Scatter(
                x=fc_m["month_start"],
                y=yf,
                mode="markers+lines",
                name="Forecast arrivals",
                line=dict(dash="dot", color="#ffa657"),
                marker=dict(size=9, symbol="diamond"),
            )
        )

    fig.update_layout(xaxis_title="Arrival month", yaxis_title=ylab)
    return fig


def fig_crude_origins(
    shipping: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
    period: str,
) -> go.Figure:
    """Top origin countries (foreign trade). Domestic has no origin country."""
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    if period == "latest" and not sub.empty:
        lm = sub["month_start"].max()
        sub = sub[sub["month_start"] == lm]
    if trade == "domestic":
        return go.Figure().update_layout(
            title="Origin countries — N/A for domestic coastal trade"
        )
    sub = sub.dropna(subset=["origin_country"])
    if sub.empty:
        return go.Figure().update_layout(title="No origin-country data")

    top = (
        sub.groupby("origin_country", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=True)
        .tail(15)
    )
    title_period = "latest month" if period == "latest" else "selected period"
    fig = go.Figure(
        go.Bar(
            x=top["volume_kt"],
            y=top["origin_country"],
            orientation="h",
            marker_color="#58a6ff",
            text=[f"{v:,.0f}" for v in top["volume_kt"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top origin countries ({title_period}, kt)",
        xaxis_title="Volume (kt)",
        yaxis_title="",
    )
    return fig


def fig_crude_provinces(
    shipping: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
    period: str,
) -> go.Figure:
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    if period == "latest" and not sub.empty:
        lm = sub["month_start"].max()
        sub = sub[sub["month_start"] == lm]
    if sub.empty:
        return go.Figure().update_layout(title="No discharge-province data")

    top = (
        sub.groupby("province", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=True)
        .tail(15)
    )
    title_period = "latest month" if period == "latest" else "selected period"
    fig = go.Figure(
        go.Bar(
            x=top["volume_kt"],
            y=top["province"],
            orientation="h",
            marker_color="#3fb950",
            text=[f"{v:,.0f}" for v in top["volume_kt"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top discharge provinces ({title_period}, kt)",
        xaxis_title="Volume (kt)",
        yaxis_title="",
    )
    return fig


def fig_crude_fleet_mix(
    shipping: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
    period: str,
) -> go.Figure:
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    if period == "latest" and not sub.empty:
        lm = sub["month_start"].max()
        sub = sub[sub["month_start"] == lm]
    if sub.empty:
        return go.Figure().update_layout(title="No fleet data")

    mix = (
        sub.groupby("ship_grade", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=False)
    )
    title_period = "latest month" if period == "latest" else "selected period"
    colors = [SHIP_GRADE_COLORS.get(str(g), "#8b949e") for g in mix["ship_grade"]]
    fig = go.Figure(
        go.Bar(
            x=mix["ship_grade"].astype(str),
            y=mix["volume_kt"],
            marker_color=colors,
            text=[f"{v:,.0f}" for v in mix["volume_kt"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Fleet mix by ship grade ({title_period}, kt)",
        xaxis_title="Ship grade",
        yaxis_title="Volume (kt)",
    )
    return fig


def fig_crude_grades(
    shipping: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
    period: str,
) -> go.Figure:
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    if period == "latest" and not sub.empty:
        lm = sub["month_start"].max()
        sub = sub[sub["month_start"] == lm]
    if sub.empty:
        return go.Figure().update_layout(title="No crude-grade data")

    top = (
        sub.groupby("crude_grade", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=True)
        .tail(15)
    )
    title_period = "latest month" if period == "latest" else "selected period"
    fig = go.Figure(
        go.Bar(
            x=top["volume_kt"],
            y=top["crude_grade"].astype(str),
            orientation="h",
            marker_color="#bc8cff",
            text=[f"{v:,.0f}" for v in top["volume_kt"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top crude grades ({title_period}, kt)",
        xaxis_title="Volume (kt)",
        yaxis_title="",
    )
    return fig


def fig_crude_quota_summary(
    quotas_summary: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
) -> go.Figure:
    """Independent-refinery crude import quota utilization."""
    if quotas_summary.empty:
        return go.Figure().update_layout(title="No crude import quota data")

    row = quotas_summary.iloc[0]
    year = int(row["quota_year"]) if pd.notna(row.get("quota_year")) else pd.Timestamp.today().year
    annual = row.get("annual_quota_kt")
    ceiling = row.get("ceiling_kt")
    total_imp = row.get("total_imports_kt")
    util = row.get("utilization")

    cum = pd.DataFrame()
    if not quotas_monthly.empty:
        m = quotas_monthly[quotas_monthly["quota_year"] == year].copy()
        m = m.sort_values("month_start")
        m = m[m["import_kt"].fillna(0) > 0]
        if not m.empty:
            m["cumulative_kt"] = m["import_kt"].cumsum()
            cum = m

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            f"Cumulative independent imports vs {year} quota (kt)",
            "Quota batches",
        ),
    )

    if not cum.empty:
        fig.add_trace(
            go.Scatter(
                x=cum["month_start"],
                y=cum["cumulative_kt"],
                mode="lines+markers",
                name="Cumulative imports",
                line=dict(color="#58a6ff"),
            ),
            row=1,
            col=1,
        )
    if pd.notna(annual):
        fig.add_hline(
            y=float(annual),
            line_dash="dash",
            line_color="#f85149",
            annotation_text=f"Annual quota {float(annual):,.0f} kt",
            row=1,
            col=1,
        )
    if pd.notna(ceiling) and (not pd.notna(annual) or abs(float(ceiling) - float(annual)) > 1):
        fig.add_hline(
            y=float(ceiling),
            line_dash="dot",
            line_color="#e3b341",
            annotation_text=f"Ceiling {float(ceiling):,.0f} kt",
            row=1,
            col=1,
        )

    batches, labels, colors = [], [], ["#3fb950", "#58a6ff", "#bc8cff", "#ffa657", "#f85149"]
    for i, col in enumerate(
        ["batch_1_kt", "batch_2_kt", "batch_3_kt", "batch_4_kt", "batch_5_kt"]
    ):
        val = row.get(col)
        if pd.notna(val) and float(val) > 0:
            batches.append(float(val))
            labels.append(f"Batch {i + 1}")
    if batches:
        fig.add_trace(
            go.Bar(x=labels, y=batches, marker_color=colors[: len(batches)], showlegend=False),
            row=1,
            col=2,
        )

    util_pct = f"{float(util) * 100:.1f}%" if pd.notna(util) else "n/a"
    ytd = f"{float(total_imp):,.0f} kt" if pd.notna(total_imp) else "n/a"
    fig.update_layout(
        title=(
            f"Independent crude import quotas {year} — YTD imports {ytd}, "
            f"utilization {util_pct}"
        )
    )
    fig.update_yaxes(title_text="kt", row=1, col=1)
    fig.update_yaxes(title_text="kt", row=1, col=2)
    return fig


def fig_crude_quotas_monthly(quotas_monthly: pd.DataFrame) -> go.Figure:
    if quotas_monthly.empty:
        return go.Figure().update_layout(title="No monthly crude quota import data")
    m = quotas_monthly.sort_values("month_start")
    fig = go.Figure(
        go.Bar(
            x=m["month_start"],
            y=m["import_kt"],
            marker_color="#58a6ff",
            name="Independent imports",
        )
    )
    fig.update_layout(
        title="Independent refinery crude imports — monthly (quota sheet, kt)",
        xaxis_title="Month",
        yaxis_title="Imports (kt)",
    )
    return fig


# ---------------------------------------------------------------------------
# Mix-over-time (100% stacked area + top-3 concentration)
# ---------------------------------------------------------------------------

_MIX_PALETTE = [
    "#58a6ff",
    "#3fb950",
    "#f85149",
    "#e3b341",
    "#bc8cff",
    "#ffa657",
    "#79c0ff",
    "#7ee787",
    "#8b949e",  # Other
]


def _top_legend_layout(**extra) -> dict:
    """
    Horizontal legend in the top paper band; chart titles are HTML outside.

    Pair with ``_apply_top_legend_domain`` so stacked areas do not draw under the legend
    (margin alone is unreliable with secondary-y subplots).
    """
    layout = dict(
        title=None,
        margin=dict(l=50, r=50, t=24, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            x=0.5,
            xanchor="center",
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            traceorder="normal",
        ),
        hovermode="x unified",
    )
    layout.update(extra)
    return layout


# Fraction of figure height reserved for the top legend (rest is the data domain).
_LEGEND_TOP_FRACTION = 0.22


def _apply_top_legend_domain(fig: go.Figure) -> go.Figure:
    """Shrink y domains so traces stay below the horizontal legend band."""
    top = 1.0 - _LEGEND_TOP_FRACTION
    # make_subplots figures support secondary_y; plain Figures do not.
    if getattr(fig, "_grid_ref", None) is not None:
        fig.update_yaxes(domain=[0.0, top], secondary_y=False)
        fig.update_yaxes(domain=[0.0, top], secondary_y=True)
    else:
        fig.update_layout(yaxis=dict(domain=[0.0, top]))
    return fig


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_top_legend_layout(),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.4,
                showarrow=False,
                font=dict(color="#8b949e", size=13),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _monthly_mix_shares(
    shipping: pd.DataFrame,
    category_col: str,
    *,
    top_n: int = 8,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Monthly volume shares by category (top_n + Other).

    Returns (long_df with month_start/category/share_pct/volume_kt,
             top3_share series indexed by month_start).
    """
    empty_long = pd.DataFrame(
        columns=["month_start", "category", "volume_kt", "share_pct"]
    )
    empty_top3 = pd.Series(dtype=float)
    if shipping.empty or category_col not in shipping.columns:
        return empty_long, empty_top3

    df = shipping.dropna(subset=[category_col, "month_start", "volume_kt"]).copy()
    if df.empty:
        return empty_long, empty_top3

    df["category"] = df[category_col].astype(str).str.strip()
    df = df[df["category"].ne("") & df["category"].str.lower().ne("nan")]

    # Rank categories by total volume in the filtered window.
    totals = (
        df.groupby("category", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=False)
    )
    keep = totals["category"].head(top_n).tolist()
    df["category"] = df["category"].where(df["category"].isin(keep), "Other")

    monthly = (
        df.groupby(["month_start", "category"], as_index=False)["volume_kt"]
        .sum()
        .sort_values("month_start")
    )
    month_tot = monthly.groupby("month_start")["volume_kt"].transform("sum")
    monthly["share_pct"] = (monthly["volume_kt"] / month_tot * 100.0).where(
        month_tot > 0
    )

    # Top-3 concentration among named categories (ex-Other) when possible.
    named = monthly[monthly["category"] != "Other"]
    if named.empty:
        top3 = monthly.groupby("month_start")["share_pct"].apply(
            lambda s: s.nlargest(3).sum()
        )
    else:
        top3 = named.groupby("month_start")["share_pct"].apply(
            lambda s: s.nlargest(min(3, len(s))).sum()
        )
    return monthly, top3


def fig_crude_mix_over_time(
    shipping: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
    category_col: str,
    title: str,
    top_n: int = 8,
) -> go.Figure:
    """100% stacked area of monthly mix + top-3 share line (secondary axis)."""
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    if category_col == "origin_country" and trade == "domestic":
        return _empty_fig("N/A for domestic coastal trade")

    mix, top3 = _monthly_mix_shares(sub, category_col, top_n=top_n)
    if mix.empty:
        return _empty_fig(f"No data — {title}")

    # Stable category order: top volume first, Other last.
    cat_order = (
        mix.groupby("category")["volume_kt"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if "Other" in cat_order:
        cat_order = [c for c in cat_order if c != "Other"] + ["Other"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i, cat in enumerate(cat_order):
        s = mix[mix["category"] == cat].sort_values("month_start")
        color = (
            SHIP_GRADE_COLORS.get(cat, _MIX_PALETTE[i % len(_MIX_PALETTE)])
            if category_col == "ship_grade"
            else _MIX_PALETTE[i % len(_MIX_PALETTE)]
        )
        fig.add_trace(
            go.Scatter(
                x=s["month_start"],
                y=s["share_pct"],
                name=cat,
                mode="lines",
                stackgroup="mix",
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate="%{x|%Y-%m}<br>%{fullData.name}: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    if not top3.empty:
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
        title_text="Top-3 (%)",
        range=[0, 100],
        showgrid=False,
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Arrival month")
    return _apply_top_legend_domain(fig)


def fig_crude_fleet_mix_trend(
    shipping: pd.DataFrame, *, trade: str, date_range: str
) -> go.Figure:
    return fig_crude_mix_over_time(
        shipping,
        trade=trade,
        date_range=date_range,
        category_col="ship_grade",
        title="Fleet mix over time",
        top_n=8,
    )


def fig_crude_origin_mix_trend(
    shipping: pd.DataFrame, *, trade: str, date_range: str
) -> go.Figure:
    return fig_crude_mix_over_time(
        shipping,
        trade=trade,
        date_range=date_range,
        category_col="origin_country",
        title="Origin-country mix over time",
        top_n=8,
    )


def fig_crude_province_mix_trend(
    shipping: pd.DataFrame, *, trade: str, date_range: str
) -> go.Figure:
    return fig_crude_mix_over_time(
        shipping,
        trade=trade,
        date_range=date_range,
        category_col="province",
        title="Discharge-province mix over time",
        top_n=8,
    )


def fig_crude_grade_mix_trend(
    shipping: pd.DataFrame, *, trade: str, date_range: str
) -> go.Figure:
    return fig_crude_mix_over_time(
        shipping,
        trade=trade,
        date_range=date_range,
        category_col="crude_grade",
        title="Crude-grade mix over time",
        top_n=8,
    )


# ---------------------------------------------------------------------------
# Voyage logistics (transit days & discharge lag)
# ---------------------------------------------------------------------------

# Drop obvious date typos / AIS gaps from transit stats.
_TRANSIT_MAX_DAYS = {"foreign": 120, "domestic": 45, "all": 120}
_LAG_MAX_DAYS = 30


def _with_voyage_metrics(shipping: pd.DataFrame) -> pd.DataFrame:
    if shipping.empty:
        return shipping
    out = shipping.copy()
    out["arrival_date"] = pd.to_datetime(out["arrival_date"], errors="coerce")
    out["departure_date"] = pd.to_datetime(out["departure_date"], errors="coerce")
    out["discharge_date"] = pd.to_datetime(out["discharge_date"], errors="coerce")
    out["transit_days"] = (out["arrival_date"] - out["departure_date"]).dt.days
    out["discharge_lag_days"] = (out["discharge_date"] - out["arrival_date"]).dt.days
    return out


def _valid_transit(df: pd.DataFrame, trade: str) -> pd.DataFrame:
    if df.empty or "transit_days" not in df.columns:
        return df.iloc[0:0]
    tmax = _TRANSIT_MAX_DAYS.get(trade, 120)
    return df[
        df["transit_days"].notna()
        & (df["transit_days"] >= 1)
        & (df["transit_days"] <= tmax)
    ]


def _monthly_percentile_band(
    df: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    g = df.groupby("month_start")[value_col]
    return pd.DataFrame(
        {
            "month_start": g.median().index,
            "p50": g.median().values,
            "p25": g.quantile(0.25).values,
            "p75": g.quantile(0.75).values,
            "n": g.count().values,
        }
    ).sort_values("month_start")


def fig_crude_transit_trend(
    shipping: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
) -> go.Figure:
    """
    Monthly median transit days (arrival − origin departure) with p25–p75 band,
    plus median lines for the main ship grades.
    """
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    sub = _valid_transit(_with_voyage_metrics(sub), trade)
    if sub.empty:
        return _empty_fig("No voyage transit data for selection")

    band = _monthly_percentile_band(sub, "transit_days")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.concat([band["month_start"], band["month_start"][::-1]]),
            y=pd.concat([band["p75"], band["p25"][::-1]]),
            fill="toself",
            fillcolor="rgba(88, 166, 255, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="p25-p75",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band["month_start"],
            y=band["p50"],
            mode="lines+markers",
            name="Median (all)",
            line=dict(color="#e6edf3", width=2.5),
            marker=dict(size=6),
            customdata=band[["n", "p25", "p75"]],
            hovertemplate=(
                "%{x|%Y-%m}<br>Median %{y:.0f}d"
                "<br>p25-p75 %{customdata[1]:.0f}-%{customdata[2]:.0f}d"
                "<br>n=%{customdata[0]}<extra></extra>"
            ),
        )
    )

    # Overlay top ship grades by voyage count.
    top_grades = (
        sub.groupby("ship_grade").size().sort_values(ascending=False).head(4).index.tolist()
    )
    for grade in top_grades:
        g = sub[sub["ship_grade"] == grade]
        med = g.groupby("month_start")["transit_days"].median().sort_index()
        if med.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=med.index,
                y=med.values,
                mode="lines",
                name=str(grade),
                line=dict(
                    color=SHIP_GRADE_COLORS.get(str(grade), "#8b949e"),
                    width=1.5,
                    dash="dot",
                ),
                hovertemplate=f"%{{x|%Y-%m}}<br>{grade} median %{{y:.0f}}d<extra></extra>",
            )
        )

    fig.update_layout(
        **_top_legend_layout(
            xaxis_title="Arrival month",
            yaxis_title="Transit days",
        )
    )
    return _apply_top_legend_domain(fig)


def fig_crude_discharge_lag_trend(
    shipping: pd.DataFrame,
    *,
    trade: str,
    date_range: str,
) -> go.Figure:
    """Monthly median days from arrival to discharge (port/berth wait proxy)."""
    sub = filter_by_date_range(_filter_trade(shipping, trade), "arrival_date", date_range)
    sub = _with_voyage_metrics(sub)
    sub = sub[
        sub["discharge_lag_days"].notna()
        & (sub["discharge_lag_days"] >= 0)
        & (sub["discharge_lag_days"] <= _LAG_MAX_DAYS)
    ]
    if sub.empty:
        return _empty_fig("No discharge-lag data for selection")

    band = _monthly_percentile_band(sub, "discharge_lag_days")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.concat([band["month_start"], band["month_start"][::-1]]),
            y=pd.concat([band["p75"], band["p25"][::-1]]),
            fill="toself",
            fillcolor="rgba(63, 185, 80, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="p25-p75",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band["month_start"],
            y=band["p50"],
            mode="lines+markers",
            name="Median lag",
            line=dict(color="#3fb950", width=2.5),
            customdata=band[["n"]],
            hovertemplate="%{x|%Y-%m}<br>Median %{y:.1f}d<br>n=%{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(
        **_top_legend_layout(
            xaxis_title="Arrival month",
            yaxis_title="Days",
        )
    )
    return _apply_top_legend_domain(fig)


def origin_transit_compare_records(
    shipping: pd.DataFrame,
    *,
    trade: str = "foreign",
    top_n: int = 12,
) -> list[dict]:
    """
    Latest complete arrival month: median transit by origin vs same month last year.
    """
    if trade == "domestic":
        return []
    sub = _filter_trade(shipping, trade if trade != "all" else "foreign")
    sub = _valid_transit(_with_voyage_metrics(sub), "foreign")
    sub = sub.dropna(subset=["origin_country"])
    if sub.empty:
        return []

    latest = sub["month_start"].max()
    prior = latest - pd.DateOffset(years=1)
    cur = sub[sub["month_start"] == latest]
    ly = sub[sub["month_start"] == prior]
    if cur.empty:
        return []

    cur_med = (
        cur.groupby("origin_country")["transit_days"]
        .agg(median_d="median", n="count", volume_kt="sum")
        .reset_index()
    )
    ly_med = (
        ly.groupby("origin_country")["transit_days"]
        .median()
        .rename("median_ly")
        .reset_index()
    )
    out = cur_med.merge(ly_med, on="origin_country", how="left")
    out = out.sort_values("volume_kt", ascending=False).head(top_n)
    month_lbl = pd.Timestamp(latest).strftime("%Y-%m")
    records = []
    for _, r in out.iterrows():
        med = float(r["median_d"])
        ly_v = float(r["median_ly"]) if pd.notna(r["median_ly"]) else None
        records.append(
            {
                "month": month_lbl,
                "origin": r["origin_country"],
                "median_days": round(med, 1),
                "same_month_ly": round(ly_v, 1) if ly_v is not None else None,
                "delta_vs_ly": round(med - ly_v, 1) if ly_v is not None else None,
                "voyages": int(r["n"]),
                "volume_kt": round(float(r["volume_kt"]), 0),
            }
        )
    return records


def origin_transit_compare_columns() -> list[dict]:
    return [
        {"name": "month", "id": "month"},
        {"name": "origin", "id": "origin"},
        {
            "name": "median_days",
            "id": "median_days",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "same_month_ly",
            "id": "same_month_ly",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "delta_vs_ly",
            "id": "delta_vs_ly",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "voyages",
            "id": "voyages",
            "type": "numeric",
            "format": {"specifier": ",d"},
        },
        {
            "name": "volume_kt",
            "id": "volume_kt",
            "type": "numeric",
            "format": {"specifier": ",.0f"},
        },
    ]


def origin_transit_delta_style() -> list[dict]:
    return [
        {
            "if": {"filter_query": "{delta_vs_ly} > 0", "column_id": "delta_vs_ly"},
            "color": "#f85149",
            "fontWeight": "600",
        },
        {
            "if": {"filter_query": "{delta_vs_ly} < 0", "column_id": "delta_vs_ly"},
            "color": "#3fb950",
            "fontWeight": "600",
        },
    ]
