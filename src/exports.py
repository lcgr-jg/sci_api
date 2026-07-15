"""Refined product export shipping, quotas, and monthly aggregate charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.filters import filter_by_date_range
from src.transforms import (
    EXPORT_TABLE_PRODUCTS,
    build_export_monthly_from_quotas,
    build_export_monthly_from_shipping,
)

EXPORT_PRODUCT_COLORS = {
    "Gasoline": "#58a6ff",
    "Jet": "#bc8cff",
    "Gasoil/Diesel": "#f85149",
    "FO": "#e3b341",
    "All": "#8b949e",
}

_SHIPPING_PRODUCTS = ["Gasoline", "Jet", "Gasoil/Diesel"]


def _filter_shipping(
    shipping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
) -> pd.DataFrame:
    if shipping.empty:
        return shipping
    out = filter_by_date_range(shipping, "departure_date", date_range)
    if product and product != "All":
        out = out[out["product"] == product]
    return out


def _latest_month(shipping: pd.DataFrame) -> pd.Timestamp | None:
    if shipping.empty:
        return None
    return pd.to_datetime(shipping["departure_date"]).max().to_period("M").to_timestamp()


def _period_slice(shipping: pd.DataFrame, period: str) -> pd.DataFrame:
    if shipping.empty:
        return shipping
    if period == "latest":
        lm = _latest_month(shipping)
        if lm is None:
            return shipping.iloc[0:0]
        return shipping[shipping["month_start"] == lm]
    return shipping


def fig_export_monthly_stacked(
    shipping: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    product: str,
    date_range: str,
) -> go.Figure:
    """Monthly export volume from shipping actuals with forecast overlay."""
    sub = _filter_shipping(shipping, product=product, date_range=date_range)
    fig = go.Figure()

    if sub.empty and (forecast.empty or product not in ("All",) and product):
        return fig.update_layout(title="No shipping data for selection")

    if product == "All" or not product:
        monthly = (
            sub.groupby(["month_start", "product"], as_index=False)["volume_kt"]
            .sum()
            .sort_values("month_start")
        )
        for prod in _SHIPPING_PRODUCTS:
            s = monthly[monthly["product"] == prod]
            if s.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=s["month_start"],
                    y=s["volume_kt"],
                    name=prod,
                    marker_color=EXPORT_PRODUCT_COLORS[prod],
                )
            )
        fig.update_layout(barmode="stack", title="Monthly exports by product (shipping actuals, kt)")
    else:
        monthly = (
            sub.groupby("month_start", as_index=False)["volume_kt"]
            .sum()
            .sort_values("month_start")
        )
        fig.add_trace(
            go.Bar(
                x=monthly["month_start"],
                y=monthly["volume_kt"],
                name=product,
                marker_color=EXPORT_PRODUCT_COLORS.get(product, "#58a6ff"),
            )
        )
        fig.update_layout(title=f"Monthly {product} exports (shipping actuals, kt)")

    if not forecast.empty:
        fc = forecast.copy()
        if product and product != "All":
            fc = fc[fc["product"] == product]
        fc_monthly = (
            fc.groupby("month_start", as_index=False)["volume_kt"]
            .sum()
            .sort_values("month_start")
        )
        if not fc_monthly.empty:
            fig.add_trace(
                go.Scatter(
                    x=fc_monthly["month_start"],
                    y=fc_monthly["volume_kt"],
                    mode="markers+lines",
                    name="Forecast cargoes",
                    line=dict(dash="dot", color="#ffa657"),
                    marker=dict(size=9, symbol="diamond"),
                )
            )

    fig.update_layout(xaxis_title="Departure month", yaxis_title="Volume (kt)")
    return fig


def fig_export_destinations(
    shipping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    period: str,
) -> go.Figure:
    sub = _period_slice(_filter_shipping(shipping, product=product, date_range=date_range), period)
    if sub.empty:
        return go.Figure().update_layout(title="No destination data")

    top = (
        sub.groupby("destination", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=True)
        .tail(15)
    )
    title_period = "latest month" if period == "latest" else "selected period"
    fig = go.Figure(
        go.Bar(
            x=top["volume_kt"],
            y=top["destination"],
            orientation="h",
            marker_color="#58a6ff",
            text=[f"{v:,.0f}" for v in top["volume_kt"]],
            textposition="outside",
        )
    )
    prod_lbl = product if product != "All" else "all products"
    fig.update_layout(
        title=f"Top destinations — {prod_lbl} ({title_period}, kt)",
        xaxis_title="Volume (kt)",
        yaxis_title="",
    )
    return fig


def fig_export_provinces(
    shipping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    period: str,
) -> go.Figure:
    sub = _period_slice(_filter_shipping(shipping, product=product, date_range=date_range), period)
    if sub.empty:
        return go.Figure().update_layout(title="No province data")

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
    prod_lbl = product if product != "All" else "all products"
    fig.update_layout(
        title=f"Top departure provinces — {prod_lbl} ({title_period}, kt)",
        xaxis_title="Volume (kt)",
        yaxis_title="",
    )
    return fig


def fig_export_sankey(
    shipping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    period: str,
) -> go.Figure:
    sub = _period_slice(_filter_shipping(shipping, product=product, date_range=date_range), period)
    if sub.empty:
        return go.Figure().update_layout(title="No flow data")

    flows = (
        sub.groupby(["province", "destination"], as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=False)
    )
    # Cap nodes for readability
    flows = flows.head(40)

    provinces = flows["province"].unique().tolist()
    destinations = flows["destination"].unique().tolist()
    labels = provinces + destinations
    src = flows["province"].map({p: i for i, p in enumerate(provinces)})
    dst = flows["destination"].map({d: i + len(provinces) for i, d in enumerate(destinations)})

    title_period = "latest month" if period == "latest" else "selected period"
    prod_lbl = product if product != "All" else "all products"
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=12,
                thickness=16,
                line=dict(color="#30363d", width=0.5),
                label=labels,
                color=["#3fb950"] * len(provinces) + ["#58a6ff"] * len(destinations),
            ),
            link=dict(
                source=src,
                target=dst,
                value=flows["volume_kt"],
                color="rgba(88, 166, 255, 0.35)",
            ),
        )
    )
    fig.update_layout(
        title=f"Province → destination flows — {prod_lbl} ({title_period}, kt)",
        font=dict(size=11),
    )
    return fig


def fig_export_quotas_monthly(quotas_monthly: pd.DataFrame) -> go.Figure:
    if quotas_monthly.empty:
        return go.Figure().update_layout(title="No quota export data")

    fig = go.Figure()
    for prod in _SHIPPING_PRODUCTS:
        s = quotas_monthly[quotas_monthly["product"] == prod].sort_values("month_start")
        if s.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=s["month_start"],
                y=s["export_kt"],
                name=prod,
                marker_color=EXPORT_PRODUCT_COLORS[prod],
            )
        )
    fig.update_layout(
        title="National export quotas — monthly utilization by product (kt)",
        barmode="group",
        xaxis_title="Month",
        yaxis_title="Exports (kt)",
    )
    return fig


def fig_export_quota_summary(
    quotas_summary: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
) -> go.Figure:
    if quotas_summary.empty:
        return go.Figure().update_layout(title="No quota summary")

    row = quotas_summary.iloc[0]
    year = int(row["quota_year"]) if pd.notna(row.get("quota_year")) else pd.Timestamp.today().year
    total = row.get("total_quota_kt")
    ytd = row.get("ytd_export_kt")
    util = row.get("utilization")

    # Cumulative monthly exports for context
    cum = pd.DataFrame()
    if not quotas_monthly.empty:
        m = quotas_monthly[quotas_monthly["year"] == year].copy()
        m = m.groupby("month_start", as_index=False)["export_kt"].sum().sort_values("month_start")
        m["cumulative_kt"] = m["export_kt"].cumsum()
        cum = m

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            f"Cumulative exports vs {year:,} quota (kt)",
            "Quota allocation",
        ),
    )

    if not cum.empty:
        fig.add_trace(
            go.Scatter(
                x=cum["month_start"],
                y=cum["cumulative_kt"],
                mode="lines+markers",
                name="Cumulative exports",
                line=dict(color="#58a6ff"),
            ),
            row=1,
            col=1,
        )
    if pd.notna(total):
        fig.add_hline(
            y=float(total),
            line_dash="dash",
            line_color="#f85149",
            annotation_text=f"Total quota {total:,.0f} kt",
            row=1,
            col=1,
        )

    batches = []
    labels = []
    colors = ["#3fb950", "#58a6ff", "#bc8cff"]
    for i, col in enumerate(["batch_1_kt", "batch_2_kt", "batch_3_kt"]):
        val = row.get(col)
        if pd.notna(val) and val > 0:
            batches.append(float(val))
            labels.append(f"Batch {i + 1}")
    if batches:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=batches,
                marker_color=colors[: len(batches)],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    util_pct = f"{float(util) * 100:.1f}%" if pd.notna(util) else "n/a"
    ytd_lbl = f"{float(ytd):,.0f} kt" if pd.notna(ytd) else "n/a"
    fig.update_layout(
        title=f"Product export quotas {year} — YTD {ytd_lbl}, utilization {util_pct}",
    )
    fig.update_yaxes(title_text="kt", row=1, col=1)
    fig.update_yaxes(title_text="kt", row=1, col=2)
    return fig


def export_monthly_table_records(
    shipping: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    *,
    source: str,
) -> list[dict]:
    """Rows for DataTable / CSV download."""
    if source == "quotas":
        wide = build_export_monthly_from_quotas(quotas_monthly)
    else:
        wide = build_export_monthly_from_shipping(shipping)

    if wide.empty:
        return []

    records = []
    for _, row in wide.sort_values("month").iterrows():
        rec = {"month": row["month"]}
        for col in EXPORT_TABLE_PRODUCTS:
            val = row.get(col, 0)
            rec[col] = int(round(val)) if pd.notna(val) else 0
        records.append(rec)
    return records


def export_monthly_table_df(
    shipping: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    records = export_monthly_table_records(shipping, quotas_monthly, source=source)
    if not records:
        return pd.DataFrame(columns=["month"] + EXPORT_TABLE_PRODUCTS)
    return pd.DataFrame(records)
