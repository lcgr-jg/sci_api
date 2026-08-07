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
    # SCI headline YTD ("Gasoline, diesel and kerosene total export") can diverge
    # from the sum of monthly product rows — utilization uses the headline.
    headline_ytd = row.get("ytd_export_kt")
    util = row.get("utilization")

    # Cumulative from monthly product rows only (drop empty future months).
    cum = pd.DataFrame()
    monthly_ytd = None
    if not quotas_monthly.empty:
        m = quotas_monthly[quotas_monthly["year"] == year].copy()
        m = m.groupby("month_start", as_index=False)["export_kt"].sum().sort_values("month_start")
        m = m[m["export_kt"].fillna(0) > 0]
        if not m.empty:
            m["cumulative_kt"] = m["export_kt"].cumsum()
            cum = m
            monthly_ytd = float(m["cumulative_kt"].iloc[-1])

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            f"Cumulative monthly exports vs {year:,} quota (kt)",
            "Quota allocation",
        ),
    )

    if not cum.empty:
        fig.add_trace(
            go.Scatter(
                x=cum["month_start"],
                y=cum["cumulative_kt"],
                mode="lines+markers",
                name="Cumulative (monthly rows)",
                line=dict(color="#58a6ff"),
            ),
            row=1,
            col=1,
        )
        # Mark SCI headline YTD when it disagrees with the monthly sum.
        if (
            pd.notna(headline_ytd)
            and monthly_ytd is not None
            and abs(float(headline_ytd) - monthly_ytd) > 1.0
        ):
            fig.add_hline(
                y=float(headline_ytd),
                line_dash="dot",
                line_color="#e3b341",
                annotation_text=f"SCI headline YTD {float(headline_ytd):,.0f}",
                annotation_position="bottom right",
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
    monthly_lbl = f"{monthly_ytd:,.0f} kt" if monthly_ytd is not None else "n/a"
    headline_lbl = f"{float(headline_ytd):,.0f} kt" if pd.notna(headline_ytd) else "n/a"
    # Title tracks the plotted series; subtitle notes SCI's separate headline used for utilization.
    title = (
        f"Product export quotas {year} — monthly YTD {monthly_lbl}"
        f"<br><sup>SCI headline YTD {headline_lbl} (drives utilization {util_pct})</sup>"
    )
    fig.update_layout(title=title)
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


def _forecast_plan_month(forecast: pd.DataFrame) -> pd.Timestamp | None:
    if forecast.empty:
        return None
    return pd.to_datetime(forecast["month_start"]).max()


def fig_export_forecast_plan(
    forecast: pd.DataFrame,
    shipping: pd.DataFrame,
    *,
    product: str = "All",
) -> go.Figure:
    """
    Sheet-41 forward cargoes as a monthly export-plan tracker.

    Company-level SCI export plans are not in the API datapack; cargo forecasts
    are the closest proxy for planned loadings in the current plan month.
    """
    if forecast.empty:
        return go.Figure().update_layout(
            title="No forward export cargoes (sheet 41) in cache"
        )

    plan_month = _forecast_plan_month(forecast)
    fc = forecast[forecast["month_start"] == plan_month].copy()
    if product and product != "All":
        fc = fc[fc["product"] == product]

    by_prod = (
        fc.groupby("product", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=False)
    )
    plan_total = float(by_prod["volume_kt"].sum()) if not by_prod.empty else 0.0

    # Same calendar month last year from shipping actuals (balance context).
    prior = None
    if not shipping.empty and plan_month is not None:
        prior_month = plan_month - pd.DateOffset(years=1)
        sh = shipping[shipping["month_start"] == prior_month]
        if product and product != "All":
            sh = sh[sh["product"] == product]
        if not sh.empty:
            prior = float(sh["volume_kt"].sum())

    fig = go.Figure()
    if not by_prod.empty:
        fig.add_trace(
            go.Bar(
                x=by_prod["product"],
                y=by_prod["volume_kt"],
                marker_color=[
                    EXPORT_PRODUCT_COLORS.get(p, "#58a6ff") for p in by_prod["product"]
                ],
                name="Planned cargoes",
                text=[f"{v:,.0f}" for v in by_prod["volume_kt"]],
                textposition="outside",
            )
        )
    if prior is not None:
        fig.add_hline(
            y=prior,
            line_dash="dot",
            line_color="#8b949e",
            annotation_text=f"Same month LY actual {prior:,.0f} kt",
        )

    month_lbl = plan_month.strftime("%b %Y") if plan_month is not None else ""
    prod_lbl = product if product != "All" else "all products"
    fig.update_layout(
        title=(
            f"Forward export plan tracker — {month_lbl} ({prod_lbl}) · "
            f"{plan_total:,.0f} kt planned"
            "<br><sup>Sheet 41 cargo forecast (company monthly plans not in API)</sup>"
        ),
        xaxis_title="Product",
        yaxis_title="Planned volume (kt)",
    )
    return fig


def export_forecast_table_records(
    forecast: pd.DataFrame,
    *,
    product: str = "All",
) -> list[dict]:
    """Cargo-level rows for the plan-month forecast table."""
    if forecast.empty:
        return []
    plan_month = _forecast_plan_month(forecast)
    fc = forecast[forecast["month_start"] == plan_month].copy()
    if product and product != "All":
        fc = fc[fc["product"] == product]
    if fc.empty:
        return []

    fc = fc.sort_values(["departure_date", "product", "destination"])
    records = []
    for _, row in fc.iterrows():
        records.append(
            {
                "departure": pd.Timestamp(row["departure_date"]).strftime("%Y-%m-%d"),
                "product": row["product"],
                "volume_kt": round(float(row["volume_kt"]), 1),
                "province": row.get("province"),
                "port": row.get("departure_port"),
                "destination": row.get("destination"),
                "ship": row.get("ship_name"),
            }
        )
    return records


def export_forecast_summary_records(
    forecast: pd.DataFrame,
    shipping: pd.DataFrame,
    *,
    product: str = "All",
) -> list[dict]:
    """Compact plan-vs-context summary by product for the plan month."""
    if forecast.empty:
        return []
    plan_month = _forecast_plan_month(forecast)
    fc = forecast[forecast["month_start"] == plan_month].copy()
    if product and product != "All":
        fc = fc[fc["product"] == product]

    planned = fc.groupby("product", as_index=False)["volume_kt"].sum()
    prior_month = plan_month - pd.DateOffset(years=1) if plan_month is not None else None
    # MTD actuals in the plan month (partial month from sheet 40).
    mtd = pd.DataFrame(columns=["product", "volume_kt"])
    ly = pd.DataFrame(columns=["product", "volume_kt"])
    if not shipping.empty and plan_month is not None:
        mtd_src = shipping[shipping["month_start"] == plan_month]
        if product and product != "All":
            mtd_src = mtd_src[mtd_src["product"] == product]
        if not mtd_src.empty:
            mtd = mtd_src.groupby("product", as_index=False)["volume_kt"].sum()
        ly_src = shipping[shipping["month_start"] == prior_month]
        if product and product != "All":
            ly_src = ly_src[ly_src["product"] == product]
        if not ly_src.empty:
            ly = ly_src.groupby("product", as_index=False)["volume_kt"].sum()

    products = sorted(set(planned["product"]) | set(mtd.get("product", [])) | set(ly.get("product", [])))
    mtd_map = dict(zip(mtd["product"], mtd["volume_kt"])) if not mtd.empty else {}
    ly_map = dict(zip(ly["product"], ly["volume_kt"])) if not ly.empty else {}
    plan_map = dict(zip(planned["product"], planned["volume_kt"])) if not planned.empty else {}

    month_lbl = plan_month.strftime("%Y-%m") if plan_month is not None else ""
    rows = []
    for prod in products:
        plan_v = float(plan_map.get(prod, 0.0))
        mtd_v = float(mtd_map.get(prod, 0.0))
        ly_v = ly_map.get(prod)
        rows.append(
            {
                "month": month_lbl,
                "product": prod,
                "planned_kt": round(plan_v, 1),
                "mtd_actual_kt": round(mtd_v, 1),
                "remaining_vs_plan_kt": round(plan_v - mtd_v, 1),
                "same_month_ly_kt": round(float(ly_v), 1) if ly_v is not None else None,
            }
        )
    return rows
