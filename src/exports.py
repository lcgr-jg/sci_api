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
from src.units import convert_volume_series, volume_label

EXPORT_PRODUCT_COLORS = {
    "Gasoline": "#58a6ff",
    "Jet": "#bc8cff",
    "Gasoil/Diesel": "#f85149",
    "FO": "#e3b341",
    "All": "#8b949e",
}

_SHIPPING_PRODUCTS = ["Gasoline", "Jet", "Gasoil/Diesel"]

# Dashboard product labels → units.py conversion keys.
_EXPORT_PRODUCT_KEYS = {
    "Gasoline": "gasoline",
    "Jet": "kerosene",
    "Gasoil/Diesel": "gasoil",
    "FO": "fuel_oil",
}


def _product_key(product: str | None) -> str | None:
    if not product or product == "All":
        return None
    return _EXPORT_PRODUCT_KEYS.get(product)


def _to_display(
    values: pd.Series,
    unit: str,
    *,
    dates: pd.Series,
    product: str | None,
) -> pd.Series:
    """Monthly kt → selected display unit (per-product barrels/mt + days-in-month)."""
    return convert_volume_series(
        values,
        unit,
        freq="monthly",
        dates=dates,
        product=_product_key(product) if product else None,
    )


def _with_display_volume(
    df: pd.DataFrame,
    unit: str,
    *,
    value_col: str = "volume_kt",
    date_col: str = "month_start",
    product_col: str = "product",
) -> pd.DataFrame:
    """Attach volume_display converted row-wise by product (needed before All-product sums)."""
    if df.empty:
        out = df.copy()
        out["volume_display"] = pd.Series(dtype=float)
        return out
    parts: list[pd.DataFrame] = []
    for prod, g in df.groupby(product_col, sort=False):
        gg = g.copy()
        gg["volume_display"] = _to_display(
            gg[value_col], unit, dates=gg[date_col], product=str(prod)
        ).to_numpy()
        parts.append(gg)
    return pd.concat(parts, axis=0)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Keep product hue visible on forecast bars (pattern + opacity alone washes to grey)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _forecast_bar_marker(color: str) -> dict:
    # Soft hatch over translucent product colour — same legend as solid actuals.
    return dict(
        color=_hex_to_rgba(color, 0.65),
        line=dict(color=color, width=1),
        pattern=dict(
            shape="/",
            size=6,
            solidity=0.15,
            fgcolor="rgba(22, 27, 34, 0.45)",
            bgcolor=_hex_to_rgba(color, 0.65),
        ),
    )


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


def _forecast_months_without_actuals(
    forecast: pd.DataFrame,
    actual_months: set,
    *,
    product: str,
) -> pd.DataFrame:
    """Sheet-41 months that do not already have shipping actual bars (avoids double-count)."""
    if forecast.empty:
        return forecast.iloc[0:0]
    fc = forecast.copy()
    if product and product != "All":
        fc = fc[fc["product"] == product]
    if fc.empty:
        return fc
    fc["month_start"] = pd.to_datetime(fc["month_start"])
    return fc[~fc["month_start"].isin(actual_months)]


def fig_export_monthly_stacked(
    shipping: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    unit: str = "kt",
) -> go.Figure:
    """Monthly export volume: solid stacked actuals + hatched stacked forecast months."""
    sub = _filter_shipping(shipping, product=product, date_range=date_range)
    fig = go.Figure()
    ylab = volume_label(unit)

    # Months already plotted as actuals — forecast bars skip these.
    actual_months: set = set()
    if not sub.empty:
        actual_months = set(pd.to_datetime(sub["month_start"]).unique())

    fc_only = _forecast_months_without_actuals(
        forecast, actual_months, product=product or "All"
    )
    has_forecast_bars = not fc_only.empty

    if sub.empty and not has_forecast_bars:
        return fig.update_layout(title="No shipping data for selection")

    show_all = product == "All" or not product
    products = _SHIPPING_PRODUCTS if show_all else [product]
    # Products that already contributed a solid legend swatch (forecast traces stay silent).
    legend_products: set[str] = set()

    if not sub.empty:
        if show_all:
            monthly = (
                sub.groupby(["month_start", "product"], as_index=False)["volume_kt"]
                .sum()
                .sort_values("month_start")
            )
            for prod in products:
                s = monthly[monthly["product"] == prod]
                if s.empty:
                    continue
                y = _to_display(s["volume_kt"], unit, dates=s["month_start"], product=prod)
                fig.add_trace(
                    go.Bar(
                        x=s["month_start"],
                        y=y,
                        name=prod,
                        legendgroup=prod,
                        marker_color=EXPORT_PRODUCT_COLORS[prod],
                    )
                )
                legend_products.add(prod)
        else:
            monthly = (
                sub.groupby("month_start", as_index=False)["volume_kt"]
                .sum()
                .sort_values("month_start")
            )
            y = _to_display(
                monthly["volume_kt"], unit, dates=monthly["month_start"], product=product
            )
            fig.add_trace(
                go.Bar(
                    x=monthly["month_start"],
                    y=y,
                    name=product,
                    legendgroup=product,
                    marker_color=EXPORT_PRODUCT_COLORS.get(product, "#58a6ff"),
                )
            )
            legend_products.add(product)

    # Forecast months: same product colours (translucent + light hatch); no extra legend rows.
    if has_forecast_bars:
        if show_all:
            fc_monthly = (
                fc_only.groupby(["month_start", "product"], as_index=False)["volume_kt"]
                .sum()
                .sort_values("month_start")
            )
            for prod in products:
                s = fc_monthly[fc_monthly["product"] == prod]
                if s.empty:
                    continue
                show_leg = prod not in legend_products
                y = _to_display(s["volume_kt"], unit, dates=s["month_start"], product=prod)
                fig.add_trace(
                    go.Bar(
                        x=s["month_start"],
                        y=y,
                        name=prod,
                        legendgroup=prod,
                        showlegend=show_leg,
                        marker=_forecast_bar_marker(EXPORT_PRODUCT_COLORS[prod]),
                        hovertemplate=(
                            f"{prod} (forecast)<br>%{{x|%b %Y}}: %{{y:,.1f}} {ylab}"
                            "<extra>unconfirmed</extra>"
                        ),
                    )
                )
                if show_leg:
                    legend_products.add(prod)
        else:
            fc_monthly = (
                fc_only.groupby("month_start", as_index=False)["volume_kt"]
                .sum()
                .sort_values("month_start")
            )
            color = EXPORT_PRODUCT_COLORS.get(product, "#58a6ff")
            y = _to_display(
                fc_monthly["volume_kt"],
                unit,
                dates=fc_monthly["month_start"],
                product=product,
            )
            fig.add_trace(
                go.Bar(
                    x=fc_monthly["month_start"],
                    y=y,
                    name=product,
                    legendgroup=product,
                    showlegend=product not in legend_products,
                    marker=_forecast_bar_marker(color),
                    hovertemplate=(
                        f"{product} (forecast)<br>%{{x|%b %Y}}: %{{y:,.1f}} {ylab}"
                        "<extra>unconfirmed</extra>"
                    ),
                )
            )

    if show_all:
        title = f"Monthly exports by product (shipping actuals, {ylab})"
    else:
        title = f"Monthly {product} exports (shipping actuals, {ylab})"
    if has_forecast_bars:
        title += (
            "<br><sup>Hatched bars = sheet-41 forecast cargoes (unconfirmed; "
            "use to track vs prior months)</sup>"
        )

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Departure month",
        yaxis_title=f"Volume ({ylab})",
    )
    return fig


def fig_export_destinations(
    shipping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    period: str,
    unit: str = "kt",
) -> go.Figure:
    sub = _period_slice(_filter_shipping(shipping, product=product, date_range=date_range), period)
    if sub.empty:
        return go.Figure().update_layout(title="No destination data")

    ylab = volume_label(unit)
    disp = _with_display_volume(sub, unit)
    top = (
        disp.groupby("destination", as_index=False)["volume_display"]
        .sum()
        .sort_values("volume_display", ascending=True)
        .tail(15)
    )
    title_period = "latest month" if period == "latest" else "selected period"
    fig = go.Figure(
        go.Bar(
            x=top["volume_display"],
            y=top["destination"],
            orientation="h",
            marker_color="#58a6ff",
            text=[f"{v:,.0f}" for v in top["volume_display"]],
            textposition="outside",
        )
    )
    prod_lbl = product if product != "All" else "all products"
    fig.update_layout(
        title=f"Top destinations — {prod_lbl} ({title_period}, {ylab})",
        xaxis_title=f"Volume ({ylab})",
        yaxis_title="",
    )
    return fig


def fig_export_provinces(
    shipping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    period: str,
    unit: str = "kt",
) -> go.Figure:
    sub = _period_slice(_filter_shipping(shipping, product=product, date_range=date_range), period)
    if sub.empty:
        return go.Figure().update_layout(title="No province data")

    ylab = volume_label(unit)
    disp = _with_display_volume(sub, unit)
    top = (
        disp.groupby("province", as_index=False)["volume_display"]
        .sum()
        .sort_values("volume_display", ascending=True)
        .tail(15)
    )
    title_period = "latest month" if period == "latest" else "selected period"
    fig = go.Figure(
        go.Bar(
            x=top["volume_display"],
            y=top["province"],
            orientation="h",
            marker_color="#3fb950",
            text=[f"{v:,.0f}" for v in top["volume_display"]],
            textposition="outside",
        )
    )
    prod_lbl = product if product != "All" else "all products"
    fig.update_layout(
        title=f"Top departure provinces — {prod_lbl} ({title_period}, {ylab})",
        xaxis_title=f"Volume ({ylab})",
        yaxis_title="",
    )
    return fig


def fig_export_sankey(
    shipping: pd.DataFrame,
    *,
    product: str,
    date_range: str,
    period: str,
    unit: str = "kt",
) -> go.Figure:
    sub = _period_slice(_filter_shipping(shipping, product=product, date_range=date_range), period)
    if sub.empty:
        return go.Figure().update_layout(title="No flow data")

    ylab = volume_label(unit)
    disp = _with_display_volume(sub, unit)
    flows = (
        disp.groupby(["province", "destination"], as_index=False)["volume_display"]
        .sum()
        .sort_values("volume_display", ascending=False)
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
                value=flows["volume_display"],
                color="rgba(88, 166, 255, 0.35)",
            ),
        )
    )
    fig.update_layout(
        title=f"Province → destination flows — {prod_lbl} ({title_period}, {ylab})",
        font=dict(size=11),
    )
    return fig


def fig_export_forecast_sankey(
    forecast: pd.DataFrame,
    *,
    product: str = "All",
    unit: str = "kt",
) -> go.Figure:
    """Sheet-41 plan-month flows: product → destination country."""
    if forecast.empty:
        return go.Figure().update_layout(
            title="No forward export cargoes (sheet 41) in cache"
        )

    plan_month = _forecast_plan_month(forecast)
    fc = forecast[forecast["month_start"] == plan_month].copy()
    if product and product != "All":
        fc = fc[fc["product"] == product]
    if fc.empty:
        return go.Figure().update_layout(title="No forecast cargoes for selection")

    ylab = volume_label(unit)
    disp = _with_display_volume(fc, unit)
    flows = (
        disp.groupby(["product", "destination"], as_index=False)["volume_display"]
        .sum()
        .sort_values("volume_display", ascending=False)
        .head(40)
    )
    flows = flows.dropna(subset=["destination"])
    if flows.empty:
        return go.Figure().update_layout(title="No destination data in forecast cargoes")

    products = flows["product"].unique().tolist()
    destinations = flows["destination"].unique().tolist()
    labels = products + destinations
    node_colors = [
        EXPORT_PRODUCT_COLORS.get(p, "#8b949e") for p in products
    ] + ["#58a6ff"] * len(destinations)
    src = flows["product"].map({p: i for i, p in enumerate(products)})
    dst = flows["destination"].map(
        {d: i + len(products) for i, d in enumerate(destinations)}
    )

    month_lbl = plan_month.strftime("%b %Y") if plan_month is not None else ""
    prod_lbl = product if product != "All" else "all products"
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=12,
                thickness=16,
                line=dict(color="#30363d", width=0.5),
                label=labels,
                color=node_colors,
            ),
            link=dict(
                source=src,
                target=dst,
                value=flows["volume_display"],
                color="rgba(88, 166, 255, 0.35)",
            ),
        )
    )
    fig.update_layout(
        title=(
            f"Forecast destinations — {month_lbl} ({prod_lbl}, {ylab})"
            "<br><sup>Sheet 41 product → destination</sup>"
        ),
        font=dict(size=11),
    )
    return fig


def fig_export_quotas_monthly(
    quotas_monthly: pd.DataFrame,
    *,
    unit: str = "kt",
) -> go.Figure:
    if quotas_monthly.empty:
        return go.Figure().update_layout(title="No quota export data")

    ylab = volume_label(unit)
    fig = go.Figure()
    for prod in _SHIPPING_PRODUCTS:
        s = quotas_monthly[quotas_monthly["product"] == prod].sort_values("month_start")
        if s.empty:
            continue
        y = _to_display(s["export_kt"], unit, dates=s["month_start"], product=prod)
        fig.add_trace(
            go.Bar(
                x=s["month_start"],
                y=y,
                name=prod,
                marker_color=EXPORT_PRODUCT_COLORS[prod],
            )
        )
    fig.update_layout(
        title=f"National export quotas — monthly utilization by product ({ylab})",
        barmode="group",
        xaxis_title="Month",
        yaxis_title=f"Exports ({ylab})",
    )
    return fig


def fig_export_quota_summary(
    quotas_summary: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    *,
    unit: str = "kt",
) -> go.Figure:
    if quotas_summary.empty:
        return go.Figure().update_layout(title="No quota summary")

    ylab = volume_label(unit)
    row = quotas_summary.iloc[0]
    year = int(row["quota_year"]) if pd.notna(row.get("quota_year")) else pd.Timestamp.today().year
    total = row.get("total_quota_kt")
    # SCI headline YTD ("Gasoline, diesel and kerosene total export") can diverge
    # from the sum of monthly product rows — utilization uses the headline.
    headline_ytd = row.get("ytd_export_kt")
    util = row.get("utilization")

    # Cumulative from monthly product rows only (drop empty future months).
    # Convert per product before summing so kb/d uses correct barrels/mt.
    cum = pd.DataFrame()
    monthly_ytd = None
    kt_ytd = None
    if not quotas_monthly.empty:
        m_raw = quotas_monthly[quotas_monthly["year"] == year].copy()
        if not m_raw.empty:
            m_kt = (
                m_raw.groupby("month_start", as_index=False)["export_kt"]
                .sum()
                .sort_values("month_start")
            )
            m_kt = m_kt[m_kt["export_kt"].fillna(0) > 0]
            if not m_kt.empty:
                kt_ytd = float(m_kt["export_kt"].sum())

            m = _with_display_volume(
                m_raw, unit, value_col="export_kt", date_col="month_start"
            )
            m = (
                m.groupby("month_start", as_index=False)["volume_display"]
                .sum()
                .sort_values("month_start")
            )
            m = m[m["volume_display"].fillna(0) > 0]
            if not m.empty:
                m["cumulative"] = m["volume_display"].cumsum()
                cum = m
                monthly_ytd = float(m["cumulative"].iloc[-1])

    # Undifferentiated SCI headline / batch totals: default barrels/mt.
    ref_date = pd.Series([pd.Timestamp(year=year, month=6, day=1)])

    def _scalar_kt(val) -> float | None:
        if pd.isna(val):
            return None
        return float(
            _to_display(pd.Series([float(val)]), unit, dates=ref_date, product=None).iloc[0]
        )

    total_disp = _scalar_kt(total)
    headline_disp = _scalar_kt(headline_ytd)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            f"Cumulative monthly exports vs {year:,} quota ({ylab})",
            "Quota allocation",
        ),
    )

    if not cum.empty:
        fig.add_trace(
            go.Scatter(
                x=cum["month_start"],
                y=cum["cumulative"],
                mode="lines+markers",
                name="Cumulative (monthly rows)",
                line=dict(color="#58a6ff"),
            ),
            row=1,
            col=1,
        )
        # Compare headline vs monthly sum in kt space; plot converted headline.
        if (
            pd.notna(headline_ytd)
            and kt_ytd is not None
            and abs(float(headline_ytd) - kt_ytd) > 1.0
            and headline_disp is not None
        ):
            fig.add_hline(
                y=headline_disp,
                line_dash="dot",
                line_color="#e3b341",
                annotation_text=f"SCI headline YTD {headline_disp:,.0f}",
                annotation_position="bottom right",
                row=1,
                col=1,
            )
    if total_disp is not None:
        fig.add_hline(
            y=total_disp,
            line_dash="dash",
            line_color="#f85149",
            annotation_text=f"Total quota {total_disp:,.0f} {ylab}",
            row=1,
            col=1,
        )

    batches = []
    labels = []
    colors = ["#3fb950", "#58a6ff", "#bc8cff"]
    for i, col in enumerate(["batch_1_kt", "batch_2_kt", "batch_3_kt"]):
        val = _scalar_kt(row.get(col))
        if val is not None and val > 0:
            batches.append(val)
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
    monthly_lbl = f"{monthly_ytd:,.0f} {ylab}" if monthly_ytd is not None else "n/a"
    headline_lbl = f"{headline_disp:,.0f} {ylab}" if headline_disp is not None else "n/a"
    title = (
        f"Product export quotas {year} — monthly YTD {monthly_lbl}"
        f"<br><sup>SCI headline YTD {headline_lbl} (drives utilization {util_pct})</sup>"
    )
    fig.update_layout(title=title)
    fig.update_yaxes(title_text=ylab, row=1, col=1)
    fig.update_yaxes(title_text=ylab, row=1, col=2)
    return fig


def export_monthly_table_records(
    shipping: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    *,
    source: str,
    unit: str = "kt",
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
        month = pd.Timestamp(row["month"])
        dates = pd.Series([month])
        rec = {"month": row["month"]}
        for col in EXPORT_TABLE_PRODUCTS:
            val = row.get(col, 0)
            raw = float(val) if pd.notna(val) else 0.0
            disp = float(
                _to_display(pd.Series([raw]), unit, dates=dates, product=col).iloc[0]
            )
            rec[col] = int(round(disp)) if unit == "kt" else round(disp, 1)
        records.append(rec)
    return records


def export_monthly_table_df(
    shipping: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    *,
    source: str,
    unit: str = "kt",
) -> pd.DataFrame:
    records = export_monthly_table_records(
        shipping, quotas_monthly, source=source, unit=unit
    )
    if not records:
        return pd.DataFrame(columns=["month"] + EXPORT_TABLE_PRODUCTS)
    return pd.DataFrame(records)


# Display labels matching the Excel-style Exports pivot (SCI cargo names).
_PIVOT_PRODUCTS = [
    ("Gasoline", "Gasoline"),
    ("Gasoil/Diesel", "Diesel"),
    ("Jet", "Kerosene"),
]


def export_monthly_pivot_columns(
    shipping: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    source: str,
    unit: str = "kt",
) -> list[dict]:
    """Dash column defs for the product × month pivot (dynamic month ids)."""
    months = _export_pivot_month_labels(shipping, quotas_monthly, forecast, source=source)
    ylab = volume_label(unit)
    cols: list[dict] = [{"name": f"Exports ({ylab})", "id": "product"}]
    for m in months:
        cols.append(
            {
                "name": m,
                "id": m,
                "type": "numeric",
                "format": {"specifier": ",.1f"},
            }
        )
    return cols


def _export_pivot_month_labels(
    shipping: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    source: str,
) -> list[str]:
    """Ordered MMM-YY labels covering actuals (selected source) plus forecast-only months."""
    if source == "quotas":
        wide = build_export_monthly_from_quotas(quotas_monthly)
    else:
        wide = build_export_monthly_from_shipping(shipping)

    month_ts: list[pd.Timestamp] = []
    if not wide.empty:
        month_ts.extend(pd.to_datetime(wide["month"]).tolist())

    # Append sheet-41 months not already in actuals so forward stubs appear.
    if not forecast.empty:
        fc_months = pd.to_datetime(forecast["month_start"]).unique()
        actual_set = set(pd.to_datetime(wide["month"]).unique()) if not wide.empty else set()
        for m in fc_months:
            if m not in actual_set:
                month_ts.append(pd.Timestamp(m))

    if not month_ts:
        return []
    ordered = sorted({pd.Timestamp(m).to_period("M").to_timestamp() for m in month_ts})
    # Excel-style pivot is year-to-date of the latest year (full history stays in the long table).
    latest_year = max(m.year for m in ordered)
    ordered = [m for m in ordered if m.year == latest_year]
    return [m.strftime("%b-%y") for m in ordered]


def export_monthly_pivot_records(
    shipping: pd.DataFrame,
    quotas_monthly: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    source: str,
    unit: str = "kt",
) -> list[dict]:
    """
    Product-as-rows × month-as-columns pivot, plus a bold Total row.

    Actuals come from the selected source; forecast-only months fill from sheet 41
    so the table mirrors the chart's forward cargo stubs. Values follow unit toggle.
    """
    month_labels = _export_pivot_month_labels(
        shipping, quotas_monthly, forecast, source=source
    )
    if not month_labels:
        return []

    if source == "quotas":
        wide = build_export_monthly_from_quotas(quotas_monthly)
    else:
        wide = build_export_monthly_from_shipping(shipping)

    label_to_ts: dict[str, pd.Timestamp] = {}
    all_ts: list = []
    if not wide.empty:
        all_ts.extend(pd.to_datetime(wide["month"]).tolist())
    if not forecast.empty:
        all_ts.extend(pd.to_datetime(forecast["month_start"]).tolist())
    for ts in all_ts:
        t = pd.Timestamp(ts).to_period("M").to_timestamp()
        label_to_ts[t.strftime("%b-%y")] = t

    # month label -> product -> volume (kt)
    values: dict[str, dict[str, float]] = {lbl: {} for lbl in month_labels}
    if not wide.empty:
        for _, row in wide.iterrows():
            lbl = pd.Timestamp(row["month"]).strftime("%b-%y")
            if lbl not in values:
                continue
            for prod, _disp in _PIVOT_PRODUCTS:
                val = row.get(prod, 0)
                values[lbl][prod] = float(val) if pd.notna(val) else 0.0

    if not forecast.empty:
        fc = forecast.copy()
        fc["month_start"] = pd.to_datetime(fc["month_start"])
        fc_g = fc.groupby(["month_start", "product"], as_index=False)["volume_kt"].sum()
        actual_labels = set()
        if not wide.empty:
            actual_labels = {pd.Timestamp(m).strftime("%b-%y") for m in wide["month"]}
        for _, row in fc_g.iterrows():
            lbl = pd.Timestamp(row["month_start"]).strftime("%b-%y")
            if lbl not in values:
                continue
            # Only inject forecast into months without actuals (same rule as the chart).
            if lbl in actual_labels:
                continue
            values[lbl][row["product"]] = float(row["volume_kt"])

    records: list[dict] = []
    totals = {lbl: 0.0 for lbl in month_labels}
    for prod, display in _PIVOT_PRODUCTS:
        rec: dict = {"product": display}
        for lbl in month_labels:
            v_kt = float(values[lbl].get(prod, 0.0))
            ts = label_to_ts.get(lbl, pd.Timestamp.today().to_period("M").to_timestamp())
            v = float(
                _to_display(
                    pd.Series([v_kt]), unit, dates=pd.Series([ts]), product=prod
                ).iloc[0]
            )
            rec[lbl] = round(v, 1)
            totals[lbl] += v
        records.append(rec)

    total_row: dict = {"product": "Total"}
    for lbl in month_labels:
        total_row[lbl] = round(totals[lbl], 1)
    records.append(total_row)
    return records


def _forecast_plan_month(forecast: pd.DataFrame) -> pd.Timestamp | None:
    if forecast.empty:
        return None
    return pd.to_datetime(forecast["month_start"]).max()


def fig_export_forecast_plan(
    forecast: pd.DataFrame,
    shipping: pd.DataFrame,
    *,
    product: str = "All",
    unit: str = "kt",
) -> go.Figure:
    """
    Sheet-41 forward cargoes as a monthly export-forecast tracker.

    Company-level SCI monthly export plans (Sinopec/PetroChina/etc.) are not in
    the API datapack; sheet 41 cargo forecasts are a separate weekly loading list.
    """
    if forecast.empty:
        return go.Figure().update_layout(
            title="No forward export cargoes (sheet 41) in cache"
        )

    ylab = volume_label(unit)
    plan_month = _forecast_plan_month(forecast)
    fc = forecast[forecast["month_start"] == plan_month].copy()
    if product and product != "All":
        fc = fc[fc["product"] == product]

    by_prod = (
        fc.groupby("product", as_index=False)["volume_kt"]
        .sum()
        .sort_values("volume_kt", ascending=False)
    )
    if not by_prod.empty and plan_month is not None:
        by_prod = by_prod.copy()
        by_prod["volume_display"] = [
            float(
                _to_display(
                    pd.Series([v]), unit, dates=pd.Series([plan_month]), product=p
                ).iloc[0]
            )
            for p, v in zip(by_prod["product"], by_prod["volume_kt"])
        ]
    elif not by_prod.empty:
        by_prod["volume_display"] = by_prod["volume_kt"]

    plan_total = float(by_prod["volume_display"].sum()) if not by_prod.empty else 0.0

    # Same calendar month last year from shipping actuals (balance context).
    prior = None
    if not shipping.empty and plan_month is not None:
        prior_month = plan_month - pd.DateOffset(years=1)
        sh = shipping[shipping["month_start"] == prior_month]
        if product and product != "All":
            sh = sh[sh["product"] == product]
        if not sh.empty:
            sh_disp = _with_display_volume(sh, unit)
            prior = float(sh_disp["volume_display"].sum())

    fig = go.Figure()
    if not by_prod.empty:
        fig.add_trace(
            go.Bar(
                x=by_prod["product"],
                y=by_prod["volume_display"],
                marker_color=[
                    EXPORT_PRODUCT_COLORS.get(p, "#58a6ff") for p in by_prod["product"]
                ],
                name="Forecasted cargoes",
                text=[f"{v:,.0f}" for v in by_prod["volume_display"]],
                textposition="outside",
            )
        )
    if prior is not None:
        fig.add_hline(
            y=prior,
            line_dash="dot",
            line_color="#8b949e",
            annotation_text=f"Same month LY actual {prior:,.0f} {ylab}",
        )

    month_lbl = plan_month.strftime("%b %Y") if plan_month is not None else ""
    prod_lbl = product if product != "All" else "all products"
    fig.update_layout(
        title=(
            f"Forward export forecast tracker — {month_lbl} ({prod_lbl}) · "
            f"{plan_total:,.0f} {ylab} forecasted"
            "<br><sup>Sheet 41 cargo forecast (company monthly plans not in API)</sup>"
        ),
        xaxis_title="Product",
        yaxis_title=f"Forecasted volume ({ylab})",
    )
    return fig


def export_forecast_table_records(
    forecast: pd.DataFrame,
    *,
    product: str = "All",
    unit: str = "kt",
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
        month = pd.Timestamp(row["month_start"])
        vol = float(
            _to_display(
                pd.Series([float(row["volume_kt"])]),
                unit,
                dates=pd.Series([month]),
                product=row["product"],
            ).iloc[0]
        )
        records.append(
            {
                "departure": pd.Timestamp(row["departure_date"]).strftime("%Y-%m-%d"),
                "product": row["product"],
                "volume": round(vol, 1),
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
    unit: str = "kt",
) -> list[dict]:
    """Compact plan-vs-context summary by product for the plan month."""
    if forecast.empty:
        return []
    plan_month = _forecast_plan_month(forecast)
    fc = forecast[forecast["month_start"] == plan_month].copy()
    if product and product != "All":
        fc = fc[fc["product"] == product]

    forecasted = fc.groupby("product", as_index=False)["volume_kt"].sum()
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

    products = sorted(
        set(forecasted["product"])
        | set(mtd.get("product", []))
        | set(ly.get("product", []))
    )
    mtd_map = dict(zip(mtd["product"], mtd["volume_kt"])) if not mtd.empty else {}
    ly_map = dict(zip(ly["product"], ly["volume_kt"])) if not ly.empty else {}
    fc_map = (
        dict(zip(forecasted["product"], forecasted["volume_kt"]))
        if not forecasted.empty
        else {}
    )

    month_lbl = plan_month.strftime("%Y-%m") if plan_month is not None else ""
    dates_plan = (
        pd.Series([plan_month]) if plan_month is not None else pd.Series([pd.Timestamp.today()])
    )
    dates_ly = pd.Series([prior_month]) if prior_month is not None else dates_plan

    def _conv(v: float, prod: str, dates: pd.Series) -> float:
        return float(_to_display(pd.Series([v]), unit, dates=dates, product=prod).iloc[0])

    rows = []
    for prod in products:
        fc_v = _conv(float(fc_map.get(prod, 0.0)), prod, dates_plan)
        mtd_v = _conv(float(mtd_map.get(prod, 0.0)), prod, dates_plan)
        ly_raw = ly_map.get(prod)
        ly_v = _conv(float(ly_raw), prod, dates_ly) if ly_raw is not None else None
        rows.append(
            {
                "month": month_lbl,
                "product": prod,
                "forecasted": round(fc_v, 1),
                "mtd_actual": round(mtd_v, 1),
                "same_month_ly": round(ly_v, 1) if ly_v is not None else None,
            }
        )
    return rows
