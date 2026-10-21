"""MoM/YoY demand tables and weekly products/stocks change tables."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from src.filters import filter_by_date_range
from src.transforms import (
    CONSUMPTION_PRODUCT_ORDER,
    JET_FUEL_PRODUCT_ORDER,
    PETCHEM_PRODUCT_ORDER,
)
from src.units import (
    CONSUMPTION_PRODUCT_MAP,
    PETCHEM_PRODUCT_MAP,
    convert_volume_series,
    volume_label,
)

# Products with both weekly national output and commercial stocks.
_STOCK_PRODUCTS = ("Gasoline", "Diesel")
_OUTPUT_PRODUCTS = ("Gasoline", "Diesel", "Kerosene")

_OUTPUT_COL = {
    "Gasoline": "gasoline_output_kt",
    "Diesel": "diesel_output_kt",
    "Kerosene": "kerosene_output_kt",
}
_STOCK_COL = {
    "Gasoline": "gasoline_stocks_kt",
    "Diesel": "diesel_stocks_kt",
}


def _safe_pct(num: float, den: float) -> float | None:
    if den is None or not math.isfinite(den) or den == 0:
        return None
    if num is None or not math.isfinite(num):
        return None
    return 100.0 * num / den


def _fmt_num(val: float | None, digits: int = 1) -> float | None:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return round(float(val), digits)


def mom_yoy_records(
    monthly: pd.DataFrame,
    *,
    value_col: str,
    product_order: list[str],
    unit: str = "kbd",
    product_key_map: dict[str, str] | None = None,
    include_total: bool = True,
    total_label: str = "Total (SCI)",
) -> list[dict]:
    """
    Build MoM/YoY change rows for the latest complete month.

    Expects columns: product, year, month, <value_col> in kt.
    """
    if monthly.empty or value_col not in monthly.columns:
        return []

    df = monthly.copy()
    df["month_start"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1)
    )
    # Prefer products that exist; keep requested order.
    present = [p for p in product_order if p in set(df["product"])]
    if not present:
        return []

    latest = df["month_start"].max()
    prev_m = latest - pd.DateOffset(months=1)
    prev_y = latest - pd.DateOffset(years=1)
    month_lbl = latest.strftime("%Y-%m")
    # Stable column ids for DataTable (avoid / in ids).
    level_id = f"level_{unit}"
    mom_id = f"mom_{unit}"
    yoy_id = f"yoy_{unit}"

    rows: list[dict] = []

    def _level(product: str, month_start: pd.Timestamp) -> float | None:
        sub = df[(df["product"] == product) & (df["month_start"] == month_start)]
        if sub.empty:
            return None
        kt = float(sub[value_col].iloc[0])
        prod_key = (product_key_map or {}).get(product)
        converted = convert_volume_series(
            pd.Series([kt]),
            unit,
            freq="monthly",
            dates=pd.Series([month_start]),
            product=prod_key,
        ).iloc[0]
        return float(converted)

    def _row(product: str, level: float | None, mom: float | None, yoy: float | None,
             base_mom: float | None, base_yoy: float | None) -> dict:
        return {
            "product": product,
            "month": month_lbl,
            level_id: _fmt_num(level),
            mom_id: _fmt_num(mom),
            "mom_pct": _fmt_num(_safe_pct(mom, base_mom) if mom is not None else None),
            yoy_id: _fmt_num(yoy),
            "yoy_pct": _fmt_num(_safe_pct(yoy, base_yoy) if yoy is not None else None),
        }

    levels_cur: list[float] = []

    for product in present:
        cur = _level(product, latest)
        pm = _level(product, prev_m)
        py = _level(product, prev_y)
        mom = (cur - pm) if cur is not None and pm is not None else None
        yoy = (cur - py) if cur is not None and py is not None else None
        rows.append(_row(product, cur, mom, yoy, pm, py))
        if cur is not None:
            levels_cur.append(cur)

    if include_total and levels_cur:
        # Total only over products present in the current month (apples-to-apples).
        tot_cur = sum(levels_cur)
        # Recompute total MoM/YoY from products that have both legs.
        mom_parts = []
        yoy_parts = []
        pm_base = []
        py_base = []
        for product in present:
            cur = _level(product, latest)
            pm = _level(product, prev_m)
            py = _level(product, prev_y)
            if cur is not None and pm is not None:
                mom_parts.append(cur - pm)
                pm_base.append(pm)
            if cur is not None and py is not None:
                yoy_parts.append(cur - py)
                py_base.append(py)
        tot_mom = sum(mom_parts) if mom_parts else None
        tot_yoy = sum(yoy_parts) if yoy_parts else None
        rows.insert(
            0,
            _row(
                total_label,
                tot_cur,
                tot_mom,
                tot_yoy,
                sum(pm_base) if pm_base else None,
                sum(py_base) if py_base else None,
            ),
        )

    return rows


def consumption_change_records(
    consumption_national: pd.DataFrame,
    jet_national: pd.DataFrame | None = None,
    *,
    unit: str = "kbd",
) -> list[dict]:
    """SCI national demand MoM/YoY (consumption products + jet refueling)."""
    frames = []
    if not consumption_national.empty:
        frames.append(
            consumption_national[
                consumption_national["product"].isin(CONSUMPTION_PRODUCT_ORDER)
            ]
        )
    if jet_national is not None and not jet_national.empty:
        frames.append(
            jet_national[jet_national["product"].isin(JET_FUEL_PRODUCT_ORDER)]
        )
    if not frames:
        return []
    monthly = pd.concat(frames, ignore_index=True)
    order = list(CONSUMPTION_PRODUCT_ORDER) + list(JET_FUEL_PRODUCT_ORDER)
    return mom_yoy_records(
        monthly,
        value_col="consumption_kt",
        product_order=order,
        unit=unit,
        product_key_map=CONSUMPTION_PRODUCT_MAP,
        total_label="Total (SCI)",
    )


def petchem_change_records(
    petchem_monthly: pd.DataFrame,
    *,
    unit: str = "kbd",
) -> list[dict]:
    """Petchem monthly output MoM/YoY."""
    if petchem_monthly.empty:
        return []
    out = petchem_monthly[petchem_monthly["metric"] == "output_kt"].copy()
    if out.empty:
        return []
    out = out.rename(columns={"value": "output_kt"})
    return mom_yoy_records(
        out,
        value_col="output_kt",
        product_order=list(PETCHEM_PRODUCT_ORDER),
        unit=unit,
        product_key_map=PETCHEM_PRODUCT_MAP,
        total_label="Total (petchem)",
    )


def change_table_columns(unit: str) -> list[dict]:
    """DataTable column defs for MoM/YoY tables."""
    unit_lbl = volume_label(unit)
    return [
        {"name": "product", "id": "product"},
        {"name": "month", "id": "month"},
        {
            "name": f"level_{unit_lbl}",
            "id": f"level_{unit}",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": f"mom_{unit_lbl}",
            "id": f"mom_{unit}",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "mom_pct",
            "id": "mom_pct",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": f"yoy_{unit_lbl}",
            "id": f"yoy_{unit}",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "yoy_pct",
            "id": "yoy_pct",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
    ]


def _product_key(product: str) -> str:
    return {
        "Gasoline": "gasoline",
        "Diesel": "diesel",
        "Kerosene": "kerosene",
    }.get(product, product.lower())


def _convert_kt(kt: float, unit: str, product: str) -> float:
    return float(
        convert_volume_series(
            pd.Series([kt]),
            unit,
            freq="weekly",
            product=_product_key(product),
        ).iloc[0]
    )


def products_scorecard_records(
    products: pd.DataFrame,
    *,
    stock_unit: str = "mb",
    flow_unit: str = "kbd",
) -> tuple[list[dict], list[dict]]:
    """
    One-row-per-product scorecard for the latest week.

    Columns: stocks, Δ stocks, days cover, output, Δ output, apparent demand, Δ app. dem.
    Apparent demand = output − Δstocks (ex-exports). Kerosene has output only.
    """
    stock_lbl = volume_label(stock_unit)
    flow_lbl = volume_label(flow_unit)

    columns = [
        {"name": "product", "id": "product"},
        {
            "name": f"stocks ({stock_lbl})",
            "id": "stocks",
            "type": "numeric",
            "format": {"specifier": ",.2f"},
        },
        {
            "name": f"Δ stocks ({stock_lbl})",
            "id": "d_stocks",
            "type": "numeric",
            "format": {"specifier": ",.2f"},
        },
        {
            "name": "days cover",
            "id": "days_cover",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": f"output ({flow_lbl})",
            "id": "output",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": f"Δ output ({flow_lbl})",
            "id": "d_output",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": f"app. demand ({flow_lbl})",
            "id": "app_demand",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": f"Δ app. dem. ({flow_lbl})",
            "id": "d_app_demand",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
    ]

    if products.empty or "week_start" not in products.columns:
        return [], columns

    df = products.sort_values("week_start").copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.dropna(subset=["week_start"])
    if len(df) < 1:
        return [], columns

    # Need prior week for deltas; use last two rows when available.
    tail = df.tail(2).reset_index(drop=True)
    cur = tail.iloc[-1]
    prev = tail.iloc[-2] if len(tail) > 1 else None

    records: list[dict] = []
    for product in _OUTPUT_PRODUCTS:
        out_col = _OUTPUT_COL[product]
        stock_col = _STOCK_COL.get(product)

        out_cur = cur.get(out_col)
        out_prev = prev.get(out_col) if prev is not None else None
        stock_cur = cur.get(stock_col) if stock_col else None
        stock_prev = prev.get(stock_col) if prev is not None and stock_col else None

        row: dict = {"product": product}

        if stock_col and pd.notna(stock_cur):
            row["stocks"] = _fmt_num(_convert_kt(float(stock_cur), stock_unit, product), 2)
            if prev is not None and pd.notna(stock_prev):
                d_stock_kt = float(stock_cur) - float(stock_prev)
                row["d_stocks"] = _fmt_num(
                    _convert_kt(d_stock_kt, stock_unit, product), 2
                )
            else:
                row["d_stocks"] = None
        else:
            row["stocks"] = None
            row["d_stocks"] = None

        if pd.notna(out_cur) and float(out_cur) > 0 and pd.notna(stock_cur):
            # Days of cover at current weekly production rate.
            row["days_cover"] = _fmt_num(float(stock_cur) * 7.0 / float(out_cur), 1)
        else:
            row["days_cover"] = None

        if pd.notna(out_cur):
            row["output"] = _fmt_num(_convert_kt(float(out_cur), flow_unit, product), 1)
            if prev is not None and pd.notna(out_prev):
                row["d_output"] = _fmt_num(
                    _convert_kt(float(out_cur) - float(out_prev), flow_unit, product), 1
                )
            else:
                row["d_output"] = None
        else:
            row["output"] = None
            row["d_output"] = None

        # Apparent demand only when stocks exist (Gasoline / Diesel).
        if (
            stock_col
            and pd.notna(out_cur)
            and pd.notna(stock_cur)
            and prev is not None
            and pd.notna(stock_prev)
        ):
            app_kt = float(out_cur) - (float(stock_cur) - float(stock_prev))
            row["app_demand"] = _fmt_num(_convert_kt(app_kt, flow_unit, product), 1)
            # Prior-week apparent demand needs week t-2.
            if len(df) >= 3:
                prev2 = df.iloc[-3]
                out_p = prev.get(out_col)
                stock_p = prev.get(stock_col)
                stock_p2 = prev2.get(stock_col)
                if pd.notna(out_p) and pd.notna(stock_p) and pd.notna(stock_p2):
                    app_prev_kt = float(out_p) - (float(stock_p) - float(stock_p2))
                    row["d_app_demand"] = _fmt_num(
                        _convert_kt(app_kt - app_prev_kt, flow_unit, product), 1
                    )
                else:
                    row["d_app_demand"] = None
            else:
                row["d_app_demand"] = None
        else:
            row["app_demand"] = None
            row["d_app_demand"] = None

        records.append(row)

    return records, columns


def products_scorecard_style() -> list[dict]:
    """Green / red text for signed delta columns."""
    delta_cols = ["d_stocks", "d_output", "d_app_demand"]
    styles = []
    for col in delta_cols:
        styles.append(
            {
                "if": {
                    "filter_query": f"{{{col}}} > 0",
                    "column_id": col,
                },
                "color": "#3fb950",
                "fontWeight": "600",
            }
        )
        styles.append(
            {
                "if": {
                    "filter_query": f"{{{col}}} < 0",
                    "column_id": col,
                },
                "color": "#f85149",
                "fontWeight": "600",
            }
        )
    return styles


# SCI weekly feedstock processing breakdowns (National is the total — no extra Total row).
_RUNS_BREAKDOWNS = ("National", "SOEs", "Independents")


def _convert_runs_kt(kt: float, unit: str) -> float:
    return float(
        convert_volume_series(
            pd.Series([kt]),
            unit,
            freq="weekly",
            product="crude",
        ).iloc[0]
    )


def runs_scorecard_records(
    runs_wide: pd.DataFrame,
    *,
    unit: str = "kbd",
) -> tuple[list[dict], list[dict], str | None]:
    """
    Latest-week feedstock processing scorecard by SCI breakdown.

    Returns (records, columns, week_label). Processing level / Δ follow ``unit``;
    operating rates stay in % / pp. No Total row — National already aggregates.
    """
    unit_lbl = volume_label(unit)
    columns = [
        {"name": "segment", "id": "segment"},
        {
            "name": f"processing ({unit_lbl})",
            "id": "processing",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": f"Δ w/w ({unit_lbl})",
            "id": "d_processing",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "CDU rate (%)",
            "id": "operating_rate",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "Δ w/w (pp)",
            "id": "d_operating_rate",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
    ]

    if runs_wide.empty or "week_start" not in runs_wide.columns:
        return [], columns, None

    df = runs_wide.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.dropna(subset=["week_start", "breakdown"])
    if df.empty:
        return [], columns, None

    weeks = sorted(df["week_start"].unique())
    latest = weeks[-1]
    prev = weeks[-2] if len(weeks) > 1 else None
    week_lbl = pd.Timestamp(latest).strftime("%d-%b-%y")

    records: list[dict] = []
    for bd in _RUNS_BREAKDOWNS:
        cur = df[(df["week_start"] == latest) & (df["breakdown"] == bd)]
        if cur.empty:
            continue
        cur_row = cur.iloc[0]
        row: dict = {"segment": bd}

        proc_cur = cur_row.get("processing_kt")
        rate_cur = cur_row.get("operating_rate")

        if pd.notna(proc_cur):
            row["processing"] = _fmt_num(_convert_runs_kt(float(proc_cur), unit), 1)
        else:
            row["processing"] = None

        if pd.notna(rate_cur):
            row["operating_rate"] = _fmt_num(float(rate_cur) * 100.0, 1)
        else:
            row["operating_rate"] = None

        if prev is not None:
            prev_rows = df[(df["week_start"] == prev) & (df["breakdown"] == bd)]
            if not prev_rows.empty:
                prev_row = prev_rows.iloc[0]
                proc_prev = prev_row.get("processing_kt")
                rate_prev = prev_row.get("operating_rate")
                if pd.notna(proc_cur) and pd.notna(proc_prev):
                    row["d_processing"] = _fmt_num(
                        _convert_runs_kt(float(proc_cur) - float(proc_prev), unit), 1
                    )
                else:
                    row["d_processing"] = None
                if pd.notna(rate_cur) and pd.notna(rate_prev):
                    row["d_operating_rate"] = _fmt_num(
                        (float(rate_cur) - float(rate_prev)) * 100.0, 1
                    )
                else:
                    row["d_operating_rate"] = None
            else:
                row["d_processing"] = None
                row["d_operating_rate"] = None
        else:
            row["d_processing"] = None
            row["d_operating_rate"] = None

        records.append(row)

    return records, columns, week_lbl


def runs_scorecard_style() -> list[dict]:
    """Green / red text for signed runs Δ columns."""
    styles = []
    for col in ("d_processing", "d_operating_rate"):
        styles.append(
            {
                "if": {"filter_query": f"{{{col}}} > 0", "column_id": col},
                "color": "#3fb950",
                "fontWeight": "600",
            }
        )
        styles.append(
            {
                "if": {"filter_query": f"{{{col}}} < 0", "column_id": col},
                "color": "#f85149",
                "fontWeight": "600",
            }
        )
    return styles


def weekly_delta_frame(
    products: pd.DataFrame,
    *,
    kind: str,
    unit: str,
    date_range: str,
) -> pd.DataFrame:
    """
    Long frame of weekly WoW changes for charting.

    kind: ``stocks`` | ``output``
    """
    if products.empty:
        return pd.DataFrame(columns=["week_start", "product", "delta"])

    df = products.sort_values("week_start").copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = filter_by_date_range(df, "week_start", date_range)

    products_list = _STOCK_PRODUCTS if kind == "stocks" else _OUTPUT_PRODUCTS
    col_map = _STOCK_COL if kind == "stocks" else _OUTPUT_COL

    rows = []
    for product in products_list:
        col = col_map.get(product)
        if not col or col not in df.columns:
            continue
        series = df[["week_start", col]].dropna().sort_values("week_start")
        if series.empty:
            continue
        series = series.copy()
        series["delta_kt"] = series[col].diff()
        for _, r in series.dropna(subset=["delta_kt"]).iterrows():
            rows.append(
                {
                    "week_start": r["week_start"],
                    "product": product,
                    "delta": _convert_kt(float(r["delta_kt"]), unit, product),
                }
            )
    return pd.DataFrame(rows)


def fig_products_delta_bars(
    products: pd.DataFrame,
    *,
    kind: str,
    unit: str,
    date_range: str,
) -> go.Figure:
    """Grouped bar chart of weekly WoW changes (inventories or production)."""
    colors = {
        "Gasoline": "#58a6ff",
        "Diesel": "#f85149",
        "Kerosene": "#bc8cff",
    }
    data = weekly_delta_frame(products, kind=kind, unit=unit, date_range=date_range)
    ylab = volume_label(unit)
    title = (
        f"Weekly inventory change ({ylab})"
        if kind == "stocks"
        else f"Weekly production change ({ylab})"
    )
    if data.empty:
        return go.Figure().update_layout(title=f"No data — {title}")

    fig = go.Figure()
    for product in data["product"].unique():
        sub = data[data["product"] == product].sort_values("week_start")
        fig.add_trace(
            go.Bar(
                x=sub["week_start"],
                y=sub["delta"],
                name=product,
                marker_color=colors.get(product, "#8b949e"),
            )
        )
    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Week",
        yaxis_title=f"WoW change ({ylab})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.add_hline(y=0, line_width=1, line_color="#8b949e")
    return fig


# Back-compat alias used by older call sites.
def products_stocks_change_records(
    products: pd.DataFrame,
    *,
    n_weeks: int = 3,
    stock_units: tuple[str, ...] = ("mb",),
    output_units: tuple[str, ...] = ("kbd",),
) -> tuple[list[dict], list[dict]]:
    del n_weeks  # scorecard is latest-week only
    stock_unit = stock_units[0] if stock_units else "mb"
    flow_unit = output_units[0] if output_units else "kbd"
    return products_scorecard_records(
        products, stock_unit=stock_unit, flow_unit=flow_unit
    )
