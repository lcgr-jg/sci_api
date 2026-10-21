"""Economics tab — profit-model WoW drivers and waterfalls."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

_TAX_LINE_ITEMS = (
    "Gasoline consumption tax",
    "Diesel consumption tax",
    "Processing cost",
    "Operating expenses",
    "Depreciation expenses",
    "Labor costs",
    "Water and electricity costs",
    "Total value-added tax",
    "Urban construction tax",
    "Education surtax",
)


def _fmt(val: float | None, digits: int = 1) -> float | None:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return round(float(val), digits)


def _factor_at(
    margins: pd.DataFrame,
    *,
    refinery_type: str,
    factor: str,
    week: pd.Timestamp,
) -> float | None:
    if margins.empty:
        return None
    sub = margins[
        (margins["refinery_type"] == refinery_type)
        & (margins["factor"] == factor)
        & (margins["week_start"] == week)
    ]
    if sub.empty:
        return None
    val = sub["value"].iloc[0]
    return float(val) if pd.notna(val) else None


def _latest_weeks(margins: pd.DataFrame, refinery_type: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if margins.empty:
        return None, None
    sub = margins[margins["refinery_type"] == refinery_type]
    if sub.empty:
        return None, None
    weeks = sorted(pd.to_datetime(sub["week_start"]).dropna().unique())
    if not weeks:
        return None, None
    latest = pd.Timestamp(weeks[-1])
    prev = pd.Timestamp(weeks[-2]) if len(weeks) > 1 else None
    return latest, prev


def profit_driver_snapshot(
    margins: pd.DataFrame,
    *,
    refinery_type: str,
) -> dict:
    """
    Latest-week P&L snapshot + WoW deltas for one SCI refinery-type model.

    Identities (Fuel / Chem / Aromatics models):
      Sales − Feedstock = Contribution
      Contribution − Taxes & other = Net profit
    """
    latest, prev = _latest_weeks(margins, refinery_type)
    empty = {
        "week_label": None,
        "prev_label": None,
        "headline": "No profit-model data for selection",
        "levels": {},
        "deltas": {},
        "tax_movers": [],
        "price_movers": [],
    }
    if latest is None:
        return empty

    def lvl(factor: str, week: pd.Timestamp | None) -> float | None:
        if week is None:
            return None
        return _factor_at(margins, refinery_type=refinery_type, factor=factor, week=week)

    keys = {
        "sales": "Total sales revenue (taxes excluded)",
        "feedstock": "Feedstock cost (taxes excluded)",
        "contribution": "Contribution margin",
        "taxes": "Taxes and other total costs",
        "net": "Net profit",
        "ppt": "Profit per ton",
        "brent": "Brent",
    }
    cur = {k: lvl(fac, latest) for k, fac in keys.items()}
    pri = {k: lvl(fac, prev) for k, fac in keys.items()} if prev is not None else {k: None for k in keys}
    deltas = {
        k: (cur[k] - pri[k]) if cur[k] is not None and pri[k] is not None else None
        for k in keys
    }

    # Top tax/opex line movers by |Δ| (exclude the aggregate total).
    tax_movers: list[dict] = []
    if prev is not None:
        for fac in _TAX_LINE_ITEMS:
            c = lvl(fac, latest)
            p = lvl(fac, prev)
            if c is None or p is None:
                continue
            tax_movers.append({"factor": fac, "level": c, "delta": c - p})
        tax_movers.sort(key=lambda r: abs(r["delta"]), reverse=True)
        tax_movers = tax_movers[:3]

    # All product-price movers (RMB/mt), largest |Δ| first.
    price_movers: list[dict] = []
    if prev is not None and not margins.empty:
        prices = margins[
            (margins["refinery_type"] == refinery_type)
            & (margins["section"] == "Product price")
            & (margins["week_start"].isin([latest, prev]))
        ]
        for fac, g in prices.groupby("factor"):
            g = g.set_index("week_start")["value"]
            if latest not in g.index or prev not in g.index:
                continue
            c, p = float(g.loc[latest]), float(g.loc[prev])
            price_movers.append({"factor": str(fac).strip(), "level": c, "delta": c - p})
        price_movers.sort(key=lambda r: abs(r["delta"]), reverse=True)

    headline = _build_headline(deltas, price_movers, tax_movers)

    return {
        "week_label": latest.strftime("%d-%b-%y"),
        "prev_label": prev.strftime("%d-%b-%y") if prev is not None else None,
        "headline": headline,
        "levels": cur,
        "prior": pri,
        "deltas": deltas,
        "tax_movers": tax_movers,
        "price_movers": price_movers,
    }


def _build_headline(
    deltas: dict,
    price_movers: list[dict],
    tax_movers: list[dict],
) -> str:
    d_net = deltas.get("net")
    d_sales = deltas.get("sales")
    d_feed = deltas.get("feedstock")
    d_taxes = deltas.get("taxes")
    d_contrib = deltas.get("contribution")

    if d_net is None:
        return "Insufficient history for WoW drivers"

    direction = "up" if d_net > 0 else "down" if d_net < 0 else "flat"
    if abs(d_net) < 1:
        return "Net profit roughly flat w/w"

    parts: list[str] = [f"Net profit {direction} w/w"]

    # Dominant P&L driver.
    if d_contrib is not None and abs(d_contrib) >= abs(d_taxes or 0):
        if d_sales is not None and d_feed is not None:
            if d_sales > 0 and (d_feed is None or d_sales > d_feed):
                parts.append("stronger sales vs feedstock")
            elif d_feed is not None and d_feed < 0:
                parts.append("lower feedstock cost")
            elif d_sales is not None and d_sales < 0 and d_feed is not None and d_feed < 0:
                parts.append("sales and feedstock both softer")
            else:
                parts.append("contribution margin move")
        else:
            parts.append("contribution margin move")
    elif d_taxes is not None and abs(d_taxes) > 0:
        parts.append(
            "lower taxes/opex" if d_taxes < 0 else "higher taxes/opex"
        )

    if price_movers:
        top = price_movers[0]
        sign = "+" if top["delta"] >= 0 else ""
        parts.append(f"top price move {top['factor']} {sign}{top['delta']:.0f} RMB/mt")

    return " — ".join(parts) + "."


def _scorecard_columns() -> list[dict]:
    return [
        {"name": "driver", "id": "driver"},
        {
            "name": "level",
            "id": "level",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {
            "name": "Δ w/w",
            "id": "delta",
            "type": "numeric",
            "format": {"specifier": ",.1f"},
        },
        {"name": "unit", "id": "unit"},
        {"name": "note", "id": "note"},
    ]


def profit_driver_scorecard(
    margins: pd.DataFrame,
    *,
    refinery_type: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], str]:
    """
    Split WoW scorecard into P&L rows and product-price rows.

    Returns (pnl_records, pnl_columns, price_records, price_columns, subtitle).
    Absolute P&L in RMB mn; product prices in RMB/mt.
    """
    snap = profit_driver_snapshot(margins, refinery_type=refinery_type)
    columns = _scorecard_columns()

    week = snap["week_label"]
    prev = snap["prev_label"]
    if not week:
        return [], columns, [], columns, snap["headline"]

    subtitle = (
        f"Week ending {week}"
        + (f" vs {prev}" if prev else "")
        + f". {snap['headline']}"
    )

    levels = snap["levels"]
    deltas = snap["deltas"]

    def row(
        driver: str,
        level: float | None,
        delta: float | None,
        unit: str,
        note: str = "",
        *,
        scale: float = 1.0,
        digits: int = 1,
        sign_mode: str = "benefit",
    ) -> dict:
        d = _fmt(delta / scale if delta is not None else None, digits)
        return {
            "driver": driver,
            "level": _fmt(level / scale if level is not None else None, digits),
            "delta": d,
            "unit": unit,
            "note": note,
            "_sign": sign_mode,
        }

    pnl_records = [
        row(
            "Net profit",
            levels.get("net"),
            deltas.get("net"),
            "RMB mn",
            "bottom line",
            scale=1e6,
        ),
        row(
            "Contribution",
            levels.get("contribution"),
            deltas.get("contribution"),
            "RMB mn",
            "sales − feedstock",
            scale=1e6,
        ),
        row(
            "Sales revenue",
            levels.get("sales"),
            deltas.get("sales"),
            "RMB mn",
            "taxes excluded",
            scale=1e6,
        ),
        row(
            "Feedstock cost",
            levels.get("feedstock"),
            deltas.get("feedstock"),
            "RMB mn",
            "↑ cost hurts margin",
            scale=1e6,
            sign_mode="cost",
        ),
        row(
            "Taxes & other",
            levels.get("taxes"),
            deltas.get("taxes"),
            "RMB mn",
            "↑ cost hurts margin",
            scale=1e6,
            sign_mode="cost",
        ),
        row(
            "Profit per ton",
            levels.get("ppt"),
            deltas.get("ppt"),
            "RMB/mt",
            "SCI unit margin",
        ),
    ]

    for tm in snap["tax_movers"]:
        if abs(tm["delta"]) < 1:
            continue
        pnl_records.append(
            row(
                f"  ↳ {tm['factor']}",
                tm["level"],
                tm["delta"],
                "RMB mn",
                "top tax/opex mover",
                scale=1e6,
                sign_mode="cost",
            )
        )

    # Markers sit with prices (not P&L).
    price_records: list[dict] = [
        row(
            "Brent",
            levels.get("brent"),
            deltas.get("brent"),
            "$/bbl",
            "feedstock marker",
            sign_mode="cost",
        ),
    ]
    # WTI when present in the model.
    latest, prev_w = _latest_weeks(margins, refinery_type)
    if latest is not None and prev_w is not None:
        wti_c = _factor_at(margins, refinery_type=refinery_type, factor="WTI", week=latest)
        wti_p = _factor_at(margins, refinery_type=refinery_type, factor="WTI", week=prev_w)
        if wti_c is not None:
            price_records.append(
                row(
                    "WTI",
                    wti_c,
                    (wti_c - wti_p) if wti_p is not None else None,
                    "$/bbl",
                    "feedstock marker",
                    sign_mode="cost",
                )
            )

    for pm in snap["price_movers"]:
        price_records.append(
            row(
                pm["factor"],
                pm["level"],
                pm["delta"],
                "RMB/mt",
                "product price",
                sign_mode="benefit",
            )
        )

    return pnl_records, columns, price_records, columns, subtitle


def profit_driver_scorecard_style() -> list[dict]:
    """Green/red for Δ — cost rows flip (rise = red)."""
    styles: list[dict] = [
        {
            "if": {
                "filter_query": '{_sign} = "benefit" && {delta} > 0',
                "column_id": "delta",
            },
            "color": "#3fb950",
            "fontWeight": "600",
        },
        {
            "if": {
                "filter_query": '{_sign} = "benefit" && {delta} < 0',
                "column_id": "delta",
            },
            "color": "#f85149",
            "fontWeight": "600",
        },
        {
            "if": {
                "filter_query": '{_sign} = "cost" && {delta} > 0',
                "column_id": "delta",
            },
            "color": "#f85149",
            "fontWeight": "600",
        },
        {
            "if": {
                "filter_query": '{_sign} = "cost" && {delta} < 0',
                "column_id": "delta",
            },
            "color": "#3fb950",
            "fontWeight": "600",
        },
        {
            "if": {"filter_query": '{driver} = "Net profit"'},
            "fontWeight": "700",
        },
    ]
    return styles


def fig_profit_waterfall(
    margins: pd.DataFrame,
    *,
    refinery_type: str,
) -> go.Figure:
    """
    WoW bridge of model net profit (RMB mn):

      Prior net → +Δ sales → −Δ feedstock → −Δ taxes/opex → Current net
    """
    snap = profit_driver_snapshot(margins, refinery_type=refinery_type)
    prior = snap.get("prior") or {}
    deltas = snap.get("deltas") or {}
    levels = snap.get("levels") or {}

    prior_net = prior.get("net")
    cur_net = levels.get("net")
    d_sales = deltas.get("sales")
    d_feed = deltas.get("feedstock")
    d_taxes = deltas.get("taxes")

    if prior_net is None or cur_net is None or d_sales is None or d_feed is None or d_taxes is None:
        return go.Figure().update_layout(
            title="Profit waterfall — need at least two weeks of model data"
        )

    # Profit impact: sales adds, feedstock/taxes subtract when they rise.
    y = [
        prior_net / 1e6,
        d_sales / 1e6,
        -d_feed / 1e6,
        -d_taxes / 1e6,
        cur_net / 1e6,
    ]
    week = snap["week_label"] or ""
    prev = snap["prev_label"] or "prior"

    fig = go.Figure(
        go.Waterfall(
            name="Net profit",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=[
                f"Net ({prev})",
                "Δ Sales",
                "Δ Feedstock",
                "Δ Taxes & opex",
                f"Net ({week})",
            ],
            y=y,
            text=[f"{v:+.2f}" if i > 0 and i < 4 else f"{v:.2f}" for i, v in enumerate(y)],
            textposition="outside",
            connector={"line": {"color": "#8b949e"}},
            increasing={"marker": {"color": "#3fb950"}},
            decreasing={"marker": {"color": "#f85149"}},
            totals={"marker": {"color": "#58a6ff"}},
            hovertemplate="%{x}<br>%{y:.2f} RMB mn<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"Net profit WoW bridge — {refinery_type}"
            f"<br><sup>{prev} → {week} · model RMB mn</sup>"
        ),
        yaxis_title="RMB mn",
        showlegend=False,
    )
    return fig
