"""
Probe SCI API for older package pubdates.

Usage (from sci_api/):
  python scripts/probe_pubdates.py
  python scripts/probe_pubdates.py --pubdates 2020-12-31T00:00:00 2023-12-31T00:00:00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client import SciClient
from src.transforms import (
    merge_weekly_summaries,
    parse_balance_monthly,
    parse_consumption,
    parse_kerosene_refueling,
    parse_profit_model,
    parse_shandong_stocks,
    parse_weekly_summary,
)

DEFAULT_PUBDATES = [
    "2020-12-31T00:00:00",
    "2021-12-31T00:00:00",
    "2022-12-31T00:00:00",
    "2023-12-31T00:00:00",
    "2024-12-31T00:00:00",
    "2025-12-31T00:00:00",
]


def _find_sheet(sheets: dict, *parts: str) -> dict | None:
    parts_l = [p.lower() for p in parts]
    for name, sheet in sheets.items():
        if all(p in name.lower() for p in parts_l):
            return sheet
    return None


def _span(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df.columns:
        return "empty"
    ts = pd.to_datetime(df[col], errors="coerce")
    return f"{ts.min():%Y-%m-%d} -> {ts.max():%Y-%m-%d} ({len(df):,} rows)"


def probe_pubdate(client: SciClient, pubdate: str) -> dict:
    result = {"pubdate": pubdate, "ok": False, "error": None}
    try:
        pkg = client.fetch_package(pubdate=pubdate)
    except Exception as exc:  # noqa: BLE001 — probe script
        result["error"] = str(exc)
        return result

    if not isinstance(pkg, dict):
        result["error"] = f"Unexpected payload type: {type(pkg).__name__!r} value={pkg!r}"
        return result

    result["ok"] = True
    result["package_pubdate"] = pkg.get("PubDate")
    sheets = {s["SheetName"]: s for s in pkg.get("SheetList", [])}
    result["sheet_count"] = len(sheets)

    weekly_sheet = _find_sheet(sheets, "weekly", "summary")
    if weekly_sheet:
        weekly = parse_weekly_summary(weekly_sheet)
        result["weekly"] = _span(weekly, "week_start")
        nat = weekly[weekly["breakdown"] == "National"]
        proc = nat[nat["metric"] == "processing_kt"]
        if not proc.empty:
            result["national_runs_kt"] = (
                f"min={proc['value'].min():,.0f} max={proc['value'].max():,.0f} "
                f"latest={proc.sort_values('week_start')['value'].iloc[-1]:,.0f}"
            )
    else:
        result["weekly"] = "sheet not found"

    for label, parts, parser, kwargs in [
        ("profit", ("profit", "model"), parse_profit_model, {}),
        ("stocks", ("stocks",), parse_shandong_stocks, {}),
        ("balance", ("balance",), parse_balance_monthly, {}),
    ]:
        sheet = _find_sheet(sheets, *parts)
        if sheet:
            frame = parser(sheet, **kwargs) if kwargs else parser(sheet)
            col = "week_start" if label != "balance" else "month_start"
            result[label] = _span(frame, col)
        else:
            result[label] = "sheet not found"

    pub_year = int(pd.to_datetime(pkg.get("PubDate"), errors="coerce").year)
    cons_sheet = _find_sheet(sheets, "consumption")
    if cons_sheet:
        cons = parse_consumption(cons_sheet, calendar_year=pub_year)
        if cons.empty:
            result["consumption"] = f"empty (calendar_year={pub_year})"
        else:
            result["consumption"] = (
                f"years {cons['year'].min()}-{cons['year'].max()} "
                f"({cons['consumption_kt'].notna().sum():,} pts)"
            )
    else:
        result["consumption"] = "sheet not found"

    ref_sheet = _find_sheet(sheets, "kerosene", "refueling")
    if ref_sheet:
        ref = parse_kerosene_refueling(ref_sheet, calendar_year=pub_year)
        result["refueling"] = (
            "empty" if ref.empty else f"years {ref['year'].min()}-{ref['year'].max()}"
        )
    else:
        result["refueling"] = "sheet not found"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SCI API pubdate availability.")
    parser.add_argument(
        "--pubdates",
        nargs="*",
        default=DEFAULT_PUBDATES,
        help="Pubdates to test (ISO format with T00:00:00).",
    )
    args = parser.parse_args()

    client = SciClient()
    client.authenticate()
    latest = client.latest_pubdate()
    print(f"Package {client.package_id}  latest_pubdate={latest}\n")

    rows = []
    for pubdate in args.pubdates:
        print(f"--- {pubdate} ---")
        row = probe_pubdate(client, pubdate)
        rows.append(row)
        if not row["ok"]:
            print(f"  FAILED: {row['error']}")
        else:
            print(f"  package PubDate: {row.get('package_pubdate')}")
            print(f"  sheets: {row.get('sheet_count')}")
            for key in ("weekly", "profit", "stocks", "balance", "consumption", "refueling"):
                if key in row:
                    print(f"  {key}: {row[key]}")
            if "national_runs_kt" in row:
                print(f"  national runs: {row['national_runs_kt']}")
        print()

    ok = [r for r in rows if r["ok"]]
    failed = [r for r in rows if not r["ok"]]
    print("=== SUMMARY ===")
    print(f"Successful: {len(ok)}/{len(rows)}")
    if failed:
        print("Unavailable pubdates:")
        for r in failed:
            print(f"  {r['pubdate']}: {r['error']}")

    if len(ok) >= 2:
        print("\n=== MERGED WEEKLY (all successful snapshots in this run) ===")
        parts = []
        for r in ok:
            pkg = client.fetch_package(pubdate=r["pubdate"])
            sheets = {s["SheetName"]: s for s in pkg.get("SheetList", [])}
            sheet = _find_sheet(sheets, "weekly", "summary")
            if sheet:
                parts.append(parse_weekly_summary(sheet))
        merged = merge_weekly_summaries(parts)
        print(_span(merged, "week_start"))

        from src.excel_archive import load_excel_weekly

        excel = load_excel_weekly()
        print("\n=== EXCEL ARCHIVE (comparison) ===")
        print(_span(excel, "week_start"))


if __name__ == "__main__":
    main()
