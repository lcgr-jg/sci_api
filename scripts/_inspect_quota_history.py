"""One-off: inspect sheet 45 across history pubdates."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client import SciClient
from src.history import HISTORY_PUBDATES
from src.transforms import parse_export_quotas_monthly, parse_export_quotas_summary

client = SciClient()
client.authenticate()
latest = client.latest_pubdate()
pubdates = list(HISTORY_PUBDATES) + ([latest] if latest not in HISTORY_PUBDATES else [])

for pubdate in pubdates:
    pkg = client.fetch_package(pubdate=pubdate)
    pub_year = int(pd.to_datetime(pkg["PubDate"]).year)
    sheets = {s["SheetName"]: s for s in pkg.get("SheetList", [])}
    sheet = None
    sheet_name = None
    for name, s in sheets.items():
        if "product quotas" in name.lower():
            sheet = s
            sheet_name = name
            break
    if sheet is None:
        print(f"{pubdate[:10]} (pub {pub_year}): NO sheet 45")
        continue
    topic = sheet.get("Topic", "")
    raw = pd.DataFrame(sheet.get("DataList", []))
    parsed = parse_export_quotas_monthly(sheet)
    summary = parse_export_quotas_summary(sheet)
    years = sorted(parsed["year"].unique()) if not parsed.empty else []
    print(
        f"{pubdate[:10]} (pub {pub_year}) | {sheet_name} | "
        f"topic: {topic[:70]} | raw rows: {len(raw)} | "
        f"parsed rows: {len(parsed)} | years: {years}"
    )
    if not summary.empty:
        print("  summary:", summary.iloc[0].to_dict())
    if not raw.empty and "Product" in raw.columns:
        prods = raw["Product"].dropna().unique()[:10]
        print("  sample products:", list(prods))
    if not parsed.empty:
        print("  month range:", parsed["month_start"].min(), "->", parsed["month_start"].max())
