"""
fetch_data.py — pull SCI China runs/stocks data and cache locally.

Usage
-----
  python fetch_data.py              # use cache if present
  python fetch_data.py --history    # merge 2021–2025 year-end + latest (recommended)
  python fetch_data.py --refresh --history
  SCI_INCLUDE_EXCEL_ARCHIVE=1 python fetch_data.py --refresh --history  # add 2020 from Excel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.history import fetch_merged_history

CACHE_DIR = _HERE / "data" / "cache"

FRAME_NAMES = [
    "meta",
    "weekly",
    "runs_wide",
    "products_stocks_wide",
    "shandong_stocks",
    "shandong_stocks_total",
    "processing_ownership",
    "operating_rate_ownership",
    "balance_monthly",
    "consumption",
    "consumption_national",
    "kerosene_refueling",
    "kerosene_refueling_national",
    "margins",
    "margins_wide",
    "provincial_processing",
    "provincial_output",
    "petchem_weekly",
    "petchem_monthly",
    "petchem_inventories",
    "export_shipping",
    "export_forecast",
    "export_quotas_monthly",
    "export_quotas_summary",
]


def save_frames(frames: dict[str, pd.DataFrame], cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pub = frames["meta"]["pub_date"].iloc[0]
    stamp = pd.Timestamp(pub).strftime("%Y%m%d") if pd.notna(pub) else "latest"
    out = cache_dir / stamp
    out.mkdir(parents=True, exist_ok=True)

    for name in FRAME_NAMES:
        if name in frames:
            frames[name].to_parquet(out / f"{name}.parquet", index=False)

    latest = cache_dir / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(out.name, target_is_directory=True)
    except OSError:
        (cache_dir / "latest.txt").write_text(out.name, encoding="utf-8")

    return out


def load_cached_frames(cache_dir: Path = CACHE_DIR) -> dict[str, pd.DataFrame] | None:
    latest = cache_dir / "latest"
    if not latest.exists():
        txt = cache_dir / "latest.txt"
        if txt.exists():
            latest = cache_dir / txt.read_text(encoding="utf-8").strip()
        else:
            dirs = sorted(p for p in cache_dir.iterdir() if p.is_dir() and p.name.isdigit())
            if not dirs:
                return None
            latest = dirs[-1]
    elif latest.is_symlink():
        latest = cache_dir / latest.readlink()

    frames = {}
    for name in FRAME_NAMES:
        path = Path(latest) / f"{name}.parquet"
        if path.exists():
            frames[name] = pd.read_parquet(path)
    if not frames:
        return None

    from src.excel_archive import maybe_merge_excel_archive

    return maybe_merge_excel_archive(frames)


def fetch_and_cache(*, include_history: bool = True) -> Path:
    frames = fetch_merged_history(include_history=include_history)
    out = save_frames(frames)
    pub = frames["meta"]["pub_date"].iloc[0]
    weekly = frames["weekly"]
    wmin = weekly["week_start"].min() if not weekly.empty else None
    wmax = weekly["week_start"].max() if not weekly.empty else None
    print(
        f"Cached {len(frames)} tables -> {out}  "
        f"(pub_date={pub.date() if pd.notna(pub) else '?'}, "
        f"weekly {wmin.date() if wmin is not None else '?'} to {wmax.date() if wmax is not None else '?'})"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SCI China runs/stocks data.")
    parser.add_argument("--refresh", action="store_true", help="Always call the API.")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Merge year-end API snapshots (2021–2025) + latest for max history.",
    )
    parser.add_argument(
        "--excel-2020",
        action="store_true",
        help="Also merge local Excel archive for 2020 (API has no 2020 snapshots).",
    )
    parser.add_argument("--latest-only", action="store_true", help="Skip historical snapshot merge.")
    args = parser.parse_args()

    if args.excel_2020:
        import os

        os.environ["SCI_INCLUDE_EXCEL_ARCHIVE"] = "1"

    include_history = not args.latest_only
    if args.refresh or args.history:
        fetch_and_cache(include_history=include_history)
        return

    if load_cached_frames():
        print("Cache exists. Use --refresh or --history to pull from API.")
        return

    fetch_and_cache(include_history=include_history)


if __name__ == "__main__":
    main()
