"""
Build the China semi-balance Excel table from SCI cache + FSU workbook.

Usage
-----
  python scripts/build_china_semi_balance.py
  python scripts/build_china_semi_balance.py --fsu FSU-2026-07_bpd_interim.xlsx
  python scripts/build_china_semi_balance.py --start 2026-01-01 --end 2026-06-01
  python scripts/build_china_semi_balance.py --refresh-crude   # pull crude_output if missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fetch_data import load_cached_frames, save_frames  # noqa: E402
from src.client import SciClient  # noqa: E402
from src.reports.china_semi_balance import load_config, write_china_semi_balance  # noqa: E402
from src.transforms import parse_crude_output  # noqa: E402


def _ensure_crude_output(frames: dict, *, refresh: bool) -> dict:
    if "crude_output" in frames and not frames["crude_output"].empty and not refresh:
        return frames

    client = SciClient()
    client.authenticate()
    pkg = client.fetch_package()
    sheets = {s["SheetName"]: s for s in pkg.get("SheetList", [])}
    sheet = next(
        (s for name, s in sheets.items() if "crude" in name.lower() and "output" in name.lower()),
        None,
    )
    if sheet is None:
        raise KeyError("SCI sheet 'Crude output' not found in latest package")
    frames = dict(frames)
    frames["crude_output"] = parse_crude_output(sheet)
    # Persist into cache so the notebook / next run does not need another API hit.
    save_frames(frames)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Build china_semi_balance.xlsx")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--fsu", type=str, default=None, help="FSU workbook path or glob")
    parser.add_argument("--start", type=str, default=None, help="Month start YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Month end YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=None, help="Output xlsx path")
    parser.add_argument(
        "--refresh-crude",
        action="store_true",
        help="Re-fetch SCI crude output even if cached",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.start:
        cfg.setdefault("months", {})["start"] = args.start
    if args.end:
        cfg.setdefault("months", {})["end"] = args.end
    if args.fsu:
        cfg.setdefault("fsu", {})["path"] = args.fsu
    if args.output:
        cfg.setdefault("output", {})["path"] = str(args.output)

    frames = load_cached_frames()
    if not frames:
        raise SystemExit("No SCI cache found. Run: python fetch_data.py --refresh --history")

    frames = _ensure_crude_output(frames, refresh=args.refresh_crude)

    out = write_china_semi_balance(frames, config=cfg)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
