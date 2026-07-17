"""Assemble the China semi-balance desk table (SCI + FSU pipeline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.reports.excel_writer import write_semi_balance_workbook
from src.reports.fsu_pipeline import parse_fsu_china_pipeline, resolve_fsu_path
from src.transforms import CRUDE_EQUITY_OWNERS
from src.units import DEFAULT_BARRELS_PER_MT

CDU_UNIT = "FEEDSTOCK PRIMARY PROCESSING UNITS"
SOE_OWNERS = ("PetroChina", "Sinopec", "CNOOC")
OWNER_ORDER = ("Sinopec", "PetroChina", "CNOOC", "Independent refinery")
PROD_OWNER_ORDER = (*CRUDE_EQUITY_OWNERS, "Others")

# Hierarchy levels drive Excel formatting (section > total > province > owner > detail).
ROW_SECTION = "section"
ROW_TOTAL = "total"
ROW_PROVINCE = "province"
ROW_OWNER = "owner"
ROW_DETAIL = "detail"

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "config" / "china_semi_balance.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def month_range(start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize().replace(day=1)
    end_ts = pd.Timestamp(end).normalize().replace(day=1)
    if end_ts < start_ts:
        raise ValueError(f"end {end} before start {start}")
    return list(pd.date_range(start_ts, end_ts, freq="MS"))


def _pivot_months(
    series_by_month: pd.Series,
    months: list[pd.Timestamp],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    index = pd.to_datetime(series_by_month.index).map(
        lambda x: pd.Timestamp(x).normalize().replace(day=1)
    )
    aligned = pd.Series(series_by_month.values, index=index)
    for m in months:
        key = m.strftime("%Y-%m-%d")
        if m in aligned.index:
            val = aligned.loc[m]
            # Duplicate index → take first
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            out[key] = float(val)
        else:
            out[key] = None
    return out


def _empty_months(months: list[pd.Timestamp]) -> dict[str, float | None]:
    return {m.strftime("%Y-%m-%d"): None for m in months}


def _row(
    *,
    section: str | None,
    sub_category: str | None,
    item: str | None,
    unit: str | None,
    value_kind: str,
    month_values: dict[str, float | None],
    row_level: str,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "section": section,
        "sub_category": sub_category,
        "item": item,
        "unit": unit,
        "value_kind": value_kind,
        "row_level": row_level,
    }
    rec.update(month_values)
    return rec


def _banner(section: str, months: list[pd.Timestamp], *, value_kind: str = "kt") -> dict[str, Any]:
    return _row(
        section=section,
        sub_category=None,
        item=None,
        unit=None,
        value_kind=value_kind,
        month_values=_empty_months(months),
        row_level=ROW_SECTION,
    )


def _national_processing(
    processing: pd.DataFrame,
    months: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    cdu = processing[processing["unit"] == CDU_UNIT].copy()
    cdu["month_start"] = pd.to_datetime(cdu["month_start"]).dt.normalize()
    cdu = cdu[cdu["month_start"].isin(months)]

    by_own = cdu.groupby(["month_start", "ownership"], as_index=False)["processing_kt"].sum()

    def series_for(mask) -> pd.Series:
        return by_own[mask].groupby("month_start")["processing_kt"].sum()

    national = series_for(by_own["ownership"].notna())
    soe = series_for(by_own["ownership"].isin(SOE_OWNERS))
    indep = series_for(by_own["ownership"] == "Independent refinery")

    rows = [
        _banner("Processing volume (national)", months),
    ]
    specs = [
        ("National total", national, ROW_TOTAL),
        ("SOE total", soe, ROW_OWNER),
        ("Independent refinery total", indep, ROW_OWNER),
    ]
    for label, ser, level in specs:
        rows.append(
            _row(
                section="Processing volume (national)",
                sub_category="Processing volume",
                item=label,
                unit="kt",
                value_kind="kt",
                month_values=_pivot_months(ser, months),
                row_level=level,
            )
        )
    return rows


def _province_processing(
    processing: pd.DataFrame,
    months: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    cdu = processing[processing["unit"] == CDU_UNIT].copy()
    cdu["month_start"] = pd.to_datetime(cdu["month_start"]).dt.normalize()
    cdu = cdu[cdu["month_start"].isin(months)]

    section = "Breakdown by province and equity owner"
    rows: list[dict[str, Any]] = [_banner(section, months)]

    provinces = sorted(cdu["province"].dropna().unique())
    province_series: list[pd.Series] = []

    for province in provinces:
        sub = cdu[cdu["province"] == province]
        if sub.empty or sub["processing_kt"].fillna(0).eq(0).all():
            continue

        prov_ser = sub.groupby("month_start")["processing_kt"].sum()
        province_series.append(prov_ser)
        rows.append(
            _row(
                section=section,
                sub_category=CDU_UNIT,
                item=f"{province} total",
                unit="kt",
                value_kind="kt",
                month_values=_pivot_months(prov_ser, months),
                row_level=ROW_PROVINCE,
            )
        )

        for owner in OWNER_ORDER:
            own = sub[sub["ownership"] == owner]
            if own.empty or own["processing_kt"].fillna(0).eq(0).all():
                continue
            own_ser = own.groupby("month_start")["processing_kt"].sum()
            rows.append(
                _row(
                    section=section,
                    sub_category="equity_owner",
                    item=owner,
                    unit="kt",
                    value_kind="kt",
                    month_values=_pivot_months(own_ser, months),
                    row_level=ROW_OWNER,
                )
            )

    if province_series:
        all_prov = pd.concat(province_series).groupby(level=0).sum()
        rows.append(
            _row(
                section=section,
                sub_category=CDU_UNIT,
                item="All provinces total",
                unit="kt",
                value_kind="kt",
                month_values=_pivot_months(all_prov, months),
                row_level=ROW_TOTAL,
            )
        )
    return rows


def _local_production(
    crude: pd.DataFrame,
    months: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    if crude is None or crude.empty:
        return []

    df = crude.copy()
    df["month_start"] = pd.to_datetime(df["month_start"]).dt.normalize()
    df = df[df["month_start"].isin(months)]

    section = "LOCAL OIL PRODUCTION"
    rows: list[dict[str, Any]] = [_banner(section, months)]

    field_order = (
        df.loc[df["metric_type"] == "field", "name"].drop_duplicates().tolist()
    )

    provinces = sorted(df["province"].dropna().unique())
    province_series: list[pd.Series] = []

    for province in provinces:
        sub = df[df["province"] == province]
        totals = sub[sub["metric_type"] == "province_total"]
        if totals.empty or totals["production_kt"].fillna(0).eq(0).all():
            if sub[sub["metric_type"] != "province_total"]["production_kt"].fillna(0).eq(0).all():
                continue

        if not totals.empty:
            ser = totals.groupby("month_start")["production_kt"].sum()
        else:
            ser = (
                sub[sub["metric_type"] == "equity_owner"]
                .groupby("month_start")["production_kt"]
                .sum()
            )
        province_series.append(ser)
        rows.append(
            _row(
                section=section,
                sub_category="province",
                item=f"{province} total",
                unit="kt",
                value_kind="kt",
                month_values=_pivot_months(ser, months),
                row_level=ROW_PROVINCE,
            )
        )

        for owner in PROD_OWNER_ORDER:
            own_rows = sub[
                (sub["metric_type"] == "equity_owner") & (sub["name"] == owner)
            ]
            if own_rows.empty or own_rows["production_kt"].fillna(0).eq(0).all():
                continue
            own_ser = own_rows.groupby("month_start")["production_kt"].sum()
            rows.append(
                _row(
                    section=section,
                    sub_category="equity_owner",
                    item=f"{owner} total",
                    unit="kt",
                    value_kind="kt",
                    month_values=_pivot_months(own_ser, months),
                    row_level=ROW_OWNER,
                )
            )

            fields = sub[
                (sub["metric_type"] == "field") & (sub["ownership"] == owner)
            ]
            for field in field_order:
                fsub = fields[fields["name"] == field]
                if fsub.empty or fsub["production_kt"].fillna(0).eq(0).all():
                    continue
                fser = fsub.groupby("month_start")["production_kt"].sum()
                rows.append(
                    _row(
                        section=section,
                        sub_category="field_name",
                        item=field,
                        unit="kt",
                        value_kind="kt",
                        month_values=_pivot_months(fser, months),
                        row_level=ROW_DETAIL,
                    )
                )

    if province_series:
        china_total = pd.concat(province_series).groupby(level=0).sum()
        rows.append(
            _row(
                section=section,
                sub_category="province",
                item="China local production total",
                unit="kt",
                value_kind="kt",
                month_values=_pivot_months(china_total, months),
                row_level=ROW_TOTAL,
            )
        )
    return rows


def _fsu_section(
    fsu: pd.DataFrame,
    months: list[pd.Timestamp],
    routes: list[str],
) -> list[dict[str, Any]]:
    section = "FSU export To China (from pl)"
    rows: list[dict[str, Any]] = [_banner(section, months, value_kind="kbd")]
    if fsu is None or fsu.empty:
        return rows

    df = fsu.copy()
    df["month_start"] = pd.to_datetime(df["month_start"]).dt.normalize()
    df = df[df["month_start"].isin(months)]

    route_series: list[pd.Series] = []
    for route in routes:
        sub = df[df["route"] == route]
        if sub.empty:
            month_values = _empty_months(months)
            ser = pd.Series(dtype=float)
        else:
            ser = sub.groupby("month_start")["kbd"].sum()
            month_values = _pivot_months(ser, months)
            route_series.append(ser)
        rows.append(
            _row(
                section=section,
                sub_category="pipeline_route",
                item=route,
                unit="kbd",
                value_kind="kbd",
                month_values=month_values,
                row_level=ROW_DETAIL,
            )
        )

    if route_series:
        total = pd.concat(route_series).groupby(level=0).sum()
        rows.append(
            _row(
                section=section,
                sub_category="pipeline_route",
                item="Total China pipeline",
                unit="kbd",
                value_kind="kbd",
                month_values=_pivot_months(total, months),
                row_level=ROW_TOTAL,
            )
        )
    return rows


def build_china_semi_balance(
    frames: dict[str, pd.DataFrame],
    *,
    fsu_path: str | Path,
    start: str,
    end: str,
    routes: list[str] | None = None,
    search_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """
    Build long-form report rows ready for the Excel writer.

    ``frames`` needs ``processing_ownership`` and ``crude_output``.
    Each row carries ``row_level`` for hierarchical formatting.
    """
    months = month_range(start, end)
    routes = routes or [
        "China via Kazakhstan",
        "China via ESPO",
    ]

    processing = frames.get("processing_ownership")
    if processing is None or processing.empty:
        raise ValueError("frames['processing_ownership'] is required")

    crude = frames.get("crude_output", pd.DataFrame())

    resolved = resolve_fsu_path(fsu_path, search_dir=search_dir or _ROOT)
    fsu = parse_fsu_china_pipeline(resolved, routes=tuple(routes))

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(_national_processing(processing, months))
    all_rows.extend(_province_processing(processing, months))
    all_rows.extend(_local_production(crude, months))
    all_rows.extend(_fsu_section(fsu, months, routes))
    return pd.DataFrame(all_rows), months


def write_china_semi_balance(
    frames: dict[str, pd.DataFrame],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    fsu_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Build + write workbook using YAML defaults with optional overrides."""
    cfg = config if config is not None else load_config(config_path)
    start = cfg["months"]["start"]
    end = cfg["months"]["end"]
    routes = cfg.get("fsu", {}).get("routes")
    barrels = float(cfg.get("barrels_per_mt", DEFAULT_BARRELS_PER_MT))

    fsu = fsu_path or cfg.get("fsu", {}).get("path", "FSU-*_bpd_interim.xlsx")
    out = Path(
        output_path
        or cfg.get("output", {}).get("path")
        or (_ROOT / "data" / "reports" / "china_semi_balance.xlsx")
    )

    rows, months = build_china_semi_balance(
        frames,
        fsu_path=fsu,
        start=start,
        end=end,
        routes=routes,
        search_dir=_ROOT,
    )
    return write_semi_balance_workbook(
        rows, months, out, barrels_per_mt=barrels
    )
