"""Fetch and merge SCI datapack snapshots for maximum history."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.client import SciClient
from src.transforms import (
    build_consumption_national,
    build_kerosene_refueling_national,
    merge_balance_monthly,
    merge_consumption,
    merge_crude_output,
    merge_crude_port_shipping,
    merge_export_quotas_monthly,
    merge_kerosene_refueling,
    merge_monthly_inventories,
    merge_monthly_petchem,
    merge_processing_by_oil_grade,
    merge_profit_models,
    merge_provincial_processing,
    merge_provincial_product_output,
    merge_refined_export_shipping,
    merge_shandong_stocks,
    merge_vehicle_possession_commercial,
    merge_vehicle_possession_passenger,
    merge_vehicle_sales_commercial,
    merge_vehicle_sales_passenger,
    merge_vehicle_sales_simple,
    merge_weekly_chemical,
    merge_weekly_summaries,
    package_to_frames,
    parse_balance_monthly,
    parse_consumption,
    parse_crude_import_quotas,
    parse_crude_output,
    parse_crude_port_shipping,
    parse_export_quotas_monthly,
    parse_export_quotas_summary,
    parse_kerosene_refueling,
    parse_monthly_inventories,
    parse_monthly_petchem,
    parse_processing_by_oil_grade,
    parse_profit_model,
    parse_provincial_processing,
    parse_provincial_product_output,
    parse_refined_export_forecast,
    parse_refined_export_shipping,
    parse_shandong_stocks,
    parse_vehicle_possession_commercial,
    parse_vehicle_possession_passenger,
    parse_vehicle_sales_commercial,
    parse_vehicle_sales_passenger,
    parse_vehicle_sales_simple,
    parse_weekly_chemical,
    parse_weekly_summary,
)

# Year-end snapshots retain full weekly history for that calendar year.
# API does not serve 2020 snapshots (tested); earliest available is 2021-12-31.
HISTORY_PUBDATES = (
    "2021-12-31T00:00:00",
    "2022-12-31T00:00:00",
    "2023-12-31T00:00:00",
    "2024-12-31T00:00:00",
    "2025-12-31T00:00:00",
)


def fetch_merged_history(
    client: SciClient | None = None,
    *,
    include_history: bool = True,
) -> dict[str, Any]:
    """
    Fetch latest package plus year-end archives and merge time series.

    Returns frames dict (same keys as package_to_frames) plus raw latest package.
    """
    client = client or SciClient()
    client.authenticate()

    latest_pubdate = client.latest_pubdate()
    pubdates = list(HISTORY_PUBDATES) if include_history else []
    if latest_pubdate not in pubdates:
        pubdates.append(latest_pubdate)

    packages: list[tuple[str, dict]] = []
    for pubdate in pubdates:
        packages.append((pubdate, client.fetch_package(pubdate=pubdate)))

    latest_package = packages[-1][1]
    base_frames = package_to_frames(latest_package)

    weekly_parts = []
    weekly_chemical_parts = []
    profit_parts = []
    stocks_parts = []
    balance_parts = []
    consumption_parts = []
    refueling_parts = []
    provincial_processing_parts = []
    provincial_output_parts = []
    crude_output_parts = []
    monthly_petchem_parts = []
    monthly_inventory_parts = []
    export_shipping_parts = []
    export_quotas_monthly_parts = []
    crude_foreign_parts = []
    crude_domestic_parts = []
    oil_grade_parts = []
    vehicle_sales_simple_parts = []
    vehicle_sales_passenger_parts = []
    vehicle_sales_commercial_parts = []
    vehicle_possession_passenger_parts = []
    vehicle_possession_commercial_parts = []

    for pubdate, pkg in packages:
        sheets = {s["SheetName"]: s for s in pkg.get("SheetList", [])}
        pub_year = int(pd.to_datetime(pkg.get("PubDate"), errors="coerce").year)

        def _find(*parts: str, exclude: tuple[str, ...] = ()) -> dict:
            parts_l = [p.lower() for p in parts]
            for name, sheet in sheets.items():
                nl = name.lower()
                if any(ex in nl for ex in exclude):
                    continue
                if all(p in nl for p in parts_l):
                    return sheet
            raise KeyError(parts)

        weekly_parts.append(parse_weekly_summary(_find("weekly", "summary")))
        weekly_chemical_parts.append(parse_weekly_chemical(_find("weekly", "summary")))
        profit_parts.append(parse_profit_model(_find("profit", "model")))
        stocks_parts.append(parse_shandong_stocks(_find("stocks")))
        balance_parts.append(parse_balance_monthly(_find("balance")))
        consumption_parts.append(parse_consumption(_find("consumption"), calendar_year=pub_year))
        refueling_parts.append(
            parse_kerosene_refueling(_find("kerosene", "refueling"), calendar_year=pub_year)
        )
        monthly_petchem_parts.append(
            parse_monthly_petchem(_find("monthly", "summary"), calendar_year=pub_year)
        )
        monthly_inventory_parts.append(
            parse_monthly_inventories(_find("monthly", "summary"), calendar_year=pub_year)
        )
        provincial_processing_parts.append(
            parse_provincial_processing(
                _find("processing", "volume", exclude=("ownership",)),
                calendar_year=pub_year,
            )
        )
        provincial_output_parts.append(
            parse_provincial_product_output(
                _find("oil product output", exclude=("ownership", "chemical")),
                calendar_year=pub_year,
            )
        )
        crude_output_parts.append(parse_crude_output(_find("crude", "output")))
        export_shipping_parts.append(
            parse_refined_export_shipping(
                _find("refined oil shipping", exclude=("forecast",))
            )
        )
        export_quotas_monthly_parts.append(
            parse_export_quotas_monthly(_find("product quotas"))
        )
        try:
            crude_foreign_parts.append(
                parse_crude_port_shipping(
                    _find("foreign", "shipping", exclude=("forecast",)),
                    trade="foreign",
                    is_forecast=False,
                )
            )
        except KeyError:
            pass
        try:
            crude_domestic_parts.append(
                parse_crude_port_shipping(
                    _find("domestic", "shipping", exclude=("forecast",)),
                    trade="domestic",
                    is_forecast=False,
                )
            )
        except KeyError:
            pass
        try:
            oil_grade_parts.append(
                parse_processing_by_oil_grade(_find("processing", "oil", "grade"))
            )
        except KeyError:
            pass
        try:
            vehicle_sales_simple_parts.append(
                parse_vehicle_sales_simple(
                    _find("vehicle sales", exclude=("commercial", "passenger")),
                    calendar_year=pub_year,
                )
            )
        except KeyError:
            pass
        try:
            vehicle_sales_passenger_parts.append(
                parse_vehicle_sales_passenger(_find("vehicle sales", "passenger"))
            )
        except KeyError:
            pass
        try:
            vehicle_sales_commercial_parts.append(
                parse_vehicle_sales_commercial(_find("vehicle sales", "commercial"))
            )
        except KeyError:
            pass
        try:
            vehicle_possession_passenger_parts.append(
                parse_vehicle_possession_passenger(
                    _find("vehicle possession", "passenger")
                )
            )
        except KeyError:
            pass
        try:
            vehicle_possession_commercial_parts.append(
                parse_vehicle_possession_commercial(
                    _find("vehicle possession", "commercial")
                )
            )
        except KeyError:
            pass

    weekly = merge_weekly_summaries(weekly_parts)
    petchem_weekly = merge_weekly_chemical(weekly_chemical_parts)
    margins = merge_profit_models(profit_parts)
    stocks = merge_shandong_stocks(stocks_parts)
    balance_monthly = merge_balance_monthly(balance_parts)
    consumption = merge_consumption(consumption_parts)
    consumption_national = build_consumption_national(consumption)
    kerosene_refueling = merge_kerosene_refueling(refueling_parts)
    kerosene_refueling_national = build_kerosene_refueling_national(kerosene_refueling)
    petchem_monthly = merge_monthly_petchem(monthly_petchem_parts)
    petchem_inventories = merge_monthly_inventories(monthly_inventory_parts)
    provincial_processing = merge_provincial_processing(provincial_processing_parts)
    provincial_output = merge_provincial_product_output(provincial_output_parts)
    crude_output = merge_crude_output(crude_output_parts)
    export_shipping = merge_refined_export_shipping(export_shipping_parts)
    export_quotas_monthly = merge_export_quotas_monthly(export_quotas_monthly_parts)
    crude_shipping = merge_crude_port_shipping(crude_foreign_parts + crude_domestic_parts)
    processing_oil_grade = merge_processing_by_oil_grade(oil_grade_parts)
    vehicle_sales_simple = merge_vehicle_sales_simple(vehicle_sales_simple_parts)
    vehicle_sales_passenger = merge_vehicle_sales_passenger(vehicle_sales_passenger_parts)
    vehicle_sales_commercial = merge_vehicle_sales_commercial(
        vehicle_sales_commercial_parts
    )
    vehicle_possession_passenger = merge_vehicle_possession_passenger(
        vehicle_possession_passenger_parts
    )
    vehicle_possession_commercial = merge_vehicle_possession_commercial(
        vehicle_possession_commercial_parts
    )

    latest_sheets = {s["SheetName"]: s for s in latest_package.get("SheetList", [])}

    def _find_latest(*parts: str, exclude: tuple[str, ...] = ()) -> dict:
        parts_l = [p.lower() for p in parts]
        for name, sheet in latest_sheets.items():
            nl = name.lower()
            if any(ex in nl for ex in exclude):
                continue
            if all(p in nl for p in parts_l):
                return sheet
        raise KeyError(parts)

    export_forecast = parse_refined_export_forecast(
        _find_latest("refined oil shipping", "forecast")
    )
    export_quotas_summary = parse_export_quotas_summary(_find_latest("product quotas"))

    crude_forecast_parts = []
    try:
        crude_forecast_parts.append(
            parse_crude_port_shipping(
                _find_latest("foreign", "shipping", "forecast"),
                trade="foreign",
                is_forecast=True,
            )
        )
    except KeyError:
        pass
    try:
        crude_forecast_parts.append(
            parse_crude_port_shipping(
                _find_latest("domestic", "shipping", "forecast"),
                trade="domestic",
                is_forecast=True,
            )
        )
    except KeyError:
        pass
    crude_forecast = merge_crude_port_shipping(crude_forecast_parts)

    try:
        # Year comes from sheet topic (e.g. "2026 Independent…").
        crude_quotas_summary, crude_quotas_monthly = parse_crude_import_quotas(
            _find_latest("crude", "quotas")
        )
    except KeyError:
        crude_quotas_summary, crude_quotas_monthly = parse_crude_import_quotas(
            {"DataList": []}
        )

    from src.transforms import (
        build_margins_wide,
        build_products_stocks_wide,
        build_runs_wide,
        shandong_stocks_totals,
    )

    base_frames["weekly"] = weekly
    base_frames["runs_wide"] = build_runs_wide(weekly)
    base_frames["products_stocks_wide"] = build_products_stocks_wide(weekly)
    base_frames["petchem_weekly"] = petchem_weekly
    base_frames["petchem_monthly"] = petchem_monthly
    base_frames["petchem_inventories"] = petchem_inventories
    base_frames["shandong_stocks"] = stocks
    base_frames["shandong_stocks_total"] = shandong_stocks_totals(stocks)
    base_frames["balance_monthly"] = balance_monthly
    base_frames["consumption"] = consumption
    base_frames["consumption_national"] = consumption_national
    base_frames["kerosene_refueling"] = kerosene_refueling
    base_frames["kerosene_refueling_national"] = kerosene_refueling_national
    base_frames["provincial_processing"] = provincial_processing
    base_frames["provincial_output"] = provincial_output
    base_frames["crude_output"] = crude_output
    base_frames["export_shipping"] = export_shipping
    base_frames["export_forecast"] = export_forecast
    base_frames["export_quotas_monthly"] = export_quotas_monthly
    base_frames["export_quotas_summary"] = export_quotas_summary
    base_frames["crude_shipping"] = crude_shipping
    base_frames["crude_forecast"] = crude_forecast
    base_frames["crude_quotas_summary"] = crude_quotas_summary
    base_frames["crude_quotas_monthly"] = crude_quotas_monthly
    base_frames["processing_oil_grade"] = processing_oil_grade
    base_frames["vehicle_sales_simple"] = vehicle_sales_simple
    base_frames["vehicle_sales_passenger"] = vehicle_sales_passenger
    base_frames["vehicle_sales_commercial"] = vehicle_sales_commercial
    base_frames["vehicle_possession_passenger"] = vehicle_possession_passenger
    base_frames["vehicle_possession_commercial"] = vehicle_possession_commercial
    base_frames["margins"] = margins
    base_frames["margins_wide"] = build_margins_wide(margins)

    meta = base_frames["meta"].copy()
    meta["history_snapshots"] = str(pubdates)
    base_frames["meta"] = meta

    from src.excel_archive import maybe_merge_excel_archive

    return maybe_merge_excel_archive(base_frames)
