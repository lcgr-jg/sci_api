"""Parse China oil infrastructure KMZ for map overlays."""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.china_geo import PROVINCE_CENTROIDS

KML_NS = "http://www.opengis.net/kml/2.2"

# KMZ folder name → SCI ownership label (None = no SCI runs data).
KMZ_OWNER_TO_SCI: dict[str, str | None] = {
    "Sinopec": "Sinopec",
    "Petrochina": "PetroChina",
    "CNOOC": "CNOOC",
    "Independants": "Independent refinery",
    "Teapots": "Independent refinery",
    "Sinochem": None,
}

# Visual layer styling — all layers shown; non-refinery infra uses neutral tones.
LAYER_COLORS: dict[str, str] = {
    "Sinopec": "#3fb950",
    "Petrochina": "#58a6ff",
    "CNOOC": "#bc8cff",
    "Independants": "#f85149",
    "Teapots": "#ffa657",
    "Sinochem": "#e3b341",
    "Terminals": "#8b949e",
    "Pipeline": "#484f58",
    "Oilfield": "#6e7681",
    "To review": "#768390",
}

REFINERY_GROUPS = frozenset(
    {"Sinopec", "Petrochina", "CNOOC", "Sinochem", "Independants", "Teapots"}
)


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _find_child(el: ET.Element, local_name: str) -> ET.Element | None:
    for child in el:
        if _local(child.tag) == local_name:
            return child
    return None


def _parse_coord_pairs(text: str) -> list[tuple[float, float]]:
    """KML coordinates are lon,lat[,alt] tuples separated by whitespace."""
    pairs: list[tuple[float, float]] = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            pairs.append((lat, lon))
    return pairs


def _geometry_parts(placemark: ET.Element) -> list[tuple[str, list[tuple[float, float]]]]:
    """Return (geometry_type, lat/lon pairs) for each sub-geometry in a placemark."""
    out: list[tuple[str, list[tuple[float, float]]]] = []
    for el in placemark.iter():
        tag = _local(el.tag)
        if tag in ("Point", "LineString", "LinearRing"):
            coord_el = _find_child(el, "coordinates")
            if coord_el is not None and coord_el.text:
                pairs = _parse_coord_pairs(coord_el.text)
                if pairs:
                    out.append((tag, pairs))
        elif tag == "Polygon":
            outer = None
            for child in el:
                if _local(child.tag) == "outerBoundaryIs":
                    ring = _find_child(child, "LinearRing")
                    if ring is not None:
                        coord_el = _find_child(ring, "coordinates")
                        if coord_el is not None and coord_el.text:
                            outer = _parse_coord_pairs(coord_el.text)
            if outer:
                out.append(("Polygon", outer))
    return out


def _centroid(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")
    lat = sum(p[0] for p in pairs) / len(pairs)
    lon = sum(p[1] for p in pairs) / len(pairs)
    return lat, lon


def _infer_province(name: str, folder_path: tuple[str, ...]) -> str | None:
    """Best-effort province from Teapot subfolder or refinery name."""
    if len(folder_path) >= 3 and folder_path[1] == "Teapots":
        candidate = folder_path[2]
        if candidate in PROVINCE_CENTROIDS:
            return candidate

    name_lower = name.lower()
    # Longer province names first to avoid partial matches (e.g. Shanxi vs Shaanxi).
    for prov in sorted(PROVINCE_CENTROIDS, key=len, reverse=True):
        if prov.lower() in name_lower:
            return prov
    return None


def load_infra_sites(kmz_path: Path | None = None) -> pd.DataFrame:
    """
    Load refinery and infrastructure sites from the KMZ.

    Returns one row per geometry part with columns:
    name, kmz_group, sci_ownership, province, lat, lon, geometry_type, is_refinery
    """
    if kmz_path is None:
        kmz_path = Path(__file__).resolve().parents[1] / "data" / "China oil infra.kmz"
    if not kmz_path.exists():
        return pd.DataFrame(
            columns=[
                "name",
                "kmz_group",
                "sci_ownership",
                "province",
                "lat",
                "lon",
                "geometry_type",
                "is_refinery",
                "coords",
            ]
        )

    with zipfile.ZipFile(kmz_path) as zf:
        kml_name = next(n for n in zf.namelist() if n.endswith(".kml"))
        root = ET.fromstring(zf.read(kml_name))

    rows: list[dict] = []

    def walk(el: ET.Element, folder_path: tuple[str, ...] = ()) -> None:
        tag = _local(el.tag)
        if tag == "Folder":
            name_el = _find_child(el, "name")
            fname = name_el.text if name_el is not None else "?"
            folder_path = folder_path + (fname,)
            for child in el:
                walk(child, folder_path)
        elif tag == "Placemark":
            name_el = _find_child(el, "name")
            pname = name_el.text if name_el is not None else "?"
            kmz_group = folder_path[1] if len(folder_path) >= 2 else "Other"
            sci_ownership = KMZ_OWNER_TO_SCI.get(kmz_group)
            province = _infer_province(pname, folder_path)
            is_refinery = kmz_group in REFINERY_GROUPS

            for geom_type, pairs in _geometry_parts(el):
                lat, lon = _centroid(pairs)
                rows.append(
                    {
                        "name": pname,
                        "kmz_group": kmz_group,
                        "sci_ownership": sci_ownership,
                        "province": province,
                        "lat": lat,
                        "lon": lon,
                        "geometry_type": geom_type,
                        "is_refinery": is_refinery,
                        "coords": pairs,
                    }
                )
        else:
            for child in el:
                walk(child, folder_path)

    walk(root)

    if not rows:
        return pd.DataFrame(
            columns=[
                "name",
                "kmz_group",
                "sci_ownership",
                "province",
                "lat",
                "lon",
                "geometry_type",
                "is_refinery",
                "coords",
            ]
        )

    return pd.DataFrame(rows)
