"""SCI99 datapack API client."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_BASE_URL = "https://open.sci99.com/datapackapi/pkg"
_DEFAULT_PKG_ID = 1233  # China Oil & Chemical Industry Data - Ownership

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


class SciClient:
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        package_id: int | None = None,
        timeout: int = 120,
    ) -> None:
        self.username = username or os.getenv("USER_NAME", "")
        self.password = password or os.getenv("PASSWORD", "")
        pkg = package_id or os.getenv("DATA_PACKAGE_ID")
        self.package_id = int(pkg) if pkg else _DEFAULT_PKG_ID
        self.timeout = timeout
        self._token: str | None = None

    def _headers(self) -> dict[str, str]:
        if not self._token:
            raise RuntimeError("Not authenticated — call authenticate() first.")
        return {"scipkgauth": self._token}

    def authenticate(self) -> str:
        response = requests.post(
            f"{_BASE_URL}/user/gettoken",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "UserName": self.username,
                    "Password": self.password,
                    "UserType": 2,
                }
            ),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("data")
        if not token:
            raise RuntimeError(f"SCI token request failed: {payload}")
        self._token = token
        return token

    def list_packages(self) -> list[dict[str, Any]]:
        response = requests.get(
            f"{_BASE_URL}/getmypkglist",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("data", [])

    def latest_pubdate(self, package_id: int | None = None) -> str:
        pkg_id = package_id or self.package_id
        for pkg in self.list_packages():
            if int(pkg["DataPackageID"]) == pkg_id:
                return pkg["LatestDate"]
        raise ValueError(f"Package {pkg_id} not found in subscription list.")

    def fetch_package(
        self,
        package_id: int | None = None,
        pubdate: str | None = None,
    ) -> dict[str, Any]:
        pkg_id = package_id or self.package_id
        if pubdate is None:
            pubdate = self.latest_pubdate(pkg_id)

        response = requests.get(
            f"{_BASE_URL}/getpkgdata",
            params={"pkgid": pkg_id, "pubdate": pubdate},
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not data:
            raise RuntimeError(f"Empty package payload: {payload}")
        if isinstance(data, str):
            raise RuntimeError(
                f"No package data for pubdate={pubdate!r} (API returned: {data!r}). "
                "This snapshot may not exist for your subscription."
            )
        return data

    def sheet_by_name(self, package: dict[str, Any], *name_parts: str) -> dict[str, Any]:
        """Find a sheet whose name contains all name_parts (case-insensitive)."""
        parts = [p.lower() for p in name_parts]
        for sheet in package.get("SheetList", []):
            name = sheet.get("SheetName", "").lower()
            if all(p in name for p in parts):
                return sheet
        available = [s.get("SheetName") for s in package.get("SheetList", [])]
        raise KeyError(f"No sheet matching {name_parts!r}. Available: {available}")


def fetch_oil_chemical_package(
    package_id: int | None = None,
    pubdate: str | None = None,
) -> dict[str, Any]:
    client = SciClient(package_id=package_id)
    client.authenticate()
    return client.fetch_package(pubdate=pubdate)
