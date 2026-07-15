"""Minimal SCI99 datapack API demo — fetches the Oil & Chemical Ownership package."""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASSWORD")
PACKAGE_ID = int(os.getenv("DATA_PACKAGE_ID", "1233"))

BASE = "https://open.sci99.com/datapackapi/pkg"

# 1) Token
resp = requests.post(
    f"{BASE}/user/gettoken",
    headers={"Content-Type": "application/json"},
    json={"UserName": USER_NAME, "Password": PASSWORD, "UserType": 2},
    timeout=30,
)
resp.raise_for_status()
token = resp.json()["data"]
headers = {"scipkgauth": token}

# 2) Package list — locate Oil & Chemical datapack
resp = requests.get(f"{BASE}/getmypkglist", headers=headers, timeout=30)
resp.raise_for_status()
packages = resp.json()["data"]
match = next((p for p in packages if int(p["DataPackageID"]) == PACKAGE_ID), packages[0])
pubdate = match["LatestDate"]
print(f"Using package {match['DataPackageID']}: {match['DataPackageName']} ({pubdate})")

# 3) Full datapack JSON
resp = requests.get(
    f"{BASE}/getpkgdata",
    params={"pkgid": match["DataPackageID"], "pubdate": pubdate},
    headers=headers,
    timeout=120,
)
resp.raise_for_status()
data = resp.json()["data"]

out = Path(__file__).resolve().parent / "data" / "latest_raw.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"Wrote {len(data.get('SheetList', []))} sheets → {out}")
