"""Province centroids for China map overlays (approx. geographic centers)."""

from __future__ import annotations

# (latitude, longitude) — sufficient for desk map context.
PROVINCE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Anhui": (31.86, 117.27),
    "Beijing": (39.90, 116.40),
    "Chongqing": (29.56, 106.55),
    "Fujian": (26.08, 117.99),
    "Gansu": (36.06, 103.83),
    "Guangdong": (23.13, 113.26),
    "Guangxi": (22.82, 108.32),
    "Guizhou": (26.60, 106.71),
    "Hainan": (19.20, 109.75),
    "Hebei": (38.04, 114.48),
    "Heilongjiang": (45.74, 126.66),
    "Henan": (34.76, 113.65),
    "Hubei": (30.98, 114.30),
    "Hunan": (28.11, 112.98),
    "Inner Mongolia": (43.40, 111.75),
    "Jiangsu": (32.06, 118.80),
    "Jiangxi": (28.68, 115.89),
    "Jilin": (43.90, 125.32),
    "Liaoning": (41.84, 123.43),
    "Ningxia": (37.47, 106.26),
    "Qinghai": (36.62, 101.78),
    "Shaanxi": (34.27, 108.95),
    "Shandong": (36.67, 117.02),
    "Shanghai": (31.23, 121.47),
    "Shanxi": (37.87, 112.55),
    "Sichuan": (30.65, 104.07),
    "Tianjin": (39.12, 117.20),
    "Tibet": (31.69, 88.09),
    "Xinjiang": (41.79, 87.63),
    "Yunnan": (25.05, 102.71),
    "Zhejiang": (29.99, 120.15),
}
