"""Shared plotting helpers for publication figures."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
COUNTRIES_GEOJSON = DATA_ROOT / "enrichment" / "ne_50m_countries.geojson"
ADMIN1_GEOJSON = DATA_ROOT / "enrichment" / "ne_10m_admin1.geojson"
OSM_CACHE = ROOT / ".cache" / "osm_tiles"

# Raster basemap sources. Standard OSM tiles carry local-script labels -- around
# Gaza that means Arabic and Hebrew place names, which is wrong for an English
# paper -- and their full-colour styling competes with plotted data. The Carto
# styles are designed as data-overlay basemaps: light, low-chroma, Latin labels.
# Every one of them is OpenStreetMap data and must be attributed as such; the
# Carto styles additionally require a CARTO credit.
TILE_SOURCES = {
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "carto_light": "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "carto_light_nolabels": "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "carto_voyager": "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    "carto_voyager_nolabels": "https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
}
TILE_ATTRIBUTION = {
    "osm": r"Basemap \copyright{} OpenStreetMap contributors",
    "carto_light": r"Basemap \copyright{} OpenStreetMap contributors, \copyright{} CARTO",
    "carto_light_nolabels": r"Basemap \copyright{} OpenStreetMap contributors, \copyright{} CARTO",
    "carto_voyager": r"Basemap \copyright{} OpenStreetMap contributors, \copyright{} CARTO",
    "carto_voyager_nolabels": r"Basemap \copyright{} OpenStreetMap contributors, \copyright{} CARTO",
}


@lru_cache(maxsize=None)
def _features(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["features"]


def _rings(geom: dict) -> Iterable[list[list[float]]]:
    if geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            yield ring
    elif geom["type"] == "MultiPolygon":
        for polygon in geom["coordinates"]:
            for ring in polygon:
                yield ring


def _ring_intersects_bbox(ring: list[list[float]], bbox: tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = bbox
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return max(xs) >= xmin and min(xs) <= xmax and max(ys) >= ymin and min(ys) <= ymax


def _feature_matches(props: dict, countries: set[str] | None, admin_names: set[str] | None) -> bool:
    if countries:
        country_values = {
            str(props.get("ISO_A2", "")),
            str(props.get("ADM0_A3", "")),
            str(props.get("ADM0_ISO", "")),
            str(props.get("iso_a2", "")),
            str(props.get("adm0_a3", "")),
            str(props.get("ADMIN", "")),
            str(props.get("admin", "")),
            str(props.get("NAME", "")),
            str(props.get("name", "")),
        }
        if country_values & countries:
            return True
    if admin_names:
        admin_values = {
            str(props.get("name", "")),
            str(props.get("name_en", "")),
            str(props.get("NAME", "")),
            str(props.get("NAME_EN", "")),
        }
        if admin_values & admin_names:
            return True
    return not countries and not admin_names


def draw_geojson_layer(
    ax: plt.Axes,
    path: Path,
    bbox: tuple[float, float, float, float],
    *,
    countries: set[str] | None = None,
    admin_names: set[str] | None = None,
    facecolor: str = "#f3efe7",
    edgecolor: str = "#776f66",
    linewidth: float = 0.55,
    alpha: float = 1.0,
    zorder: int = 0,
) -> None:
    """Draw matching GeoJSON polygon rings that intersect bbox.

    The bbox is (xmin, xmax, ymin, ymax) in lon/lat. Holes are drawn as outlines
    but not cut from the filled polygon; for these small context maps the visual
    priority is coastline and administrative context, not exact areal masking.
    """

    countries = {str(c) for c in countries} if countries else None
    admin_names = {str(a) for a in admin_names} if admin_names else None
    for feature in _features(str(path)):
        props = feature.get("properties", {})
        if not _feature_matches(props, countries, admin_names):
            continue
        for i, ring in enumerate(_rings(feature["geometry"])):
            if not _ring_intersects_bbox(ring, bbox):
                continue
            xy = [(lon, lat) for lon, lat in ring]
            if i == 0:
                patch = Polygon(
                    xy,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
                ax.add_patch(patch)
            else:
                xs, ys = zip(*xy)
                ax.plot(xs, ys, color=edgecolor, linewidth=max(linewidth * 0.55, 0.2), alpha=alpha, zorder=zorder + 0.1)


def setup_context_map(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    countries: set[str],
    admin_names: set[str] | None = None,
    label_points: list[tuple[str, float, float]] | None = None,
    high_res: bool = False,
) -> None:
    """Style an axis as a clipped lon/lat context map with local basemap data.

    `high_res` matters for small extents. The default path fills land from
    `ne_50m_countries` (1:50M) and then strokes administrative boundaries from
    `ne_10m_admin1` (1:10M) over the top. Those two sources do not agree
    geometrically, so on a bbox only a fraction of a degree wide the same
    coastline or border is drawn two or three times at visibly different places
    -- a 40 km-wide territory like the Gaza Strip comes out as a bundle of
    offset, inaccurate-looking lines.

    With `high_res=True` everything comes from `ne_10m_admin1` instead: land is
    filled from the admin-1 units of `countries` with no stroke at all, so
    neighbouring units abut seamlessly into one landmass and the coastline is
    the fill boundary against the water colour. Only `admin_names` is stroked.
    One source, one resolution, each line drawn exactly once.
    """

    xmin, xmax, ymin, ymax = bbox
    ax.set_facecolor("#dceaf2")
    if high_res:
        _setup_high_res(ax, bbox, countries=countries, admin_names=admin_names)
        _finish_context_map(ax, bbox, label_points)
        return
    draw_geojson_layer(
        ax,
        COUNTRIES_GEOJSON,
        bbox,
        countries=countries,
        facecolor="#f5f1e8",
        edgecolor="#69635c",
        linewidth=0.55,
        alpha=1.0,
        zorder=0,
    )
    draw_geojson_layer(
        ax,
        ADMIN1_GEOJSON,
        bbox,
        countries=countries,
        facecolor="none",
        edgecolor="#a69d91",
        linewidth=0.35,
        alpha=0.9,
        zorder=0.8,
    )
    if admin_names:
        draw_geojson_layer(
            ax,
            ADMIN1_GEOJSON,
            bbox,
            admin_names=admin_names,
            facecolor="#ede2cf",
            edgecolor="#514b45",
            linewidth=0.75,
            alpha=0.75,
            zorder=0.9,
        )
    _finish_context_map(ax, bbox, label_points)


def _setup_high_res(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    countries: set[str],
    admin_names: set[str] | None,
) -> None:
    """Land and borders from ne_10m_admin1 alone. See `setup_context_map`."""

    # Land: every admin-1 unit of the requested countries, filled, unstroked.
    # Unstroked is the point -- these units tile their country, so any stroke
    # would draw internal seams, and it is the fill boundary against the axes
    # water colour that forms the coastline.
    draw_geojson_layer(
        ax,
        ADMIN1_GEOJSON,
        bbox,
        countries=countries,
        facecolor="#f5f1e8",
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
        zorder=0,
    )
    # The one boundary this map is about, stroked exactly once.
    if admin_names:
        draw_geojson_layer(
            ax,
            ADMIN1_GEOJSON,
            bbox,
            admin_names=admin_names,
            facecolor="#ede2cf",
            edgecolor="#4a453f",
            linewidth=0.7,
            alpha=0.95,
            zorder=0.9,
        )


def _finish_context_map(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    label_points: list[tuple[str, float, float]] | None,
) -> None:
    xmin, xmax, ymin, ymax = bbox
    if label_points:
        for label, lon, lat in label_points:
            ax.text(
                lon,
                lat,
                label,
                fontsize=7.5,
                color="#3f3a35",
                ha="center",
                va="center",
                zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.2},
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, color="#ffffff", alpha=0.65)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.0511), -85.0511)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2**zoom
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lon_e, lat_s, lat_n


def _load_osm_tile(x: int, y: int, zoom: int, source: str = "osm") -> Image.Image | None:
    # Cache per source: the styles differ, so they cannot share a directory.
    root = OSM_CACHE if source == "osm" else OSM_CACHE.with_name(f"tiles_{source}")
    path = root / str(zoom) / str(x) / f"{y}.png"
    if path.exists():
        return Image.open(path).convert("RGB")
    url = TILE_SOURCES[source].format(z=zoom, x=x, y=y)
    req = Request(url, headers={"User-Agent": "cell-geolocation-paper/0.1 (research plotting)"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = resp.read()
    except Exception:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return Image.open(path).convert("RGB")


def add_osm_basemap(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    zoom: int = 13,
    alpha: float = 0.72,
    grayscale: bool = True,
    zorder: int = 0,
    source: str = "osm",
) -> bool:
    """Draw cached OpenStreetMap raster tiles under a lon/lat plot.

    Returns True if at least one tile was drawn. The script remains usable
    offline after tiles are cached, and simply falls back to the caller's base
    map styling if the tile server is unavailable.
    """

    xmin, xmax, ymin, ymax = bbox
    x0, y1 = _lonlat_to_tile(xmin, ymin, zoom)
    x1, y0 = _lonlat_to_tile(xmax, ymax, zoom)
    drawn = False
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            img = _load_osm_tile(x, y, zoom, source)
            if img is None:
                continue
            if grayscale:
                img = ImageEnhance.Brightness(img.convert("L").convert("RGB")).enhance(1.08)
                img = ImageEnhance.Contrast(img).enhance(0.82)
            lon_w, lon_e, lat_s, lat_n = _tile_bounds(x, y, zoom)
            ax.imshow(img, extent=(lon_w, lon_e, lat_s, lat_n), origin="upper", alpha=alpha, zorder=zorder)
            drawn = True
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return drawn
