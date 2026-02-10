"""Shared geospatial and date-parsing utilities used by all downloaders.

This module is the single source of truth for functions that were previously
copy-pasted across every script in ``src/baixar_dados/``.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date(value: str) -> date:
    """Parse a date string in ``DD/MM/YYYY`` or ``YYYY-MM-DD`` format.

    Raises ``ValueError`` for unrecognised formats.
    """
    value = value.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    # pandas fallback (handles other ISO forms)
    try:
        dt = pd.to_datetime(value, errors="raise")
        return dt.date()
    except Exception:
        pass
    raise ValueError(f"Invalid date format: {value}. Use DD/MM/YYYY or YYYY-MM-DD")


def format_ddmmyyyy(value: str | date) -> str:
    """Convert a date (or date-string) to ``DD/MM/YYYY``."""
    dt = parse_date(value) if isinstance(value, str) else value
    return dt.strftime("%d/%m/%Y")


# ---------------------------------------------------------------------------
# Shapefile helpers
# ---------------------------------------------------------------------------

def _require_geopandas():
    if gpd is None:
        raise RuntimeError(
            "geopandas is not installed; please install it to use shapefile operations."
        )


def centroid_from_shapefile(shp_path: str | Path) -> Tuple[float, float]:
    """Return ``(latitude, longitude)`` of the shapefile's centroid."""
    _require_geopandas()
    gdf = gpd.read_file(str(shp_path))
    try:
        union_geom = gdf.union_all()
    except AttributeError:  # geopandas < 0.13
        union_geom = gdf.unary_union
    centroid = union_geom.centroid
    return float(centroid.y), float(centroid.x)


def bbox_from_shapefile(shp_path: str | Path) -> List[float]:
    """Return ``[north, west, south, east]`` bounding box (CDS convention)."""
    _require_geopandas()
    gdf = gpd.read_file(str(shp_path))
    minx, miny, maxx, maxy = gdf.total_bounds
    return [float(maxy), float(minx), float(miny), float(maxx)]


# ---------------------------------------------------------------------------
# IBGE code resolution
# ---------------------------------------------------------------------------

def get_ibge_code(shapefile_path: str | Path) -> Optional[int]:
    """Extract the 7-digit IBGE municipality code.

    Resolution order:
    1. Read directly from the shapefile attribute table
       (columns ``code_muni``, ``CD_MUN``, ``CD_GEOCMU``).
    2. Fallback: look for a ``*_ibge.csv`` sidecar file in the same directory.

    Returns ``None`` when the code cannot be determined.
    """
    shapefile_path = Path(shapefile_path)

    # 1. Try shapefile columns
    if gpd is not None:
        try:
            gdf = gpd.read_file(str(shapefile_path))
            for col in ("code_muni", "CD_MUN", "CD_GEOCMU"):
                if col in gdf.columns:
                    return int(gdf[col].iloc[0])
        except Exception as exc:
            LOGGER.warning("Could not read shapefile for IBGE code: %s", exc)

    # 2. Fallback: sidecar CSV
    directory = shapefile_path.parent
    for csv_file in sorted(directory.glob("*_ibge.csv")):
        try:
            df = pd.read_csv(csv_file)
            for col in ("codigo_ibge", "cod_ibge", "COD_IBGE", "CD_MUN", "CD_GEOCMU"):
                if col in df.columns and not df.empty:
                    return int(df[col].iloc[0])
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", csv_file, exc)

    return None


def get_ibge_code_str(shapefile_path: str | Path) -> Optional[str]:
    """Like :func:`get_ibge_code` but returns a 7-char zero-padded string."""
    code = get_ibge_code(shapefile_path)
    if code is not None:
        return str(code).zfill(7)
    return None
