"""Unit tests for the shared geo/date utilities in src/utils/geo.py."""
from __future__ import annotations

import os
import pytest
from datetime import date
from pathlib import Path

from src.utils.geo import (
    parse_date,
    format_ddmmyyyy,
    centroid_from_shapefile,
    bbox_from_shapefile,
    get_ibge_code,
    get_ibge_code_str,
)


# --------------------------------------------------------------------------- #
# parse_date
# --------------------------------------------------------------------------- #

class TestParseDate:
    def test_ddmmyyyy(self):
        assert parse_date("15/03/2024") == date(2024, 3, 15)

    def test_yyyy_mm_dd(self):
        assert parse_date("2024-03-15") == date(2024, 3, 15)

    def test_whitespace_stripped(self):
        assert parse_date("  2024-01-01  ") == date(2024, 1, 1)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("not-a-date")


# --------------------------------------------------------------------------- #
# format_ddmmyyyy
# --------------------------------------------------------------------------- #

class TestFormatDdmmyyyy:
    def test_from_date_object(self):
        assert format_ddmmyyyy(date(2024, 1, 5)) == "05/01/2024"

    def test_from_iso_string(self):
        assert format_ddmmyyyy("2024-01-05") == "05/01/2024"

    def test_from_ddmmyyyy_string(self):
        # Should round-trip cleanly
        assert format_ddmmyyyy("05/01/2024") == "05/01/2024"


# --------------------------------------------------------------------------- #
# Shapefile-dependent tests (skipped if test shapefile missing)
# --------------------------------------------------------------------------- #

_TEST_SHP = Path("data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp")
_HAS_SHP = _TEST_SHP.exists()


@pytest.mark.skipif(not _HAS_SHP, reason="Test shapefile not available")
class TestCentroidFromShapefile:
    def test_returns_lat_lon_tuple(self):
        lat, lon = centroid_from_shapefile(str(_TEST_SHP))
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        # São Paulo is roughly at -23.5, -46.6
        assert -25 < lat < -22, f"lat {lat} out of expected range"
        assert -48 < lon < -45, f"lon {lon} out of expected range"


@pytest.mark.skipif(not _HAS_SHP, reason="Test shapefile not available")
class TestBboxFromShapefile:
    def test_returns_four_element_list(self):
        bbox = bbox_from_shapefile(str(_TEST_SHP))
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        north, west, south, east = bbox
        assert north > south, "North must be greater than south"


@pytest.mark.skipif(not _HAS_SHP, reason="Test shapefile not available")
class TestGetIbgeCode:
    def test_returns_int_or_none(self):
        code = get_ibge_code(str(_TEST_SHP))
        if code is not None:
            assert isinstance(code, int)
            # São Paulo municipality code starts with 355030
            assert str(code).startswith("355030")

    def test_str_version_zero_padded(self):
        code_str = get_ibge_code_str(str(_TEST_SHP))
        if code_str is not None:
            assert len(code_str) == 7
            assert code_str.isdigit()
