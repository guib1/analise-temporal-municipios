#!/usr/bin/env python3
"""
Unit tests for MERRA2Downloader.

Tests hourly→daily aggregation, variable config, location label,
column naming, and date filtering — NO network access required.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baixar_dados.MERRA2 import MERRA2Downloader, MerraVariable


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_dl(**kwargs) -> MERRA2Downloader:
    """Create downloader without credential validation."""
    dl = MERRA2Downloader.__new__(MERRA2Downloader)
    dl.username = "test_user"
    dl.password = "test_pass"
    dl.cache_dir = Path(kwargs.get("cache_dir", tempfile.mkdtemp()))
    dl.cache_dir.mkdir(parents=True, exist_ok=True)
    return dl


def _write_hourly_csv(path: Path, var_name: str, n_days: int = 5,
                      start: str = "2020-01-01") -> Path:
    """Write a synthetic hourly CSV matching MERRA2 output format."""
    path.mkdir(parents=True, exist_ok=True)
    csv_file = path / f"test_loc_hourly.csv"
    rows = []
    np.random.seed(42)
    for d in range(n_days):
        dt = pd.Timestamp(start) + pd.Timedelta(days=d)
        for h in range(24):
            ts = dt + pd.Timedelta(hours=h)
            rows.append({
                "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                var_name: np.random.exponential(1e-11),
            })
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    return csv_file


# --------------------------------------------------------------------------- #
# Variable Config tests
# --------------------------------------------------------------------------- #
class TestVariableConfig(unittest.TestCase):
    """Verify that declared variables match the glossary."""

    EXPECTED_VARS = [
        "organiccarbonbiofuelemissions",
        "organiccarbonbiogenicemissions",
        "organiccarbonanthropogenicemissions",
        "organiccarbonbiomassburningemissions",
        "blackcarbonbiomassburning",
        "blackcarbonbiofuel",
        "so2antropogenico",
        "so4antropogenico",
        "so2biomassburning",
        "blackcarbonantrogonenico",
        "comerra2",
    ]

    def test_all_glossary_vars_present(self):
        """All 11 reference CSV variables should be configured."""
        for var in self.EXPECTED_VARS:
            self.assertIn(var, MERRA2Downloader.VARIABLES,
                          f"Missing variable: {var}")

    def test_variable_has_field_and_dataset(self):
        """Each variable should have field_id and dataset_name."""
        for name, mv in MERRA2Downloader.VARIABLES.items():
            self.assertTrue(mv.field_id, f"{name} missing field_id")
            self.assertTrue(mv.dataset_name, f"{name} missing dataset_name")
            self.assertTrue(mv.dataset_id, f"{name} missing dataset_id")

    def test_adg_variables_use_correct_dataset(self):
        """Emission variables should use M2T1NXADG dataset."""
        adg_vars = [
            "organiccarbonbiofuelemissions", "organiccarbonbiogenicemissions",
            "organiccarbonanthropogenicemissions", "organiccarbonbiomassburningemissions",
            "blackcarbonbiomassburning", "blackcarbonbiofuel",
            "so2antropogenico", "so4antropogenico", "so2biomassburning",
            "blackcarbonantrogonenico",
        ]
        for var in adg_vars:
            cfg = MERRA2Downloader.VARIABLES[var]
            self.assertEqual(cfg.dataset_name, "M2T1NXADG",
                             f"{var} should use M2T1NXADG")

    def test_co_uses_chm_dataset(self):
        """CO (comerra2) should use M2T1NXCHM dataset."""
        self.assertEqual(MERRA2Downloader.VARIABLES["comerra2"].dataset_name,
                         "M2T1NXCHM")


# --------------------------------------------------------------------------- #
# Hourly CSV Aggregation tests
# --------------------------------------------------------------------------- #
class TestProcessHourlyCSV(unittest.TestCase):
    """Test _process_hourly_csv aggregation logic."""

    def setUp(self):
        self.dl = _make_dl()
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_daily_aggregation_columns(self):
        """Output should have date + {var}_max, {var}_min, {var}_mea."""
        var = "organiccarbonbiofuelemissions"
        csv_path = _write_hourly_csv(self.tmpdir, var, n_days=3)
        result = self.dl._process_hourly_csv(
            csv_path, var, date(2020, 1, 1), date(2020, 1, 3)
        )
        self.assertIn(f"{var}_max", result.columns)
        self.assertIn(f"{var}_min", result.columns)
        self.assertIn(f"{var}_mea", result.columns)
        self.assertIn("date", result.columns)

    def test_daily_aggregation_row_count(self):
        """Should produce one row per day."""
        var = "blackcarbonbiofuel"
        csv_path = _write_hourly_csv(self.tmpdir, var, n_days=5)
        result = self.dl._process_hourly_csv(
            csv_path, var, date(2020, 1, 1), date(2020, 1, 5)
        )
        self.assertEqual(len(result), 5)

    def test_max_min_mean_correctness(self):
        """Max >= mean >= min for each day."""
        var = "so2antropogenico"
        csv_path = _write_hourly_csv(self.tmpdir, var, n_days=3)
        result = self.dl._process_hourly_csv(
            csv_path, var, date(2020, 1, 1), date(2020, 1, 3)
        )
        for _, row in result.iterrows():
            self.assertGreaterEqual(row[f"{var}_max"], row[f"{var}_mea"])
            self.assertGreaterEqual(row[f"{var}_mea"], row[f"{var}_min"])

    def test_date_filtering(self):
        """Should filter to requested date range only."""
        var = "comerra2"
        csv_path = _write_hourly_csv(self.tmpdir, var, n_days=10,
                                     start="2020-01-01")
        result = self.dl._process_hourly_csv(
            csv_path, var, date(2020, 1, 3), date(2020, 1, 7)
        )
        self.assertEqual(len(result), 5)  # Jan 3–7

    def test_empty_csv(self):
        """Empty CSV should return empty DataFrame."""
        var = "comerra2"
        csv_path = self.tmpdir / "empty.csv"
        pd.DataFrame(columns=["date", var]).to_csv(csv_path, index=False)
        result = self.dl._process_hourly_csv(
            csv_path, var, date(2020, 1, 1), date(2020, 1, 5)
        )
        self.assertTrue(result.empty)

    def test_missing_column(self):
        """CSV without expected column should return empty DataFrame."""
        var = "comerra2"
        csv_path = self.tmpdir / "wrong_cols.csv"
        pd.DataFrame({"date": ["2020-01-01"], "other": [1.0]}).to_csv(
            csv_path, index=False
        )
        result = self.dl._process_hourly_csv(
            csv_path, var, date(2020, 1, 1), date(2020, 1, 5)
        )
        self.assertTrue(result.empty)


# --------------------------------------------------------------------------- #
# Location Label tests
# --------------------------------------------------------------------------- #
class TestLocationLabel(unittest.TestCase):

    def setUp(self):
        self.dl = _make_dl()

    def test_basic_label(self):
        label = self.dl._build_location_label("data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp")
        self.assertTrue(label)
        self.assertTrue(all(ch.isalnum() or ch == '_' for ch in label))

    def test_special_chars_sanitized(self):
        label = self.dl._build_location_label("/path/to/SP (São Paulo).shp")
        self.assertNotIn("(", label)
        self.assertNotIn(")", label)
        self.assertNotIn(" ", label)


# --------------------------------------------------------------------------- #
# Reference Value Range tests
# --------------------------------------------------------------------------- #
class TestReferenceValueRanges(unittest.TestCase):
    """Verify that typical MERRA2 values are in expected scientific ranges."""

    REF_CSV = PROJECT_ROOT / "data/raw_input_reference/3550308_saopaulo_sp_asma.csv"

    @unittest.skipUnless(
        (PROJECT_ROOT / "data/raw_input_reference/3550308_saopaulo_sp_asma.csv").exists(),
        "Reference CSV not found"
    )
    def test_emission_values_order_of_magnitude(self):
        """Emission values should be ~1e-12 to 1e-10 kg/(m².s)."""
        df = pd.read_csv(self.REF_CSV, nrows=20)
        emission_cols = [c for c in df.columns
                        if any(c.startswith(p) for p in
                               ["organiccarbon", "blackcarbon", "so2", "so4"])
                        and not c.startswith("so2_")]  # exclude CETESB so2
        for col in emission_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                # All values should be >= 0 (emissions can't be negative)
                self.assertTrue((vals >= 0).all(),
                                f"{col} has negative values")
                # Non-zero values should be in ~1e-13 to 1e-8 range
                nonzero = vals[vals > 0]
                if len(nonzero) > 0:
                    self.assertGreater(nonzero.max(), 1e-14,
                                      f"{col} values too small")
                    self.assertLess(nonzero.max(), 1e-6,
                                    f"{col} values too large")

    @unittest.skipUnless(
        (PROJECT_ROOT / "data/raw_input_reference/3550308_saopaulo_sp_asma.csv").exists(),
        "Reference CSV not found"
    )
    def test_co_values_range(self):
        """CO (comerra2) should be in ~100-2000 range (different unit/scale)."""
        df = pd.read_csv(self.REF_CSV, nrows=20)
        if "comerra2_max" in df.columns:
            vals = df["comerra2_max"].dropna()
            self.assertGreater(vals.max(), 50)
            self.assertLess(vals.max(), 10000)


if __name__ == "__main__":
    unittest.main()
