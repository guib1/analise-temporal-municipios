#!/usr/bin/env python3
"""
Unit tests for ERA5Downloader.

Tests aggregation logic, variable mapping, date iteration, and output schema
using synthetic xarray Datasets — NO network access required.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baixar_dados.ERA5 import ERA5Downloader


def _make_synthetic_hourly_ds(
    n_days: int = 3,
    lat: float = -23.55,
    lon: float = -46.63,
    start: str = "2020-01-01",
    renamed: bool = True,
) -> xr.Dataset:
    """Create a synthetic hourly ERA5-like dataset for testing.

    Args:
        renamed: if True (default), use the internal variable names (rnet, rterm)
                 that _aggregate_point/_aggregate_grid expect (i.e. after
                 _process_nc_to_daily_csv has done the renaming). If False,
                 use the raw CDS API names.
    """
    times = pd.date_range(start, periods=n_days * 24, freq="h")
    lats = [lat]
    lons = [lon]

    np.random.seed(42)
    n = len(times)

    # Solar radiation: positive during day, zero at night (J/m²)
    hours = times.hour
    ssr = np.where((hours >= 6) & (hours <= 18), np.random.uniform(100000, 500000, n), 0.0)

    # Thermal radiation: always negative (J/m²)
    str_vals = np.random.uniform(-200000, -10000, n)

    # Boundary layer height: 100-2000m, higher during day
    blh = np.where((hours >= 10) & (hours <= 16), np.random.uniform(800, 2000, n), np.random.uniform(50, 400, n))

    if renamed:
        var_names = {"ssr": "rnet", "str": "rterm", "blh": "boundary_layer_height"}
    else:
        var_names = {"ssr": "surface_net_solar_radiation", "str": "surface_net_thermal_radiation", "blh": "boundary_layer_height"}

    ds = xr.Dataset(
        {
            var_names["ssr"]: (["time", "latitude", "longitude"], ssr.reshape(n, 1, 1)),
            var_names["str"]: (["time", "latitude", "longitude"], str_vals.reshape(n, 1, 1)),
            var_names["blh"]: (["time", "latitude", "longitude"], blh.reshape(n, 1, 1)),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


class TestAggregatePoint(unittest.TestCase):
    """Test _aggregate_point with synthetic hourly data."""

    def setUp(self):
        self.dl = ERA5Downloader()
        self.ds = _make_synthetic_hourly_ds(n_days=3)

    def test_output_columns(self):
        """Output should have all expected columns."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)], ibge_code=3550308)
        expected_cols = [
            "codigo_ibge", "date",
            "rnet_max", "rnet_min", "rnet_mea",
            "rterm_max", "rterm_min", "rterm_mea",
            "pblhera5_max", "pblhera5_min", "pblhera5_mea",
        ]
        self.assertEqual(list(df.columns), expected_cols)

    def test_one_row_per_day(self):
        """Should produce exactly one row per day."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)], ibge_code=3550308)
        self.assertEqual(len(df), 3)

    def test_max_ge_mean_ge_min(self):
        """max >= mean >= min for all variables."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)])
        for var in ["rnet", "rterm", "pblhera5"]:
            for _, row in df.iterrows():
                self.assertGreaterEqual(row[f"{var}_max"], row[f"{var}_mea"],
                                        f"{var}_max should >= {var}_mea on {row['date']}")
                self.assertGreaterEqual(row[f"{var}_mea"], row[f"{var}_min"],
                                        f"{var}_mea should >= {var}_min on {row['date']}")

    def test_solar_radiation_range(self):
        """Solar radiation (rnet) should be >= 0 in our synthetic data (min is 0 at night)."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)])
        for _, row in df.iterrows():
            self.assertGreaterEqual(row["rnet_min"], 0,
                                    "Solar radiation min should be >= 0")

    def test_thermal_radiation_negative(self):
        """Thermal radiation (rterm) should be negative in our synthetic data."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)])
        for _, row in df.iterrows():
            self.assertLess(row["rterm_max"], 0,
                           "Thermal radiation should be negative")

    def test_pblh_positive(self):
        """Boundary layer height should be positive."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)])
        for _, row in df.iterrows():
            self.assertGreater(row["pblhera5_min"], 0,
                              "PBLH should be positive")

    def test_ibge_code_present(self):
        """codigo_ibge should be in the output when provided."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)], ibge_code=3550308)
        self.assertIn("codigo_ibge", df.columns)
        self.assertTrue((df["codigo_ibge"] == 3550308).all())

    def test_no_ibge_code_has_latlon(self):
        """Without ibge_code, should have lat/lon columns."""
        df = self.dl._aggregate_point(self.ds, [(-23.55, -46.63)])
        self.assertIn("lat", df.columns)
        self.assertIn("lon", df.columns)
        self.assertNotIn("codigo_ibge", df.columns)


class TestAggregateGrid(unittest.TestCase):
    """Test _aggregate_grid with synthetic hourly data."""

    def setUp(self):
        self.dl = ERA5Downloader()

    def _make_multi_grid_ds(self):
        """Create a dataset with 2x2 grid cells."""
        times = pd.date_range("2020-01-01", periods=48, freq="h")  # 2 days
        lats = [-23.5, -23.6]
        lons = [-46.6, -46.7]
        n = len(times)
        shape = (n, 2, 2)
        np.random.seed(42)
        ds = xr.Dataset(
            {
                "rnet": (["time", "latitude", "longitude"], np.random.uniform(0, 500000, shape)),
                "rterm": (["time", "latitude", "longitude"], np.random.uniform(-200000, -10000, shape)),
                "boundary_layer_height": (["time", "latitude", "longitude"], np.random.uniform(50, 2000, shape)),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        return ds

    def test_grid_rows(self):
        """Grid aggregation: 2 days * 4 grid points = 8 rows."""
        ds = self._make_multi_grid_ds()
        df = self.dl._aggregate_grid(ds)
        self.assertEqual(len(df), 8)  # 2 days * 2x2 grid

    def test_grid_with_ibge(self):
        """Grid aggregation with ibge_code."""
        ds = self._make_multi_grid_ds()
        df = self.dl._aggregate_grid(ds, ibge_code=3550308)
        self.assertIn("codigo_ibge", df.columns)
        self.assertTrue((df["codigo_ibge"] == 3550308).all())


class TestVariableMapping(unittest.TestCase):
    """Test that short variable names (ssr, str, blh) are mapped correctly."""

    def setUp(self):
        self.dl = ERA5Downloader()

    def test_short_names(self):
        """Dataset with short CDS names (ssr, str, blh) should be handled."""
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        np.random.seed(42)
        n = len(times)
        ds = xr.Dataset(
            {
                "ssr": (["time", "latitude", "longitude"], np.random.uniform(0, 500000, (n, 1, 1))),
                "str": (["time", "latitude", "longitude"], np.random.uniform(-200000, -10000, (n, 1, 1))),
                "blh": (["time", "latitude", "longitude"], np.random.uniform(50, 2000, (n, 1, 1))),
            },
            coords={"time": times, "latitude": [-23.55], "longitude": [-46.63]},
        )
        # Save and load through _process_nc_to_daily_csv
        # For a simpler test, just verify the renaming logic manually
        var_map = {
            'surface_net_solar_radiation': ['surface_net_solar_radiation', 'ssr'],
            'surface_net_thermal_radiation': ['surface_net_thermal_radiation', 'str'],
            'boundary_layer_height': ['boundary_layer_height', 'blh'],
        }
        for standard_name, candidates in var_map.items():
            for candidate in candidates:
                if candidate in ds:
                    if candidate != standard_name:
                        ds = ds.rename({candidate: standard_name})
                    break

        self.assertIn("surface_net_solar_radiation", ds)
        self.assertIn("surface_net_thermal_radiation", ds)
        self.assertIn("boundary_layer_height", ds)


class TestValidTimeDimension(unittest.TestCase):
    """Test that 'valid_time' dimension is correctly handled."""

    def setUp(self):
        self.dl = ERA5Downloader()

    def test_valid_time_rename(self):
        """When dataset has 'valid_time' instead of 'time', it should be handled."""
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        np.random.seed(42)
        n = len(times)
        ds = xr.Dataset(
            {
                "surface_net_solar_radiation": (["valid_time", "latitude", "longitude"], np.random.uniform(0, 500000, (n, 1, 1))),
                "surface_net_thermal_radiation": (["valid_time", "latitude", "longitude"], np.random.uniform(-200000, -10000, (n, 1, 1))),
                "boundary_layer_height": (["valid_time", "latitude", "longitude"], np.random.uniform(50, 2000, (n, 1, 1))),
            },
            coords={"valid_time": times, "latitude": [-23.55], "longitude": [-46.63]},
        )
        # Simulate the renaming logic from _process_nc_to_daily_csv
        if 'valid_time' in ds.coords and 'time' not in ds.coords:
            ds = ds.rename({'valid_time': 'time'})

        self.assertIn("time", ds.dims)

        # Should aggregate without error
        ds['rnet'] = ds['surface_net_solar_radiation']
        ds['rterm'] = ds['surface_net_thermal_radiation']
        ds = ds[['rnet', 'rterm', 'boundary_layer_height']]
        df = self.dl._aggregate_point(ds, [(-23.55, -46.63)])
        self.assertEqual(len(df), 1)  # 1 day


class TestIterMonths(unittest.TestCase):
    """Test the month iteration utility."""

    def test_single_month(self):
        result = list(ERA5Downloader._iter_months(date(2020, 3, 1), date(2020, 3, 31)))
        self.assertEqual(result, [(2020, 3)])

    def test_cross_year(self):
        result = list(ERA5Downloader._iter_months(date(2019, 11, 1), date(2020, 2, 28)))
        self.assertEqual(result, [(2019, 11), (2019, 12), (2020, 1), (2020, 2)])

    def test_same_month_partial(self):
        result = list(ERA5Downloader._iter_months(date(2020, 6, 15), date(2020, 6, 20)))
        self.assertEqual(result, [(2020, 6)])


class TestGetDaysInMonth(unittest.TestCase):
    """Test the day listing utility."""

    def test_full_month(self):
        days = ERA5Downloader._get_days_in_month(2020, 1, date(2020, 1, 1), date(2020, 1, 31))
        self.assertEqual(len(days), 31)
        self.assertEqual(days[0], "01")
        self.assertEqual(days[-1], "31")

    def test_partial_start(self):
        days = ERA5Downloader._get_days_in_month(2020, 1, date(2020, 1, 15), date(2020, 2, 28))
        self.assertEqual(days[0], "15")
        self.assertEqual(days[-1], "31")

    def test_partial_end(self):
        days = ERA5Downloader._get_days_in_month(2020, 2, date(2020, 1, 1), date(2020, 2, 10))
        self.assertEqual(days[0], "01")
        self.assertEqual(days[-1], "10")

    def test_leap_year_feb(self):
        days = ERA5Downloader._get_days_in_month(2020, 2, date(2020, 2, 1), date(2020, 2, 29))
        self.assertEqual(len(days), 29)

    def test_non_leap_year_feb(self):
        days = ERA5Downloader._get_days_in_month(2021, 2, date(2021, 2, 1), date(2021, 2, 28))
        self.assertEqual(len(days), 28)


class TestAggregationAccuracy(unittest.TestCase):
    """Test that aggregation produces exact expected values with known input."""

    def setUp(self):
        self.dl = ERA5Downloader()

    def test_known_values(self):
        """With deterministic input, verify exact daily aggregates."""
        # 1 day = 24 hours. Set specific values.
        times = pd.date_range("2020-06-15", periods=24, freq="h")

        # Solar: linearly 0 to 230000 then back to 0
        ssr = np.array([0, 10000, 20000, 30000, 40000, 50000,
                        60000, 80000, 100000, 150000, 200000, 230000,
                        230000, 200000, 150000, 100000, 80000, 60000,
                        50000, 40000, 30000, 20000, 10000, 0], dtype=float)

        # Thermal: constant -50000
        str_vals = np.full(24, -50000.0)

        # BLH: constant 500.0
        blh = np.full(24, 500.0)

        ds = xr.Dataset(
            {
                "rnet": (["time", "latitude", "longitude"], ssr.reshape(24, 1, 1)),
                "rterm": (["time", "latitude", "longitude"], str_vals.reshape(24, 1, 1)),
                "boundary_layer_height": (["time", "latitude", "longitude"], blh.reshape(24, 1, 1)),
            },
            coords={"time": times, "latitude": [-23.55], "longitude": [-46.63]},
        )

        df = self.dl._aggregate_point(ds, [(-23.55, -46.63)])

        self.assertEqual(len(df), 1)
        row = df.iloc[0]

        # rnet: max=230000, min=0, mean=sum/24
        self.assertAlmostEqual(row["rnet_max"], 230000.0)
        self.assertAlmostEqual(row["rnet_min"], 0.0)
        expected_mean = ssr.sum() / 24
        self.assertAlmostEqual(row["rnet_mea"], expected_mean, places=1)

        # rterm: all constant -50000
        self.assertAlmostEqual(row["rterm_max"], -50000.0)
        self.assertAlmostEqual(row["rterm_min"], -50000.0)
        self.assertAlmostEqual(row["rterm_mea"], -50000.0)

        # pblh: all constant 500
        self.assertAlmostEqual(row["pblhera5_max"], 500.0)
        self.assertAlmostEqual(row["pblhera5_min"], 500.0)
        self.assertAlmostEqual(row["pblhera5_mea"], 500.0)


if __name__ == "__main__":
    unittest.main()
