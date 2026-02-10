#!/usr/bin/env python3
"""
Unit tests for DiversosDownloader (INDICE-CALCULADO.py).

Tests heat index calculation, heat/cold wave detection, SPI, El Niño parsing,
and output schema — NO network access required (El Niño tested via mock).
"""
from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baixar_dados import __path__  # noqa: ensure package importable

# We need to handle the import carefully since the module has a hyphenated name
import importlib
INDICE_MOD = importlib.import_module("src.baixar_dados.INDICE-CALCULADO")
DiversosDownloader = INDICE_MOD.DiversosDownloader


def _make_dl() -> DiversosDownloader:
    """Create DiversosDownloader without network calls."""
    dl = DiversosDownloader.__new__(DiversosDownloader)
    dl.inmet_downloader = None
    return dl


def _make_daily_inmet(n_days: int = 90, start: str = "2020-01-01") -> pd.DataFrame:
    """Create synthetic daily INMET output for testing computed indices."""
    np.random.seed(42)
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({
        "date": dates.date,
        "globalradiation_max": np.random.uniform(3, 11, n_days),
        "globalradiation_min": np.random.uniform(0, 2, n_days),
        "globalradiation_mea": np.random.uniform(1, 6, n_days),
        "precipitation_sum": np.random.exponential(5, n_days),  # right-skewed
        "temperature_max": np.random.uniform(25, 38, n_days),
        "temperature_min": np.random.uniform(12, 22, n_days),
        "temperature_mea": np.random.uniform(18, 28, n_days),
        "humidity_min": np.random.uniform(25, 60, n_days),
        "humidity_mea": np.random.uniform(50, 85, n_days),
        "wind_mea": np.random.uniform(1, 5, n_days),
        "heatindex_max": np.random.uniform(26, 42, n_days),
        "heatindex_min": np.random.uniform(12, 22, n_days),
        "heatindex_mea": np.random.uniform(18, 30, n_days),
    })


# --------------------------------------------------------------------------- #
# Heat Index tests
# --------------------------------------------------------------------------- #
class TestHeatIndex(unittest.TestCase):

    def setUp(self):
        self.dl = _make_dl()

    def test_below_80f_simple_formula(self):
        """When T < 80°F (~26.7°C), simplified formula is used."""
        row = {"temp_max": 20.0, "hum_min": 50.0}
        hi = self.dl._calculate_heat_index(row)
        self.assertIsNotNone(hi)
        self.assertFalse(np.isnan(hi))
        # Below 80°F, HI ≈ T (but uses simple formula, not identity)
        self.assertAlmostEqual(hi, 20.0, delta=3.0)

    def test_above_80f_regression(self):
        """When T > 80°F (~26.7°C), Rothfusz regression used."""
        row = {"temp_max": 35.0, "hum_min": 40.0}
        hi = self.dl._calculate_heat_index(row)
        self.assertGreater(hi, 30.0)
        self.assertLess(hi, 60.0)

    def test_high_humidity_adjustment(self):
        """High humidity + moderate temp should trigger adjustment."""
        row = {"temp_max": 30.0, "hum_min": 90.0}
        hi = self.dl._calculate_heat_index(row)
        # At 30°C (86°F) + 90% RH the adjustment kicks in
        self.assertGreater(hi, 30.0)

    def test_missing_temp(self):
        row = {"hum_min": 50.0}
        hi = self.dl._calculate_heat_index(row)
        self.assertTrue(np.isnan(hi))

    def test_missing_humidity_fallback(self):
        """Falls back to hum_mean when hum_min is missing."""
        row = {"temp_max": 30.0, "hum_mean": 60.0}
        hi = self.dl._calculate_heat_index(row)
        self.assertFalse(np.isnan(hi))


# --------------------------------------------------------------------------- #
# SPI tests
# --------------------------------------------------------------------------- #
class TestSPI(unittest.TestCase):

    def setUp(self):
        self.dl = _make_dl()

    def test_spi_output_length(self):
        """SPI series should have same length as input."""
        df = _make_daily_inmet(n_days=365)
        df.rename(columns={"precipitation_sum": "precip_total"}, inplace=True)
        spi = self.dl._calculate_spi(df, "precip_total", period=30)
        self.assertEqual(len(spi), len(df))

    def test_spi_mostly_nan_at_start(self):
        """First 'period' values should be NaN (insufficient rolling window)."""
        df = _make_daily_inmet(n_days=365)
        df.rename(columns={"precipitation_sum": "precip_total"}, inplace=True)
        spi = self.dl._calculate_spi(df, "precip_total", period=30)
        # First 29 should be NaN
        self.assertTrue(spi.iloc[:29].isna().all())

    def test_spi_range(self):
        """Non-NaN SPI values should be finite and within reasonable range."""
        df = _make_daily_inmet(n_days=365)
        df.rename(columns={"precipitation_sum": "precip_total"}, inplace=True)
        spi = self.dl._calculate_spi(df, "precip_total", period=30)
        valid = spi.dropna()
        if len(valid) > 0:
            self.assertTrue((valid > -4).all(), "SPI should be > -4")
            self.assertTrue((valid < 4).all(), "SPI should be < 4")

    def test_spi_missing_column(self):
        """Missing precip column should return all NaN."""
        df = _make_daily_inmet(n_days=90)
        spi = self.dl._calculate_spi(df, "nonexistent_col", period=30)
        self.assertTrue(spi.isna().all())

    def test_spi_short_series(self):
        """Short series (< period) should return all NaN."""
        df = _make_daily_inmet(n_days=10)
        df.rename(columns={"precipitation_sum": "precip_total"}, inplace=True)
        spi = self.dl._calculate_spi(df, "precip_total", period=30)
        self.assertTrue(spi.isna().all())


# --------------------------------------------------------------------------- #
# Heat/Cold Wave tests
# --------------------------------------------------------------------------- #
class TestHeatWaveDetection(unittest.TestCase):
    """Test heat wave detection using Meehl & Tebaldi (2004) criteria."""

    def setUp(self):
        self.dl = _make_dl()

    def test_no_heatwave_normal_temps(self):
        """Normal temps (all similar) should not trigger heat wave."""
        df = _make_daily_inmet(n_days=90)
        df["temperature_max"] = 28.0  # constant — no outlier days
        # Mock El Niño to avoid network call
        with patch.object(self.dl, "_get_elnino_data", return_value=pd.DataFrame()):
            result = self.dl.fetch_data("dummy.shp", "01/01/2020", "31/03/2020",
                                        output_csv="/tmp/test_indice_out.csv",
                                        inmet_df=df)
        self.assertEqual(result["heatwave_has"].sum(), 0)

    def test_extreme_temps_trigger_heatwave(self):
        """Extreme temperature spike should trigger heat wave detection."""
        n = 90
        df = _make_daily_inmet(n_days=n)
        # Constant baseline with extreme spike — ensures 97.5th percentile < spike
        temps = np.full(n, 25.0)
        temps[40:51] = 50.0  # 11 consecutive extreme days (well above any pctile)
        df["temperature_max"] = temps
        with patch.object(self.dl, "_get_elnino_data", return_value=pd.DataFrame()):
            result = self.dl.fetch_data("dummy.shp", "01/01/2020", "31/03/2020",
                                        output_csv="/tmp/test_indice_out2.csv",
                                        inmet_df=df)
        self.assertGreater(result["heatwave_has"].sum(), 0)

    def test_coldwave_extreme_cold(self):
        """Extreme cold spike should trigger cold wave detection."""
        n = 90
        df = _make_daily_inmet(n_days=n)
        # Constant baseline with extreme cold spike
        temps = np.full(n, 18.0)
        temps[40:51] = -5.0  # 11 consecutive extreme cold days
        df["temperature_min"] = temps
        with patch.object(self.dl, "_get_elnino_data", return_value=pd.DataFrame()):
            result = self.dl.fetch_data("dummy.shp", "01/01/2020", "31/03/2020",
                                        output_csv="/tmp/test_indice_out3.csv",
                                        inmet_df=df)
        self.assertGreater(result["coldwave_has"].sum(), 0)

    def test_duration_columns(self):
        """Duration and intensity columns should be present."""
        df = _make_daily_inmet(n_days=90)
        with patch.object(self.dl, "_get_elnino_data", return_value=pd.DataFrame()):
            result = self.dl.fetch_data("dummy.shp", "01/01/2020", "31/03/2020",
                                        output_csv="/tmp/test_indice_out4.csv",
                                        inmet_df=df)
        self.assertIn("heatwaveduration_sum", result.columns)
        self.assertIn("heatwaveintensity_ind", result.columns)
        self.assertIn("coldwaveduration_sum", result.columns)
        self.assertIn("coldwaveintensity", result.columns)


# --------------------------------------------------------------------------- #
# Output Schema tests
# --------------------------------------------------------------------------- #
class TestOutputSchema(unittest.TestCase):

    def setUp(self):
        self.dl = _make_dl()

    def test_all_expected_columns(self):
        """Output should contain all glossary columns."""
        df = _make_daily_inmet(n_days=90)
        with patch.object(self.dl, "_get_elnino_data", return_value=pd.DataFrame()):
            result = self.dl.fetch_data("dummy.shp", "01/01/2020", "31/03/2020",
                                        output_csv="/tmp/test_indice_schema.csv",
                                        inmet_df=df)
        expected_cols = [
            "date",
            "heatindex_max", "heatindex_min", "heatindex_mea",
            "heatwave_has", "heatwaveduration_sum", "heatwaveintensity_ind",
            "coldwave_has", "coldwaveduration_sum", "coldwaveintensity",
            "spi_ind", "elnino_ind",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_row_count(self):
        """Should have one row per day."""
        df = _make_daily_inmet(n_days=90)
        with patch.object(self.dl, "_get_elnino_data", return_value=pd.DataFrame()):
            result = self.dl.fetch_data("dummy.shp", "01/01/2020", "31/03/2020",
                                        output_csv="/tmp/test_indice_rows.csv",
                                        inmet_df=df)
        self.assertEqual(len(result), 90)


# --------------------------------------------------------------------------- #
# El Niño parsing tests
# --------------------------------------------------------------------------- #
class TestElNinoParsing(unittest.TestCase):

    def setUp(self):
        self.dl = _make_dl()

    def test_parse_oni_data(self):
        """Should parse NOAA ONI format correctly."""
        # Mock the requests.get response
        mock_text = (
            " ONI DATA\n"
            " 2020   0.5  0.4  0.3  0.1 -0.1 -0.3 -0.5 -0.6 -0.9 -1.2 -1.3 -1.0\n"
            " 2021  -0.7 -0.5 -0.3  0.0  0.2  0.3  0.4  0.4  0.5  0.5  0.5  0.7\n"
            "  -99.9\n"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_text

        with patch("requests.get", return_value=mock_response):
            result = self.dl._get_elnino_data("01/01/2020", "31/12/2021")

        self.assertFalse(result.empty)
        self.assertIn("elnino_ind", result.columns)
        self.assertIn("date", result.columns)
        # Should have daily entries via ffill
        self.assertGreater(len(result), 24)  # at least more than 24 months

    def test_elnino_failed_request(self):
        """Failed HTTP should return empty DataFrame."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("requests.get", return_value=mock_response):
            result = self.dl._get_elnino_data("01/01/2020", "31/12/2020")

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
