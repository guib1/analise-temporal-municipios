#!/usr/bin/env python3
"""
Unit tests for INMETDownloader.

Tests column standardization, aggregation, heat index calculation,
station selection, and output schema — NO network access required.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baixar_dados.INMET import INMETDownloader


def _make_synthetic_hourly_inmet(n_days: int = 3, start: str = "2020-01-01") -> pd.DataFrame:
    """Create synthetic hourly INMET data matching standardized column names."""
    np.random.seed(42)
    rows = []
    for day_offset in range(n_days):
        dt = pd.Timestamp(start) + pd.Timedelta(days=day_offset)
        for hour in range(24):
            rows.append({
                "DT_MEDICAO": dt,
                "RAD_GLO": np.random.uniform(0, 5000) if 6 <= hour <= 18 else 0.0,  # kJ/m²
                "CHUVA": np.random.uniform(0, 5),  # mm
                "TEM_MAX": np.random.uniform(25, 35),  # °C
                "TEM_MIN": np.random.uniform(15, 22),  # °C
                "TEM_INS": np.random.uniform(18, 30),  # °C
                "UMD_MIN": np.random.uniform(30, 60),  # %
                "UMD_INS": np.random.uniform(50, 90),  # %
                "VEN_VEL": np.random.uniform(0.5, 5.0),  # m/s
            })
    return pd.DataFrame(rows)


class TestProcessData(unittest.TestCase):
    """Test _process_data daily aggregation."""

    def setUp(self):
        self.dl = INMETDownloader.__new__(INMETDownloader)  # skip __init__ (no network)
        self.df = _make_synthetic_hourly_inmet(n_days=3)

    def test_output_columns(self):
        """Output should have all expected columns."""
        result = self.dl._process_data(self.df.copy())
        expected = [
            "date",
            "globalradiation_max", "globalradiation_min", "globalradiation_mea",
            "precipitation_sum",
            "temperature_max", "temperature_min", "temperature_mea",
            "humidity_min", "humidity_mea",
            "wind_mea",
            "heatindex_max", "heatindex_min", "heatindex_mea",
        ]
        self.assertEqual(list(result.columns), expected)

    def test_one_row_per_day(self):
        result = self.dl._process_data(self.df.copy())
        self.assertEqual(len(result), 3)

    def test_radiation_in_mj(self):
        """Radiation should be converted from kJ/m² to MJ/m² (÷1000)."""
        result = self.dl._process_data(self.df.copy())
        # kJ values are 0-5000, so MJ should be 0-5.0
        self.assertTrue((result["globalradiation_max"] <= 6.0).all(),
                        "Radiation max should be ≤ 6 MJ/m²")
        self.assertTrue((result["globalradiation_min"] >= 0.0).all(),
                        "Radiation min should be ≥ 0 MJ/m²")

    def test_precipitation_is_sum(self):
        """Precipitation should be summed across hours."""
        result = self.dl._process_data(self.df.copy())
        # 24 hours * ~2.5mm avg = ~60mm/day
        for _, row in result.iterrows():
            self.assertGreater(row["precipitation_sum"], 0)

    def test_temperature_ranges(self):
        """Temperature max/min/mean should be in reasonable ranges."""
        result = self.dl._process_data(self.df.copy())
        self.assertTrue((result["temperature_max"] >= 25).all())
        self.assertTrue((result["temperature_max"] <= 36).all())
        self.assertTrue((result["temperature_min"] >= 14).all())
        self.assertTrue((result["temperature_min"] <= 23).all())

    def test_humidity_ranges(self):
        """Humidity should be in 0-100% range."""
        result = self.dl._process_data(self.df.copy())
        self.assertTrue((result["humidity_min"] >= 0).all())
        self.assertTrue((result["humidity_mea"] <= 100).all())

    def test_wind_positive(self):
        result = self.dl._process_data(self.df.copy())
        self.assertTrue((result["wind_mea"] > 0).all())


class TestHeatIndex(unittest.TestCase):
    """Test heat index calculation."""

    def setUp(self):
        self.dl = INMETDownloader.__new__(INMETDownloader)

    def test_heat_index_below_80f(self):
        """When T < 80°F (~26.7°C), heat index = temperature."""
        df = pd.DataFrame({
            "TEM_INS": [20.0, 25.0],
            "UMD_INS": [50.0, 50.0],
        })
        result = self.dl._calculate_heat_index(df)
        # Below 80°F, HI_c ≈ T_c
        np.testing.assert_allclose(result["HEAT_INDEX"].values, [20.0, 25.0], atol=0.01)

    def test_heat_index_above_80f(self):
        """When T > 80°F (~26.7°C), Rothfusz regression should be used."""
        df = pd.DataFrame({
            "TEM_INS": [35.0],
            "UMD_INS": [70.0],
        })
        result = self.dl._calculate_heat_index(df)
        hi = result["HEAT_INDEX"].values[0]
        # At 35°C, 70% RH, heat index should be > 35 (feels hotter)
        self.assertGreater(hi, 35.0)
        # Should be reasonable (not above 70°C)
        self.assertLess(hi, 70.0)

    def test_heat_index_missing_columns(self):
        """If columns missing, should return df unchanged."""
        df = pd.DataFrame({"TEM_INS": [30.0]})
        result = self.dl._calculate_heat_index(df)
        self.assertNotIn("HEAT_INDEX", result.columns)


class TestStandardizeColumns(unittest.TestCase):
    """Test column renaming from Portuguese INMET headers."""

    def setUp(self):
        self.dl = INMETDownloader.__new__(INMETDownloader)

    def test_rename_columns(self):
        df = pd.DataFrame({
            "DATA (YYYY-MM-DD)": ["2020-01-01"],
            "RADIACAO GLOBAL (Kj/m²)": [1500.0],
            "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": [2.5],
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": [25.0],
            "UMIDADE RELATIVA DO AR, HORARIA (%)": [65.0],
            "VENTO, VELOCIDADE HORARIA (m/s)": [3.0],
        })
        result = self.dl._standardize_columns(df)
        self.assertIn("DT_MEDICAO", result.columns)
        self.assertIn("RAD_GLO", result.columns)
        self.assertIn("CHUVA", result.columns)
        self.assertIn("TEM_INS", result.columns)
        self.assertIn("UMD_INS", result.columns)
        self.assertIn("VEN_VEL", result.columns)

    def test_date_parsing(self):
        df = pd.DataFrame({"DATA (YYYY-MM-DD)": ["2020/01/15", "2020-06-30"]})
        result = self.dl._standardize_columns(df)
        self.assertEqual(result["DT_MEDICAO"].iloc[0], pd.Timestamp("2020-01-15"))
        self.assertEqual(result["DT_MEDICAO"].iloc[1], pd.Timestamp("2020-06-30"))

    def test_comma_decimal(self):
        """INMET CSV uses comma as decimal separator."""
        df = pd.DataFrame({
            "DT_MEDICAO": ["2020-01-01"],
            "RADIACAO GLOBAL (Kj/m²)": ["1500,5"],
        })
        result = self.dl._standardize_columns(df)
        self.assertAlmostEqual(result["RAD_GLO"].iloc[0], 1500.5)

    def test_unnamed_columns_dropped(self):
        df = pd.DataFrame({
            "DT_MEDICAO": ["2020-01-01"],
            "Unnamed: 10": [None],
            "Unnamed: 11": [None],
        })
        result = self.dl._standardize_columns(df)
        unnamed = [c for c in result.columns if "Unnamed" in str(c)]
        self.assertEqual(len(unnamed), 0)


class TestStationOperatesInRange(unittest.TestCase):

    def test_station_active(self):
        station = {"DT_INICIO_OPERACAO": "2010-01-01", "DT_FIM_OPERACAO": None}
        result = INMETDownloader._station_operates_in_range(station, date(2015, 1, 1), date(2020, 12, 31))
        self.assertTrue(result)

    def test_station_too_new(self):
        station = {"DT_INICIO_OPERACAO": "2020-01-01", "DT_FIM_OPERACAO": None}
        result = INMETDownloader._station_operates_in_range(station, date(2010, 1, 1), date(2015, 12, 31))
        self.assertFalse(result)

    def test_station_closed(self):
        station = {"DT_INICIO_OPERACAO": "2000-01-01", "DT_FIM_OPERACAO": "2005-12-31"}
        result = INMETDownloader._station_operates_in_range(station, date(2010, 1, 1), date(2015, 12, 31))
        self.assertFalse(result)

    def test_station_no_dates(self):
        station = {}
        result = INMETDownloader._station_operates_in_range(station, date(2010, 1, 1), date(2015, 12, 31))
        self.assertTrue(result)


class TestDedupeStations(unittest.TestCase):

    def test_removes_duplicates(self):
        stations = [
            {"CD_STATION": "A771", "name": "first"},
            {"CD_STATION": "A771", "name": "second"},
            {"CD_STATION": "A801", "name": "other"},
        ]
        result = INMETDownloader._dedupe_stations(stations)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "first")

    def test_skips_no_code(self):
        stations = [{"name": "no code"}, {"CD_STATION": "A771", "name": "has code"}]
        result = INMETDownloader._dedupe_stations(stations)
        self.assertEqual(len(result), 1)


class TestReadHistoricalCsv(unittest.TestCase):
    """Test the CSV parser for INMET historical data format."""

    def test_basic_csv(self):
        # 8 header metadata lines + actual CSV data
        lines = [
            "REGIAO: SE",
            "UF: SP",
            "ESTACAO: SAO PAULO - MIRANTE",
            "CODIGO: A771",
            "LATITUDE: -23.49",
            "LONGITUDE: -46.61",
            "ALTITUDE: 786.00",
            "DATA: 01/01/2020 A 31/12/2020",
            "Data;Hora;RADIACAO GLOBAL (Kj/m²);PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
            "2020/01/01;0000;0;0,0",
            "2020/01/01;0100;0;0,2",
        ]
        raw = "\n".join(lines)
        df = INMETDownloader._read_historical_csv_text(raw)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)

    def test_too_few_lines(self):
        raw = "just\nfew\nlines"
        df = INMETDownloader._read_historical_csv_text(raw)
        self.assertTrue(df.empty)


class TestDailyToWeekly(unittest.TestCase):
    """Test weekly aggregation."""

    def setUp(self):
        self.dl = INMETDownloader.__new__(INMETDownloader)

    def test_weekly_aggregation(self):
        """7 daily rows should produce 1 weekly row."""
        dates = pd.date_range("2020-01-06", periods=7, freq="D")  # Mon-Sun
        daily = pd.DataFrame({
            "date": dates,
            "globalradiation_max": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 4.0],
            "globalradiation_min": [0.0, 0.5, 1.0, 0.2, 0.0, 0.3, 0.1],
            "globalradiation_mea": [2.0, 3.0, 4.0, 3.5, 4.5, 5.0, 2.5],
            "precipitation_sum": [0.0, 5.0, 0.0, 10.0, 0.0, 2.0, 0.0],
            "temperature_max": [28, 29, 30, 31, 30, 29, 27],
            "temperature_min": [18, 19, 20, 21, 20, 19, 17],
            "temperature_mea": [23, 24, 25, 26, 25, 24, 22],
            "humidity_min": [40, 35, 30, 38, 42, 45, 50],
            "humidity_mea": [65, 60, 55, 62, 68, 70, 75],
            "wind_mea": [2.0, 2.5, 3.0, 2.8, 2.2, 1.8, 2.0],
            "heatindex_max": [30, 31, 32, 33, 32, 31, 29],
            "heatindex_min": [18, 19, 20, 21, 20, 19, 17],
            "heatindex_mea": [24, 25, 26, 27, 26, 25, 23],
        })
        weekly = INMETDownloader._daily_to_weekly(daily)
        self.assertEqual(len(weekly), 1)

        row = weekly.iloc[0]
        # globalradiation_max should be max of daily maxes
        self.assertAlmostEqual(row["globalradiation_max"], 10.0)
        # globalradiation_min should be min of daily mins
        self.assertAlmostEqual(row["globalradiation_min"], 0.0)
        # precipitation_sum should be sum across the week
        self.assertAlmostEqual(row["precipitation_sum"], 17.0)
        # temperature_max should be max of daily maxes
        self.assertAlmostEqual(row["temperature_max"], 31.0)
        # wind_mea should be mean of daily means
        expected_wind = np.mean([2.0, 2.5, 3.0, 2.8, 2.2, 1.8, 2.0])
        self.assertAlmostEqual(row["wind_mea"], expected_wind, places=4)


if __name__ == "__main__":
    unittest.main()
