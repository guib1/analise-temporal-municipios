from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Mapping from the first two IBGE digits to UF acronym used by DataSUS file names.
UF_BY_CODE = {
    "11": "RO",
    "12": "AC",
    "13": "AM",
    "14": "RR",
    "15": "PA",
    "16": "AP",
    "17": "TO",
    "21": "MA",
    "22": "PI",
    "23": "CE",
    "24": "RN",
    "25": "PB",
    "26": "PE",
    "27": "AL",
    "28": "SE",
    "29": "BA",
    "31": "MG",
    "32": "ES",
    "33": "RJ",
    "35": "SP",
    "41": "PR",
    "42": "SC",
    "43": "RS",
    "50": "MS",
    "51": "MT",
    "52": "GO",
    "53": "DF",
}

# CID-10 codes that cover asthma diagnoses.
DEFAULT_ASMA_CODES = ("J45", "J450", "J451", "J452", "J453", "J454", "J455", "J456", "J459")

def _read_parquet_directory(path: Path) -> Optional[List[pd.DataFrame]]:
    """Load all parquet files inside a directory as dataframes."""
    if not path.exists() or not path.is_dir():
        return None
    parquet_files = sorted(path.glob("*.parquet"))
    if not parquet_files:
        return None
    frames: List[pd.DataFrame] = []
    for file in parquet_files:
        try:
            frames.append(pd.read_parquet(file))
        except Exception as exc:
            LOGGER.warning("Failed to read parquet %s: %s", file, exc)
    return frames or None


def _normalize_ibge_codes(cod_ibge: str | int) -> Tuple[str, str]:
    code = str(cod_ibge).strip()
    if not code.isdigit():
        raise ValueError(f"Invalid IBGE code {cod_ibge!r}")
    code7 = code.zfill(7)
    code6 = code7[:6]
    return code6, code7

def _iter_year_month(start: date, end: date) -> Iterable[Tuple[int, int]]:
    if start > end:
        raise ValueError("start date must be <= end date")
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1

def _decode_age_to_years(raw_age: Optional[str | int]) -> Optional[float]:
    """
    Convert the SIH/SUS encoded age field to age in years.

    The age is stored in four digits where the first digit represents the unit:
        0 = ignored, 1 = hours, 2 = days, 3 = months, 4 = years.
    The remaining three digits contain the magnitude.
    """
    if raw_age in (None, "", "   "):
        return None
    try:
        value = int(raw_age)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    # Some PySUS versions already return age directly in years (0-999).
    if value < 1000:
        return float(value)
    unit = value // 1000
    magnitude = value % 1000
    if unit == 0:
        return None
    if unit == 1:  # hours
        return magnitude / (24 * 365.25)
    if unit == 2:  # days
        return magnitude / 365.25
    if unit == 3:  # months
        return magnitude / 12
    if unit == 4:  # years
        return float(magnitude)
    return None

def _categorise_age(df: pd.DataFrame) -> pd.DataFrame:
    age_years = df["age_years"]
    df["age_0_sum"] = ((age_years.notna()) & (age_years <= 0)).astype("Int64")
    df["age_1_10_sum"] = ((age_years >= 1) & (age_years <= 10)).astype("Int64")
    df["age_11_20_sum"] = ((age_years >= 11) & (age_years <= 20)).astype("Int64")
    df["age_21_30_sum"] = ((age_years >= 21) & (age_years <= 30)).astype("Int64")
    df["age_31_40_sum"] = ((age_years >= 31) & (age_years <= 40)).astype("Int64")
    df["age_41_50_sum"] = ((age_years >= 41) & (age_years <= 50)).astype("Int64")
    df["age_51_60_sum"] = ((age_years >= 51) & (age_years <= 60)).astype("Int64")
    df["age_61_70_sum"] = ((age_years >= 61) & (age_years <= 70)).astype("Int64")
    df["age_m70_sum"] = ((age_years >= 71)).astype("Int64")
    return df


def _parse_sih_dates(series: pd.Series) -> pd.Series:
    # The field may already be datetime, integer yyyymmdd, or string.
    return pd.to_datetime(series, errors="coerce", format="%Y%m%d")


@dataclass
class DataSUSDownloader:

    cache_dir: Path = Path("data/datasus/raw")
    storage_format: str = "parquet"  # "parquet" (default from PySUS) or "csv"
    cleanup_cache: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage_format = self.storage_format.lower()
        if self.storage_format not in {"parquet", "csv"}:
            raise ValueError("storage_format must be 'parquet' or 'csv'")

    def _monthly_base_name(self, uf_sigla: str, year: int, month: int) -> str:
        return f"RD{uf_sigla}{str(year)[-2:]}{month:02d}"

    def _cleanup_month_files(self, base_name: str) -> None:
        targets = [
            self.cache_dir / f"{base_name}.csv",
            self.cache_dir / f"{base_name}.parquet",
        ]
        for path in targets:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:
                LOGGER.warning("Failed to remove cached file %s: %s", path, exc)

    def _download_month(self, uf_sigla: str, year: int, month: int) -> pd.DataFrame:
        base_name = self._monthly_base_name(uf_sigla, year, month)
        if self.storage_format == "csv":
            csv_path = self.cache_dir / f"{base_name}.csv"
            if csv_path.exists():
                LOGGER.info("Loading cached CSV %s", csv_path)
                return pd.read_csv(csv_path)

        parquet_dir = self.cache_dir / f"{base_name}.parquet"
        if parquet_dir.exists():
            frames_opt = _read_parquet_directory(parquet_dir)
            if frames_opt is not None:
                df_cached = pd.concat(frames_opt, ignore_index=True)
                if self.storage_format == "csv":
                    csv_path = self.cache_dir / f"{base_name}.csv"
                    df_cached.to_csv(csv_path, index=False)
                return df_cached

        try:
            from pysus.online_data import SIH as pysus_sih
        except Exception as exc:  # pragma: no cover - requires network
            raise RuntimeError(
                "PySUS SIH module could not be initialised. "
                "Check your network access to ftp.datasus.gov.br."
            ) from exc

        LOGGER.info("Requesting SIH data for %s %04d-%02d via PySUS", uf_sigla, year, month)
        try:
            parquet_sets = pysus_sih.download(
                states=[uf_sigla],
                years=[year],
                months=[month],
                groups=["RD"],
                data_dir=str(self.cache_dir),
            )
        except Exception as exc:  # pragma: no cover - requires network
            raise RuntimeError(
                f"PySUS download failed for {uf_sigla} {year:04d}-{month:02d}."
            ) from exc

        # pysus returns a ParquetSet or list of ParquetSets.
        if not isinstance(parquet_sets, list):
            parquet_sets = [parquet_sets]

        frames: List[pd.DataFrame] = []
        for item in parquet_sets:
            if hasattr(item, "to_dataframe"):
                frames.append(item.to_dataframe())
            else:
                raise RuntimeError(
                    "Unexpected object returned by PySUS download: "
                    f"{type(item)!r}"
                )

        if not frames:
            return pd.DataFrame()

        df_month = pd.concat(frames, ignore_index=True)
        if self.storage_format == "csv":
            csv_path = self.cache_dir / f"{base_name}.csv"
            LOGGER.info("Saving monthly CSV %s", csv_path)
            df_month.to_csv(csv_path, index=False)
        return df_month

    # --------------------------------------------------------------------- #
    # Public API>
    # --------------------------------------------------------------------- #
    def fetch_sih_asthma_weekly(
        self,
        cod_ibge: str | int,
        start: str | date | datetime,
        end: str | date | datetime,
        cid10_prefixes: Sequence[str] = DEFAULT_ASMA_CODES,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Download SIH/SUS records for the given municipality and aggregate weekly metrics.

        Parameters
        ----------
        cod_ibge:
            Municipality IBGE code (6 or 7 digits).
        start, end:
            Date boundaries (inclusive). Accepts `datetime.date`, `datetime.datetime` or ISO strings.
        cid10_prefixes:
            Iterable of CID-10 codes or prefixes that identify the diagnoses of interest.

        Returns
        -------
        pandas.DataFrame
            Data frame with the weekly aggregations aligned with the glossary columns.

        If ``output_csv`` is provided, the aggregated dataframe is also written to disk.
        """
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()
        if start_date > end_date:
            raise ValueError("start must be <= end")

        code6, code7 = _normalize_ibge_codes(cod_ibge)
        uf_digits = code7[:2]
        try:
            uf_sigla = UF_BY_CODE[uf_digits]
        except KeyError as exc:
            raise KeyError(f"Unknown UF for IBGE code {code7}") from exc

        months = list(_iter_year_month(start_date.replace(day=1), end_date.replace(day=1)))
        frames: List[pd.DataFrame] = []
        try:
            for year, month in months:
                df_month = self._download_month(uf_sigla, year, month)
                if not df_month.empty:
                    frames.append(df_month)
        finally:
            if self.cleanup_cache:
                for year, month in months:
                    base_name = self._monthly_base_name(uf_sigla, year, month)
                    self._cleanup_month_files(base_name)

        if not frames:
            LOGGER.warning("No data downloaded for %s between %s and %s", code7, start, end)
            return _empty_weekly_dataframe()

        raw = pd.concat(frames, ignore_index=True)
        filtered = self._filter_records(
            raw,
            municipality_codes=(code6, code7),
            cid10_prefixes=cid10_prefixes,
            start=start_date,
            end=end_date,
        )
        LOGGER.info(
            "Found %d SIH records for %s between %s and %s after filtering",
            len(filtered),
            code7,
            start_date,
            end_date,
        )
        if filtered.empty:
            LOGGER.warning("No SIH records found for %s in the selected period", code7)
            return _empty_weekly_dataframe()
        aggregated = self._aggregate_weekly(filtered)
        aggregated["codibge"] = code7
        aggregated = aggregated.sort_values("date").reset_index(drop=True)

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Saving weekly aggregation to CSV %s", output_path)
            aggregated.to_csv(output_path, index=False)

        return aggregated

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _filter_records(
        self,
        df: pd.DataFrame,
        municipality_codes: Tuple[str, str],
        cid10_prefixes: Sequence[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        df = df.copy()

        # Municipality filter (DataSUS stores MUNIC_RES as numeric without the check digit).
        muni_str = df.get("MUNIC_RES")
        if muni_str is None:
            raise KeyError("Column 'MUNIC_RES' not present in SIH dataset.")
        df["MUNIC_RES_STR"] = muni_str.astype(str).str.zfill(6)
        df = df[df["MUNIC_RES_STR"].isin(municipality_codes)]

        if df.empty:
            return df

        # Diagnosis filter.
        diag_col = None
        for candidate in ("DIAG_PRINC", "CIDPRI", "DIAG_PRINCIPAL"):
            if candidate in df.columns:
                diag_col = candidate
                break
        if diag_col is None:
            raise KeyError("Could not find the principal diagnosis column in SIH dataset.")
        prefixes = tuple(code.upper() for code in cid10_prefixes)
        df["CID_PRINCIPAL"] = df[diag_col].astype(str).str.upper().str.strip()
        df = df[df["CID_PRINCIPAL"].str.startswith(prefixes)]
        if df.empty:
            return df

        # Date filtering.
        if "DT_INTER" not in df.columns:
            raise KeyError("Column 'DT_INTER' (admission date) not found in SIH dataset.")
        df["DT_INTER"] = _parse_sih_dates(df["DT_INTER"])
        df = df.dropna(subset=["DT_INTER"])
        df = df[(df["DT_INTER"].dt.date >= start) & (df["DT_INTER"].dt.date <= end)]

        # Sex classification.
        df["man_sum"] = (df["SEXO"] == 1).astype("Int64") if "SEXO" in df.columns else 0
        df["woman_sum"] = (df["SEXO"] == 3).astype("Int64") if "SEXO" in df.columns else 0
        df["unknownsex_sum"] = 1 - df["man_sum"] - df["woman_sum"]

        # Death indicator.
        if "MORTE" in df.columns:
            df["deaths_sum"] = (df["MORTE"] == 1).astype("Int64")
        else:
            df["deaths_sum"] = 0

        df["cases_sum"] = 1

        # Age decoding.
        if "IDADE" in df.columns:
            df["age_years"] = df["IDADE"].apply(_decode_age_to_years)
        else:
            df["age_years"] = None
        df = _categorise_age(df)
        return df

    def _aggregate_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        week_end = (
            df["DT_INTER"]
            .dt.to_period("W-SUN")
            .dt.to_timestamp(how="end")
        )
        normalized = pd.DatetimeIndex(week_end).normalize()
        df["date"] = pd.Series(normalized, index=df.index)

        agg_dict = {
            "cases_sum": "sum",
            "deaths_sum": "sum",
            "man_sum": "sum",
            "woman_sum": "sum",
            "unknownsex_sum": "sum",
            "age_0_sum": "sum",
            "age_1_10_sum": "sum",
            "age_11_20_sum": "sum",
            "age_21_30_sum": "sum",
            "age_31_40_sum": "sum",
            "age_41_50_sum": "sum",
            "age_51_60_sum": "sum",
            "age_61_70_sum": "sum",
            "age_m70_sum": "sum",
        }
        weekly = df.groupby("date", as_index=False).agg(agg_dict)
        iso = weekly["date"].dt.isocalendar()
        weekly["week_number_ind"] = iso.week.astype(int)
        weekly["Data"] = weekly["date"]  # duplicate column kept for compatibility
        return weekly[
            [
                "date",
                "cases_sum",
                "deaths_sum",
                "week_number_ind",
                "age_0_sum",
                "age_1_10_sum",
                "age_11_20_sum",
                "age_21_30_sum",
                "age_31_40_sum",
                "age_41_50_sum",
                "age_51_60_sum",
                "age_61_70_sum",
                "age_m70_sum",
                "man_sum",
                "woman_sum",
                "unknownsex_sum",
                "Data",
            ]
        ]


def _empty_weekly_dataframe() -> pd.DataFrame:
    """Return an empty dataframe with the expected columns."""
    columns = [
        "date",
        "cases_sum",
        "deaths_sum",
        "week_number_ind",
        "age_0_sum",
        "age_1_10_sum",
        "age_11_20_sum",
        "age_21_30_sum",
        "age_31_40_sum",
        "age_41_50_sum",
        "age_51_60_sum",
        "age_61_70_sum",
        "age_m70_sum",
        "man_sum",
        "woman_sum",
        "unknownsex_sum",
        "Data",
        "codibge",
    ]
    return pd.DataFrame(columns=columns)

if __name__ == "__main__":
    # Basic manual test (requires network access and supporting libraries).
    logging.basicConfig(level=logging.INFO)
    downloader = DataSUSDownloader(storage_format="csv")
    df_result = downloader.fetch_sih_asthma_weekly(
        cod_ibge="3550308",
        start="2025-01-01",
        end="2025-12-31",
        output_csv="data/output/datasus/weekly_3550308_asma.csv",
    )
    print(df_result.head())
