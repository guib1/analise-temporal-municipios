from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _coerce_int(series: pd.Series) -> pd.Series:
    """Best-effort conversion to pandas nullable Int64."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _norm_str(series: pd.Series) -> pd.Series:
    """Normalize to uppercase trimmed strings (NA-safe)."""
    return series.astype("string").str.upper().str.strip()

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

# ═══════════════════════════════════════════════════════════════════════════
# CID-10 codes by disease group
# ═══════════════════════════════════════════════════════════════════════════
#
# Organizados em 3 grandes categorias conforme solicitado:
#   • Respiratórias (J)
#   • Cardiovasculares (I)
#   • Cerebrovasculares (I + G)
#
# O filtro usa str.startswith(), então "J00" já captura J000, J001, etc.
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# RESPIRATÓRIAS
# ---------------------------------------------------------------------------

# IVAS – Infecções Agudas das Vias Aéreas Superiores
# J00 = Nasofaringite aguda (resfriado comum)
# J01 = Sinusite aguda
# J02 = Faringite aguda
# J03 = Amigdalite aguda
# J04 = Laringite e traqueíte agudas
# J05 = Laringite obstrutiva aguda (crupe) e epiglotite
# J06 = Infecções agudas das VAS de localizações múltiplas / não especificadas
DEFAULT_IVAS_CODES = (
    "J00", "J01", "J02", "J03", "J04", "J05", "J06",
)

# Influenza / Gripe
# J09 = Influenza devida a vírus zoonótico ou pandêmico identificado (H1N1, H5N1, etc.)
# J10 = Influenza devida a outro vírus da influenza identificado
# J11 = Influenza devida a vírus não identificado
DEFAULT_INFLUENZA_CODES = (
    "J09", "J10", "J11",
)

# Pneumonia
# J12 = Pneumonia viral não classificada em outra parte
# J13 = Pneumonia devida a Streptococcus pneumoniae
# J14 = Pneumonia devida a Haemophilus influenzae
# J15 = Pneumonia bacteriana não classificada em outra parte
# J16 = Pneumonia devida a outros microrganismos infecciosos
# J17 = Pneumonia em doenças classificadas em outra parte
# J18 = Pneumonia por microrganismo não especificado
DEFAULT_PNEUMONIA_CODES = (
    "J12", "J13", "J14", "J15", "J16", "J17", "J18",
)

# Infecções Agudas das Vias Aéreas Inferiores
# J20 = Bronquite aguda
# J21 = Bronquiolite aguda
# J22 = Infecção aguda não especificada das vias aéreas inferiores
DEFAULT_INFEC_VIAS_AEREAS_INF_CODES = (
    "J20", "J21", "J22",
)

# Rinite Alérgica e Sinusite Crônica
# J30 = Rinite alérgica e vasomotora
# J32 = Sinusite crônica
DEFAULT_RINITE_SINUSITE_CODES = (
    "J30", "J32",
)

# DPOC – Doença Pulmonar Obstrutiva Crônica (inclui Bronquiectasia)
# J40 = Bronquite não especificada como aguda ou crônica
# J44 = Outras doenças pulmonares obstrutivas crônicas
# J47 = Bronquiectasia
DEFAULT_DPOC_CODES = (
    "J40", "J44", "J47",
)

# Asma
# J45 = Asma
# J46 = Estado de mal asmático (status asthmaticus)
DEFAULT_ASMA_CODES = (
    "J45", "J46",
)

# ---------------------------------------------------------------------------
# CARDIOVASCULARES
# ---------------------------------------------------------------------------

# Hipertensão Arterial
# I10 = Hipertensão essencial (primária)
# I11 = Doença cardíaca hipertensiva
# I15 = Hipertensão secundária
DEFAULT_HIPERTENSAO_CODES = (
    "I10", "I11", "I15",
)

# Doença Isquêmica do Coração
# I20 = Angina pectoris
# I21 = Infarto agudo do miocárdio
# I22 = Infarto do miocárdio recorrente
# I24 = Outras doenças isquêmicas agudas do coração
# I25 = Doença isquêmica crônica do coração
DEFAULT_DOENCA_ISQUEMICA_CODES = (
    "I20", "I21", "I22", "I24", "I25",
)

# Embolia Pulmonar
# I26 = Embolia pulmonar
DEFAULT_EMBOLIA_PULMONAR_CODES = (
    "I26",
)

# Arritmias Cardíacas
# I45 = Outros transtornos de condução
# I47 = Taquicardia paroxística
# I48 = Flutter e fibrilação atrial
# I49 = Outras arritmias cardíacas
DEFAULT_ARRITMIAS_CODES = (
    "I45", "I47", "I48", "I49",
)

# Insuficiência Cardíaca
# I50 = Insuficiência cardíaca
DEFAULT_INSUFICIENCIA_CARDIACA_CODES = (
    "I50",
)

# ---------------------------------------------------------------------------
# CEREBROVASCULARES
# ---------------------------------------------------------------------------

# AVC – Acidente Vascular Cerebral
# I60 = Hemorragia subaracnóidea
# I61 = Hemorragia intracerebral
# I62 = Outras hemorragias intracranianas não-traumáticas
# I63 = Infarto cerebral
# I64 = AVC não especificado como hemorrágico ou isquêmico
# G45 = Acidentes isquêmicos cerebrais transitórios (AIT)
DEFAULT_AVC_CODES = (
    "I60", "I61", "I62", "I63", "I64", "G45",
)

# ---------------------------------------------------------------------------
# Mapeamento doença → códigos CID-10
# ---------------------------------------------------------------------------
DISEASE_CID10_MAP: dict[str, tuple[str, ...]] = {
    # Respiratórias
    "ivas": DEFAULT_IVAS_CODES,
    "influenza": DEFAULT_INFLUENZA_CODES,
    "pneumonia": DEFAULT_PNEUMONIA_CODES,
    "infec_vias_aereas_inf": DEFAULT_INFEC_VIAS_AEREAS_INF_CODES,
    "rinite_sinusite": DEFAULT_RINITE_SINUSITE_CODES,
    "dpoc": DEFAULT_DPOC_CODES,
    "asma": DEFAULT_ASMA_CODES,
    # Cardiovasculares
    "hipertensao": DEFAULT_HIPERTENSAO_CODES,
    "doenca_isquemica": DEFAULT_DOENCA_ISQUEMICA_CODES,
    "embolia_pulmonar": DEFAULT_EMBOLIA_PULMONAR_CODES,
    "arritmias": DEFAULT_ARRITMIAS_CODES,
    "insuficiencia_cardiaca": DEFAULT_INSUFICIENCIA_CARDIACA_CODES,
    # Cerebrovasculares
    "avc": DEFAULT_AVC_CODES,
}


def get_cid10_codes(disease: str) -> tuple[str, ...]:
    """Return CID-10 codes for a given disease key (case-insensitive)."""
    key = disease.strip().lower().replace(" ", "_")
    if key not in DISEASE_CID10_MAP:
        raise ValueError(
            f"Doença '{disease}' não suportada. "
            f"Opções: {', '.join(sorted(DISEASE_CID10_MAP))}"
        )
    return DISEASE_CID10_MAP[key]

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


def _decode_age_fields_to_years(
    idade: Optional[str | int],
    cod_idade: Optional[str | int],
) -> Optional[float]:
    """Decode SIH age using IDADE (magnitude) + COD_IDADE (unit).

    COD_IDADE conventions (commonly observed in SIH extracts):
      1 = hours, 2 = days, 3 = months, 4 = years

    If COD_IDADE is missing/invalid, falls back to `_decode_age_to_years(idade)`.
    """
    if cod_idade in (None, "", "   "):
        return _decode_age_to_years(idade)
    try:
        unit = int(cod_idade)
    except (TypeError, ValueError):
        return _decode_age_to_years(idade)

    if unit <= 0:
        return _decode_age_to_years(idade)
    if idade in (None, "", "   "):
        return None
    try:
        magnitude = int(idade)
    except (TypeError, ValueError):
        return None
    if magnitude < 0:
        return None

    if unit == 1:  # hours
        return magnitude / (24 * 365.25)
    if unit == 2:  # days
        return magnitude / 365.25
    if unit == 3:  # months
        return magnitude / 12
    if unit == 4:  # years
        return float(magnitude)
    return _decode_age_to_years(idade)

def _categorise_age(df: pd.DataFrame) -> pd.DataFrame:
    age_years = df["age_years"]
    # SIH may encode newborn ages in hours/days/months. Those become fractions (<1 year).
    df["age_0_sum"] = ((age_years.notna()) & (age_years >= 0) & (age_years < 1)).astype("Int64")
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

    cache_dir: Path = Path("data/cache_data")
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
    # Public API
    # --------------------------------------------------------------------- #
    def fetch_sih_daily(
        self,
        cod_ibge: str | int,
        start: str | date | datetime,
        end: str | date | datetime,
        disease: str = "asma",
        cid10_prefixes: Sequence[str] | None = None,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Download SIH/SUS records for *any* supported disease and aggregate daily.

        Parameters
        ----------
        disease : str
            Key from ``DISEASE_CID10_MAP`` (e.g. "dengue", "malaria").
            Ignored when *cid10_prefixes* is provided explicitly.
        cid10_prefixes : sequence of str, optional
            Override the CID-10 filter.  When ``None`` the codes are looked up
            from ``DISEASE_CID10_MAP[disease]``.
        """
        if cid10_prefixes is None:
            cid10_prefixes = get_cid10_codes(disease)

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
            empty = _empty_daily_dataframe(start_date, end_date)
            empty["codigo_ibge"] = code7
            return empty

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
            empty = _empty_daily_dataframe(start_date, end_date)
            empty["codigo_ibge"] = code7
            return empty

        aggregated = self._aggregate_daily(filtered)
        aggregated["codigo_ibge"] = code7

        # Reindex to the full date range so days without hospitalizations
        # get 0 (zero cases) instead of NaN (unknown).
        full_dates = pd.date_range(start_date, end_date, freq="D").normalize()
        aggregated = aggregated.set_index("date").reindex(full_dates).rename_axis("date").reset_index()
        # Fill numeric columns with 0 (no cases), keep codigo_ibge
        numeric_cols = [
            c for c in aggregated.columns
            if c not in {"date", "codigo_ibge"}
        ]
        aggregated[numeric_cols] = aggregated[numeric_cols].fillna(0).astype(int)
        aggregated["codigo_ibge"] = code7
        aggregated = aggregated.sort_values("date").reset_index(drop=True)

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Saving daily aggregation to CSV %s", output_path)
            aggregated.to_csv(output_path, index=False)

        return aggregated

    # Backward-compatible alias
    def fetch_sih_asthma_daily(
        self,
        cod_ibge: str | int,
        start: str | date | datetime,
        end: str | date | datetime,
        cid10_prefixes: Sequence[str] = DEFAULT_ASMA_CODES,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """Convenience wrapper – delegates to ``fetch_sih_daily``."""
        return self.fetch_sih_daily(
            cod_ibge=cod_ibge,
            start=start,
            end=end,
            disease="asma",
            cid10_prefixes=cid10_prefixes,
            output_csv=output_csv,
        )

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
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(1, unit="ns")
        dt_inter = pd.to_datetime(df["DT_INTER"], errors="coerce")
        df = df[dt_inter.between(start_ts, end_ts)]

        # Sex classification.
        if "SEXO" in df.columns:
            sexo_raw = df["SEXO"]
            sexo_num = _coerce_int(sexo_raw)
            sexo_str = _norm_str(sexo_raw)
            man = (sexo_num == 1) | sexo_str.isin({"M", "MAS", "MASC", "MASCULINO"})
            woman = (sexo_num == 3) | sexo_str.isin({"F", "FEM", "FEMININO"})
            df["man_sum"] = man.astype("Int64")
            df["woman_sum"] = woman.astype("Int64")
            unknown = (1 - df["man_sum"] - df["woman_sum"]).clip(lower=0)
            df["unknownsex_sum"] = unknown.astype("Int64")
        else:
            df["man_sum"] = 0
            df["woman_sum"] = 0
            df["unknownsex_sum"] = 0

        # Death indicator.
        if "MORTE" in df.columns:
            morte_raw = df["MORTE"]
            morte_num = _coerce_int(morte_raw)
            morte_str = _norm_str(morte_raw)
            deaths = (morte_num == 1) | morte_str.isin({"S", "SIM", "Y", "YES", "TRUE"})
            df["deaths_sum"] = deaths.astype("Int64")
        else:
            df["deaths_sum"] = 0

        df["cases_sum"] = 1

        # Age decoding.
        if "IDADE" in df.columns:
            if "COD_IDADE" in df.columns:
                df["age_years"] = [
                    _decode_age_fields_to_years(idade, cod)
                    for idade, cod in zip(df["IDADE"].tolist(), df["COD_IDADE"].tolist(), strict=False)
                ]
            else:
                df["age_years"] = df["IDADE"].apply(_decode_age_to_years)
        else:
            df["age_years"] = None
        df = _categorise_age(df)
        return df

    def _aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dt_inter = pd.to_datetime(df["DT_INTER"], errors="coerce")
        df["date"] = pd.Series(pd.DatetimeIndex(dt_inter).normalize(), index=df.index)

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
        daily = df.groupby("date", as_index=False).agg(agg_dict)
        return daily[
            [
                "date",
                "cases_sum",
                "deaths_sum",
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
            ]
        ]

def _empty_daily_dataframe(
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Return a dataframe with the expected columns filled with 0 for the full date range.

    If *start* and *end* are provided, the frame covers every day in the
    range (0 cases per day).  Otherwise an empty (no-row) frame is returned
    for backward-compatibility.
    """
    numeric_cols = [
        "cases_sum",
        "deaths_sum",
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
    ]
    all_cols = ["date"] + numeric_cols + ["codigo_ibge"]

    if start is not None and end is not None:
        dates = pd.date_range(start, end, freq="D").normalize()
        df = pd.DataFrame({"date": dates})
        for col in numeric_cols:
            df[col] = 0
        df["codigo_ibge"] = pd.NA
        return df

    return pd.DataFrame(columns=all_cols)

if __name__ == "__main__":
    # Basic manual test (requires network access and supporting libraries).
    logging.basicConfig(level=logging.INFO)
    downloader = DataSUSDownloader(storage_format="csv")
    df_result = downloader.fetch_sih_asthma_daily(
        cod_ibge="3550308",
        start="2020-01-01",
        end="2020-01-31",
        output_csv="data/output/datasus/daily_3550308_asma.csv",
    )
    print(df_result.head())
