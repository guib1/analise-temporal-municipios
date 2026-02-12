from __future__ import annotations

import logging
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.utils.geo import parse_date, centroid_from_shapefile, get_ibge_code

# Ensure local 'merradownload' package can be imported when running as a script
PROJECT_SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_SRC.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

# Load environment variables from `.env` regardless of current working directory
dotenv_loaded = load_dotenv()
if not dotenv_loaded:
    load_dotenv(PROJECT_ROOT / ".env")

from merradownload.merra_scraping import baixar_merra
from merradownload.opendap_download.multi_processing_download import (
    AuthenticationError,
    DownloadError,
)

LOGGER = logging.getLogger(__name__)

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

@dataclass(frozen=True)
class MerraVariable:
    field_id: str
    dataset_name: str
    dataset_id: str
    description: str = ""


class MERRA2Downloader:
    """Wrapper around the merradownload package to fetch pollutant metrics."""

    CACHE_ROOT = PROJECT_ROOT / "data/cache/merra2"
    OUTPUT_ROOT = PROJECT_ROOT / "data/output/merra2"

    VARIABLES: Dict[str, MerraVariable] = {
        "organiccarbonbiofuelemissions": MerraVariable("OCEMBF", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "organiccarbonbiogenicemissions": MerraVariable("OCEMBG", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "organiccarbonanthropogenicemissions": MerraVariable("OCEMAN", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "organiccarbonbiomassburningemissions": MerraVariable("OCEMBB", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "blackcarbonbiomassburning": MerraVariable("BCEMBB", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "blackcarbonbiofuel": MerraVariable("BCEMBF", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "so2antropogenico": MerraVariable("SO2EMAN", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "so4antropogenico": MerraVariable("SO4EMAN", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "so2biomassburning": MerraVariable("SO2EMBB", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "blackcarbonantrogonenico": MerraVariable("BCEMAN", "M2T1NXADG", "tavg1_2d_adg_Nx"),
        "comerra2": MerraVariable("COEM", "M2T1NXCHM", "tavg1_2d_chm_Nx"),
        "o3merra2": MerraVariable("TO3", "M2T1NXSLV", "tavg1_2d_slv_Nx"),
        "o3omi": MerraVariable("TO3", "M2T1NXSLV", "tavg1_2d_slv_Nx"),
        "aodterramodis": MerraVariable("TOTEXTTAU", "M2T1NXAER", "tavg1_2d_aer_Nx"),
    }

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.username = os.getenv("NASA_USER")
        self.password = os.getenv("NASA_PASSWORD")

        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_ROOT
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.username or not self.password:
            netrc_path = Path.home() / ".netrc"
            if not netrc_path.exists():  # pragma: no cover - requires local credentials
                raise ValueError(
                    "Credenciais Earthdata não encontradas. "
                    "Por favor, defina 'NASA_USER' e 'NASA_PASSWORD' no arquivo .env."
                )

    def fetch_daily_data(
        self,
        shapefile_path: str,
        start: str,
        end: str,
        out_csv: Optional[str] = None,
        variables: Optional[Iterable[str]] = None,
        force_download: bool = False,
        cleanup: bool = True,
    ) -> pd.DataFrame:
        if out_csv is None:
            out_csv = self.OUTPUT_ROOT / "merra2_daily.csv"
        
        # Ensure out_csv is absolute or relative to PROJECT_ROOT if it looks like a data path
        out_csv_path = Path(out_csv)
        if not out_csv_path.is_absolute():
            # If it starts with 'data/', assume relative to PROJECT_ROOT
            if str(out_csv_path).startswith("data"):
                out_csv_path = PROJECT_ROOT / out_csv_path
            else:
                out_csv_path = Path.cwd() / out_csv_path

        start_date = parse_date(start)
        end_date = parse_date(end)
        if end_date < start_date:
            raise ValueError("end date must be after start date")

        ibge_code = get_ibge_code(shapefile_path)
        lat, lon = centroid_from_shapefile(shapefile_path)
        loc_label = self._build_location_label(shapefile_path)

        selected = list(variables) if variables else list(self.VARIABLES.keys())
        years = list(range(start_date.year, end_date.year + 1))

        base = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
        frames = [base]

        for var_name in selected:
            config = self.VARIABLES.get(var_name)
            if not config:
                LOGGER.warning("Variable %s not configured, skipping.", var_name)
                continue

            df_var = self._download_variable(
                var_name=var_name,
                config=config,
                loc_label=loc_label,
                lat=lat,
                lon=lon,
                years=years,
                start_date=start_date,
                end_date=end_date,
                force_download=force_download,
            )

            if not df_var.empty:
                frames.append(df_var)
            else:
                LOGGER.warning("No data returned for %s", var_name)

            if cleanup:
                self._cleanup_variable_cache(var_name)

        if len(frames) == 1:
            LOGGER.warning("No MERRA-2 data was gathered for the requested range.")
            if cleanup:
                self._clear_cache_root()
            return pd.DataFrame()

        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.merge(frame, on="date", how="left")

        combined = combined.sort_values("date").reset_index(drop=True)
        combined.insert(0, "shapefile_nome", Path(shapefile_path).stem)
        combined.insert(0, "codigo_ibge", ibge_code if ibge_code is not None else None)

        output_path = out_csv_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        LOGGER.info("CSV final gerado -> %s", output_path)

        # Optional: save a copy to default location if different
        default_final_csv = self.OUTPUT_ROOT / f"{Path(shapefile_path).stem}_final.csv"
        if default_final_csv.resolve() != output_path.resolve():
            default_final_csv.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(default_final_csv, index=False)
            LOGGER.info("Cópia adicional salva em -> %s", default_final_csv)

        if cleanup:
            self._clear_cache_root()

        return combined

    def _download_variable(
        self,
        var_name: str,
        config: MerraVariable,
        loc_label: str,
        lat: float,
        lon: float,
        years: List[int],
        start_date: date,
        end_date: date,
        force_download: bool,
    ) -> pd.DataFrame:
        hourly_csv = self.cache_dir / var_name / f"{loc_label}_hourly.csv"

        if force_download or not hourly_csv.exists():
            try:
                with self._temporary_workdir(self.cache_dir):
                    baixar_merra(
                        username=self.username,
                        password=self.password,
                        years=years,
                        field_id=config.field_id,
                        field_name=var_name,
                        database_name=config.dataset_name,
                        database_id=config.dataset_id,
                        locs=[(loc_label, lat, lon)],
                        conversion_function=self._identity,
                        aggregator="mean",
                        start_date=start_date,
                        end_date=end_date,
                    )
            except AuthenticationError:
                raise
            except Exception as exc:
                LOGGER.error(
                    "Failed to download %s (%s) for %s: %s",
                    var_name,
                    config.dataset_id,
                    loc_label,
                    exc,
                )
                return pd.DataFrame()

        if not hourly_csv.exists():
            LOGGER.warning("Hourly CSV for %s not found at %s", var_name, hourly_csv)
            return pd.DataFrame()

        df_daily = self._process_hourly_csv(hourly_csv, var_name, start_date, end_date)
        return df_daily

    @staticmethod
    def _identity(value):
        return value

    def _process_hourly_csv(
        self,
        csv_path: Path,
        column: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame()

        if "date" not in df.columns or column not in df.columns:
            LOGGER.warning("Unexpected column layout in %s", csv_path)
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
        if df.empty:
            return pd.DataFrame()

        grouped = (
            df.groupby(df["date"].dt.normalize())[column]
            .agg(["max", "min", "mean"])
            .rename(
                columns={
                    "max": f"{column}_max",
                    "min": f"{column}_min",
                    "mean": f"{column}_mea",
                }
            )
            .reset_index()
        )
        grouped.rename(columns={"date": "date"}, inplace=True)
        return grouped

    def _cleanup_variable_cache(self, var_name: str) -> None:
        target = self.cache_dir / var_name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def _clear_cache_root(self) -> None:
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_location_label(self, shapefile_path: str) -> str:
        stem = Path(shapefile_path).stem
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in stem)
        return sanitized or "merra_location"

    @contextmanager
    def _temporary_workdir(self, path: Path):
        previous = Path.cwd()
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)
        try:
            yield
        finally:  # pragma: no cover - simple filesystem op
            os.chdir(previous)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = MERRA2Downloader()
    shapefile = "data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp"
    if os.path.exists(shapefile):
        try:
            df_result = downloader.fetch_daily_data(
                shapefile_path=shapefile,
                start="2020-01-01",
                end="2020-01-31",
                out_csv="data/output/merra2/sp_sao_paulo_merra2.csv",
            )
            print(df_result.head())
        except Exception as exc:  # pragma: no cover - requires credentials/network
            LOGGER.error("An error occurred during the MERRA-2 download: %s", exc)
    else:
        LOGGER.warning("Shapefile not found at %s", shapefile)
