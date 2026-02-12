from __future__ import annotations
import os
import logging
import zipfile
import requests
import io
import re
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.utils.geo import parse_date, centroid_from_shapefile, get_ibge_code

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None

LOGGER = logging.getLogger(__name__)

class INMETDownloader:
    """
    Downloads INMET data for a given shapefile/municipality by finding the nearest station.
    Uses historical CSV data from portal.inmet.gov.br as the API is deprecated/restricted.
    """

    CACHE_DIR = "data/cache/inmet_zips"

    def __init__(self):
        # self.inmet = InmetStation()
        self._stations_cache = None
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _get_stations(self) -> pd.DataFrame:
        if self._stations_cache is None:
            # Fetch stations directly from API (metadata endpoint seems to still work)
            try:
                # Add timeout to prevent hanging
                response = requests.get("https://apitempo.inmet.gov.br/estacoes/T", timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    self._stations_cache = pd.DataFrame(data)
                    
                    # Rename columns to match previous inmetpy structure if needed
                    # inmetpy used: CD_STATION, STATION_NAME, TP_STATION, LATITUDE, LONGITUDE
                    # API returns: CD_ESTACAO, DC_NOME, TP_ESTACAO, VL_LATITUDE, VL_LONGITUDE
                    rename_map = {
                        'CD_ESTACAO': 'CD_STATION',
                        'DC_NOME': 'STATION_NAME',
                        'TP_ESTACAO': 'TP_STATION',
                        'VL_LATITUDE': 'LATITUDE',
                        'VL_LONGITUDE': 'LONGITUDE'
                    }
                    self._stations_cache.rename(columns=rename_map, inplace=True)
                    
                    # Map TP_STATION values: 'Automatica' -> 'Automatic'
                    self._stations_cache['TP_STATION'] = self._stations_cache['TP_STATION'].replace({
                        'Automatica': 'Automatic',
                        'Convencional': 'Conventional'
                    })
                    
                else:
                    LOGGER.error(f"Failed to fetch stations. Status: {response.status_code}")
                    return pd.DataFrame()
            except Exception as e:
                LOGGER.error(f"Error fetching stations: {e}")
                return pd.DataFrame()

            # Extract IBGE code from CD_WSI
            # Format: 0-76-0-{IBGE_CODE}00000...
            def extract_ibge(wsi):
                if pd.isna(wsi):
                    return None
                parts = str(wsi).split('-')
                if len(parts) >= 4:
                    code_part = parts[3]
                    if len(code_part) >= 7:
                        return code_part[:7]
                return None
            
            if 'CD_WSI' in self._stations_cache.columns:
                self._stations_cache['IBGE_CODE'] = self._stations_cache['CD_WSI'].apply(extract_ibge)
            else:
                self._stations_cache['IBGE_CODE'] = None
            
        return self._stations_cache

    def fetch_daily_data(
        self,
        shapefile_path: str,
        start: str,
        end: str,
        out_csv: str = 'data/output/inmet/inmet_daily.csv',
        ibge_code: Optional[int] = None,
        frequency: str = 'daily',
    ) -> pd.DataFrame:
        """
        Downloads INMET data for the nearest station to the shapefile centroid.
        """
        start_date = parse_date(start)
        end_date = parse_date(end)
        
        # Try to get IBGE code if not provided
        if not ibge_code:
            ibge_code = get_ibge_code(shapefile_path)

        stations = []
        
        # 1. Try to find station by IBGE code
        if ibge_code:
            LOGGER.info(f"Searching for station with IBGE code: {ibge_code}")
            stations_by_ibge = self._find_stations_by_ibge(ibge_code)
            for st in stations_by_ibge:
                LOGGER.info(f"Found station by IBGE code: {st['STATION_NAME']} ({st['CD_STATION']})")
            stations.extend(stations_by_ibge)
        
        # 2. If no station found by IBGE, fallback to nearest stations
        if not stations:
            LOGGER.info("No station found by IBGE code. Falling back to nearest stations.")
            # Get centroid
            lat, lon = centroid_from_shapefile(shapefile_path)
            stations = self._find_nearest_stations(lat, lon, n=5)

        # 3. If we found stations by IBGE, still add nearest as fallback candidates
        if stations:
            try:
                lat, lon = centroid_from_shapefile(shapefile_path)
                stations.extend(self._find_nearest_stations(lat, lon, n=10))
            except Exception:
                pass

        stations = self._dedupe_stations(stations)

        # Filter stations by operational period when available
        stations = [
            st for st in stations
            if self._station_operates_in_range(st, start_date, end_date)
        ]

        if not stations:
            LOGGER.warning(
                "No candidate INMET stations appear to operate in the requested period. "
                "For older years, INMET automatic station coverage may start later (e.g., many SP stations start in 2006+)."
            )
            return pd.DataFrame()
        
        if not stations:
            LOGGER.warning("No INMET stations found.")
            return pd.DataFrame()
            
        df = pd.DataFrame()
        for station in stations:
            station_id = station['CD_STATION']
            station_name = station['STATION_NAME']
            LOGGER.info(f"Trying station: {station_name} ({station_id})")

            try:
                # Iterate over years
                years = range(start_date.year, end_date.year + 1)
                df_station = pd.DataFrame()
                
                for year in years:
                    LOGGER.info(f"Processing year {year} for station {station_id}...")
                    df_year = self._get_data_from_zip(year, station_id)
                    if not df_year.empty:
                        df_station = pd.concat([df_station, df_year])
                
                if not df_station.empty:
                    # Standardize columns first to ensure DT_MEDICAO exists
                    df_station = self._standardize_columns(df_station)
                    
                    # Filter by date range
                    # df_station['DT_MEDICAO'] is already datetime from _standardize_columns
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    mask = df_station['DT_MEDICAO'].between(start_ts, end_ts)
                    df = df_station.loc[mask].copy()
                    
                    if not df.empty:
                        LOGGER.info(f"Successfully retrieved data from {station_name} ({station_id})")
                        break
                    else:
                        min_dt = df_station['DT_MEDICAO'].min()
                        max_dt = df_station['DT_MEDICAO'].max()
                        LOGGER.warning(
                            f"Data found for {station_id} but outside requested range. "
                            f"Available range: {min_dt.date() if pd.notna(min_dt) else 'n/a'} to {max_dt.date() if pd.notna(max_dt) else 'n/a'}."
                        )
                else:
                    LOGGER.warning(f"No data found in ZIPs for station {station_id}.")

            except Exception as e:
                LOGGER.error(f"Error processing data for {station_id}: {e}")
                continue

        if df.empty:
            LOGGER.warning(f"No data returned for any of the stations.")
            return pd.DataFrame()

        # Process and aggregate
        daily_df = self._process_data(df)

        if frequency not in {'daily', 'weekly'}:
            raise ValueError("frequency must be 'daily' or 'weekly'")
        if frequency == 'weekly':
            daily_df = self._daily_to_weekly(daily_df)
        
        if ibge_code:
            daily_df.insert(0, 'codigo_ibge', ibge_code)
        else:
            inferred_ibge = get_ibge_code(shapefile_path)
            if inferred_ibge:
                daily_df.insert(0, 'codigo_ibge', inferred_ibge)
            else:
                lat, lon = centroid_from_shapefile(shapefile_path)
                daily_df['lat'] = lat
                daily_df['lon'] = lon

        # Save
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        daily_df.to_csv(out_csv, index=False)
        LOGGER.info(f"CSV generated -> {out_csv}")
        
        return daily_df

    def _get_data_from_zip(self, year: int, station_code: str) -> pd.DataFrame:
        zip_filename = f"{year}.zip"
        zip_path = os.path.join(self.CACHE_DIR, zip_filename)
        url = f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"

        # Download if not exists or if size is extremely small (corrupt)
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100:
            LOGGER.info(f"Downloading historical data for {year}...")
            try:
                response = requests.get(url, stream=True, timeout=60) # Added timeout
                if response.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Verify validity immediately
                    if not zipfile.is_zipfile(zip_path):
                        LOGGER.error(f"Downloaded file {zip_path} is not a valid ZIP. Removing.")
                        os.remove(zip_path)
                        return pd.DataFrame()
                        
                else:
                    LOGGER.error(f"Failed to download {url}. Status: {response.status_code}")
                    return pd.DataFrame()
            except Exception as e:
                LOGGER.error(f"Download error: {e}")
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                    except:
                        pass
                return pd.DataFrame()

        # Extract specific station file
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find file matching station code (case-insensitive; supports subfolders like 2000/)
                # Typical pattern: INMET_SE_SP_A771_... .CSV
                station_code_up = str(station_code).upper()
                target_file = None
                for filename in z.namelist():
                    base = os.path.basename(filename).upper()
                    if f"_{station_code_up}_" in base:
                        target_file = filename
                        break
                
                if target_file:
                    LOGGER.info(f"Found file {target_file} in {zip_filename}")
                    with z.open(target_file) as f:
                        raw = f.read().decode('latin1', errors='replace')
                        return self._read_historical_csv_text(raw)
                else:
                    LOGGER.info(f"Station {station_code} not found in {year}.zip")
                    return pd.DataFrame()
        except zipfile.BadZipFile:
            LOGGER.error(f"Bad ZIP file: {zip_path}")
            # Optionally remove bad zip
            # os.remove(zip_path)
            return pd.DataFrame()
        except Exception as e:
            LOGGER.error(f"Error reading ZIP {zip_path}: {e}")
            return pd.DataFrame()


    @staticmethod
    def _read_historical_csv_text(raw_text: str) -> pd.DataFrame:
        """Read INMET historical CSV content.

        Some INMET ZIP CSVs have a header line that breaks into the next line, and the
        first data row begins on the same line as the header continuation. This function
        normalizes that into a proper 1-line header + data rows before parsing.
        """
        lines = raw_text.splitlines()
        if len(lines) <= 8:
            return pd.DataFrame()

        data_lines = lines[8:]
        if not data_lines:
            return pd.DataFrame()

        date_re = re.compile(r"\d{4}[/-]\d{2}[/-]\d{2}")

        def starts_with_date(s: str) -> bool:
            return bool(re.match(r"^\s*\d{4}[/-]\d{2}[/-]\d{2}", s))

        # If the second line is not a data row but contains a date later, it's header continuation
        if len(data_lines) >= 2 and (not starts_with_date(data_lines[1])) and date_re.search(data_lines[1]):
            m = date_re.search(data_lines[1])
            if m is not None:
                header = data_lines[0].rstrip(';') + data_lines[1][:m.start()].lstrip().rstrip(';')
                first_row = data_lines[1][m.start():]
                data_lines = [header, first_row] + data_lines[2:]

        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=';',
            decimal=',',
            engine='python',
            on_bad_lines='skip',
        )
        # Drop trailing empty columns from extra ';'
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        return df


    def _find_stations_by_ibge(self, ibge_code: int) -> List[dict]:
        stations = self._get_stations()
        if stations is None or stations.empty:
            return []
            
        # Convert input to string and take first 6 or 7 digits
        ibge_str = str(ibge_code)
        
        # Prefer automatic, but allow conventional as fallback
        stations = stations.copy()
        
        # Try exact match (7 digits)
        match = stations[stations['IBGE_CODE'] == ibge_str]
        if not match.empty:
            match = match.copy()
            match['__tp_rank'] = (match['TP_STATION'] != 'Automatic').astype(int)
            match = match.sort_values(['__tp_rank']).drop(columns=['__tp_rank'])
            return match.to_dict('records')
            
        # Try 6 digits match
        if len(ibge_str) >= 6:
            ibge_6 = ibge_str[:6]
            # Check if IBGE_CODE starts with ibge_6
            # Handle NaN in IBGE_CODE
            stations_valid = stations.dropna(subset=['IBGE_CODE'])
            match = stations_valid[stations_valid['IBGE_CODE'].str.startswith(ibge_6)]
            if not match.empty:
                match = match.copy()
                match['__tp_rank'] = (match['TP_STATION'] != 'Automatic').astype(int)
                match = match.sort_values(['__tp_rank']).drop(columns=['__tp_rank'])
                return match.to_dict('records')
                
        return []


    def _find_nearest_stations(self, lat: float, lon: float, n: int = 5) -> List[dict]:
        stations = self._get_stations()
        if stations is None or stations.empty:
            return []
            
        # Calculate distance
        # Simple Euclidean distance for now (sufficient for finding nearest)
        # Or use Haversine if needed, but for small distances Euclidean on lat/lon is okay-ish for selection
        # Better: convert to geometry and use distance
        
        # Prefer automatic, but allow conventional as fallback
        stations = stations.copy()
        
        # Ensure coordinates are float
        # Column names changed in recent inmetpy versions or API: VL_LATITUDE -> LATITUDE, VL_LONGITUDE -> LONGITUDE
        lat_col = 'LATITUDE' if 'LATITUDE' in stations.columns else 'VL_LATITUDE'
        lon_col = 'LONGITUDE' if 'LONGITUDE' in stations.columns else 'VL_LONGITUDE'
        
        stations[lat_col] = pd.to_numeric(stations[lat_col], errors='coerce')
        stations[lon_col] = pd.to_numeric(stations[lon_col], errors='coerce')
        stations = stations.dropna(subset=[lat_col, lon_col])

        stations['dist'] = np.sqrt(
            (stations[lat_col] - lat)**2 + (stations[lon_col] - lon)**2
        )

        stations['__tp_rank'] = (stations['TP_STATION'] != 'Automatic').astype(int)
        nearest = stations.sort_values(['__tp_rank', 'dist']).head(n).drop(columns=['__tp_rank'])
        return nearest.to_dict('records')

    @staticmethod
    def _dedupe_stations(stations: List[dict]) -> List[dict]:
        seen = set()
        out: List[dict] = []
        for st in stations:
            code = st.get('CD_STATION')
            if not code or code in seen:
                continue
            seen.add(code)
            out.append(st)
        return out

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            'Data': 'DT_MEDICAO',
            'DATA (YYYY-MM-DD)': 'DT_MEDICAO',
            'RADIACAO GLOBAL (Kj/m²)': 'RAD_GLO',
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'CHUVA',
            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'TEM_MAX',
            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'TEM_MIN',
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'TEM_INS',
            'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'UMD_MIN',
            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'UMD_INS',
            'VENTO, VELOCIDADE HORARIA (m/s)': 'VEN_VEL'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)

        # Drop empty trailing columns from extra separators
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        
        # Convert date and time
        if 'DT_MEDICAO' in df.columns:
            # Historical data 'Data' is usually YYYY/MM/DD or DD/MM/YYYY
            df['DT_MEDICAO'] = df['DT_MEDICAO'].astype(str).str.replace('/', '-')
            df['DT_MEDICAO'] = pd.to_datetime(df['DT_MEDICAO'], errors='coerce')
        
        # Ensure numeric
        cols_to_numeric = ['RAD_GLO', 'CHUVA', 'TEM_MAX', 'TEM_MIN', 'TEM_INS', 'UMD_MIN', 'UMD_INS', 'VEN_VEL']
        for col in cols_to_numeric:
            if col in df.columns:
                # Replace comma with dot if it's a string
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    @staticmethod
    def _station_operates_in_range(station: dict, start_date: date, end_date: date) -> bool:
        """Return True if station operational dates overlap requested range.

        If start/end operation dates are missing, we keep the station as a candidate.
        """
        start_raw = station.get('DT_INICIO_OPERACAO')
        end_raw = station.get('DT_FIM_OPERACAO')

        try:
            start_op = pd.to_datetime(start_raw, errors='coerce') if start_raw else pd.NaT
            end_op = pd.to_datetime(end_raw, errors='coerce') if end_raw else pd.NaT
        except Exception:
            return True

        if pd.notna(start_op):
            start_op_date = start_op.date()
            if end_date < start_op_date:
                return False
        if pd.notna(end_op):
            end_op_date = end_op.date()
            if start_date > end_op_date:
                return False

        return True

    def _calculate_heat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'TEM_INS' not in df.columns or 'UMD_INS' not in df.columns:
            return df
            
        T_c = df['TEM_INS']
        RH = df['UMD_INS']
        
        # Convert to Fahrenheit
        T_f = T_c * 1.8 + 32
        
        # Formula provided by user
        HI_f = (-42.379 + 
                (2.04901523 * T_f) + 
                (10.14333127 * RH) - 
                (0.22475541 * T_f * RH) - 
                (0.00683783 * T_f**2) - 
                (0.05481717 * RH**2) + 
                (0.00122874 * T_f**2 * RH) + 
                (0.00085282 * T_f * RH**2) - 
                (0.00000199 * T_f**2 * RH**2))
        
        # Use T_f if T_f < 80 (Standard Heat Index definition)
        HI_f = np.where(T_f < 80, T_f, HI_f)
        
        # Convert back to Celsius
        HI_c = (HI_f - 32) / 1.8
        
        df['HEAT_INDEX'] = HI_c
        return df

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate Heat Index
        df = self._calculate_heat_index(df)

        # Group by date
        agg_funcs = {}

        # globalradiation (RAD_GLO)
        # INMET provides hourly radiation in kJ/m². We compute max/min/mean of hourly values
        # for each day, then convert to MJ/m² (divide by 1000).
        if 'RAD_GLO' in df.columns:
            agg_funcs['RAD_GLO'] = ['max', 'min', 'mean']

        # precipitation (CHUVA)
        if 'CHUVA' in df.columns:
            agg_funcs['CHUVA'] = ['sum']

        # temperature (TEM_MAX, TEM_MIN, TEM_INS)
        if 'TEM_MAX' in df.columns:
            agg_funcs['TEM_MAX'] = ['max']
        if 'TEM_MIN' in df.columns:
            agg_funcs['TEM_MIN'] = ['min']
        if 'TEM_INS' in df.columns:
            agg_funcs['TEM_INS'] = ['mean']

        # humidity (UMD_MIN, UMD_INS)
        if 'UMD_MIN' in df.columns:
            agg_funcs['UMD_MIN'] = ['min']
        if 'UMD_INS' in df.columns:
            agg_funcs['UMD_INS'] = ['mean']

        # wind (VEN_VEL)
        if 'VEN_VEL' in df.columns:
            agg_funcs['VEN_VEL'] = ['mean']

        # heat index
        if 'HEAT_INDEX' in df.columns:
            agg_funcs['HEAT_INDEX'] = ['max', 'min', 'mean']

        if 'DT_MEDICAO' not in df.columns:
            LOGGER.error("DT_MEDICAO column missing after processing. Columns found: %s", df.columns)
            return pd.DataFrame()

        daily = df.groupby('DT_MEDICAO').agg(agg_funcs)
        
        # Flatten columns
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        daily = daily.reset_index()
        daily.rename(columns={'DT_MEDICAO': 'date'}, inplace=True)
        
        # Rename to requested names (compatible with reference base)
        rename_map = {
            'RAD_GLO_max': 'globalradiation_max_kj',
            'RAD_GLO_min': 'globalradiation_min_kj',
            'RAD_GLO_mean': 'globalradiation_mea_kj',
            'CHUVA_sum': 'precipitation_sum',
            'TEM_MAX_max': 'temperature_max',
            'TEM_MIN_min': 'temperature_min',
            'TEM_INS_mean': 'temperature_mea',
            'UMD_MIN_min': 'humidity_min',
            'UMD_INS_mean': 'humidity_mea',
            'VEN_VEL_mean': 'wind_mea',
            'HEAT_INDEX_max': 'heatindex_max',
            'HEAT_INDEX_min': 'heatindex_min',
            'HEAT_INDEX_mean': 'heatindex_mea'
        }
        
        daily.rename(columns=rename_map, inplace=True)

        # Convert radiation from kJ/m² to MJ/m² (1 MJ = 1000 kJ)
        for suffix in ['max', 'min', 'mea']:
            kj_col = f'globalradiation_{suffix}_kj'
            mj_col = f'globalradiation_{suffix}'
            if kj_col in daily.columns:
                daily[mj_col] = daily[kj_col] / 1000.0
                daily = daily.drop(columns=[kj_col])

        # Ensure all requested columns exist (fill with NaN if missing)
        requested_cols = [
            'globalradiation_max', 'globalradiation_min', 'globalradiation_mea',
            'precipitation_sum',
            'temperature_max', 'temperature_min', 'temperature_mea',
            'humidity_min', 'humidity_mea',
            'wind_mea',
            'heatindex_max', 'heatindex_min', 'heatindex_mea'
        ]
        
        for col in requested_cols:
            if col not in daily.columns:
                daily[col] = np.nan
                
        return daily[['date'] + requested_cols]

    @staticmethod
    def _daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily rows into base-style weekly rows ending on Sunday (W-SUN).
        
        For globalradiation: daily values already have max/min/mean per day (MJ/m²).
        Weekly aggregation produces max of daily max, min of daily min, mean of daily mean.
        """
        if daily.empty:
            return daily

        df = daily.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).set_index('date').sort_index()

        # Aggregation rules for weekly
        agg = {
            'globalradiation_max': 'max',  # Weekly max = max of daily maxes
            'globalradiation_min': 'min',  # Weekly min = min of daily mins  
            'globalradiation_mea': 'mean', # Weekly mean = mean of daily means
            'precipitation_sum': 'sum',
            'temperature_max': 'max',
            'temperature_min': 'min',
            'temperature_mea': 'mean',
            'humidity_min': 'min',
            'humidity_mea': 'mean',
            'wind_mea': 'mean',
            'heatindex_max': 'max',
            'heatindex_min': 'min',
            'heatindex_mea': 'mean',
        }

        weekly = df.resample('W-SUN').agg(agg)  # type: ignore[arg-type]
        
        weekly = weekly.reset_index()
        weekly['date'] = pd.to_datetime(weekly['date'], errors='coerce').dt.normalize()
        return weekly



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    downloader = INMETDownloader()
    
    shapefile = 'data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp'
    
    # Check if shapefile exists before running
    if os.path.exists(shapefile):
        try:
            df_result = downloader.fetch_daily_data(
                shapefile_path=shapefile,
                start='2020-01-01',
                end='2020-01-31',
                out_csv='data/output/inmet/São_Paulo_inmet_2020.csv'
            )
            print("\n--- Download and processing successful ---")
            print(df_result.head())
        except Exception as e:
            LOGGER.error("An error occurred during the main execution: %s", e)
    else:
        LOGGER.warning(f"Shapefile not found at {shapefile}. Please run the shapefile downloader first.")
