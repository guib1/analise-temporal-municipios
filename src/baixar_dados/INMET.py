from __future__ import annotations
import os
import logging
import zipfile
import requests
import io
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
# from inmetpy.stations import InmetStation  # Deprecated

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
                response = requests.get("https://apitempo.inmet.gov.br/estacoes/T")
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
        ibge_code: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Downloads INMET data for the nearest station to the shapefile centroid.
        """
        start_date = self._parse_date(start)
        end_date = self._parse_date(end)
        
        # Try to get IBGE code if not provided
        if not ibge_code:
            ibge_code = self._get_ibge_code(shapefile_path)

        stations = []
        
        # 1. Try to find station by IBGE code
        if ibge_code:
            LOGGER.info(f"Searching for station with IBGE code: {ibge_code}")
            station_by_ibge = self._find_station_by_ibge(ibge_code)
            if station_by_ibge:
                LOGGER.info(f"Found station by IBGE code: {station_by_ibge['STATION_NAME']} ({station_by_ibge['CD_STATION']})")
                stations.append(station_by_ibge)
        
        # 2. If no station found by IBGE, fallback to nearest stations
        if not stations:
            LOGGER.info("No station found by IBGE code. Falling back to nearest stations.")
            # Get centroid
            lat, lon = self._centroid_from_shapefile(shapefile_path)
            stations = self._find_nearest_stations(lat, lon, n=5)
        
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
                    mask = (df_station['DT_MEDICAO'].dt.date >= start_date) & (df_station['DT_MEDICAO'].dt.date <= end_date)
                    df = df_station.loc[mask].copy()
                    
                    if not df.empty:
                        LOGGER.info(f"Successfully retrieved data from {station_name} ({station_id})")
                        break
                    else:
                        LOGGER.warning(f"Data found for {station_id} but outside requested range.")
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
        
        if ibge_code:
            daily_df.insert(0, 'codigo_ibge', ibge_code)
        else:
            inferred_ibge = self._get_ibge_code(shapefile_path)
            if inferred_ibge:
                daily_df.insert(0, 'codigo_ibge', inferred_ibge)
            else:
                lat, lon = self._centroid_from_shapefile(shapefile_path)
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

        # Download if not exists
        if not os.path.exists(zip_path):
            LOGGER.info(f"Downloading historical data for {year}...")
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    LOGGER.error(f"Failed to download {url}. Status: {response.status_code}")
                    return pd.DataFrame()
            except Exception as e:
                LOGGER.error(f"Download error: {e}")
                return pd.DataFrame()

        # Extract specific station file
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find file matching station code
                # Pattern: INMET_SE_SP_A771_...
                # We look for `_{station_code}_` in the filename
                target_file = None
                for filename in z.namelist():
                    if f"_{station_code}_" in filename:
                        target_file = filename
                        break
                
                if target_file:
                    LOGGER.info(f"Found file {target_file} in {zip_filename}")
                    with z.open(target_file) as f:
                        # Read CSV. INMET CSVs usually have header at line 9 (skip 8 rows)
                        # Separator is ';' and decimal is ','
                        # Encoding is usually latin1 or utf-8 (try latin1 first for legacy)
                        return pd.read_csv(
                            f, 
                            sep=';', 
                            decimal=',', 
                            skiprows=8, 
                            encoding='latin1',
                            on_bad_lines='skip'
                        )
                else:
                    LOGGER.warning(f"Station {station_code} not found in {year}.zip")
                    return pd.DataFrame()
        except zipfile.BadZipFile:
            LOGGER.error(f"Bad ZIP file: {zip_path}")
            # Optionally remove bad zip
            # os.remove(zip_path)
            return pd.DataFrame()
        except Exception as e:
            LOGGER.error(f"Error reading ZIP {zip_path}: {e}")
            return pd.DataFrame()


    def _find_station_by_ibge(self, ibge_code: int) -> Optional[dict]:
        stations = self._get_stations()
        if stations is None or stations.empty:
            return None
            
        # Convert input to string and take first 6 or 7 digits
        ibge_str = str(ibge_code)
        
        # Filter only automatic stations?
        stations = stations[stations['TP_STATION'] == 'Automatic'].copy()
        
        # Try exact match (7 digits)
        match = stations[stations['IBGE_CODE'] == ibge_str]
        if not match.empty:
            return match.iloc[0].to_dict()
            
        # Try 6 digits match
        if len(ibge_str) >= 6:
            ibge_6 = ibge_str[:6]
            # Check if IBGE_CODE starts with ibge_6
            # Handle NaN in IBGE_CODE
            stations_valid = stations.dropna(subset=['IBGE_CODE'])
            match = stations_valid[stations_valid['IBGE_CODE'].str.startswith(ibge_6)]
            if not match.empty:
                return match.iloc[0].to_dict()
                
        return None


    def _find_nearest_stations(self, lat: float, lon: float, n: int = 5) -> List[dict]:
        stations = self._get_stations()
        if stations is None or stations.empty:
            return []
            
        # Calculate distance
        # Simple Euclidean distance for now (sufficient for finding nearest)
        # Or use Haversine if needed, but for small distances Euclidean on lat/lon is okay-ish for selection
        # Better: convert to geometry and use distance
        
        # Filter only automatic stations? User link points to "estações automáticas"
        stations = stations[stations['TP_STATION'] == 'Automatic'].copy()
        
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
        
        nearest = stations.sort_values('dist').head(n)
        return nearest.to_dict('records')

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
            
        # humidity (UMD_MAX, UMD_MIN, UMD_INS)
        if 'UMD_MAX' in df.columns:
            agg_funcs['UMD_MAX'] = ['max']
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
        
        # Rename to requested names
        rename_map = {
            'RAD_GLO_max': 'globalradiation_max',
            'RAD_GLO_min': 'globalradiation_min',
            'RAD_GLO_mean': 'globalradiation_mea',
            'CHUVA_sum': 'precipitation_sum',
            'TEM_MAX_max': 'temperature_max',
            'TEM_MIN_min': 'temperature_min',
            'TEM_INS_mean': 'temperature_med',
            'UMD_MAX_max': 'humidity_max',
            'UMD_MIN_min': 'humidity_min',
            'UMD_INS_mean': 'humidity_mea',
            'VEN_VEL_mean': 'wind_mea',
            'HEAT_INDEX_max': 'heatindex_max',
            'HEAT_INDEX_min': 'heatindex_min',
            'HEAT_INDEX_mean': 'heatindex_mea'
        }
        
        daily.rename(columns=rename_map, inplace=True)
        
        # Ensure all requested columns exist (fill with NaN if missing)
        requested_cols = [
            'globalradiation_max', 'globalradiation_min', 'globalradiation_mea',
            'precipitation_sum',
            'temperature_max', 'temperature_min', 'temperature_med',
            'humidity_max', 'humidity_min', 'humidity_mea',
            'wind_mea',
            'heatindex_max', 'heatindex_min', 'heatindex_mea'
        ]
        
        for col in requested_cols:
            if col not in daily.columns:
                daily[col] = np.nan
                
        return daily[ ['date'] + requested_cols ]

    @staticmethod
    def _parse_date(s: str) -> date:
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        raise ValueError(f"Invalid date format: {s}. Use DD/MM/YYYY or YYYY-MM-DD")

    @staticmethod
    def _centroid_from_shapefile(shp_path: str) -> Tuple[float, float]:
        if gpd is None:
            raise RuntimeError("geopandas is not installed.")
        gdf = gpd.read_file(shp_path)
        try:
            union_geom = gdf.union_all()
        except AttributeError:
            union_geom = gdf.unary_union
        centroid = union_geom.centroid
        return float(centroid.y), float(centroid.x)

    def _get_ibge_code(self, shapefile_path: str) -> Optional[int]:
        if gpd is not None:
            try:
                gdf = gpd.read_file(shapefile_path)
                for col in ['code_muni', 'CD_MUN', 'CD_GEOCMU']:
                    if col in gdf.columns:
                        return int(gdf[col].iloc[0])
            except Exception as e:
                LOGGER.warning(f"Could not read shapefile for IBGE code: {e}")
        
        try:
            directory = os.path.dirname(shapefile_path)
            for f in os.listdir(directory):
                if f.endswith('_ibge.csv'):
                    path = os.path.join(directory, f)
                    df = pd.read_csv(path)
                    if 'codigo_ibge' in df.columns:
                        return int(df['codigo_ibge'].iloc[0])
        except Exception as e:
            LOGGER.warning(f"Could not read IBGE CSV: {e}")
            
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    downloader = INMETDownloader()
    
    # Example usage for Diadema
    shapefile = 'data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp'
    
    # Check if shapefile exists before running
    if os.path.exists(shapefile):
        try:
            df_result = downloader.fetch_daily_data(
                shapefile_path=shapefile,
                start='2024-01-01',
                end='2024-06-01', # 10 days example
                out_csv='data/output/inmet/São_Paulo_inmet_2024.csv'
            )
            print("\n--- Download and processing successful ---")
            print(df_result.head())
        except Exception as e:
            LOGGER.error("An error occurred during the main execution: %s", e)
    else:
        LOGGER.warning(f"Shapefile not found at {shapefile}. Please run the shapefile downloader first.")
