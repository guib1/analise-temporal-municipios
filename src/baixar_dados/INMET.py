from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from inmetpy.stations import InmetStation

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None

LOGGER = logging.getLogger(__name__)

class INMETDownloader:
    """
    Downloads INMET data for a given shapefile/municipality by finding the nearest station.
    """

    def __init__(self):
        self.inmet = InmetStation()
        self._stations_cache = None

    def _get_stations(self) -> pd.DataFrame:
        if self._stations_cache is None:
            self._stations_cache = self.inmet.get_stations()
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
            
            self._stations_cache['IBGE_CODE'] = self._stations_cache['CD_WSI'].apply(extract_ibge)
            
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
            
        df = None
        for station in stations:
            station_id = station['CD_STATION']
            station_name = station['STATION_NAME']
            LOGGER.info(f"Trying station: {station_name} ({station_id})")

            # Download data using inmetpy
            try:
                df_temp = self.inmet.get_data_station(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    'hour',
                    [station_id]
                )
                
                if df_temp is not None and not df_temp.empty:
                    df = df_temp
                    LOGGER.info(f"Successfully downloaded data from {station_name} ({station_id})")
                    break
                else:
                    LOGGER.warning(f"No data returned for station {station_id}. Trying next...")
            except Exception as e:
                LOGGER.error(f"Error downloading data from {station_id}: {e}")
                continue

        if df is None or df.empty:
            LOGGER.warning(f"No data returned for any of the nearest stations.")
            return pd.DataFrame()

        # Process and aggregate
        daily_df = self._process_data(df)
        
        if ibge_code:
            daily_df.insert(0, 'codigo_ibge', ibge_code)
        else:
            # Try to get IBGE code from shapefile if not provided
            # (Already tried above, but just in case)
            inferred_ibge = self._get_ibge_code(shapefile_path)
            if inferred_ibge:
                daily_df.insert(0, 'codigo_ibge', inferred_ibge)
            else:
                # Fallback to lat/lon if no IBGE code
                lat, lon = self._centroid_from_shapefile(shapefile_path)
                daily_df['lat'] = lat
                daily_df['lon'] = lon

        # Save
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        daily_df.to_csv(out_csv, index=False)
        LOGGER.info(f"CSV generated -> {out_csv}")
        
        return daily_df

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

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Columns mapping and aggregation
        # INMET columns (hourly):
        # DT_MEDICAO, HR_MEDICAO, TEM_INS, TEM_MAX, TEM_MIN, UMD_INS, UMD_MAX, UMD_MIN, 
        # PTO_INS, PTO_MAX, PTO_MIN, PRE_INS, PRE_MAX, PRE_MIN, RAD_GLO, CHUVA, VEN_DIR, VEN_VEL, VEN_RAJ
        
        # Convert date and time
        df['DT_MEDICAO'] = pd.to_datetime(df['DT_MEDICAO'])
        
        # Ensure numeric
        cols_to_numeric = ['RAD_GLO', 'CHUVA', 'TEM_MAX', 'TEM_MIN', 'TEM_INS', 'UMD_MIN', 'UMD_INS', 'VEN_VEL']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
            
        # humidity (UMD_MIN, UMD_INS)
        if 'UMD_MIN' in df.columns:
            agg_funcs['UMD_MIN'] = ['min']
        if 'UMD_INS' in df.columns:
            agg_funcs['UMD_INS'] = ['mean']
            
        # wind (VEN_VEL)
        if 'VEN_VEL' in df.columns:
            agg_funcs['VEN_VEL'] = ['mean']

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
            'UMD_MIN_min': 'humidity_min',
            'UMD_INS_mean': 'humidity_mea',
            'VEN_VEL_mean': 'wind_mea'
        }
        
        daily.rename(columns=rename_map, inplace=True)
        
        # Ensure all requested columns exist (fill with NaN if missing)
        requested_cols = [
            'globalradiation_max', 'globalradiation_min', 'globalradiation_mea',
            'precipitation_sum',
            'temperature_max', 'temperature_min', 'temperature_med',
            'humidity_min', 'humidity_mea',
            'wind_mea'
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
    shapefile = 'data/shapefiles/SP-Diadema/SP_Diadema.shp'
    
    # Check if shapefile exists before running
    if os.path.exists(shapefile):
        try:
            df_result = downloader.fetch_daily_data(
                shapefile_path=shapefile,
                start='2024-01-01',
                end='2024-06-01', # 10 days example
                out_csv='data/output/inmet/diadema_inmet_2024.csv'
            )
            print("\n--- Download and processing successful ---")
            print(df_result.head())
        except Exception as e:
            LOGGER.error("An error occurred during the main execution: %s", e)
    else:
        LOGGER.warning(f"Shapefile not found at {shapefile}. Please run the shapefile downloader first.")
