import logging
import os
import re
import earthaccess
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pyhdf.SD import SD, SDC
from pyproj import CRS, Transformer
from shapely.geometry import mapping
from rasterio import features, Affine

# Carrega variáveis de ambiente
load_dotenv()

LOGGER = logging.getLogger(__name__)

class ModisDownloader:
    """
    Baixa e processa dados do MODIS (MCD19A2 - AOD) usando earthaccess e processamento manual.
    """
    
    # MODIS Sinusoidal Projection Parameters
    MODIS_SINU = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    # Tile size in meters
    TILE_SIZE = 1111950.537
    # Global Origin (Top-Left)
    GLOBAL_X_MIN = -20015109.354
    GLOBAL_Y_MAX = 10007554.677
    
    def __init__(self):
        self.user = os.getenv("NASA_USER")
        self.password = os.getenv("NASA_PASSWORD")
        
        # Configure env for earthaccess
        if self.user and self.password:
            os.environ["EARTHDATA_USERNAME"] = self.user
            os.environ["EARTHDATA_PASSWORD"] = self.password
            
        # Authenticate
        self.auth = earthaccess.login(strategy="environment")
        if not self.auth:
            LOGGER.warning("Autenticação via variáveis de ambiente falhou. Tentando interativa (não recomendado em automação).")
            self.auth = earthaccess.login()
            
    def _get_tile_transform(self, h, v, rows=1200, cols=1200):
        """
        Calcula a transformação Afim para um tile MODIS (h, v).
        """
        # Calcular limites do tile em metros
        x_min = self.GLOBAL_X_MIN + (h * self.TILE_SIZE)
        y_max = self.GLOBAL_Y_MAX - (v * self.TILE_SIZE)
        
        # Tamanho do pixel
        pixel_size = self.TILE_SIZE / rows # ~926m
        
        # Transform: x_min, pixel_w, 0, y_max, 0, -pixel_h
        return Affine(pixel_size, 0, x_min, 0, -pixel_size, y_max)
        
    def process_file(self, file_path, shape_sinu):
        """
        Lê o arquivo HDF4, extrai AOD e aplica a máscara do shapefile.
        """
        try:
            # Parse h/v from filename (e.g., MCD19A2...h13v11...)
            match = re.search(r'\.h(\d+)v(\d+)\.', os.path.basename(file_path))
            if not match:
                LOGGER.warning(f"Não foi possível extrair Tile ID de {file_path}")
                return None
                
            h = int(match.group(1))
            v = int(match.group(2))
            
            
            # --- CRITICAL SECTION: HDF4 ACCESS ---
            from src.utils.locks import HDF5_LOCK
            with HDF5_LOCK:
                # Read HDF
                hdf = SD(str(file_path), SDC.READ)
                try:
                    # Select Variable
                    # MCD19A2 v061 'Optical_Depth_055' (Green Band AOD)
                    var_name = 'Optical_Depth_055'
                    if var_name not in hdf.datasets():
                        LOGGER.warning(f"Variável {var_name} não encontrada em {file_path}")
                        return None
                        
                    sds = hdf.select(var_name)
                    try:
                        data = sds.get().astype(float)
                        attrs = sds.attributes()
                    finally:
                        sds.endaccess()
                finally:
                    hdf.end()
            
            # Handle Fill Value, Scale, Offset (Numpy ops, safe outside lock? 
            # ideally yes, but 'data' is numpy array now. 
            # If we want to minimize lock time, we copy data out.
            # sds.get() returns a numpy array copy usually.
            
            # Handle Fill Value, Scale, Offset
            fill_value = attrs.get('_FillValue')
            scale = attrs.get('scale_factor', 1.0)
            offset = attrs.get('add_offset', 0.0)
            
            # Apply Mask from FillValue
            if fill_value is not None:
                data[data == fill_value] = np.nan
                
            # Apply Scale and Offset
            data = data * scale + offset
            
            # Get Grid Dimensions
            if data.ndim == 2:
                 rows, cols = data.shape
            elif data.ndim == 3:
                 # Usually (orbits, rows, cols) for MCD19A2
                 _, rows, cols = data.shape
            else:
                 LOGGER.warning(f"Dimensões inesperadas para variável {var_name}: {data.shape}")
                 return None
            
            # Build Transform
            transform = self._get_tile_transform(h, v, rows, cols)
            
            # Create Mask (Rasterize Shapefile)
            # shape_sinu is a list of geometries in Sinusoidal projection
            mask = features.geometry_mask(
                shape_sinu,
                transform=transform,
                out_shape=(rows, cols),
                invert=True # True where shapes ARE present
            )
            
            # Extract Data
            if data.ndim == 2:
                subset = data[mask]
            else:
                # For 3D (orbits, y, x), apply mask to last 2 dims
                # values[:, mask] returns flattened valid pixels from all orbits
                subset = data[:, mask].flatten()
            
            # Valid pixels (not NaN)
            valid_subset = subset[~np.isnan(subset)]
            
            result = None
            if valid_subset.size > 0:
                result = {
                    'aod_max': float(np.max(valid_subset)),
                    'aod_min': float(np.min(valid_subset)),
                    'aod_mean': float(np.mean(valid_subset)),
                    'count': int(valid_subset.size)
                }
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Erro ao processar {file_path}: {e}")
            return None

    def fetch_data(self, shapefile_path, start_date, end_date, output_csv="data/output/modis/modis_data.csv", cache_dir=None):
        """
        Fluxo principal usando earthaccess.
        """
        # 1. Carregar e Projetar Shapefile for BBox and Masking
        gdf = gpd.read_file(shapefile_path)
        
        # Reproject to WGS84 for earthaccess query (bbox)
        if gdf.crs != "EPSG:4326":
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
        else:
            gdf_wgs84 = gdf
            
        bbox = gdf_wgs84.total_bounds # (minx, miny, maxx, maxy)
        
        # Reproject to Sinusoidal for Masking
        crs_sinu = CRS.from_proj4(self.MODIS_SINU)
        gdf_sinu = gdf_wgs84.to_crs(crs_sinu)
        geoms_sinu = [mapping(g) for g in gdf_sinu.geometry]
        
        # 2. Search Granules
        LOGGER.info(f"Buscando grânulos MCD19A2 entre {start_date} e {end_date}...")
        # Convert output dates to YYYY-MM-DD
        # Assume input is DD/MM/YYYY or YYYY-MM-DD. 
        # earthaccess handles ISO strings roughly. Best to ensure YYYY-MM-DD.
        # Assuming parse_date util (removed import to simplify, doing robust logic here)
        try:
             s_dt = datetime.strptime(start_date, "%d/%m/%Y").strftime("%Y-%m-%d")
             e_dt = datetime.strptime(end_date, "%d/%m/%Y").strftime("%Y-%m-%d")
        except:
             s_dt = start_date
             e_dt = end_date
             
        results = earthaccess.search_data(
            short_name="MCD19A2",
            bounding_box=tuple(bbox),
            temporal=(s_dt, e_dt)
        )
        
        LOGGER.info(f"Encontrados {len(results)} grânulos.")
        
        if not results:
            return pd.DataFrame()
            
        # 3. Download
        if cache_dir is None:
            download_dir = "data/cache/modis_ea"
        else:
            download_dir = str(cache_dir)
            
        os.makedirs(download_dir, exist_ok=True)
        files = earthaccess.download(results, download_dir)
        
        # 4. Process files
        daily_stats = {} # Date -> List of stats (one per tile, usually one, maybe two)
        
        for file in files:
            fpath = str(file)
            
            # Extract date from filename: MCD19A2.A2023001...
            fname = os.path.basename(fpath)
            try:
                parts = fname.split('.')
                # parts[1] is AYYYYDDD
                yd_str = parts[1][1:] # YYYYDDD
                f_date = datetime.strptime(yd_str, "%Y%j").date()
            except:
                LOGGER.warning(f"Data ilegível no arquivo {fname}")
                continue
                
            LOGGER.info(f"Processando {fname} ({f_date})")
            
            stats = self.process_file(fpath, geoms_sinu)
            
            if stats:
                if f_date not in daily_stats:
                    daily_stats[f_date] = []
                daily_stats[f_date].append(stats)
        
        # 5. Aggregate Daily
        final_rows = []
        for d, s_list in daily_stats.items():
            # If multiple tiles cover the area (rare for small city, but possible), average them weighted by pixel count?
            # Or just take global mean.
            # Simplified: weighted mean
            total_count = sum(s['count'] for s in s_list)
            if total_count == 0: 
                continue
                
            w_mean = sum(s['aod_mean'] * s['count'] for s in s_list) / total_count
            g_max = max(s['aod_max'] for s in s_list)
            g_min = min(s['aod_min'] for s in s_list)
            
            final_rows.append({
                'date': d,
                'aodterramodis_max': g_max,
                'aodterramodis_min': g_min,
                'aodterramodis_mea': w_mean
            })
            
        df = pd.DataFrame(final_rows)
        if not df.empty:
            df = df.sort_values('date')
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            LOGGER.info(f"Salvo: {output_csv}")
            
        # Cleanup
        try:
            import shutil
            shutil.rmtree(download_dir, ignore_errors=True)
            LOGGER.info(f"Cache limpo: {download_dir}")
        except Exception as e:
            LOGGER.warning(f"Erro ao limpar cache {download_dir}: {e}")
            
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dl = ModisDownloader()
    # Teste
    dl.fetch_data("data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp", "01/01/2023", "05/01/2023")
