import logging
import os
import earthaccess
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
from datetime import datetime
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

LOGGER = logging.getLogger(__name__)

class OMIDownloader:
    """
    Baixa e processa dados do satélite Aura/OMI (NO2 e Ozônio).
    Produtos:
    - OMNO2d: OMI/Aura Nitrogen Dioxide (NO2) Total and Tropospheric Column 1-orbit L3 Daily 0.25x0.25 deg
    - OMTO3d: OMI/Aura Ozone (O3) Total Column 1-orbit L3 Daily 1.0x1.0 deg
    """
    
    # Mapeamento de produtos e variáveis
    PRODUCTS = {
        "no2omi": {
            "short_name": "OMNO2d",
            "variable": "ColumnAmountNO2Trop", 
            "grid_group": "HDFEOS/GRIDS/ColumnAmountNO2/Data Fields",
            "scale": 1.0, 
            "unit_convert": True, # Concersion logic: molec/cm2 -> 1e15 molec/cm2 for readability? Or keep raw?
                                  # Let's keep raw but maybe clean up output. 
                                  # Actually, typical analysis uses raw or mol/m2. 
                                  # Let's add a comment but keep False for now unless requested.
                                  # WAIT: The values are huge 1e16. Let's leave as is for scientific accuracy unless asked.
            "unit_label": "molec/cm2"
        },
        "o3omi": {
            "short_name": "OMTO3d",
            "variable": "ColumnAmountO3",
            "grid_group": "HDFEOS/GRIDS/OMI Column Amount O3/Data Fields",
            "scale": 1.0,
            "unit_convert": False, # Dobson Units (DU)
            "unit_label": "DU"
        }
    }

    def __init__(self):
        self.user = os.getenv("NASA_USER")
        self.password = os.getenv("NASA_PASSWORD")
        
        if self.user and self.password:
            os.environ["EARTHDATA_USERNAME"] = self.user
            os.environ["EARTHDATA_PASSWORD"] = self.password
            
        self.auth = earthaccess.login(strategy="environment")
        if not self.auth:
             # Fallback interactive
             self.auth = earthaccess.login()

    def _process_file(self, file_path, product_info, gdf_bbox):
        """
        Lê arquivo HE5, recorta e extrai estatísticas.
        """
        try:
            # Tenta abrir com xarray engine='netcdf4' ou 'h5netcdf'
            # OMI L3 é HDF-EOS5, que é baseado em HDF5.
            # As vezes é necessário especificar group.
            
            # Group path for data fields
            group = product_info.get("grid_group")
            var_name = product_info.get("variable")
            
            LOGGER.info(f"DEBUG: Processing {file_path}")
            LOGGER.info(f"DEBUG: derived var_name='{var_name}', group='{group}'")
            LOGGER.info(f"DEBUG: product_info keys: {list(product_info.keys())}")
            
            
            # --- CRITICAL SECTION: HDF5 ACCESS ---
            from src.utils.locks import HDF5_LOCK
            with HDF5_LOCK:
                ds = None
                try:
                    ds = xr.open_dataset(file_path, group=group, engine="netcdf4", decode_coords=False)
                except Exception:
                    try:
                        ds = xr.open_dataset(file_path, group=group, engine="h5netcdf", decode_coords=False)
                    except Exception as e:
                        LOGGER.warning(f"Erro ao abrir {file_path} (group={group}): {e}")
                        return None
                
                if var_name not in ds:
                    LOGGER.warning(f"Variável {var_name} não encontrada em {file_path}")
                    ds.close()
                    return None
                    
                data_da = ds[var_name]
                
                # Identificar dimensões e construir coordenadas
                shape = data_da.shape
                lat_dim_name = data_da.dims[0]
                lon_dim_name = data_da.dims[1]
                
                lats = None
                lons = None
                
                if shape == (720, 1440):
                    # 0.25 degree
                    lats = np.arange(-90 + 0.125, 90, 0.25)
                    lons = np.arange(-180 + 0.125, 180, 0.25)
                elif shape == (180, 360):
                     # 1.0 degree
                    lats = np.arange(-90 + 0.5, 90, 1.0)
                    lons = np.arange(-180 + 0.5, 180, 1.0)
                
                if lats is not None and lons is not None:
                    # Assign coords
                    ds = ds.assign_coords({
                        lat_dim_name: lats,
                        lon_dim_name: lons
                    })
                    
                    # Recorte espacial
                    minx, miny, maxx, maxy = gdf_bbox
                    
                    # Select using nearest/slice
                    ds_subset = ds.sel({
                        lat_dim_name: slice(miny, maxy),
                        lon_dim_name: slice(minx, maxx)
                    })
                    
                    # Data array subset - FORCE LOAD
                    vals = ds_subset[var_name].values.astype(float)
                else:
                     LOGGER.warning(f"Shape não reconhecido para {file_path}: {shape}")
                     ds.close()
                     return None

                ds.close()


            # Filter fill values (usually very large negative numbers or defined in attrs)
            fill_val = data_da.attrs.get('_FillValue')
            missing_val = data_da.attrs.get('MissingValue')
            
            if fill_val is not None:
                vals[vals == fill_val] = np.nan
            if missing_val is not None:
                vals[vals == missing_val] = np.nan
                
            # Filter unlikely values (e.g. < -1e20)
            vals[vals < -1e20] = np.nan
            
            # Valid data
            valid = vals[~np.isnan(vals)]
            
            result = None
            if valid.size > 0:
                result = {
                    'max': float(np.max(valid)),
                    'min': float(np.min(valid)),
                    'mea': float(np.mean(valid))
                }
                
            ds.close()
            return result
            
        except Exception as e:
            LOGGER.error(f"Erro processando {file_path}: {e}")
            return None

    def fetch_data(self, shapefile_path, start_date, end_date, output_csv="data/output/omi/omi_data.csv", cache_dir=None):
        
        # Carregar Shapefile
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        bbox = gdf.total_bounds # (minx, miny, maxx, maxy)
        
        # Parse Dates
        try:
             s_dt = datetime.strptime(start_date, "%d/%m/%Y").strftime("%Y-%m-%d")
             e_dt = datetime.strptime(end_date, "%d/%m/%Y").strftime("%Y-%m-%d")
        except:
             s_dt = start_date
             e_dt = end_date
             
        # Armazenar resultados diários consolidados
        # Estrutura: { date: { 'no2omi_max': ..., 'o3omi_mea': ... } }
        daily_records = {}
        
        if cache_dir is None:
            cache_dir = "data/cache/omi"
        
        os.makedirs(cache_dir, exist_ok=True)
        
        for pol_key, info in self.PRODUCTS.items():
            LOGGER.info(f"Buscando OMI {info['short_name']} ({pol_key})...")
            
            # Create subfolder per product to avoid confusion
            prod_cache = os.path.join(cache_dir, pol_key)
            os.makedirs(prod_cache, exist_ok=True)
            
            results = earthaccess.search_data(
                short_name=info['short_name'],
                bounding_box=tuple(bbox),
                temporal=(s_dt, e_dt)
            )
            
            LOGGER.info(f"Encontrados {len(results)} grânulos para {pol_key}.")
            if not results:
                continue
                
            files = earthaccess.download(results, prod_cache)
            
            for file in files:
                fpath = str(file)
                fname = os.path.basename(fpath)
                
                # Extract date
                # Format: OMI-Aura_L3-OMNO2d_2023m0101_v003...
                # or OMTO3d usually has date in name too.
                # Regex generic for YYYYmMMDD
                try:
                    date_str = fname.split('_')[2] # 2023m0101
                    # remove 'm'
                    date_str = date_str.replace('m', '') # 20230101
                    f_date = datetime.strptime(date_str, "%Y%m%d").date()
                except:
                    LOGGER.warning(f"Data não extraída de {fname}")
                    continue
                
                LOGGER.info(f"Processando {pol_key} - {f_date}")
                stats = self._process_file(fpath, info, bbox)
                
                if stats:
                    if f_date not in daily_records:
                        daily_records[f_date] = {'date': f_date}
                        
                    daily_records[f_date][f"{pol_key}_max"] = stats['max']
                    daily_records[f_date][f"{pol_key}_min"] = stats['min']
                    daily_records[f_date][f"{pol_key}_mea"] = stats['mea']

        # Converter para DataFrame
        rows = list(daily_records.values())
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df = df.sort_values('date')
            
            # Reordenar e Renomear para seguir o padrão de referência
            # Referência: no2omi_max, no2omi_mea, no2omi_min (confirmado pelo grep)
            # OMI.py gera: _max, _min, _mea
            
            # Conversão de Unidade para NO2 (molec/cm2 -> DU)
            # 1 DU = 2.69e16 molec/cm2
            cols_no2 = [c for c in df.columns if 'no2omi' in c]
            for c in cols_no2:
                df[c] = df[c] / 2.69e16
                
            # Garantir ordem das colunas
            # date, no2omi_max, no2omi_mea, no2omi_min, o3omi_max, o3omi_mea, o3omi_min
            desired_order = ['date']
            for pol in ['no2omi', 'o3omi']:
                desired_order.append(f"{pol}_max")
                desired_order.append(f"{pol}_mea")
                desired_order.append(f"{pol}_min")
                
            # Filter standard columns that exist
            final_cols = [c for c in desired_order if c in df.columns]
            df = df[final_cols]
            
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            LOGGER.info(f"Salvo: {output_csv}")
            
        # Cleanup
        try:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
            LOGGER.info(f"Cache OMI limpo: {cache_dir}")
        except Exception as e:
             LOGGER.warning(f"Erro ao limpar cache OMI {cache_dir}: {e}")
             
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dl = OMIDownloader()
    dl.fetch_data("data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp", "01/01/2020", "05/01/2020")
