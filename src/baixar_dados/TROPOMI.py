import logging
import os
import shutil
import sys
import tempfile
import zipfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dotenv import load_dotenv
from shapely.geometry import Point, Polygon, box
from shapely.wkt import dumps as wkt_dumps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.utils.geo import parse_date

# Carrega variáveis de ambiente
load_dotenv()

LOGGER = logging.getLogger(__name__)

class TropomiDownloader:
    """
    Baixa e processa dados do Sentinel-5P (TROPOMI) para O3 e NO2
    usando a API OData do Copernicus Data Space Ecosystem (CDSE).
    """
    
    # Endpoints do CDSE
    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    DOWNLOAD_URL_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

    # Constantes de conversão
    # 1 mol/m^2 = 2241.15 DU (Dobson Units)
    MOL_M2_TO_DU = 2241.15
    
    # Mapeamento de produtos
    PRODUCTS = {
        "o3": {
            "product_type": "L2__O3____",
            "variable": "ozone_total_vertical_column",
            "qa_threshold": 0.5,
            "unit_convert": True
        },
        "no2": {
            "product_type": "L2__NO2___",
            "variable": "nitrogendioxide_tropospheric_column",
            "qa_threshold": 0.75,
            "unit_convert": True
        },
    }

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.getenv("COPERNICUS_USER")
        self.password = password or os.getenv("COPERNICUS_PASSWORD")

        if not self.username or not self.password:
            raise ValueError(
                "Credenciais Copernicus não encontradas. "
                "Defina COPERNICUS_USER e COPERNICUS_PASSWORD no arquivo .env."
            )
        
        self.access_token = None
        self.token_expiry = 0

    def _get_token(self) -> str:
        """Obtém ou renova o token de acesso OIDC."""
        if self.access_token and time.time() < self.token_expiry:
            return self.access_token

        LOGGER.info("Autenticando no Copernicus Data Space Ecosystem...")
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        try:
            response = requests.post(self.AUTH_URL, data=data, timeout=30)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # Define expiração com margem de segurança de 60s
            self.token_expiry = time.time() + token_data.get("expires_in", 3600) - 60
            LOGGER.info("Autenticação realizada com sucesso.")
            return self.access_token
        except Exception as e:
            LOGGER.error(f"Falha na autenticação CDSE: {e}")
            if 'response' in locals() and response.content:
                try:
                    error_data = response.json()
                    if error_data.get("error_description") == "Account is not fully set up":
                        LOGGER.critical(">>> AÇÃO NECESSÁRIA: Sua conta CDSE não está ativada ou completa. <<<")
                        LOGGER.critical("Acesse https://dataspace.copernicus.eu/, faça login e complete o cadastro/aceite os termos.")
                except Exception:
                    pass
                LOGGER.error(f"Detalhes: {response.content.decode()}")
            raise

    def fetch_data(self, shapefile_path, start_date, end_date, output_csv="data/output/tropomi/tropomi_data.csv", cache_dir="data/cache/tropomi"):
        """
        Baixa, processa e agrega dados TROPOMI dia a dia.
        Salva incrementalmente no CSV e limpa o cache.
        """
        # Carregar shapefile
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            
        # Simplificar geometria para query: Usar Bounding Box (Box)
        minx, miny, maxx, maxy = gdf.total_bounds
        # Formato WKT: POLYGON((minx miny, maxx miny, maxx maxy, minx maxy, minx miny))
        wkt_poly = f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
        
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Inicializar CSV se não existir
        # Headers esperados: date, no2tropomi_max, no2tropomi_min, no2tropomi_mea, o3tropomi_max, ...
        expected_cols = ['date']
        for pol in ['no2tropomi', 'o3tropomi']:
            for stat in ['max', 'min', 'mea']:
                expected_cols.append(f"{pol}_{stat}")
                
        # Validate existing file header
        if os.path.exists(output_csv):
            try:
                existing_cols = pd.read_csv(output_csv, nrows=0).columns.tolist()
                if existing_cols != expected_cols:
                    LOGGER.warning(f"Schema mismatch in {output_csv}. Expected {expected_cols}, found {existing_cols}. Overwriting.")
                    os.remove(output_csv)
            except Exception as e:
                LOGGER.warning(f"Error reading {output_csv}: {e}. Overwriting.")
                os.remove(output_csv)

        if not os.path.exists(output_csv):
            pd.DataFrame(columns=expected_cols).to_csv(output_csv, index=False)
            
        # Loop dia a dia
        current_date = start_dt
        while current_date <= end_dt:
            day_str = current_date.strftime("%Y-%m-%d")
            LOGGER.info(f"=== Processando data: {day_str} ===")
            
            # Dicionário para armazenar stats do dia
            daily_stats = {'date': current_date}
            
            # Data range para o dia (00:00 a 23:59)
            day_start_iso = current_date.strftime("%Y-%m-%dT00:00:00.000Z")
            day_end_iso = (current_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z")
            
            # Para cada poluente (NO2, O3)
            for pol_key, config in self.PRODUCTS.items():
                LOGGER.info(f"  > Buscando {pol_key.upper()}...")
                
                # Query
                filter_query = (
                    f"Collection/Name eq 'SENTINEL-5P' and "
                    f"ContentDate/Start ge {day_start_iso} and "
                    f"ContentDate/Start lt {day_end_iso} and "
                    f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_poly}') and "
                    f"contains(Name, '{config['product_type']}')"
                )
                
                try:
                    token = self._get_token()
                    headers = {"Authorization": f"Bearer {token}"}
                    query_url = f"{self.CATALOGUE_URL}?$filter={filter_query}&$top=20" # Max 20 files per day per pol usually enough
                    
                    r = requests.get(query_url, headers=headers, timeout=30)
                    r.raise_for_status()
                    products = r.json().get('value', [])
                except Exception as e:
                    LOGGER.error(f"Erro na busca {pol_key}: {e}")
                    continue
                    
                if not products:
                    LOGGER.info(f"    Nenhum produto encontrado.")
                    continue
                    
                LOGGER.info(f"    Encontrados {len(products)} arquivos.")
                
                pol_stats_list = []
                
                # Download e processamento "On the Fly"
                for prod in products:
                    prod_id = prod['Id']
                    prod_name = prod['Name']
                    file_path = os.path.join(cache_dir, f"{prod_name}.nc")
                    
                    try:
                        # Download
                        if not os.path.exists(file_path):
                            download_url = f"{self.DOWNLOAD_URL_BASE}({prod_id})/$value"
                            # Add 60s timeout
                            with requests.get(download_url, headers=headers, stream=True, timeout=60) as r_down:
                                r_down.raise_for_status()
                                temp_name = os.path.join(cache_dir, f"{prod_name}_temp")
                                with open(temp_name, 'wb') as f:
                                    shutil.copyfileobj(r_down.raw, f)
                                os.rename(temp_name, file_path)
                        
                        # Processar
                        stats = self._process_product(file_path, pol_key, config, gdf)
                        if stats:
                            pol_stats_list.append(stats)
                            
                    except Exception as e:
                        LOGGER.error(f"Erro processando {prod_name}: {e}")
                    finally:
                        # APAGAR CACHE IMEDIATAMENTE
                        # Clean up original downloaded file if it exists
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except Exception:
                                pass
                        
                        # Clean up temp file if it exists (using variable defined in try scope if possible)
                        # We use 'prod_name' which is defined in the loop
                        try:
                            possible_temp_name = os.path.join(cache_dir, f"{prod_name}_temp")
                            if os.path.exists(possible_temp_name):
                                os.remove(possible_temp_name)
                        except Exception:
                            pass

                # Média do dia para este poluente
                if pol_stats_list:
                    df_pol = pd.DataFrame(pol_stats_list)
                    # Colunas retornadas: no2_max, no2_min, etc.
                    # Renomear para tropomi
                    cols_renamed = {}
                    for c in df_pol.columns:
                        cols_renamed[c] = f"{pol_key}tropomi_{c.split('_')[1]}"
                    df_pol = df_pol.rename(columns=cols_renamed)
                    
                    # Calcular médias das passagens
                    mean_stats = df_pol.mean(numeric_only=True).to_dict()
                    daily_stats.update(mean_stats)
            
            # Salvar dia no CSV se houver dados
            if len(daily_stats) > 1: # Tem mais que só a data
                df_day = pd.DataFrame([daily_stats])
                # Reordenar colunas para bater com header (se possível)
                # Adicionar colunas faltantes com NaN
                for col in expected_cols:
                    if col not in df_day.columns:
                        df_day[col] = np.nan
                
                df_day = df_day[expected_cols]
                
                # Append com header=False
                df_day.to_csv(output_csv, mode='a', header=False, index=False)
                LOGGER.info(f"Dados de {day_str} salvos.")
            else:
                LOGGER.warning(f"Sem dados completos para {day_str}.")
            
            current_date += timedelta(days=1)
            
        return pd.read_csv(output_csv)

    def _process_product(self, nc_path: str, pol_name: str, config: dict, gdf_roi: gpd.GeoDataFrame) -> Optional[dict]:
        """
        Abre o NetCDF, filtra por QA e ROI, calcula estatísticas.
        """
        try:
            actual_nc_path = nc_path
            temp_dir = None
            
            # Verificar se é ZIP
            if zipfile.is_zipfile(nc_path):
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(nc_path, 'r') as zip_ref:
                    # Procurar o arquivo .nc dentro
                    nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                    if nc_files:
                        zip_ref.extract(nc_files[0], temp_dir)
                        actual_nc_path = os.path.join(temp_dir, nc_files[0])
                    else:
                        # Talvez extrair tudo
                        zip_ref.extractall(temp_dir)
                        pass

            
            # --- CRITICAL SECTION: HDF5 ACCESS ---
            from src.utils.locks import HDF5_LOCK
            with HDF5_LOCK:
                try:
                    ds = xr.open_dataset(actual_nc_path, group='PRODUCT')
                except Exception:
                    # Se falhar, pode não ter grupo PRODUCT ou estar corrompido
                    # Tentar abrir root
                    ds = xr.open_dataset(actual_nc_path)

                # Carregar variáveis
                if 'latitude' in ds:
                    lat = ds['latitude'].values[0]
                    lon = ds['longitude'].values[0]
                    val = ds[config['variable']].values[0]
                    qa = ds['qa_value'].values[0]
                else:
                    # Fallback se estrutura for diferente
                    LOGGER.warning(f"Estrutura inesperada no arquivo {nc_path}")
                    ds.close()
                    return None
                
                ds.close()

            
            if temp_dir:
                shutil.rmtree(temp_dir)
            
            # Filtro de Qualidade
            mask_qa = qa > config['qa_threshold']
            
            # Filtro Bbox Rápido (numpy)
            minx, miny, maxx, maxy = gdf_roi.total_bounds
            mask_bbox = (lat >= miny) & (lat <= maxy) & (lon >= minx) & (lon <= maxx)
            
            final_mask = mask_qa & mask_bbox
            
            if not np.any(final_mask):
                return None
            
            # Extrair dados válidos
            valid_vals = val[final_mask]
            valid_lats = lat[final_mask]
            valid_lons = lon[final_mask]
            
            points = [Point(x, y) for x, y in zip(valid_lons, valid_lats)]
            gdf_points = gpd.GeoDataFrame({'val': valid_vals}, geometry=points, crs=gdf_roi.crs)
            
            # Clip espacial exato
            gdf_clipped = gpd.clip(gdf_points, gdf_roi)
            
            if gdf_clipped.empty:
                return None
            
            data_values = gdf_clipped['val'].values.astype(float)
            
            # Conversão de Unidade
            if config['unit_convert']:
                data_values = data_values * self.MOL_M2_TO_DU
            
            return {
                f"{pol_name}_max": float(np.max(data_values)),
                f"{pol_name}_min": float(np.min(data_values)),
                f"{pol_name}_mea": float(np.mean(data_values))
            }

        except Exception as e:
            LOGGER.error(f"Erro ao processar {nc_path}: {e}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Exemplo
    user = os.getenv("COPERNICUS_USER")
    pwd = os.getenv("COPERNICUS_PASSWORD")
    
    if not user:
        print("Defina COPERNICUS_USER e COPERNICUS_PASSWORD em .env")
    else:
        downloader = TropomiDownloader(user, pwd)
        # Teste com São Paulo (Jan 2023)
        shp = "data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp"
        if not os.path.exists(shp):
            print(f"Shapefile não encontrado: {shp}")
        else:
            downloader.fetch_data(shp, "01/01/2023", "05/01/2023", output_csv="data/output/tropomi/tropomi_test.csv")
