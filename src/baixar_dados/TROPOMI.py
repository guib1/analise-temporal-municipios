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
        "no2omi": {
            "product_type": "L2__NO2___",
            "variable": "nitrogendioxide_tropospheric_column",
            "qa_threshold": 0.75,
            "unit_convert": True
        }
    }

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.getenv("COPERNICUS_USER")
        self.password = password or os.getenv("COPERNICUS_PASSWORD")

        if not self.username or not self.password:
            raise ValueError("Credenciais Copernicus não encontradas. Defina COPERNICUS_USER e COPERNICUS_PASSWORD.")
        
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
            response = requests.post(self.AUTH_URL, data=data)
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

    def fetch_data(
        self,
        shapefile_path: str,
        start_date: str,
        end_date: str,
        output_csv: str = "data/output/tropomi/tropomi_data.csv",
        cache_dir: str = "data/cache/tropomi"
    ) -> pd.DataFrame:
        """
        Baixa, processa e agrega dados TROPOMI.
        """
        start_dt = datetime.strptime(start_date, "%d/%m/%Y")
        end_dt = datetime.strptime(end_date, "%d/%m/%Y") + timedelta(days=1)
        
        # Formato ISO para OData
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Ler shapefile
        gdf = gpd.read_file(shapefile_path)
        # Converter geometria para WKT Polygon para o filtro OData
        # OData espera: geography'SRID=4326;POLYGON((...))'
        try:
            geom = gdf.geometry.union_all()
        except AttributeError:
            geom = gdf.geometry.unary_union
            
        if geom.geom_type != 'Polygon':
            geom = geom.convex_hull # Simplificação se for MultiPolygon complexo
            
        wkt_poly = wkt_dumps(geom)
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        all_stats = []

        for pol_name, config in self.PRODUCTS.items():
            LOGGER.info(f"Buscando produtos {pol_name.upper()} ({config['product_type']})...")
            
            # Construir query OData
            # Filtra por Coleção, Data, Interseção e Tipo de Produto (no nome)
            filter_query = (
                f"Collection/Name eq 'SENTINEL-5P' and "
                f"ContentDate/Start ge {start_iso} and "
                f"ContentDate/Start lt {end_iso} and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_poly}') and "
                f"contains(Name, '{config['product_type']}')"
            )
            
            token = self._get_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            try:
                # Busca paginada (simplificada para primeira página por enquanto, max 1000)
                query_url = f"{self.CATALOGUE_URL}?$filter={filter_query}&$top=100&$orderby=ContentDate/Start asc"
                LOGGER.debug(f"Query URL: {query_url}")
                
                r = requests.get(query_url, headers=headers)
                r.raise_for_status()
                results = r.json()
                products = results.get('value', [])
                
            except Exception as e:
                LOGGER.error(f"Erro na busca OData: {e}")
                continue

            if not products:
                LOGGER.warning(f"Nenhum produto encontrado para {pol_name}")
                continue
            
            LOGGER.info(f"Encontrados {len(products)} produtos para {pol_name}. Iniciando download e processamento...")

            for prod in products:
                prod_id = prod['Id']
                prod_name = prod['Name']
                file_path = os.path.join(cache_dir, f"{prod_name}.nc") 
                
                if not os.path.exists(file_path):
                    LOGGER.info(f"Baixando {prod_name}...")
                    download_url = f"{self.DOWNLOAD_URL_BASE}({prod_id})/$value"
                    
                    try:
                        # Stream download
                        with requests.get(download_url, headers=headers, stream=True) as r_down:
                            r_down.raise_for_status()
                            # Salvar temporariamente
                            temp_name = os.path.join(cache_dir, f"{prod_name}_temp")
                            with open(temp_name, 'wb') as f:
                                shutil.copyfileobj(r_down.raw, f)
                            
                            # Renomear para .nc
                            os.rename(temp_name, file_path)
                            
                    except Exception as e:
                        LOGGER.error(f"Erro ao baixar {prod_name}: {e}")
                        if os.path.exists(temp_name): os.remove(temp_name)
                        continue
                
                # Processar
                stats = self._process_product(file_path, pol_name, config, gdf)
                if stats:
                    acq_date_str = prod['ContentDate']['Start']
                    # Parse ISO date
                    acq_date = datetime.fromisoformat(acq_date_str.replace('Z', '+00:00')).date()
                    stats['date'] = acq_date
                    all_stats.append(stats)

        if not all_stats:
            return pd.DataFrame()

        # Consolidar
        df_final = pd.DataFrame(all_stats)
        
        # Agrupar por data
        df_daily = df_final.groupby('date').mean().reset_index()
        
        # Renomear colunas
        rename_map = {}
        for col in df_daily.columns:
            if col == 'date': continue
            parts = col.split('_')
            if len(parts) == 2:
                pol, stat = parts
                if pol.endswith('omi'):
                    rename_map[col] = col
                else:
                    rename_map[col] = f"{pol}tropomi_{stat}"
        
        df_daily = df_daily.rename(columns=rename_map)
        
        # Salvar
        df_daily.to_csv(output_csv, index=False)
        LOGGER.info(f"Dados salvos em {output_csv}")
        
        return df_daily

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
        # Exemplo fictício
        downloader.fetch_data("data/shapefiles/SP-Diadema/SP_Diadema.shp", "01/01/2024", "02/01/2024")
