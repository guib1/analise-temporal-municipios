import logging
import os
import time
import json
import requests
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.utils.geo import parse_date

# Carrega variáveis de ambiente
load_dotenv()

LOGGER = logging.getLogger(__name__)

class ModisDownloader:
    """
    Baixa e processa dados do MODIS (AOD - Aerosol Optical Depth) usando a API do AppEEARS.
    Produto recomendado: MCD19A2 (1km resolution) ou MOD04_L2 (10km).
    """
    
    APPEEARS_API_URL = "https://appeears.earthdatacloud.nasa.gov/api"
    
    def __init__(self):
        self.user = os.getenv("NASA_USER")
        self.password = os.getenv("NASA_PASSWORD")
        self.token = None
        
        if not self.user or not self.password:
            LOGGER.warning("Credenciais EARTHDATA_USER e EARTHDATA_PASSWORD não encontradas no .env. O download automático falhará.")

    def _login(self):
        """Realiza login na API do AppEEARS para obter o token."""
        if self.token:
            return self.token
            
        LOGGER.info("Autenticando no NASA Earthdata (AppEEARS)...")
        try:
            response = requests.post(
                f"{self.APPEEARS_API_URL}/login", 
                auth=(self.user, self.password)
            )
            response.raise_for_status()
            self.token = response.json()['token']
            LOGGER.info("Autenticação realizada com sucesso.")
            return self.token
        except Exception as e:
            LOGGER.error(f"Falha na autenticação AppEEARS: {e}")
            raise

    def _submit_task(self, task_name, shapefile_path, start_date, end_date, layers):
        """Submete uma tarefa de processamento no AppEEARS."""
        token = self._login()
        headers = {'Authorization': f'Bearer {token}'}
        
        # Ler shapefile e converter para GeoJSON
        gdf = gpd.read_file(shapefile_path)
        # Reprojetar para WGS84 se necessário
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        # Simplificar geometria se for muito complexa para a API
        # O AppEEARS aceita GeoJSON feature collection
        geojson = json.loads(gdf.to_json())
        
        # Formatar datas (MM-DD-YYYY)
        start_dt = parse_date(start_date).strftime("%m-%d-%Y")
        end_dt = parse_date(end_date).strftime("%m-%d-%Y")
        
        task = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [
                    {
                        "startDate": start_dt,
                        "endDate": end_dt
                    }
                ],
                "layers": layers,
                "output": {
                    "format": {
                        "type": "netcdf4"
                    },
                    "projection": "geographic"
                },
                "geo": geojson
            }
        }
        
        LOGGER.info(f"Submetendo tarefa '{task_name}' para o AppEEARS...")
        response = requests.post(
            f"{self.APPEEARS_API_URL}/task", 
            json=task, 
            headers=headers
        )
        
        if response.status_code == 202:
            task_id = response.json()['task_id']
            LOGGER.info(f"Tarefa submetida com sucesso. ID: {task_id}")
            return task_id
        else:
            LOGGER.error(f"Erro ao submeter tarefa: {response.text}")
            raise Exception(f"AppEEARS Error: {response.text}")

    def _wait_for_task(self, task_id):
        """Aguarda a conclusão da tarefa."""
        token = self._login()
        headers = {'Authorization': f'Bearer {token}'}
        
        while True:
            response = requests.get(
                f"{self.APPEEARS_API_URL}/task/{task_id}", 
                headers=headers
            )
            status = response.json()['status']
            LOGGER.info(f"Status da tarefa {task_id}: {status}")
            
            if status == 'done':
                return True
            elif status == 'error':
                LOGGER.error("Tarefa falhou no servidor.")
                return False
            
            time.sleep(60) # Verifica a cada 1 minuto

    def _download_bundle(self, task_id, download_dir):
        """Baixa os arquivos resultantes."""
        token = self._login()
        headers = {'Authorization': f'Bearer {token}'}
        
        response = requests.get(
            f"{self.APPEEARS_API_URL}/bundle/{task_id}", 
            headers=headers
        )
        bundle = response.json()
        
        files_downloaded = []
        os.makedirs(download_dir, exist_ok=True)
        
        for file_info in bundle['files']:
            file_id = file_info['file_id']
            filename = file_info['file_name']
            
            # Baixar apenas NetCDF
            if not filename.endswith('.nc'):
                continue
                
            filepath = os.path.join(download_dir, filename)
            if os.path.exists(filepath):
                LOGGER.info(f"Arquivo já existe: {filename}")
                files_downloaded.append(filepath)
                continue
                
            LOGGER.info(f"Baixando {filename}...")
            stream = requests.get(
                f"{self.APPEEARS_API_URL}/bundle/{task_id}/{file_id}", 
                headers=headers, 
                stream=True
            )
            
            with open(filepath, 'wb') as f:
                for chunk in stream.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            
            files_downloaded.append(filepath)
            
        return files_downloaded

    def process_files(self, nc_files, output_csv):
        """
        Processa os arquivos NetCDF baixados e gera o CSV diário.
        Calcula Max, Min, Mean para AOD.
        """
        results = []
        
        for nc_path in nc_files:
            try:
                ds = xr.open_dataset(nc_path)
                
                # O nome da variável depende do produto. 
                # Para MCD19A2 é geralmente 'Optical_Depth_055' ou similar.
                # O AppEEARS costuma manter o nome original ou prefixar.
                
                # Procurar variável de AOD
                aod_var = None
                for var in ds.data_vars:
                    if 'Optical_Depth' in var or 'AOD' in var:
                        aod_var = var
                        break
                
                if not aod_var:
                    LOGGER.warning(f"Variável de AOD não encontrada em {nc_path}")
                    continue
                
                # Extrair data do nome do arquivo ou metadados
                # AppEEARS filename format: MCD19A2.A2023001...
                filename = os.path.basename(nc_path)
                try:
                    # Tenta extrair data do nome (ex: MCD19A2.AYYYYDDD...)
                    # Ajuste conforme o padrão exato retornado
                    parts = filename.split('.')
                    if len(parts) > 1 and parts[1].startswith('A'):
                        year_doy = parts[1][1:8] # YYYYDDD
                        date_obj = datetime.strptime(year_doy, "%Y%j").date()
                    else:
                        # Tenta pegar do tempo dentro do NetCDF
                        if 'time' in ds.coords:
                            date_obj = pd.to_datetime(ds.time.values[0]).date()
                        else:
                            continue
                except:
                    LOGGER.warning(f"Não foi possível extrair data de {filename}")
                    continue

                data = ds[aod_var].values
                # Filtrar valores inválidos (fill value)
                valid_data = data[~np.isnan(data)]
                
                if valid_data.size > 0:
                    row = {
                        'date': date_obj,
                        'aodterramodis_max': float(np.max(valid_data)),
                        'aodterramodis_min': float(np.min(valid_data)),
                        'aodterramodis_mea': float(np.mean(valid_data))
                    }
                    results.append(row)
                    
                ds.close()
                
            except Exception as e:
                LOGGER.error(f"Erro ao processar {nc_path}: {e}")

        if results:
            df = pd.DataFrame(results)
            # Agrupar por data (caso haja múltiplos tiles/passagens no mesmo dia)
            df_daily = df.groupby('date').agg({
                'aodterramodis_max': 'max',
                'aodterramodis_min': 'min',
                'aodterramodis_mea': 'mean'
            }).reset_index()
            
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_daily.to_csv(output_csv, index=False)
            LOGGER.info(f"Dados MODIS salvos em {output_csv}")
            return df_daily
        else:
            LOGGER.warning("Nenhum dado processado.")
            return pd.DataFrame()

    def fetch_data(self, shapefile_path, start_date, end_date, output_csv="data/output/modis/modis_data.csv"):
        """
        Fluxo principal: Submete tarefa -> Baixa -> Processa.
        """
        # Produto MCD19A2 (MAIAC) - AOD 550nm
        # Layer: Optical_Depth_047_055 (Blue band AOD, often used as proxy or check specific 550nm layer)
        # Para MCD19A2 V6.1, a layer comum é 'Optical_Depth_047_055' (Green band AOD)
        # Ou MOD04_L2: 'Optical_Depth_Land_And_Ocean'
        
        product_layer = [
            {
                "product": "MCD19A2.061",
                "layer": "Optical_Depth_047_055" 
            }
        ]
        
        task_name = f"MODIS_AOD_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            # 1. Submeter e Baixar (Requer credenciais)
            if self.user and self.password:
                task_id = self._submit_task(task_name, shapefile_path, start_date, end_date, product_layer)
                if self._wait_for_task(task_id):
                    download_dir = os.path.join("data", "cache", "modis_nc")
                    nc_files = self._download_bundle(task_id, download_dir)
                    return self.process_files(nc_files, output_csv)
            else:
                LOGGER.warning("Sem credenciais. Tentando processar arquivos locais em data/cache/modis_nc se existirem.")
                download_dir = os.path.join("data", "cache", "modis_nc")
                if os.path.exists(download_dir):
                    nc_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith('.nc')]
                    return self.process_files(nc_files, output_csv)
                else:
                    LOGGER.error("Nenhum arquivo local encontrado e sem credenciais para baixar.")
                    return pd.DataFrame()
                    
        except Exception as e:
            LOGGER.error(f"Falha no fluxo MODIS: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Exemplo de uso
    dl = ModisDownloader()
    dl.fetch_data("data/shapefiles/SP-Diadema/SP_Diadema.shp", "01/01/2023", "05/01/2023")
