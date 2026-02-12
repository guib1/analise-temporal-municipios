import io
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.utils.geo import parse_date, format_ddmmyyyy

# Carrega variáveis de ambiente
load_dotenv()

LOGGER = logging.getLogger(__name__)

class CETESBDownloader:    
    LOGIN_URL = "https://qualar.cetesb.sp.gov.br/qualar/autenticador"
    EXPORT_URL = "https://qualar.cetesb.sp.gov.br/qualar/exportaDados.do"
    HOME_URL = "https://qualar.cetesb.sp.gov.br/qualar/home.do"
    
    # IDs dos parâmetros no QUALAR
    PARAMS = {
        "pm10": 12,   # Partículas Inaláveis
        "pm25": 57,   # Partículas Inaláveis Finas
        "o3": 63,     # Ozônio
        "no": 17,     # Monóxido de Nitrogênio
        "no2": 15,    # Dióxido de Nitrogênio
        "nox": 18,    # Óxidos de Nitrogênio
        "so2": 13,    # Dióxido de Enxofre
    }

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.getenv("CETESB_USER")
        self.password = password or os.getenv("CETESB_PASSWORD")
        
        if not self.username or not self.password:
            raise ValueError(
                "Credenciais CETESB não encontradas. "
                "Defina CETESB_USER e CETESB_PASSWORD no arquivo .env."
            )
            
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, como Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        self.station_map = {}
        
        # Carrega estações do CSV
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'estacoes_cetesb.csv')
        try:
            self.stations_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            LOGGER.error(f"Arquivo de estações não encontrado em {csv_path}")
            self.stations_df = pd.DataFrame()

    def login(self):
        """Realiza login no sistema QUALAR."""
        LOGGER.info("Autenticando no QUALAR/CETESB...")
        try:
            # Acessa página de login para cookies (home.do contém o formulário)
            self.session.get(self.HOME_URL, timeout=30)
            
            payload = {
                "cetesb_login": self.username,
                "cetesb_password": self.password,
                "enviar": "OK"
            }
            
            response = self.session.post(self.LOGIN_URL, data=payload, timeout=30)
            
            # Verifica redirecionamento ou sucesso
            if response.url == self.HOME_URL or "Logoff" in response.text or "Bem Vindo" in response.text:
                LOGGER.info("Login realizado com sucesso.")
            elif response.status_code == 302:
                 # Se houve redirect não seguido automaticamente (raro com requests, mas possível se configurado)
                 LOGGER.info("Redirecionamento detectado no login.")
            else:
                # Tenta identificar erro
                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.get_text()
                if "inválida" in text_content or "não cadastrado" in text_content:
                    raise ValueError(f"Erro no login: Credenciais inválidas ou usuário não cadastrado.")
                
                # Se não achou erro explícito, mas não está na home, avisa
                LOGGER.warning("Login pode ter falhado (indicador de sucesso não encontrado).")
                LOGGER.warning(f"URL Final: {response.url}")
                
        except Exception as e:
            LOGGER.error(f"Falha ao logar no CETESB: {e}")
            raise

    def _get_nearest_station(self, gdf: gpd.GeoDataFrame) -> Optional[int]:
        """Encontra a estação mais próxima do centroide do município."""
        if self.stations_df.empty:
            LOGGER.error("Lista de estações vazia.")
            return None

        # Verifica se é SP
        if 'abbrev_sta' not in gdf.columns or gdf.iloc[0]['abbrev_sta'] != 'SP':
            LOGGER.warning("O shapefile fornecido não é do estado de São Paulo (SP). Operação cancelada.")
            return None
            
        # Calcula centroide
        # O CRS original é geográfico (lat/lon), então o centroide é em graus.
        # Para distância precisa, idealmente projetaríamos, mas para "mais próximo" em SP,
        # a distância euclidiana em graus funciona razoavelmente bem.
        centroid = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326).iloc[0]
        
        # Calcula distância Euclidiana simples (graus)
        # d = sqrt((lat1-lat2)^2 + (lon1-lon2)^2)
        self.stations_df['distance'] = ((self.stations_df['lat'] - centroid.y)**2 + (self.stations_df['lon'] - centroid.x)**2)**0.5
        
        # Pega a mais próxima
        nearest = self.stations_df.loc[self.stations_df['distance'].idxmin()]
        
        LOGGER.info(f"Estação mais próxima: {nearest['name']} (Distância aprox: {nearest['distance']:.4f} graus)")
        return int(nearest['code'])

    def fetch_data(
        self,
        shapefile_path: str,
        start_date: str,
        end_date: str,
        output_csv: str = "data/output/cetesb/cetesb_data.csv"
    ) -> pd.DataFrame:
        """
        Baixa dados, agrega e salva em CSV.
        """
        # Login
        self.login()
        
        # Identificar Município
        gdf = gpd.read_file(shapefile_path)
        
        # Encontrar estação mais próxima
        station_id = self._get_nearest_station(gdf)
        
        if not station_id:
            LOGGER.error("Não foi possível determinar a estação CETESB.")
            return pd.DataFrame()
            
        LOGGER.info(f"Usando estação ID: {station_id}")
        
        # Ensure dates are dd/mm/yyyy
        start_date_fmt = format_ddmmyyyy(start_date)
        end_date_fmt = format_ddmmyyyy(end_date)

        all_dfs = []

        def _pick_qualar_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
            """Pick the best QUALAR table containing Date+Time+Value columns.

            QUALAR pages may include summary tables (e.g., weekly). We prefer the
            table that contains both a date column ("Data"), a time column ("Hora"),
            and a value column ("Média Horária" or "Concentração").
            """
            for t in tables:
                if t is None or t.empty:
                    continue
                cols = [str(c).strip() for c in t.columns]
                has_date = any('Data' in c for c in cols)
                has_time = any('Hora' in c for c in cols)
                has_value = any(('Média Horária' in c) or ('Concentração' in c) for c in cols)
                if has_date and has_time and has_value:
                    return t
            # Fallback: if none match perfectly, return the first non-empty table
            for t in tables:
                if t is not None and not t.empty:
                    return t
            return None
        
        for var_name, param_id in self.PARAMS.items():
            LOGGER.info(f"Baixando {var_name} (ID {param_id})...")
            
            # 1. Initialize Export Page (Important for session state)
            init_url = f"{self.EXPORT_URL}?method=pesquisarInit"
            self.session.headers.update({"Referer": self.HOME_URL})
            self.session.get(init_url, timeout=30)
            
            # 2. POST Data Request
            post_url = f"{self.EXPORT_URL}?method=pesquisar"
            
            payload = {
                "irede": "A", # Automática
                "dataInicialStr": start_date_fmt,
                "dataFinalStr": end_date_fmt,
                "iTipoDado": "P", # Média Horária (P)
                # Importante: NÃO incluir dados inválidos (evita zeros/sentinelas afetando min/média)
                "estacaoVO.nestcaMonto": str(station_id),
                "parametroVO.nparmt": str(param_id),
                "btnPesquisar": "Pesquisar"
            }
            
            try:
                # Update Referer to the init page
                self.session.headers.update({"Referer": init_url})
                resp = self.session.post(post_url, data=payload, timeout=60)
                
                if resp.status_code != 200:
                    LOGGER.error(f"Erro HTTP {resp.status_code} ao baixar {var_name}")
                    continue

                # Parse HTML Tables
                dfs: List[pd.DataFrame] = []
                for hdr in (1, 0):
                    try:
                        dfs = pd.read_html(io.StringIO(resp.text), decimal=',', thousands='.', header=hdr)
                        if dfs:
                            break
                    except ValueError:
                        dfs = []
                if not dfs:
                    LOGGER.warning(f"Nenhuma tabela encontrada no HTML para {var_name}")
                    continue
                
                df = _pick_qualar_table(dfs)
                if df is None or df.empty:
                    LOGGER.warning(f"Sem dados retornados para {var_name}")
                    continue
                
                # Check for "Nenhum Registro Encontrado"
                if "Nenhum Registro Encontrado" in str(df.columns) or (not df.empty and "Nenhum Registro Encontrado" in str(df.iloc[0].values)):
                    LOGGER.warning(f"Nenhum registro encontrado para {var_name}")
                    continue

                # Clean columns
                df.columns = [str(c).strip() for c in df.columns]
                
                # Identify columns
                date_col = None
                time_col = None
                value_col = None
                
                for col in df.columns:
                    if "Data" in col:
                        date_col = col
                    elif "Hora" in col:
                        time_col = col
                
                # Priority selection for value column
                # "Média Horária" is preferred for hourly data (iTipoDado="P")
                # "Concentração" is a fallback
                if "Média Horária" in df.columns:
                    value_col = "Média Horária"
                elif "Concentração" in df.columns:
                    value_col = "Concentração"
                else:
                    # Fallback: search for substrings if exact match fails
                    for col in df.columns:
                        if "Média Horária" in col or "Concentração" in col:
                            value_col = col
                            break
                
                if not date_col or not time_col or not value_col:
                    # Fallback to indices if names don't match
                    # Expected: 4=Data, 5=Hora, 10=Value
                    if len(df.columns) > 10:
                        LOGGER.warning(f"Colunas não identificadas pelo nome. Tentando índices fixos.")
                        date_col = df.columns[4]
                        time_col = df.columns[5]
                        value_col = df.columns[10]
                    else:
                        LOGGER.warning(f"Estrutura da tabela inesperada para {var_name}. Colunas: {df.columns.tolist()}")
                        continue

                # Limpeza
                df = df.rename(columns={value_col: 'value'})
                
                # Replace '-' with NA
                df['value'] = df['value'].replace('-', pd.NA)
                
                LOGGER.info(f"Sample values before conversion for {var_name}: {df['value'].head().tolist()}")
                
                # Force string conversion then replace comma with dot if needed
                if df['value'].dtype == 'object':
                     df['value'] = df['value'].astype(str).str.replace(',', '.', regex=False)
                
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                LOGGER.info(f"Sample values after conversion for {var_name}: {df['value'].head().tolist()}")
                
                # Combine Date and Time
                try:
                    df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], format='%d/%m/%Y %H:%M', errors='coerce')
                    df['date'] = df['datetime'].dt.date
                except Exception:
                     df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce').dt.date
                
                df = df.dropna(subset=['date'])
                
                # Agregação Diária
                daily = df.groupby('date')['value'].agg(['max', 'min', 'mean']).reset_index()
                daily.columns = ['date', f'{var_name}_max', f'{var_name}_min', f'{var_name}_mea']
                
                all_dfs.append(daily)
                
                # Delay para não sobrecarregar
                time.sleep(1)
                
            except Exception as e:
                LOGGER.error(f"Erro ao processar {var_name}: {e}")
                continue
                
        if not all_dfs:
            LOGGER.warning("Nenhum dado obtido. Gerando DataFrame com colunas vazias (NaN).")
            try:
                d_start = datetime.strptime(start_date_fmt, "%d/%m/%Y")
                d_end = datetime.strptime(end_date_fmt, "%d/%m/%Y")
                date_range = pd.date_range(start=d_start, end=d_end, freq='D')
                final_df = pd.DataFrame({'date': date_range.date})
            except Exception as e:
                LOGGER.error(f"Erro ao gerar datas: {e}")
                return pd.DataFrame()
        else:
            # Merge all dataframes
            final_df = all_dfs[0]
            for df in all_dfs[1:]:
                final_df = pd.merge(final_df, df, on='date', how='outer')

        # Garantir que todas as colunas existam (mantendo vazio/NaN quando não houver dado)
        for var_name in self.PARAMS.keys():
            for metric in ['max', 'min', 'mea']:
                col_name = f'{var_name}_{metric}'
                if col_name not in final_df.columns:
                    final_df[col_name] = np.nan
            
        # Save to CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        LOGGER.info(f"Dados salvos em {output_csv}")
        
        return final_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Exemplo de uso
    dl = CETESBDownloader()
    dl.fetch_data("data/shapefiles/SP-São_Paulo/SP_São_Paulo.shp", "01/01/2008", "01/02/2008")
