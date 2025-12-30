import logging
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import scipy.stats as stats
from pathlib import Path

LOGGER = logging.getLogger(__name__)

class DiversosDownloader:
    """
    Calcula índices meteorológicos e climáticos (Heat Index, Ondas de Calor/Frio, SPI, El Niño).
    Utiliza dados do INMET como base para os cálculos meteorológicos.
    """
    
    def __init__(self):
        self.inmet_downloader = None

    def _get_inmet_downloader(self):
        if self.inmet_downloader is None:
            from INMET import INMETDownloader  # import local para evitar cache antigo em notebook
            self.inmet_downloader = INMETDownloader()
        return self.inmet_downloader

    def _calculate_heat_index(self, row):
        """
        Calcula Heat Index (Índice de Calor) usando a equação de regressão do NWS/NOAA.
        T deve estar em Fahrenheit e RH em %.
        Retorna em Celsius.
        """
        T_c = row.get('temp_max', np.nan) # Usando temp máxima para o HI máximo
        RH = row.get('hum_min', np.nan)   # Usando hum mínima (geralmente ocorre na Tmax) - Aproximação
        
        # Se não tivermos hum_min, tentamos hum_mean
        if pd.isna(RH):
            RH = row.get('hum_mean', np.nan)
            
        if pd.isna(T_c) or pd.isna(RH):
            return np.nan
            
        # Conversão para Fahrenheit
        T = (T_c * 9/5) + 32
        
        # Simple formula first
        HI = 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (RH*0.094))
        
        # Full regression if HI > 80 F
        if HI > 80:
            HI = -42.379 + 2.04901523*T + 10.14333127*RH - .22475541*T*RH - .00683783*T*T - .05481717*RH*RH + .00122874*T*T*RH + .00085282*T*RH*RH - .00000199*T*T*RH*RH
            
            if (RH < 13) and (T > 80) and (T < 112):
                adj = ((13-RH)/4)*np.sqrt((17-abs(T-95.))/17)
                HI -= adj
            elif (RH > 85) and (T > 80) and (T < 87):
                adj = ((RH-85)/10) * ((87-T)/5)
                HI += adj
                
        # Convert back to Celsius
        return (HI - 32) * 5/9

    def _get_elnino_data(self, start_date, end_date):
        """
        Baixa dados do ONI (Oceanic Nino Index) da NOAA.
        """
        url = "https://psl.noaa.gov/data/correlation/oni.data"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                LOGGER.error("Falha ao baixar dados do El Niño")
                return pd.DataFrame()
            
            lines = response.text.split('\n')
            data = []
            # O arquivo tem um cabeçalho e rodapé que precisam ser ignorados
            # Formato: Year  Jan Feb Mar ...
            
            start_reading = False
            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) == 4:
                    year = int(parts[0])
                    if 1950 <= year <= 2030: # Filtro básico de validade
                        # parts[1:] são os meses
                        for month, val in enumerate(parts[1:], 1):
                            try:
                                val_float = float(val)
                                if val_float > -99: # -99.90 é missing
                                    # O usuário quer o valor do índice (ONI), não apenas a classificação binária
                                    date_obj = datetime(year, month, 1)
                                    data.append({'date': date_obj, 'elnino_ind': val_float})
                            except ValueError:
                                continue
                                
            df_oni = pd.DataFrame(data)
            # Remove duplicates if any
            df_oni = df_oni.drop_duplicates(subset=['date'])
            
            # Expandir para diário (ffill)
            if not df_oni.empty:
                full_idx = pd.date_range(start=df_oni['date'].min(), end=df_oni['date'].max() + pd.offsets.MonthEnd(1), freq='D')
                df_oni = df_oni.set_index('date').reindex(full_idx, method='ffill').reset_index().rename(columns={'index': 'date'})
                df_oni['date'] = df_oni['date'].dt.date
                return df_oni
                
        except Exception as e:
            LOGGER.error(f"Erro ao processar El Niño: {e}")
            
        return pd.DataFrame()

    def _calculate_spi(self, df, precip_col='precip_total', period=30):
        """
        Calcula SPI (Standardized Precipitation Index).
        Baseado em: McKee, T. B., N. J. Doesken, and J. Kleist. 1993.
        
        NOTA: O SPI requer uma série histórica longa (30-50 anos) para ajustar corretamente
        a distribuição Gamma para cada mês. O cálculo abaixo agrupa os dados por mês
        disponíveis no DataFrame. Se o DataFrame for curto (ex: 1 ano), o resultado
        será apenas uma aproximação da anomalia dentro desse próprio período, 
        não um SPI climatológico real.
        """
        if precip_col not in df.columns or len(df) < period:
            return pd.Series(np.nan, index=df.index)
            
        # 1. Calcular soma móvel (Rolling Sum)
        # O SPI é calculado sobre totais acumulados móveis (ex: 30 dias)
        df_working = df.copy()
        df_working['rolling_precip'] = df_working[precip_col].rolling(window=period, min_periods=period).sum()
        
        # Garantir que temos coluna de mês para sazonalidade
        if 'date' in df_working.columns:
            # Se for string, converte. Se já for date/datetime, extrai mês.
            dates = pd.to_datetime(df_working['date'])
            df_working['month'] = dates.dt.month
        else:
            # Se não tiver data, não podemos fazer ajuste sazonal. Retorna NaN ou erro.
            return pd.Series(np.nan, index=df.index)
            
        spi_series = pd.Series(np.nan, index=df_working.index)
        
        # 2. Ajuste da Distribuição Gamma por Mês
        # O SPI compara a precipitação acumulada com a distribuição histórica PARA AQUELE MÊS/PERÍODO.
        for month in range(1, 13):
            # Seleciona dados deste mês (de todos os anos disponíveis)
            mask = df_working['month'] == month
            month_data = df_working.loc[mask, 'rolling_precip'].dropna()
            
            if len(month_data) < 10:
                # Insuficiente para ajuste estatístico
                continue
                
            try:
                # Dados devem ser > 0 para ajuste Gamma
                zeros = month_data[month_data == 0]
                non_zeros = month_data[month_data > 0]
                
                # Probabilidade de chuva zero (q)
                q = len(zeros) / len(month_data)
                
                if len(non_zeros) > 0:
                    # Ajuste dos parâmetros Gamma (alpha=shape, loc, beta=scale)
                    # Fixamos loc=0 pois precipitação não pode ser negativa
                    alpha, loc, beta = stats.gamma.fit(non_zeros, floc=0)
                    
                    # Calcular Probabilidade Acumulada H(x)
                    # H(x) = q + (1-q) * G(x)
                    
                    # Vamos aplicar a transformação para todos os dados do mês
                    current_cdf = np.zeros(len(month_data))
                    
                    # Indices relativos ao slice month_data
                    is_zero = (month_data == 0)
                    is_pos = (month_data > 0)
                    
                    # Para zeros: H(0) = q
                    current_cdf[is_zero] = q
                    
                    # Para positivos
                    current_cdf[is_pos] = q + (1 - q) * stats.gamma.cdf(month_data[is_pos], alpha, loc, beta)
                    
                    # Clip para evitar infinito na transformação Z (norm.ppf(0) = -inf, norm.ppf(1) = inf)
                    current_cdf = np.clip(current_cdf, 0.0001, 0.9999)
                    
                    # 3. Transformação para Normal Padrão (Z-score)
                    spi_vals = stats.norm.ppf(current_cdf)
                    
                    # Atribuir de volta à série original
                    spi_series.loc[month_data.index] = spi_vals
                    
                else:
                    # Caso extremo: Mês inteiro sem chuva em todo o histórico
                    # Se q=1, H(x)=1 -> SPI = inf. 
                    # Se sempre chove 0, 0 é o normal (SPI=0).
                    if q == 1:
                        spi_series.loc[month_data.index] = 0
                        
            except Exception as e:
                LOGGER.warning(f"Erro no cálculo do SPI para mês {month}: {e}")
                
        return spi_series

    def fetch_data(
        self,
        shapefile_path,
        start_date,
        end_date,
        output_csv="data/output/diversos/diversos_data.csv",
        inmet_df: pd.DataFrame | None = None,
    ):
        LOGGER.info("Iniciando processamento de dados diversos...")
        
        # 1. Obter dados base do INMET (Temp, Hum, Precip)
        # Usamos um arquivo temporário para o output do INMET.
        # Precisa ser único para permitir execução multi-município sem sobrescrever.
        df_inmet = pd.DataFrame()
        if inmet_df is not None and not inmet_df.empty:
            df_inmet = inmet_df.copy()
        else:
            shp_stem = Path(str(shapefile_path)).stem
            temp_inmet_csv = f"data/cache/temp_inmet_diversos_{shp_stem}.csv"
            try:
                # INMETDownloader usa fetch_daily_data
                df_inmet = self._get_inmet_downloader().fetch_daily_data(
                    shapefile_path,
                    start_date,
                    end_date,
                    out_csv=temp_inmet_csv,
                )
            except Exception as e:
                LOGGER.error(f"Erro ao baixar dados base do INMET: {e}")
                df_inmet = pd.DataFrame()

        if df_inmet.empty:
            LOGGER.warning("Sem dados meteorológicos base. Gerando colunas vazias.")
            # Cria dataframe vazio com datas
            try:
                dates = pd.date_range(start=datetime.strptime(start_date, "%d/%m/%Y"), 
                                     end=datetime.strptime(end_date, "%d/%m/%Y"))
                df = pd.DataFrame({'date': dates.date})
            except:
                return pd.DataFrame()
        else:
            df = df_inmet.copy()
            # Ensure date is date object
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Renomear colunas do INMET para o padrão esperado aqui
            rename_map = {
                'temperature_max': 'temp_max',
                'temperature_min': 'temp_min',
                'temperature_med': 'temp_mean',  # compat legado
                'temperature_mea': 'temp_mean',
                'humidity_max': 'hum_max',       # não existe no INMET atual (mantido por compat)
                'humidity_min': 'hum_min',
                'humidity_mea': 'hum_mean',
                'precipitation_sum': 'precip_total'
            }
            df.rename(columns=rename_map, inplace=True)

        # 2. Heat Index
        # Se já veio do INMET (cálculo horário preciso), usamos.
        if 'heatindex_max' in df.columns and 'heatindex_min' in df.columns and 'heatindex_mea' in df.columns:
            LOGGER.info("Usando Heat Index calculado pelo INMETDownloader.")
        elif 'temp_max' in df.columns and 'hum_min' in df.columns:
            LOGGER.info("Calculando Heat Index aproximado.")
            df['heatindex_max'] = df.apply(self._calculate_heat_index, axis=1)
            # Para min e mean, precisaríamos de temp_min/hum_max e temp_mean/hum_mean
            # Aproximação:
            df['heatindex_min'] = df.apply(lambda row: self._calculate_heat_index({'temp_max': row.get('temp_min'), 'hum_min': row.get('hum_max')}), axis=1)
            df['heatindex_mea'] = df.apply(lambda row: self._calculate_heat_index({'temp_max': row.get('temp_mean'), 'hum_min': row.get('hum_mean')}), axis=1)
        else:
            df['heatindex_max'] = np.nan
            df['heatindex_min'] = np.nan
            df['heatindex_mea'] = np.nan

        # 3. Ondas de Calor / Frio
        # Definição: Meehl and Tebaldi (2004)
        # Heat Wave:
        # 1. Tmax > T2 (81st percentile) every day
        # 2. Duration >= 3 days
        # 3. Tmax > T1 (97.5th percentile) for at least 3 days
        # 4. Average Tmax > T1
        
        if 'temp_max' in df.columns and pd.to_numeric(df['temp_max'], errors='coerce').notna().any():
            # Calculate percentiles based on the available data
            # Note: Ideally this should be a long-term historical baseline
            T1 = df['temp_max'].quantile(0.975)
            T2 = df['temp_max'].quantile(0.81)
            
            # Identify candidate days (Tmax > T2)
            is_candidate = (df['temp_max'] > T2)
            
            # Group consecutive candidates
            df['group_hw'] = (is_candidate != is_candidate.shift()).cumsum()
            
            # Initialize columns
            df['heatwave_has'] = 0
            df['heatwaveduration_sum'] = 0
            df['heatwaveintensity_ind'] = 0.0
            
            # Process groups
            groups = df[is_candidate].groupby('group_hw')
            
            for _, group in groups:
                # Condition: Duration >= 3
                if len(group) < 3:
                    continue
                
                # Condition: Tmax > T1 for at least 3 days
                if (group['temp_max'] > T1).sum() < 3:
                    continue
                    
                # Condition: Average Tmax > T1
                if group['temp_max'].mean() <= T1:
                    continue
                
                # It is a heat wave
                indices = group.index
                df.loc[indices, 'heatwave_has'] = 1
                
                duration = len(group)
                df.loc[indices, 'heatwaveduration_sum'] = duration
                
                # Intensity: Qtd de dias / temperatura máxima no período
                max_t = group['temp_max'].max()
                intensity = duration / max_t if max_t != 0 else 0
                df.loc[indices, 'heatwaveintensity_ind'] = intensity
                
            # Cleanup aux column
            df.drop(columns=['group_hw'], inplace=True)
                    
        else:
            df['heatwave_has'] = pd.NA
            df['heatwaveduration_sum'] = pd.NA
            df['heatwaveintensity_ind'] = np.nan

        if 'temp_min' in df.columns and pd.to_numeric(df['temp_min'], errors='coerce').notna().any():
            # Cold Wave (Symmetric definition assumption)
            # T1 = 2.5th percentile (symmetric to 97.5)
            # T2 = 19th percentile (symmetric to 81)
            T1_cold = df['temp_min'].quantile(0.025)
            T2_cold = df['temp_min'].quantile(0.19)
            
            # Identify candidate days (Tmin < T2_cold)
            is_candidate_cold = (df['temp_min'] < T2_cold)
            
            df['group_cw'] = (is_candidate_cold != is_candidate_cold.shift()).cumsum()
            
            df['coldwave_has'] = 0
            df['coldwaveduration_sum'] = 0
            df['coldwaveintensity'] = 0.0
            
            groups_cold = df[is_candidate_cold].groupby('group_cw')
            
            for _, group in groups_cold:
                # Condition: Duration >= 3
                if len(group) < 3:
                    continue
                
                # Condition: Tmin < T1_cold for at least 3 days
                if (group['temp_min'] < T1_cold).sum() < 3:
                    continue
                
                # Condition: Average Tmin < T1_cold
                if group['temp_min'].mean() >= T1_cold:
                    continue
                
                # It is a cold wave
                indices = group.index
                df.loc[indices, 'coldwave_has'] = 1
                
                duration = len(group)
                df.loc[indices, 'coldwaveduration_sum'] = duration
                
                # Intensity: Qtd de dias / temperatura mínima (pico de frio)
                min_t = group['temp_min'].min()
                intensity = duration / min_t if min_t != 0 else 0
                df.loc[indices, 'coldwaveintensity'] = intensity
                
            # Cleanup aux column
            df.drop(columns=['group_cw'], inplace=True)
        else:
            df['coldwave_has'] = pd.NA
            df['coldwaveduration_sum'] = pd.NA
            df['coldwaveintensity'] = np.nan

        # 4. SPI
        if 'precip_total' in df.columns and pd.to_numeric(df['precip_total'], errors='coerce').notna().any():
            df['spi_ind'] = self._calculate_spi(df, 'precip_total')
        else:
            df['spi_ind'] = np.nan

        # 5. El Niño
        df_elnino = self._get_elnino_data(start_date, end_date)
        if not df_elnino.empty:
            df = pd.merge(df, df_elnino, on='date', how='left')
            # keep NaN when missing
        else:
            df['elnino_ind'] = np.nan

        # Limpeza final
        cols_to_keep = [
            'date', 
            'heatindex_max', 'heatindex_min', 'heatindex_mea',
            'heatwave_has', 'heatwaveduration_sum', 'heatwaveintensity_ind',
            'coldwave_has', 'coldwaveduration_sum', 'coldwaveintensity',
            'spi_ind', 'elnino_ind'
        ]
        
        # Filtrar colunas existentes
        final_cols = [c for c in cols_to_keep if c in df.columns]
        final_df = df[final_cols]
        
        # Salvar
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        LOGGER.info(f"Dados diversos salvos em {output_csv}")
        
        return final_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dl = DiversosDownloader()
    # Exemplo de teste (requer shapefile válido)
    dl.fetch_data("data/shapefiles/SP-Diadema/SP_Diadema.shp", "01/01/2023", "01/12/2023")
