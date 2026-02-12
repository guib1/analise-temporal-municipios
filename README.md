# Análise Temporal de Municípios - Coleta de Dados Ambientais e de Saúde

Este projeto automatiza a coleta, processamento e consolidação de dados ambientais (poluentes, meteorologia) e de saúde (internações) para análise temporal em municípios brasileiros.

## 🚀 Funcionalidades Principais

*   **Download Paralelo**: Orquestrador inteligente que baixa dados de múltiplas fontes simultaneamente (`ThreadPoolExecutor`).
*   **Gestão de Dependências**: Garante que índices calculados só rodem após a disponibilidade dos dados primários (ex: INMET).
*   **Economia de Disco**: Limpeza automática de arquivos temporários pesados (NetCDF, HDF) após o processamento.
*   **Dados Unificados**: Gera um CSV final consolidado por município e data, pronto para análise (Machine Learning/Estatística).

---

## 📦 Fontes de Dados

| Fonte | Tipo | Variáveis Principais | Cobertura | Autenticação |
| :--- | :--- | :--- | :--- | :--- |
| **ERA5** | Reanálise | Vento, Temperatura, Precipitação | Global | CDS API |
| **MERRA-2** | Reanálise | Carbono (BC/OC), SO2, O3, Aerossóis | Global | NASA Earthdata |
| **MODIS** | Satélite | AOD (Profundidade Óptica de Aerossóis) | Global | NASA Earthdata |
| **OMI** | Satélite | NO2, Ozônio (Coluna Total) | Global | NASA Earthdata |
| **TROPOMI** | Satélite | NO2, Ozônio (Alta Resolução) | Global | Copernicus Data Space |
| **INMET** | Estação | Temp, Umidade, Vento (Dados Observados) | Brasil | Aberto |
| **CETESB** | Estação | Poluentes (PM10, O3, NO2, etc.) | SP | Qualar (Login) |
| **DATASUS** | Saúde | Internações (SIH/RD) - Asma, etc. | Brasil | Aberto |

---

## 🛠️ Instalação

### Pré-requisitos
- Python 3.11
- Bibliotecas em requirements.txt

### Instalação das Dependências
```bash
pip install -r requirements.txt
```

---

## 🔑 Configuração de Credenciais (.env)

O sistema exige credenciais para acessar as APIs da NASA, Copernicus e CETESB. Crie um arquivo `.env` na raiz do projeto seguindo o modelo abaixo:

```ini
# --- NASA Earthdata (MERRA-2, MODIS, OMI) ---
# Cadastro: https://urs.earthdata.nasa.gov/
# IMPORTANTE: Autorize os apps "NASA GESDISC DATA ARCHIVE" e "LP DAAC" no seu perfil.
NASA_USER=seu_usuario
NASA_PASSWORD=sua_senha

# --- Copernicus Climate Data Store (ERA5) ---
# Cadastro: https://cds.climate.copernicus.eu/
# O formato da chave mudou recentemente. Verifique seu perfil no site.
CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
CDSAPI_KEY=UID:API_KEY

# --- Copernicus Data Space Ecosystem (TROPOMI) ---
# Cadastro: https://dataspace.copernicus.eu/
COPERNICUS_USER=seu_email@dominio.com
COPERNICUS_PASSWORD=sua_senha

# --- CETESB Qualar (Estações SP) ---
# Cadastro: https://qualar.cetesb.sp.gov.br/
CETESB_USER=seu_login
CETESB_PASSWORD=sua_senha
```

---

## ▶️ Fluxo de Trabalho (Como Usar)

O projeto foi desenhado para funcionar através de um **Jupyter Notebook interativo**, que atua como o frontend da aplicação.

### 1. Interface Principal (`src/baixar_shapefiles.ipynb`)

1.  **Abra o Notebook**: Inicie o Jupyter e abra `src/baixar_shapefiles.ipynb`.
2.  **Passo 1: Seleção e Download do Shapefile**:
    *   Esta é a etapa pré-requisitada. O sistema precisa da geometria do município para recortar os dados de satélite.
    *   Nos menus interativos, escolha o **Estado** e o **Município**.
    *   Clique em **"Baixar e Plotar Shapefile"**.
    *   O script salvará automaticamente os arquivos em `data/shapefiles/{UF}-{MUNICIPIO}/`.
3.  **Passo 2: Download e Processamento dos Dados**:
    *   Avance para a seção "Criar arquivo final" no mesmo notebook.
    *   Defina as datas de **Início** e **Fim**.
    *   Escolha o município (que aparecerá na lista após o passo 1);
    *   Clique em **"Executar"**.
    *   Isso acionará o orquestrador (`download_all.py`) para coletar dados de todas as fontes (ERA5, MERRA2, INMET, etc.) em paralelo.

### 2. Modo Avançado (Headless / Script)

Se você já possui os shapefiles baixados (via notebook ou manualmente) e deseja rodar em um servidor ou em lote, pode chamar o orquestrador diretamente via Python:

```python
from src.baixar_dados.download_all import download_all

# Baixar dados de Janeiro/2023 para todos os shapefiles na pasta data/shapefiles
df_final = download_all(
    start="2023-01-01",
    end="2023-01-31",
    shapefiles_dir="data/shapefiles",
    max_workers=4
)
```

### Argumentos Principais
- `start`, `end`: Datas de início e fim (YYYY-MM-DD).
- `shapefiles_dir`: Diretório contendo os shapefiles dos municípios alvo.
- `output_dir`: Onde os CSVs serão salvos (padrão: `data/output`).
- `final_schema`:
    - `"all"`: Mantém todas as colunas encontradas.
    - `"reference"`: Filtra e ordena as colunas conforme um CSV modelo (ex: `data/reference.csv`).
    - `"inmet"`: Força o padrão do INMET.

---

## 📁 Estrutura de Saída

```text
data/
├── output/
│   ├── cetesb/          # CSVs brutos da CETESB
│   ├── era5/            # CSVs brutos do ERA5
│   ├── ...              # (outras fontes)
│   ├── final/
│   │   ├── by_municipio/
│   │   │   └── final_3550308.csv  # CSV consolidado de cada município
│   │   └── final_by_ibge_date.csv # ARQUIVO FINAL UNIFICADO (Todos os municípios)
└── cache/               # Arquivos temporários (autolimpeza ativa)
```

---

## ✅ Status de Validação

| Scraper | Status | Obs |
| :--- | :--- | :--- |
| **DATASUS** | ✅ OK | Validado (SIH/RD) |
| **ERA5** | ✅ OK | Validado |
| **INMET** | ✅ OK | Validado |
| **Índice Calc.** | ✅ OK | Validado (Ondas de calor, frio, SPI) |
| **MERRA-2** | ✅ OK | Validado (Correção de datas e agregação) |
| **CETESB** | ✅ OK | Validado (Unidades µg/m³ verificadas) |
| **MODIS** | ✅ OK | Implementado via `earthaccess` (sem GDAL) |
| **OMI** | ✅ OK | Validado (Conversão para DU) |
| **TROPOMI** | ✅ OK | Pipeline otimizado (Stream day-by-day) |
| **Orquestrador** | ✅ OK | Paralelismo e Dependências implementados |
