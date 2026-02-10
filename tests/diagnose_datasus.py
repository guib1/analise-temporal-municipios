#!/usr/bin/env python3
"""
Diagnóstico: por que o scraper DATASUS encontra menos registros que a referência?

Trace step-by-step dos filtros para entender onde os registros somem.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

import pandas as pd

COD_IBGE = "3550308"
START = date(2000, 1, 1)
END = date(2000, 1, 31)

# ---- Step 1: Download raw month data ----
print("=" * 70)
print("  DIAGNÓSTICO DATASUS — Jan/2000 São Paulo")
print("=" * 70)

from src.baixar_dados.DATASUS import (
    DataSUSDownloader, _normalize_ibge_codes, _parse_sih_dates,
    DEFAULT_ASMA_CODES,
)

cache = PROJECT_ROOT / "data" / "cache_data"
dl = DataSUSDownloader(cache_dir=cache, storage_format="parquet", cleanup_cache=False)

print("\n📥 Baixando dados brutos de Jan/2000...")
raw = dl._download_month("SP", 2000, 1)
print(f"   Total registros brutos (todo SP, todas as doenças): {len(raw)}")
print(f"   Colunas: {list(raw.columns)[:20]}...")

# ---- Step 2: Municipality filter ----
code6, code7 = _normalize_ibge_codes(COD_IBGE)
print(f"\n🏙️  Filtro por município: MUNIC_RES in ({code6}, {code7})")

if "MUNIC_RES" in raw.columns:
    muni_raw = raw["MUNIC_RES"].astype(str).str.strip()
    print(f"   Valores únicos MUNIC_RES (amostra): {sorted(muni_raw.unique())[:20]}")
    print(f"   Tamanho típico: {muni_raw.str.len().value_counts().to_dict()}")
    
    muni_padded = muni_raw.str.zfill(6)
    match6 = muni_padded == code6
    match7_raw = muni_raw == code7
    match_either = muni_padded.isin({code6, code7})
    
    print(f"   Match code6 ({code6}): {match6.sum()}")
    print(f"   Match code7 ({code7}): {match7_raw.sum()}")
    print(f"   Match either (isin): {match_either.sum()}")
    
    # Check also with str contains
    contains_355030 = muni_raw.str.contains("355030", na=False)
    print(f"   Contains '355030': {contains_355030.sum()}")
    
    df_muni = raw[match_either].copy()
else:
    print("   ❌ Coluna MUNIC_RES não encontrada!")
    df_muni = pd.DataFrame()

print(f"   Após filtro município: {len(df_muni)} registros")

# ---- Step 3: Diagnosis filter ----
if not df_muni.empty:
    diag_col = None
    for candidate in ("DIAG_PRINC", "CIDPRI", "DIAG_PRINCIPAL"):
        if candidate in df_muni.columns:
            diag_col = candidate
            break
    
    if diag_col:
        print(f"\n🩺 Filtro por diagnóstico: coluna={diag_col}")
        diag_vals = df_muni[diag_col].astype(str).str.upper().str.strip()
        
        # Show what diagnoses exist for SP
        j45_matches = diag_vals.str.startswith("J45")
        j46_matches = diag_vals.str.startswith("J46")  # status asthmaticus
        print(f"   Diagnósticos J45*: {j45_matches.sum()}")
        print(f"   Diagnósticos J46*: {j46_matches.sum()}")
        print(f"   Cap X (J00-J99) respiratório: {diag_vals.str.startswith('J').sum()}")
        
        # Distribution of J45 subcodes
        j45_vals = diag_vals[j45_matches].value_counts()
        print(f"   Distribuição J45: {j45_vals.to_dict()}")
        
        prefixes = tuple(code.upper() for code in DEFAULT_ASMA_CODES)
        matches = diag_vals.str.startswith(prefixes)
        df_diag = df_muni[matches].copy()
        print(f"   Após filtro CID-10 ({len(DEFAULT_ASMA_CODES)} codes): {len(df_diag)}")
    else:
        print("   ❌ Coluna de diagnóstico não encontrada!")
        df_diag = df_muni

# ---- Step 4: Date filter ----
if not df_diag.empty and "DT_INTER" in df_diag.columns:
    print(f"\n📅 Filtro por data: DT_INTER entre {START} e {END}")
    dt_raw = df_diag["DT_INTER"]
    print(f"   Tipo DT_INTER: {dt_raw.dtype}")
    print(f"   Amostra valores: {dt_raw.head(10).tolist()}")
    
    dt_parsed = _parse_sih_dates(dt_raw)
    na_count = dt_parsed.isna().sum()
    print(f"   Após parse: {na_count} NaT (não parseados)")
    
    if na_count > 0:
        failed = dt_raw[dt_parsed.isna()]
        print(f"   Valores que falharam no parse: {failed.unique()[:20].tolist()}")
    
    valid = dt_parsed.dropna()
    if len(valid) > 0:
        print(f"   Range de datas válidas: {valid.min()} a {valid.max()}")
    
    start_ts = pd.Timestamp(START)
    end_ts = pd.Timestamp(END) + pd.Timedelta(days=1) - pd.Timedelta(1, unit="ns")
    in_range = dt_parsed.between(start_ts, end_ts)
    print(f"   Dentro do range [{start_ts}, {end_ts}]: {in_range.sum()}")
    
    df_date = df_diag[in_range.reindex(df_diag.index, fill_value=False)].copy()
    print(f"   Após filtro data: {len(df_date)} registros")

# ---- Step 5: Check if DT_SAIDA would give more records ----
if "DT_SAIDA" in raw.columns or "DT_NASC" in raw.columns:
    print(f"\n📋 Outras colunas de data disponíveis:")
    for col in ["DT_INTER", "DT_SAIDA", "DT_NASC", "ANO_CMPT", "MES_CMPT"]:
        if col in raw.columns:
            print(f"   {col}: presente")

# ---- Summary ----
print(f"\n{'='*70}")
print(f"  RESUMO")
print(f"{'='*70}")
print(f"  Raw (todo SP):           {len(raw)}")
print(f"  Após filtro município:   {len(df_muni)}")
print(f"  Após filtro diagnóstico: {len(df_diag)}")
if 'df_date' in dir():
    print(f"  Após filtro data:        {len(df_date)}")
print(f"  Referência (semanal):    619 cases em ~5 semanas")
print(f"{'='*70}")
