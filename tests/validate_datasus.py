#!/usr/bin/env python3
"""
Validação do scraper DATASUS contra o CSV de referência.

Uso:
    python tests/validate_datasus.py

O script:
  1. Baixa dados SIH/RD de Jan/2000 para São Paulo via PySUS
  2. Agrega diariamente (como o scraper faz)
  3. Compara os totais mensais com o CSV de referência
  4. Imprime um relatório de validação

Requer: acesso de rede ao ftp.datasus.gov.br
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

# Garante que o projeto está no sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

REFERENCE_CSV = PROJECT_ROOT / "data" / "raw_input_reference" / "3550308_saopaulo_sp_asma.csv"
COD_IBGE = "3550308"

# Período de teste — apenas Jan/2000 (1 mês, rápido de baixar)
TEST_START = date(2000, 1, 1)
TEST_END = date(2000, 1, 31)


def load_reference(start: date, end: date):
    """Carrega o CSV de referência e filtra pelo período."""
    import pandas as pd

    df = pd.read_csv(REFERENCE_CSV)
    df["date"] = pd.to_datetime(df["date"])

    # O CSV de referência é semanal — para comparar com dados diários,
    # somamos o total do período
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    ref = df.loc[mask]

    cols = [
        "cases_sum", "deaths_sum",
        "man_sum", "woman_sum", "unknownsex_sum",
        "age_0_sum", "age_1_10_sum", "age_11_20_sum", "age_21_30_sum",
        "age_31_40_sum", "age_41_50_sum", "age_51_60_sum", "age_61_70_sum",
        "age_m70_sum",
    ]

    totals = {}
    for c in cols:
        if c in ref.columns:
            totals[c] = int(ref[c].sum())
        else:
            totals[c] = None
    totals["_weeks"] = len(ref)
    totals["_date_range"] = f"{ref['date'].min()} — {ref['date'].max()}"
    return totals


def run_scraper(start: date, end: date):
    """Roda o scraper DATASUS para o período e retorna totais."""
    from src.baixar_dados.DATASUS import DataSUSDownloader

    cache = PROJECT_ROOT / "data" / "cache_data"
    dl = DataSUSDownloader(cache_dir=cache, storage_format="parquet", cleanup_cache=False)

    df = dl.fetch_sih_asthma_daily(
        cod_ibge=COD_IBGE,
        start=start,
        end=end,
    )

    if df.empty:
        print("⚠️  Scraper retornou DataFrame vazio!")
        return {}

    cols = [
        "cases_sum", "deaths_sum",
        "man_sum", "woman_sum", "unknownsex_sum",
        "age_0_sum", "age_1_10_sum", "age_11_20_sum", "age_21_30_sum",
        "age_31_40_sum", "age_41_50_sum", "age_51_60_sum", "age_61_70_sum",
        "age_m70_sum",
    ]

    totals = {}
    for c in cols:
        if c in df.columns:
            totals[c] = int(df[c].sum())
        else:
            totals[c] = None
    totals["_days"] = len(df)
    totals["_date_range"] = f"{df['date'].min()} — {df['date'].max()}"

    # Salva CSV intermediário para inspeção
    out = PROJECT_ROOT / "data" / "output" / "datasus" / "validate_jan2000.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"📁 CSV intermediário salvo em: {out}")

    return totals


def compare(ref: dict, scraper: dict):
    """Compara os totais e imprime relatório."""
    print("\n" + "=" * 70)
    print(f"  VALIDAÇÃO DATASUS — São Paulo (3550308) — Jan/2000")
    print("=" * 70)

    print(f"\n  Referência: {ref.get('_weeks', '?')} semanas, período {ref.get('_date_range', '?')}")
    print(f"  Scraper:    {scraper.get('_days', '?')} dias, período {scraper.get('_date_range', '?')}")

    cols = [
        "cases_sum", "deaths_sum",
        "man_sum", "woman_sum", "unknownsex_sum",
        "age_0_sum", "age_1_10_sum", "age_11_20_sum", "age_21_30_sum",
        "age_31_40_sum", "age_41_50_sum", "age_51_60_sum", "age_61_70_sum",
        "age_m70_sum",
    ]

    print(f"\n  {'Coluna':<25} {'Referência':>12} {'Scraper':>12} {'Diff':>8} {'Status':>8}")
    print("  " + "-" * 67)

    all_ok = True
    for c in cols:
        r = ref.get(c)
        s = scraper.get(c)
        if r is None or s is None:
            status = "⚠️ N/A"
            diff = "—"
        else:
            diff_val = s - r
            diff = f"{diff_val:+d}"
            # Tolerância: o CSV de referência é semanal e as semanas podem
            # não se alinhar perfeitamente com o mês. Aceita até ±20%.
            if r == 0:
                status = "✅" if s == 0 else "⚠️ DIFF"
            elif abs(diff_val) / max(r, 1) <= 0.20:
                status = "✅"
            else:
                status = "❌ DIFF"
                all_ok = False

        print(f"  {c:<25} {str(r):>12} {str(s):>12} {str(diff):>8} {status:>8}")

    print()
    if all_ok:
        print("  ✅ RESULTADO: Todos os valores estão dentro da tolerância (±20%)")
        print("     (tolerância alta porque CSV de referência é semanal e o scraper é diário,")
        print("      semanas na borda do mês podem ter dias fora do range)")
    else:
        print("  ❌ RESULTADO: Alguns valores estão fora da tolerância")
        print("     Verifique os dados manualmente")

    print()
    print("  NOTA: Para uma validação mais precisa, compare um ano inteiro:")
    print("    python -c \"")
    print("    from tests.validate_datasus import *")
    print("    ref = load_reference(date(2000,1,1), date(2000,12,31))")
    print("    scr = run_scraper(date(2000,1,1), date(2000,12,31))")
    print("    compare(ref, scr)\"")
    print("=" * 70)


if __name__ == "__main__":
    print("🔍 Carregando CSV de referência...")
    ref = load_reference(TEST_START, TEST_END)
    print(f"   Referência Jan/2000: {ref.get('cases_sum', '?')} cases, {ref.get('deaths_sum', '?')} deaths")

    print("\n🌐 Rodando scraper DATASUS (requer rede)...")
    try:
        scraper = run_scraper(TEST_START, TEST_END)
    except Exception as exc:
        print(f"\n❌ Erro ao rodar scraper: {exc}")
        print("   Verifique se há acesso ao ftp.datasus.gov.br")
        sys.exit(1)

    if not scraper:
        print("❌ Scraper não retornou dados")
        sys.exit(1)

    compare(ref, scraper)
