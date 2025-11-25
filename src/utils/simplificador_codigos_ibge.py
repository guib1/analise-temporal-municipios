from pathlib import Path

import pandas as pd

SOURCE_XLS = Path("data/raw_input_reference/RELATORIO_DTB_BRASIL_2024_DISTRITOS.xls")
TARGET_CSV = Path("data/utils/municipios_ibge.csv")
COLUMNS = ["Código Município Completo", "Nome_UF", "Nome_Município"]

df = pd.read_excel(SOURCE_XLS, engine="xlrd", skiprows=6, usecols=COLUMNS)
df = df.dropna(subset=["Código Município Completo"])
df["Código Município Completo"] = (
    df["Código Município Completo"].astype(int).astype(str).str.zfill(7)
)
df = df.drop_duplicates(subset=["Código Município Completo"])
df = df.sort_values(["Nome_UF", "Nome_Município"]).reset_index(drop=True)
df.rename(columns={"Código Município Completo": "codigo_ibge"}, inplace=True)
df = df.set_index("codigo_ibge")

print(df.head(5))

TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(TARGET_CSV)