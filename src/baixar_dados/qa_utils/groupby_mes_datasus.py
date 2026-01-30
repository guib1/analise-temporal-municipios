import pandas as pd
import datetime

df = pd.read_csv("data/output/datasus/daily_3550308_asma.csv")
df["date"] = pd.to_datetime(df["date"])
df["mes"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")  # type: ignore[union-attr]
resultado = df.groupby("mes", as_index=False).sum(numeric_only=True)
print(resultado)
