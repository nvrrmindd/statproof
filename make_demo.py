"""Создаёт демо-наборы для веб-приложения аудита из синтетических микроданных организаторов.

    python3 qol_ai/make_demo.py
Результат: qol_ai/demo_data/ (в git не хранится: это производные от данных организаторов).
"""
from pathlib import Path
import shutil
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "demo_data"
OUT.mkdir(exist_ok=True)

shutil.copy(HERE / "out" / "households.csv", OUT / "домохозяйства.csv")

parts = []
for y in (2023, 2024):
    for q in range(1, 5):
        parts.append(pd.read_csv(ROOT / f"d004/{y}/{q}kv/kv_vopr11.csv",
                                 usecols=["TE", "K", "VOZ", "POL", "GR1", "GR2", "GR4", "GR13", "GR22", "KVARTAL", "GOD"]))
d = pd.concat(parts).rename(columns={"TE": "регион", "K": "город_село", "VOZ": "возраст", "POL": "пол", "GR1": "зарплата",
                                     "GR2": "доход_от_самозанятости", "GR4": "пенсия", "GR13": "соцвыплаты",
                                     "GR22": "натуральные_доходы", "KVARTAL": "квартал", "GOD": "год"})
d.sample(n=min(60000, len(d)), random_state=1).to_csv(OUT / "d004_доходы_людей.csv", index=False)
print("готово:", [p.name for p in OUT.iterdir()])
