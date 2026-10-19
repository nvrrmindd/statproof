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
# таблица не о семьях, с известным ответом: проверка, что Provable работает на любых данных.
# «сотрудников» и «выручка» связаны у каждого предприятия; «зарплата» и «текучесть» — только через отрасль (ловушка);
# «налоги» ни с чем не связаны.
import numpy as np
rng = np.random.default_rng(3)
n = 5000
branches = np.array(["агро", "IT", "торговля", "стройка", "добыча", "транспорт", "финансы", "образование",
                     "медицина", "энергетика", "связь", "туризм"])
b = rng.integers(0, len(branches), n)
eff = rng.normal(0, 1, len(branches))[b]
h = rng.normal(0, 1, n)
pd.DataFrame({"id_предприятия": np.arange(1, n + 1), "отрасль": branches[b], "год": rng.integers(2023, 2025, n),
              "сотрудников": np.exp(h + 3).round(), "выручка_млн": np.exp(h * 0.8 + eff + rng.normal(0, 1, n)).round(2),
              "зарплата_тыс": (300 + 80 * eff + rng.normal(0, 40, n)).round(),
              "текучесть_%": (15 + 5 * eff + rng.normal(0, 5, n)).round(1),
              "налоги_млн": np.exp(rng.normal(0, 1, n)).round(2)}).to_csv(OUT / "предприятия_синтетика.csv", index=False)
print("готово:", [p.name for p in OUT.iterdir()])
