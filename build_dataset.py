"""Шаг 1. Сборка панели домохозяйство×год из форм D002, D004, D006, D008.

Запуск:  python3 qol_ai/build_dataset.py
Результат: qol_ai/out/households.parquet (или .csv)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
YEARS = [2023, 2024]  # одинаковая схема D002; 2021-2022 используются в robustness.py

# 9 пунктов материальной депривации (D002, часть 3). 1 = «может позволить», иное = лишение.
# Для пунктов с вариантом «нет, по другой причине» лишением считается только «не хватает средств» (=2).
DEPRIVATION = {
    "arrears":   ("GR31_*", "Просрочки по аренде/коммуналке/кредиту/рассрочке"),
    "heat":      ("GR32",   "Не может поддерживать тепло в доме"),
    "furniture": ("GR33",   "Не может заменить изношенную мебель"),
    "protein":   ("GR34",   "Нет мяса/рыбы/белка раз в два дня"),
    "shock":     ("GR35",   "Не покроет непредвиденный расход 30–40 тыс. тг"),
    "holiday":   ("GR36",   "Нет недельного отпуска вне дома"),
    "social":    ("GR37",   "Не может раз в месяц встречаться с близкими"),
    "ritual":    ("GR38",   "Не может провести обряды без долгов"),
    "shoes":     ("GR39",   "Нет двух пар сезонной обуви"),
    "clothes":   ("GR310",  "Не может заменить изношенную одежду"),
    "leisure":   ("GR312",  "Нет средств на досуг (кино, концерты)"),
}
# Шкала продовольственной небезопасности FAO FIES, 8 пунктов (1 = да)
FIES = ["GR26", "GR27", "GR29", "GR210", "GR211", "GR212", "GR213", "GR214"]

INCOME_LABELS = {1: "Зарплата", 2: "Самозанятость", 4: "Пенсии", 8: "Пособия (8)", 9: "Пособия (9)",
                 13: "АСП/соцвыплаты", 22: "Натуральные/прочие"}


def read(path):
    return pd.read_csv(path, dtype={"NOMER": str, "NOMP": str}, low_memory=False)


def deprivation(o):
    d = pd.DataFrame(index=o.index)
    arrears_cols = [c for c in ["GR31_1", "GR31_2", "GR31_3", "GR31_4"] if c in o]
    # коды 1,2 = «да, один раз / два и более» — была просрочка
    d["arrears"] = o[arrears_cols].isin([1, 2]).any(axis=1).astype(int)
    for k in ["heat", "protein", "shock", "holiday", "ritual"]:
        d[k] = (o[DEPRIVATION[k][0]] == 2).astype(int)
    for k in ["furniture", "social", "shoes", "clothes", "leisure"]:
        d[k] = (o[DEPRIVATION[k][0]] == 2).astype(int)  # только «не хватает средств»
    return d


def income(year):
    parts = []
    for q in range(1, 5):
        p = ROOT / f"d004/{year}/{q}kv/kv_vopr11.csv"
        if p.exists():
            parts.append(read(p))
    i = pd.concat(parts)
    g = [f"GR{k}" for k in range(1, 24)]
    i["total"] = i[g].sum(axis=1)
    hh = i.groupby("NOMER").agg(income_total=("total", "sum"), n_quarters=("KVARTAL", "nunique"),
                                inc_wage=("GR1", "sum"), inc_self=("GR2", "sum"), inc_pension=("GR4", "sum"),
                                inc_social=("GR13", "sum"), inc_other=("GR22", "sum"))
    hh["inc_benefits"] = i.groupby("NOMER")[["GR8", "GR9", "GR13"]].sum().sum(axis=1)
    # годовой доход: нормируем на число кварталов участия
    hh["income_annual"] = hh.income_total / hh.n_quarters * 4
    return hh


def nonfood_spend(year):
    tot = {}
    for q in range(1, 5):
        for t in [1, 2, 3, 4, 5, 6, 7]:
            p = ROOT / f"d004/{year}/{q}kv/kv_vopr{t}.csv"
            if p.exists():
                x = pd.read_csv(p, usecols=["NOMER", "STOIMK"], dtype={"NOMER": str})
                s = x.groupby("NOMER").STOIMK.sum()
                for k, v in s.items():
                    tot[k] = tot.get(k, 0) + v
    return pd.Series(tot, name="spend_nonfood")


def debt(year):
    parts = [pd.read_csv(ROOT / f"d004/{year}/{q}kv/kv_vopr12.csv", dtype={"NOMER": str}, usecols=["NOMER", "STOIMK"])
             for q in range(1, 5) if (ROOT / f"d004/{year}/{q}kv/kv_vopr12.csv").exists()]
    return pd.concat(parts).groupby("NOMER").STOIMK.sum().rename("borrowed")


def roster(year):
    r = read(ROOT / f"d008/{year}/kontr_k.csv")
    r["age"] = year - pd.to_numeric(r.GOD_ROJD, errors="coerce")
    g = r.groupby("NOMER")
    out = pd.DataFrame({
        "hh_size": g.size(),
        "n_children": g.age.apply(lambda a: (a < 18).sum()),
        "n_elderly": g.age.apply(lambda a: (a >= 63).sum()),
        "n_working_age": g.age.apply(lambda a: ((a >= 18) & (a < 63)).sum()),
        "max_edu": g.UROV.max(),
    })
    # STATUS (форма D008, вопрос 9): 1 по найму, 2 не по найму, 3 безработный, 7 нетрудоспособный
    r["employed"] = r.STATUS.isin([1, 2])
    r["unemployed"] = r.STATUS == 3
    r["disabled"] = r.STATUS == 7
    r["higher_edu"] = r.UROV.isin([5, 6]) & (r.age >= 22)
    g = r.groupby("NOMER")
    out["n_employed"] = g.employed.sum()
    out["n_unemployed"] = g.unemployed.sum()
    out["n_disabled"] = g.disabled.sum()
    out["any_higher_edu"] = (g.higher_edu.sum() > 0).astype(int)
    head = r[r.RODSTVO == 1].drop_duplicates("NOMER").set_index("NOMER")
    out["head_female"] = (head.POL == 2).astype(int)
    out["head_age"] = head.age
    return out


def dwelling(year):
    name = {2023: "rio2023.csv", 2024: "rio_2024.csv", 2022: "rio2022.csv", 2021: "rio.csv"}[year]
    d = read(ROOT / f"d006/{year}/{name}").drop_duplicates("NOMER").set_index("NOMER")
    out = pd.DataFrame({"dwelling_type": d.TIP_J, "owner": (d.VLAD1 == 1).astype(int),
                        "living_area": d.J_PL, "rooms": d.KOL_K, "total_area": d.OB_PL})
    # U1..U42 — удобства (1 = есть). Число отсутствующих удобств как индекс жилищной депривации.
    ucols = [c for c in d.columns if c.startswith("U") and c[1:].isdigit()]
    out["amenities_missing"] = (d[ucols] == 2).sum(axis=1)
    return out


def build_year(year):
    s = read(ROOT / f"d002/{year}/subject.csv").set_index("NOMER")
    o = read(ROOT / f"d002/{year}/ocenka.csv").set_index("NOMER")
    df = pd.DataFrame(index=s.index)
    df["year"] = year
    df["region"] = s.TE
    df["urban"] = (s.K == 1).astype(int)
    df["resp_female"] = (s.POL == 2).astype(int)
    df["resp_age"] = s.VOZ
    df["life_sat"] = s.GR1.where(s.GR1 <= 10)
    df["fin_sat"] = s.GR4.where(s.GR4 <= 10)
    df["family_econ_sat"] = s.GR7.where(s.GR7 <= 10)
    df["self_class"] = s.GR21          # 1 низкий ... 6 высокий
    df["wellbeing_change"] = s.GR23    # 1 не изм, 2 лучше, 3 хуже
    df["optimism"] = s.GR24            # 1 уверены лучше ... 5 хуже
    df["support"] = s.GR19.where(s.GR19 <= 10)
    df["fies_raw"] = (s[FIES] == 1).sum(axis=1)
    dep = deprivation(o.loc[df.index])
    for c in dep:
        df[f"dep_{c}"] = dep[c].values
    df = df.join(income(year)).join(nonfood_spend(year)).join(debt(year)).join(roster(year)).join(dwelling(year))
    df["borrowed"] = df.borrowed.fillna(0)
    df["income_pc"] = df.income_annual / df.hh_size / 12   # тенге в месяц на человека
    df["spend_pc"] = df.spend_nonfood / df.hh_size
    return df.reset_index().rename(columns={"index": "NOMER"})


if __name__ == "__main__":
    full = pd.concat([build_year(y) for y in YEARS], ignore_index=True)
    full.to_csv(OUT / "households.csv", index=False)
    print(full.shape)
    print(full.isna().mean().round(3)[lambda s: s > 0])
