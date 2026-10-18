"""Файлы для сотрудников БНС: карта доказательности и библиотека доказанных выводов.

    python3 qol_ai/export_bns.py
Результат: out/bns/provable_для_БНС.xlsx (три листа) и те же таблицы в CSV.
"""
import json
from pathlib import Path

import pandas as pd

import engine as E

OUT = Path(__file__).resolve().parent / "out"
BNS = OUT / "bns"
LEVEL_RU = {"HOUSEHOLD": "подтверждено для семей", "REGIONAL": "только между регионами", "NONE": "не подтверждается"}
NOUN = {"income_pc": "доход на человека", "spend_pc": "расходы на человека", "borrowed": "заёмные средства", "inc_social": "соцвыплаты",
        "dep_n": "число лишений", "dep_shock": "доля семей без финансовой подушки", "dep_ritual": "доля «обрядов в долг»",
        "dep_holiday": "доля семей без отпуска", "dep_heat": "доля семей без тепла", "life_sat": "удовлетворённость жизнью",
        "fin_sat": "удовлетворённость финансами", "self_class": "самооценка достатка", "optimism": "пессимизм ожиданий",
        "fies_raw": "продовольственная небезопасность", "hh_size": "размер семьи", "n_children": "число детей",
        "n_elderly": "число пожилых", "n_employed": "число работающих", "total_area": "площадь жилья",
        "amenities_missing": "число отсутствующих удобств"}
BIN = {"urban": "город, а не село", "head_female": "глава семьи — женщина", "any_higher_edu": "в семье есть высшее образование"}


def statement(p):
    """Та же формулировка, что в разделе «Библиотека доказанных выводов» приложения."""
    reg = p["level"] == "REGIONAL"
    r = list(p["reg"].values()) if reg else [p["partial"]]
    up = r[0] > 0
    a, b = (p["a"], p["b"]) if (p["a"] in BIN and p["b"] not in BIN) else (p["b"], p["a"])
    if a in BIN and b in BIN:
        txt = f"«{BIN[a]}» и «{BIN[b]}» {'встречаются вместе чаще' if up else 'встречаются вместе реже'}"
    elif a in BIN:
        txt = f"Где {BIN[a]}, там {NOUN[b]} {'выше' if up else 'ниже'}"
    else:
        txt = f"Чем больше {NOUN[a]}, тем {'больше' if up else 'меньше'} {NOUN[b]}"
    if reg:
        txt = "Между регионами: " + txt[0].lower() + txt[1:]
    return txt, min(abs(x) for x in r) if reg else abs(p["partial"])


def main():
    f = json.loads((OUT / "findings.json").read_text())
    rows, lib = [], []
    for p in f["pairs"]:
        la, lb = E.CATALOG[p["a"]][0], E.CATALOG[p["b"]][0]
        rows.append({"Показатель 1": la, "Форма 1": p["form_a"], "Показатель 2": lb, "Форма 2": p["form_b"],
                     "Вердикт": LEVEL_RU[p["level"]], "ρ по всем семьям": round(p["raw"], 3),
                     "ρ внутри регион × город/село × год": round(p["partial"], 3),
                     "95% ДИ (внутри)": f"{p['partial_ci'][0]:+.3f} … {p['partial_ci'][1]:+.3f}",
                     "p (перестановки)": round(p["p_perm"], 3),
                     "ρ между регионами, 2023": p["reg"]["2023"], "ρ между регионами, 2024": p["reg"]["2024"]})
        if p["level"] != "NONE" and (p["level"] == "REGIONAL" or abs(p["partial"]) >= 0.05):
            txt, strength = statement(p)
            ev = (f"ρ = {p['reg']['2023']:+.2f} (2023), {p['reg']['2024']:+.2f} (2024)" if p["level"] == "REGIONAL"
                  else f"ρ = {p['partial']:+.2f} внутри регионов, p < 0,01")
            lib.append({"Утверждение": txt, "Доказательство": ev, "Уровень": LEVEL_RU[p["level"]], "_s": strength})
    m = pd.DataFrame(rows).sort_values(["Вердикт", "ρ внутри регион × город/село × год"], key=lambda c: c.abs() if c.dtype != object else c)
    l = pd.DataFrame(lib).sort_values("_s", ascending=False).drop(columns="_s")
    forms = (pd.DataFrame([{"Формы": " × ".join(sorted([p["form_a"], p["form_b"]])), "Вердикт": LEVEL_RU[p["level"]]} for p in f["pairs"]])
             .pivot_table(index="Формы", columns="Вердикт", aggfunc="size", fill_value=0).reset_index())
    BNS.mkdir(exist_ok=True)
    with pd.ExcelWriter(BNS / "provable_для_БНС.xlsx") as xw:
        l.to_excel(xw, sheet_name="Доказанные выводы", index=False)
        m.to_excel(xw, sheet_name="Карта доказательности", index=False)
        forms.to_excel(xw, sheet_name="Сводка по формам", index=False)
    l.to_csv(BNS / "доказанные_выводы.csv", index=False)
    m.to_csv(BNS / "карта_доказательности.csv", index=False)
    print(f"доказанных выводов: {len(l)}; пар в карте: {len(m)}; файлы: {BNS}")
    print(forms.to_string(index=False))


if __name__ == "__main__":
    main()
