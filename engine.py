"""Шаг 2. Аналитический движок StatProof.

Каждый вывод получает «уровень доказательности»:
  * HOUSEHOLD  — связь подтверждена на уровне домохозяйств и переживает контроль
                 на регион × тип поселения × год (перестановочный тест внутри ячеек);
  * REGIONAL   — на уровне домохозяйств связь не отличима от артефакта (исчезает после
                 контроля), но её можно изучать на уровне регионов; приводится региональная оценка;
  * NONE       — данных недостаточно, вывод не делается.
Все числа в ответах берутся только отсюда.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
SEED = 20261012
N_BOOT = 200
N_PERM = 200

REGIONS = {10: "Абайская", 11: "Акмолинская", 15: "Актюбинская", 19: "Алматинская", 23: "Атырауская",
           27: "Западно-Казахстанская", 31: "Жамбылская", 33: "Жетысуская", 35: "Карагандинская",
           39: "Костанайская", 43: "Кызылординская", 47: "Мангистауская", 55: "Павлодарская",
           59: "Северо-Казахстанская", 61: "Туркестанская", 62: "Улытауская", 63: "Восточно-Казахстанская",
           71: "г. Астана", 75: "г. Алматы", 79: "г. Шымкент"}

# Каталог переменных: метка, форма-источник, ключевые слова для разбора вопросов
CATALOG = {
    "income_pc":       ("Доход на человека в месяц", "D004", ["доход", "заработ", "денег", "богат", "достаток дох"]),
    "spend_pc":        ("Непродовольственные расходы на человека", "D004", ["расход", "трат", "потреблен"]),
    "borrowed":        ("Заёмные средства за год", "D004", ["заём", "заем", "кредит", "долг", "займ"]),
    "inc_social":      ("Социальные выплаты (АСП)", "D004", ["асп", "соцвыплат", "социальн выплат", "пособи"]),
    "dep_n":           ("Число лишений (0–11)", "D002-ocenka", ["лишени", "депривац", "бедност", "нужд"]),
    "dep_shock":       ("Не покроет расход 30–40 тыс. тг", "D002-ocenka", ["непредвид", "шок", "подушк", "30"]),
    "dep_ritual":      ("Обряды только в долг", "D002-ocenka", ["обряд", "той", "ритуал", "похорон", "свадьб"]),
    "dep_holiday":     ("Нет отпуска вне дома", "D002-ocenka", ["отпуск", "отдых"]),
    "dep_heat":        ("Не может поддерживать тепло", "D002-ocenka", ["тепл", "отоплен"]),
    "life_sat":        ("Удовлетворённость жизнью (1–10)", "D002-subject", ["удовлетвор", "счаст", "довол", "жизнью"]),
    "fin_sat":         ("Удовлетворённость финансами (1–10)", "D002-subject", ["финанс", "материальн"]),
    "self_class":      ("Самооценка достатка (1–6)", "D002-subject", ["самооцен", "средний класс", "считают себя", "класс"]),
    "optimism":        ("Пессимизм ожиданий (1 лучше … 5 хуже)", "D002-subject", ["ожидан", "оптимиз", "пессимиз", "будущ"]),
    "fies_raw":        ("Продовольственная небезопасность FIES (0–8)", "D002-subject", ["еда", "голод", "питани", "продовольств", "fies"]),
    "hh_size":         ("Размер домохозяйства", "D008", ["размер", "многодет", "членов", "большие семьи", "семьи"]),
    "n_children":      ("Число детей до 18", "D008", ["дет", "ребен", "ребён"]),
    "n_elderly":       ("Число лиц 63+", "D008", ["пожил", "пенсион", "стар"]),
    "n_employed":      ("Число занятых", "D008", ["занят", "работающ", "работа"]),
    "any_higher_edu":  ("Есть член с высшим образованием", "D008", ["образован", "высш", "вуз"]),
    "head_female":     ("Глава — женщина", "D008", ["женщин", "глава", "пол"]),
    "total_area":      ("Общая площадь жилья", "D006", ["площад", "жиль", "квартир", "дом "]),
    "amenities_missing": ("Число отсутствующих удобств", "D006", ["удобств", "водопровод", "канализац"]),
    "urban":           ("Город (1) / село (0)", "каркас", ["город", "сел", "аул", "сельск"]),
}
GROUPERS = {"urban": "Город/село", "region": "Регион"}


def load(path: Path | None = None) -> pd.DataFrame:
    d = pd.read_csv(path or OUT / "households.csv")
    dep = [c for c in d if c.startswith("dep_") and c != "dep_n"]
    d["dep_n"] = d[dep].sum(axis=1)
    d["dep_severe"] = (d.dep_n >= 3).astype(int)
    d["cell"] = d.year.astype(str) + "_" + d.region.astype(str) + "_" + d.urban.astype(str)
    return d


# ---------------------------------------------------------------- статистика
def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def _corr(a, b) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _demean(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    s = np.bincount(codes, weights=x); n = np.bincount(codes)
    return x - (s / np.maximum(n, 1))[codes]


def audit_pair(d: pd.DataFrame, a: str, b: str, rng=None) -> dict:
    """Связь a~b на уровне домохозяйств: сырая vs внутри ячеек регион×тип×год."""
    rng = rng or np.random.default_rng(SEED)
    m = d[[a, b]].notna().all(axis=1).to_numpy()
    if a == "urban" or b == "urban":   # контроль на тип поселения «съедает» саму переменную
        codes = pd.factorize(d.loc[m, "year"].astype(str) + "_" + d.loc[m, "region"].astype(str))[0]
    else:
        codes = pd.factorize(d.loc[m, "cell"])[0]
    ra, rb = _rank(d.loc[m, a].to_numpy(float)), _rank(d.loc[m, b].to_numpy(float))
    raw = _corr(ra, rb)
    pa, pb = _demean(ra, codes), _demean(rb, codes)
    part = _corr(pa, pb)
    n = int(m.sum())
    boot_raw, boot_part = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        boot_raw.append(_corr(ra[i], rb[i])); boot_part.append(_corr(pa[i], pb[i]))
    # нулевое распределение: перестановка b внутри ячеек
    order = np.argsort(codes, kind="stable"); bounds = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, bounds)
    null = []
    for _ in range(N_PERM):
        pb2 = pb.copy()
        for g in groups:
            pb2[g] = pb[rng.permutation(g)]
        null.append(_corr(pa, pb2))
    null = np.abs(np.array(null))
    p = float((1 + (null >= abs(part)).sum()) / (N_PERM + 1))
    ci_raw = np.percentile(boot_raw, [2.5, 97.5]); ci_part = np.percentile(boot_part, [2.5, 97.5])
    retained = abs(part) / abs(raw) if abs(raw) > 1e-9 else 0.0
    survives = (p < 0.01) and abs(part) >= 0.03 and (retained >= 0.5 or abs(part) >= 0.1)
    return {"a": a, "b": b, "n": n, "raw": raw, "raw_ci": ci_raw.tolist(), "partial": part,
            "partial_ci": ci_part.tolist(), "null95": float(np.quantile(null, 0.95)), "p_perm": p,
            "retained": retained, "household_valid": bool(survives),
            "form_a": CATALOG.get(a, ("", "?"))[1], "form_b": CATALOG.get(b, ("", "?"))[1]}


def regional_table(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    agg = d.groupby(["year", "region"])[cols].mean()
    agg["n"] = d.groupby(["year", "region"]).size()
    return agg.reset_index()


def regional_pair(d: pd.DataFrame, a: str, b: str, rng=None) -> dict:
    """Экологическая связь между регионами; бутстрап по домохозяйствам внутри регионов
    и проверка воспроизводимости в двух годах."""
    rng = rng or np.random.default_rng(SEED)
    res = {}
    for y, g in d.groupby("year"):
        g = g[["region", a, b]].dropna()
        m = g.groupby("region")[[a, b]].mean()
        rho = _corr(_rank(m[a].to_numpy()), _rank(m[b].to_numpy()))
        # бутстрап: регионы пересэмплируются (неопределённость выбора регионов)
        # + домохозяйства внутри регионов (неопределённость оценки средних)
        boots = []
        gi = {r: x.index.to_numpy() for r, x in g.groupby("region")}
        regs = np.array(list(gi))
        for _ in range(N_BOOT):
            pick = rng.choice(regs, len(regs))
            ma, mb = [], []
            for r in pick:
                idx = rng.choice(gi[r], len(gi[r]))
                ma.append(g.loc[idx, a].mean()); mb.append(g.loc[idx, b].mean())
            ma, mb = np.array(ma), np.array(mb)
            ma = ma + rng.normal(0, 1e-9, len(ma))  # разрыв совпадающих рангов
            boots.append(_corr(_rank(ma), _rank(mb)))
        res[int(y)] = {"rho": rho, "ci": np.percentile(boots, [2.5, 97.5]).tolist(), "k": len(m)}
    signs = {np.sign(v["rho"]) for v in res.values()}
    robust = len(signs) == 1 and all((v["ci"][0] > 0) or (v["ci"][1] < 0) for v in res.values())
    return {"a": a, "b": b, "by_year": res, "robust": bool(robust)}


# ---------------------------------------------------------------- вопросы
def parse_question(q: str) -> list[str]:
    ql = q.lower().replace("ё", "е")
    hits = []
    for var, (_, _, kws) in CATALOG.items():
        for kw in kws:
            pos = ql.find(kw.replace("ё", "е"))
            if pos >= 0:
                hits.append((pos, -len(kw), var)); break
    hits.sort()
    seen = []
    for _, _, v in hits:
        if v not in seen:
            seen.append(v)
    return seen[:2]


def fmt(x: float, nd=2) -> str:
    return f"{x:+.{nd}f}".replace("+-", "−").replace("-", "−")


def answer(d: pd.DataFrame, q: str, cache: dict | None = None) -> dict:
    vars_ = parse_question(q)
    if len(vars_) < 2:
        return {"question": q, "vars": vars_, "level": "NONE",
                "verdict": "Не удалось сопоставить вопрос с двумя показателями из каталога. "
                           "Сформулируйте связь двух показателей (например, «доход и лишения»).",
                "claims": []}
    a, b = vars_
    key = tuple(sorted((a, b)))
    hh = (cache or {}).get(key) or audit_pair(d, a, b)
    rg = regional_pair(d, a, b)
    la, lb = CATALOG[a][0], CATALOG[b][0]
    claims = [
        {"id": "raw", "text": f"Наивная корреляция Спирмена по {hh['n']} домохозяйствам",
         "value": round(hh["raw"], 3), "ci": [round(c, 3) for c in hh["raw_ci"]]},
        {"id": "partial", "text": "Та же связь внутри ячеек регион × город/село × год",
         "value": round(hh["partial"], 3), "ci": [round(c, 3) for c in hh["partial_ci"]],
         "p": round(hh["p_perm"], 4)},
    ]
    for y, v in rg["by_year"].items():
        claims.append({"id": f"reg{y}", "text": f"Связь между {v['k']} регионами, {y}",
                       "value": round(v["rho"], 3), "ci": [round(c, 3) for c in v["ci"]]})
    if hh["household_valid"]:
        level = "HOUSEHOLD"
        verdict = (f"Связь «{la}» — «{lb}» подтверждена на уровне домохозяйств: "
                   f"ρ = {fmt(hh['partial'])} после контроля региона и типа поселения "
                   f"(95% ДИ {fmt(hh['partial_ci'][0])}…{fmt(hh['partial_ci'][1])}, перестановочный p = {hh['p_perm']:.3f}).")
    elif rg["robust"]:
        level = "REGIONAL"
        ys = sorted(rg["by_year"])
        verdict = (f"На уровне домохозяйств ответ дать нельзя: наивная ρ = {fmt(hh['raw'])} "
                   f"падает до {fmt(hh['partial'])} после контроля региона. Связь существует только между регионами: "
                   + ", ".join(f"ρ = {fmt(rg['by_year'][y]['rho'])} в {y}" for y in ys)
                   + "; знак воспроизводится в обоих годах. Это экологический вывод, переносить его на отдельные семьи нельзя.")
    else:
        level = "NONE"
        verdict = (f"Данные не подтверждают связь «{la}» — «{lb}». На уровне домохозяйств она исчезает после контроля "
                   f"(ρ {fmt(hh['raw'])} → {fmt(hh['partial'])}), между регионами не воспроизводится в двух годах. Вывод не делается.")
    caveat = ""
    if hh["form_a"] != hh["form_b"] and not hh["household_valid"]:
        caveat = (f"Показатели получены из разных форм ({hh['form_a']} и {hh['form_b']}). В синтетических данных "
                  f"такие формы связаны только через общий каркас домохозяйства (регион, тип поселения, состав).")
    return {"question": q, "vars": [a, b], "labels": [la, lb], "level": level, "verdict": verdict,
            "caveat": caveat, "claims": claims, "audit": hh, "regional": rg}


# ---------------------------------------------------------------- защита от выдуманных чисел
NUM_RE = re.compile(r"[−-]?\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> set[str]:
    return {n.replace("−", "-").replace(",", ".").lstrip("+") for n in NUM_RE.findall(text)}


def grounding_check(text: str, facts: dict) -> list[str]:
    """Вернёт числа из текста, которых нет среди фактов (с допуском округления)."""
    import json
    allowed = numbers_in(json.dumps(facts, ensure_ascii=False))
    allowed_f = []
    for s in allowed:
        try:
            allowed_f.append(float(s))
        except ValueError:
            pass
    bad = []
    for s in numbers_in(text):
        try:
            v = float(s)
        except ValueError:
            continue
        if not any(abs(v - f) <= 0.006 or abs(abs(v) - abs(f)) <= 0.006 for f in allowed_f):
            bad.append(s)
    return bad
