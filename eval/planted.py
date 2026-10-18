"""Проверка метода на внесённых связях: знаем правильный ответ — смотрим, угадывает ли Provable.

Данные моделируются по устройству реального обследования: 20 регионов, 2 года, город/село.
В каждую пару показателей заранее закладывается один из трёх сценариев:
  * HOUSEHOLD — настоящая связь между семьями (разной силы);
  * REGIONAL  — связь только через различия регионов, внутри региона связи нет;
  * NONE      — связи нет вовсе.
Затем к паре применяется тот же движок, что и к реальным данным (engine.audit_pair, engine.regional_pair),
и вердикт сравнивается с заложенным.

    python3 qol_ai/eval/planted.py     # около 3 минут
Результат: eval/planted_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine as E  # noqa: E402

N = 12000          # домохозяйств (половина реального объёма — консервативно)
REPS = 6           # повторов на сценарий
SCENARIOS = [("HOUSEHOLD", 0.20), ("HOUSEHOLD", 0.10), ("HOUSEHOLD", 0.05),
             ("REGIONAL", 1.0), ("REGIONAL", 0.5), ("NONE", 0.0)]


def simulate(kind: str, strength: float, rng) -> pd.DataFrame:
    region = rng.integers(0, 20, N)
    year = rng.integers(2023, 2025, N)
    urban = rng.integers(0, 2, N)
    df = pd.DataFrame({"year": year, "region": region, "urban": urban})
    df["cell"] = df.year.astype(str) + "_" + df.region.astype(str) + "_" + df.urban.astype(str)
    reg_a = rng.normal(0, 1, 20)[region]          # у каждого региона свой уровень показателей
    reg_b = rng.normal(0, 1, 20)[region]
    if kind == "HOUSEHOLD":
        z = rng.normal(0, 1, N)
        a = reg_a + np.sqrt(strength) * z + rng.normal(0, 1, N)
        b = reg_b + np.sqrt(strength) * z + rng.normal(0, 1, N)
    elif kind == "REGIONAL":
        shared = rng.normal(0, 1, 20)[region]     # общий региональный фактор, внутри региона связи нет
        a = reg_a * 0.3 + strength * shared + rng.normal(0, 1, N)
        b = reg_b * 0.3 + strength * shared + rng.normal(0, 1, N)
    else:
        a = reg_a + rng.normal(0, 1, N)
        b = reg_b + rng.normal(0, 1, N)
    df["a"], df["b"] = a, b
    return df


def verdict(df: pd.DataFrame, rng) -> tuple[str, float]:
    hh = E.audit_pair(df, "a", "b", rng)
    if hh["household_valid"]:
        return "HOUSEHOLD", hh["partial"]
    rg = E.regional_pair(df, "a", "b", rng)
    return ("REGIONAL" if rg["robust"] else "NONE"), hh["partial"]


def main():
    rng = np.random.default_rng(E.SEED)
    rows = []
    for kind, strength in SCENARIOS:
        for rep in range(REPS):
            df = simulate(kind, strength, rng)
            got, part = verdict(df, rng)
            rows.append({"truth": kind, "strength": strength, "rep": rep, "verdict": got, "partial": round(part, 4)})
            print(f"{kind:9} {strength:4}  → {got:9}  ρ в ячейках {part:+.3f}", flush=True)
    res = pd.DataFrame(rows)
    acc = float((res.truth == res.verdict).mean())
    by = {k: f"{(g.truth == g.verdict).sum()}/{len(g)}" for k, g in res.groupby(["truth", "strength"])}
    false_household = int(((res.truth != "HOUSEHOLD") & (res.verdict == "HOUSEHOLD")).sum())
    summary = {"pairs": len(res), "correct": int((res.truth == res.verdict).sum()), "accuracy": round(acc, 3),
               "false_household": false_household,
               "by_scenario": {f"{k[0]} {k[1]}": v for k, v in by.items()}, "n_households": N, "reps": REPS}
    out = Path(__file__).resolve().parent / "planted_results.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
