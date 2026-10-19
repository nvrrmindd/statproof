"""Точность и полнота анализа на таблицах с заранее известным ответом (универсальный движок generic.py).

300 смоделированных таблиц разного устройства: 2 000–10 000 строк, 10–30 групп («регионов»), город/село, 2 года.
В каждую таблицу закладывается одна связь между показателями a и b:
  * UNIT  — настоящая связь между объектами (сила 0,05 / 0,1 / 0,2 / 0,4);
  * GROUP — связь только через различия групп, внутри групп связи нет (сила 0,5 / 1,0);
  * NONE  — связи нет.
По 100 таблиц на каждый тип. Затем движок выносит вердикт, и он сравнивается с заложенным.

Метрики, как у оценки извлечения правил: точность (precision) — какая доля вердиктов «верно для каждого»
действительно верна; полнота (recall) — какую долю настоящих связей движок нашёл. Отдельно — главная ошибка:
вердикт «верно для каждого» там, где связи между объектами нет.

    python3 qol_ai/eval/planted_generic.py     # около 2 минут
Результат: eval/planted_generic_results.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generic as G  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZES, GROUPS = [2000, 5000, 10000], [10, 20, 30]
PLAN = [("UNIT", s) for s in (0.05, 0.1, 0.2, 0.4) for _ in range(25)] + \
       [("GROUP", s) for s in (0.5, 1.0) for _ in range(50)] + [("NONE", 0.0)] * 100


def table(kind, s, n, g, rng):
    grp = rng.integers(0, g, n)
    df = pd.DataFrame({"группа": grp, "город_село": rng.integers(0, 2, n), "год": rng.integers(2023, 2025, n)})
    ra, rb = rng.normal(0, 1, g)[grp], rng.normal(0, 1, g)[grp]
    if kind == "UNIT":
        z = rng.normal(0, 1, n)
        a, b = ra + np.sqrt(s) * z + rng.normal(0, 1, n), rb + np.sqrt(s) * z + rng.normal(0, 1, n)
    elif kind == "GROUP":
        shared = rng.normal(0, 1, g)[grp]
        a, b = 0.3 * ra + s * shared + rng.normal(0, 1, n), 0.3 * rb + s * shared + rng.normal(0, 1, n)
    else:
        a, b = ra + rng.normal(0, 1, n), rb + rng.normal(0, 1, n)
    df["a"], df["b"] = a, b
    return df


def main():
    rng = np.random.default_rng(20261020)
    rows, t0 = [], time.time()
    for i, (kind, s) in enumerate(PLAN):
        n, g = SIZES[i % 3], GROUPS[(i // 3) % 3]
        rep = G.audit_table(table(kind, s, n, g, rng), ["a", "b"], ["группа", "город_село", "год"], "группа", "год", seed=1000 + i)
        rows.append({"truth": kind, "strength": s, "rows": n, "groups": g, "verdict": rep["pairs"][0]["level"],
                     "partial": round(rep["pairs"][0]["partial"], 4)})
    pred_unit = [r for r in rows if r["verdict"] == "UNIT"]
    true_unit = [r for r in rows if r["truth"] == "UNIT"]
    tp = sum(r["truth"] == "UNIT" for r in pred_unit)
    pred_grp = [r for r in rows if r["verdict"] == "GROUP"]
    true_grp = [r for r in rows if r["truth"] == "GROUP"]
    tpg = sum(r["truth"] == "GROUP" for r in pred_grp)
    summary = {
        "tables": len(rows),
        "accuracy": round(sum(r["truth"] == r["verdict"] for r in rows) / len(rows), 4),
        "unit_precision": round(tp / max(len(pred_unit), 1), 4), "unit_recall": round(tp / len(true_unit), 4),
        "group_precision": round(tpg / max(len(pred_grp), 1), 4), "group_recall": round(tpg / len(true_grp), 4),
        "false_unit": sum(r["truth"] != "UNIT" for r in pred_unit),
        "none_correct": sum(r["verdict"] == "NONE" for r in rows if r["truth"] == "NONE"),
        "unit_recall_by_strength": {str(s): round(sum(r["verdict"] == "UNIT" for r in true_unit if r["strength"] == s) / 25, 2) for s in (0.05, 0.1, 0.2, 0.4)},
        "confusion": {f"{t}->{v}": c for (t, v), c in sorted(Counter((r["truth"], r["verdict"]) for r in rows).items())},
        "seconds": round(time.time() - t0, 1)}
    (HERE / "planted_generic_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
