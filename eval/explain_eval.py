"""Замер ИИ-объяснений: сколько пересказов вердикта модель пишет без чисел вне расчётов.

    python3 qol_ai/eval/explain_eval.py
Берутся вопросы контрольного набора test3, для каждого распознанной пары — объяснение на русском и казахском.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine as E  # noqa: E402
import llm_api  # noqa: E402

HERE = Path(__file__).resolve().parent
d = E.load()
bench = json.loads((HERE / "benchmark_test3.json").read_text())
rows, seen = [], set()
for q in bench["questions"]:
    if q["gold"] == "REFUSE":
        continue
    key = tuple(sorted(q["gold"]))
    if key in seen:
        continue
    seen.add(key)
    a, b = q["gold"]
    r = E.answer(d, f"{E.CATALOG[a][0]} и {E.CATALOG[b][0]}")
    if len(r.get("vars", [])) < 2:
        continue
    facts = {"labels": r["labels"], "level": r["level"], "verdict": r["verdict"], "claims": r["claims"],
             "k_regions": 20, "n": r["audit"]["n"]}
    for lang in ("ru", "kk"):
        x = llm_api.explain(facts, lang)
        rows.append({"pair": key, "lang": lang, "ok": x["text"] is not None and not x["blocked"],
                     "blocked": x["blocked"], "bad": x["bad"], "text": x["text"], "model": x["model"]})
        print(lang, key, "OK" if rows[-1]["ok"] else f"БЛОК {x['bad']}", flush=True)
summ = {"explanations": len(rows), "passed": sum(r["ok"] for r in rows), "blocked": sum(r["blocked"] for r in rows)}
(HERE / "explain_results.json").write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1))
print(summ)
