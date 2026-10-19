"""Сравнение с обычной языковой моделью (как чат-бот без доступа к расчётам).

Каждое утверждение из контрольных наборов задаётся Gemini с одинаковым запросом: подтверждается ли оно
данными обследования домохозяйств Казахстана. Ответ: подтверждается / не подтверждается / нельзя проверить.
«Подтверждается» засчитывается как принятие; всё остальное — как непринятие.

Наборы: test2 (30 утверждений), test3 (20), team (36). Правильные ответы — те же, что для Provable.

    python3 qol_ai/eval/llm_baseline.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import llm_api as L  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPT = ("Ты аналитик официальной статистики. Ниже утверждение о домохозяйствах Казахстана по данным обследования "
          "(формы о качестве жизни, бюджете, составе семьи и жилье; около 24 тысяч семей, 20 регионов, 2023–2024). "
          "Подтверждается ли это утверждение такими данными? Ответь одним словом: «подтверждается», «не подтверждается» "
          "или «нельзя проверить».\n\nУтверждение: ")


def _norm(a: str) -> str:
    a = a.strip().lower()
    if "не подтверждается" in a:
        return "не подтверждается"
    if "нельзя" in a:
        return "нельзя проверить"
    return "подтверждается" if "подтверждается" in a else a[:40]


def ask(text: str):
    """Ответ модели и имя модели. При лимите (429) — пауза и повтор, затем следующая модель цепочки."""
    import time, urllib.error
    for prov, model, key in L.chain():
        body = {"model": model, "temperature": 0, "max_tokens": 2000,
                "messages": [{"role": "user", "content": PROMPT + text}]}
        if prov == "gemini":
            body["reasoning_effort"] = "low"
        for wait in (0, 15, 30, 45):
            time.sleep(wait)
            try:
                req = urllib.request.Request(L.BASE[prov] + "/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=60, context=L.SSL) as r:
                    return _norm(json.load(r)["choices"][0]["message"]["content"]), f"{prov}:{model}"
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503):
                    break
            except Exception:
                break
    return None, None


def main():
    out, summary = [], {}
    for name in ["test2", "test3", "team"]:
        B = json.loads((HERE / f"benchmark_{name}.json").read_text())
        acc = fa = fr = n_false = 0
        for c in B["claims"]:
            v, used = ask(c["text"])
            if v is None:
                raise SystemExit("ни одна модель не ответила")
            time.sleep(1)
            accepted = v == "подтверждается"
            acc += accepted == c["gold_ok"]
            fa += accepted and not c["gold_ok"]
            fr += (not accepted) and c["gold_ok"]
            n_false += not c["gold_ok"]
            out.append({"set": name, "text": c["text"], "gold_ok": c["gold_ok"], "llm": v, "model": used})
        summary[name] = {"claims": len(B["claims"]), "correct": acc, "false_accepts": fa, "false_claims": n_false, "false_rejects": fr}
        print(name, summary[name], flush=True)
    from collections import Counter
    model = dict(Counter(r["model"] for r in out))
    (HERE / "llm_baseline_results.json").write_text(json.dumps({"model": model, "summary": summary, "rows": out}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
