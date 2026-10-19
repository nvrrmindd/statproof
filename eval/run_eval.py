"""Оценка качества ИИ-части Provable на наборах с известными ответами.

Прогоняет вопросы и утверждения через само приложение (out/provable.html) в браузере
без окна (Google Chrome), а не через копию логики, и считает точность.

    python3 qol_ai/eval/run_eval.py dev           # набор для разработки
    python3 qol_ai/eval/run_eval.py test          # контрольный набор, не используется для настройки
    python3 qol_ai/eval/run_eval.py dev --details # показать ошибки

Результат: eval/results_<набор>.json
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "out" / "provable.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def run_in_app(bench: dict, llm_map=None, strategy=None) -> dict:
    inj = """<script>setTimeout(() => { try {
      const B = %s;
      const L = %s, S = %s;
      const U = t => L ? understand(t, L[t], S) : understandRules(t);
      const q = B.questions.map(x => U(x.text).vars);
      const c = B.claims.map(x => { const r = claimVerdict(x.text, U(x.text)); return {ok: r.ok, vars: r.vars}; });
      console.log('EVALRES ' + JSON.stringify({q, c}));
    } catch (e) { console.log('EVALERR ' + e); } }, 300);</script></body>""" % (json.dumps(bench, ensure_ascii=False), json.dumps(llm_map, ensure_ascii=False), json.dumps(strategy))
    html = APP.read_text().replace("</body>", inj)
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "eval.html"
        page.write_text(html)
        out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--enable-logging=stderr", "--v=0",
                              "--virtual-time-budget=6000", "--dump-dom", page.as_uri()],
                             capture_output=True, text=True, timeout=120)
    m = re.search(r'"EVALRES (.*)", source', out.stderr)
    if not m:
        err = re.search(r'"EVALERR (.*?)"', out.stderr)
        raise SystemExit("Оценка не выполнилась: " + (err.group(1) if err else "нет ответа от страницы"))
    return json.loads(m.group(1).encode().decode("unicode_escape").encode("latin-1").decode("utf-8")) \
        if "\\u" in m.group(1) else json.loads(m.group(1))


def score(bench: dict, res: dict) -> dict:
    qrows, crows = [], []
    for item, pred in zip(bench["questions"], res["q"]):
        gold = item["gold"]
        if gold == "REFUSE":
            ok = len(pred) < 2
        else:
            ok = set(pred) == set(gold)
        qrows.append({"text": item["text"], "gold": gold, "pred": pred, "correct": ok})
    for item, pred in zip(bench["claims"], res["c"]):
        p = pred["ok"]
        kind = ("unverifiable" if p is None else
                "correct" if p == item["gold_ok"] else
                "false_accept" if p and not item["gold_ok"] else "false_reject")
        crows.append({"text": item["text"], "gold_ok": item["gold_ok"], "pred_ok": p, "vars": pred["vars"],
                      "result": kind, "why": item["why"]})
    nq, nc = len(qrows), len(crows)
    bad_claims = [r for r in crows if not r["gold_ok"]]
    summary = {
        "questions_total": nq,
        "questions_correct": sum(r["correct"] for r in qrows),
        "refusals_correct": sum(r["correct"] for r in qrows if r["gold"] == "REFUSE"),
        "refusals_total": sum(r["gold"] == "REFUSE" for r in qrows),
        "claims_total": nc,
        "claims_correct": sum(r["result"] == "correct" for r in crows),
        "false_claims_total": len(bad_claims),
        "false_claims_caught": sum(r["result"] in ("correct", "unverifiable") for r in bad_claims),
        "false_accepts": sum(r["result"] == "false_accept" for r in crows),
        "false_rejects": sum(r["result"] == "false_reject" for r in crows),
        "unverifiable": sum(r["result"] == "unverifiable" for r in crows),
    }
    return {"summary": summary, "questions": qrows, "claims": crows}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "dev"
    bench = json.loads((HERE / f"benchmark_{name}.json").read_text())
    llm_map, strategy = None, None
    if "--llm" in sys.argv:   # облачная модель: python3 eval/run_eval.py <набор> --llm <стратегия>
        sys.path.insert(0, str(HERE.parent))
        import llm_api
        strategy = sys.argv[sys.argv.index("--llm") + 1]
        llm_map = {}
        for t in [x["text"] for x in bench["questions"] + bench["claims"]]:
            u = llm_api.understand(t)
            if u is None:
                raise SystemExit("Облачная модель не ответила: проверьте интернет и ключи в qol_ai/.env")
            llm_map[t] = u
        name = f"{name}_llm_{strategy}"
    out = score(bench, run_in_app(bench, llm_map, strategy))
    (HERE / f"results_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    s = out["summary"]
    print(f"[{name}] вопросы: {s['questions_correct']}/{s['questions_total']} верно "
          f"(отказы {s['refusals_correct']}/{s['refusals_total']})")
    print(f"[{name}] утверждения: {s['claims_correct']}/{s['claims_total']} верно; "
          f"ложных пойманы {s['false_claims_caught']}/{s['false_claims_total']}; "
          f"ложно приняты {s['false_accepts']}, ложно отклонены {s['false_rejects']}, не проверено {s['unverifiable']}")
    if "--details" in sys.argv:
        for r in out["questions"]:
            if not r["correct"]:
                print("  В:", r["text"], "| ожидалось", r["gold"], "| получено", r["pred"])
        for r in out["claims"]:
            if r["result"] != "correct":
                print("  У:", r["result"], "|", r["text"], "| распознано", r["vars"], "|", r["why"])


if __name__ == "__main__":
    main()
