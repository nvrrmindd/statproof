"""Проверка логики Provable на всех 253 парах показателей.

Для каждой пары автоматически строятся утверждения семи типов с известным правильным ответом:
  1. вывод о семьях без чисел                → верен, только если уровень HOUSEHOLD
  2. вывод о регионах                        → верен, если уровень не NONE
  3. вывод о семьях с верным ρ               → верен, только если уровень HOUSEHOLD
  4. вывод с неверным ρ (сдвиг на 0,3)       → всегда неверен
  5. вывод о семьях с верным направлением    → верен, только если уровень HOUSEHOLD
  6. вывод о семьях с обратным направлением  → неверен
  7. «связи нет» (о семьях)                  → верен, только если уровень не HOUSEHOLD

Понимание языка здесь не проверяется: показатели передаются логике напрямую. Проверяется, что вердикт
приложения (функция claimVerdict) применяет уровни, числа, направление и отрицание без ошибок.
Правильные ответы берутся из тех же расчётов (out/findings.json), поэтому это проверка логики, а не метода;
метод проверяется отдельно на данных с заложенным ответом (eval/planted.py).

    python3 qol_ai/eval/logic_eval.py
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "out" / "provable.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def fmt(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):.2f}".replace(".", ",")


def build():
    f = json.loads((HERE.parent / "out" / "findings.json").read_text())
    items = []
    for p in f["pairs"]:
        a, b, lv, part = p["a"], p["b"], p["level"], p["partial"]
        A, B = f["catalog"][a]["label"], f["catalog"][b]["label"]
        sign = 1 if part > 0 else -1
        base = {"vars": [a, b], "regional": False, "negated": False, "direction": 0, "fracPct": None}
        add = lambda t, kind, u, gold: items.append({"text": t, "kind": kind, "u": {**base, **u}, "gold": gold, "level": lv})
        add(f"{A} связано с показателем «{B}» у семей", "семьи", {}, lv == "HOUSEHOLD")
        add(f"Между регионами {A} связано с показателем «{B}»", "регионы", {"regional": True}, lv != "NONE")
        add(f"У семей {A} и {B} связаны: ρ = {fmt(part)}", "верное число", {}, lv == "HOUSEHOLD")
        add(f"У семей {A} и {B} связаны: ρ = {fmt(part + 0.3 if part < 0.5 else part - 0.3)}", "неверное число", {}, False)
        if abs(part) >= 0.01:
            add(f"У семей {A} и {B}: прямое или обратное направление по данным", "верное направление", {"direction": sign}, lv == "HOUSEHOLD")
            add(f"У семей {A} и {B}: направление противоположное данным", "обратное направление", {"direction": -sign}, False)
        add(f"У семей {A} и {B} не связаны", "«связи нет»", {"negated": True}, lv != "HOUSEHOLD")
    return items


def main():
    items = build()
    inj = """<script>setTimeout(() => { try { const I = %s;
      const r = I.map(x => { const v = claimVerdict(x.text, x.u); return v.ok === true; });
      console.log('EVALRES ' + JSON.stringify(r)); } catch (e) { console.log('EVALERR ' + e); } }, 300);</script></body>""" % json.dumps(items, ensure_ascii=False)
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "logic.html"
        page.write_text(APP.read_text().replace("</body>", inj))
        out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--enable-logging=stderr", "--v=0",
                              "--virtual-time-budget=8000", "--dump-dom", page.as_uri()], capture_output=True, text=True, timeout=180)
    m = re.search(r'"EVALRES (\[.*?\])"', out.stderr)
    if not m:
        raise SystemExit("Оценка не выполнилась")
    pred = json.loads(m.group(1))
    by = {}
    for it, ok in zip(items, pred):
        k = by.setdefault(it["kind"], [0, 0])
        k[0] += ok == it["gold"]
        k[1] += 1
    false_acc = sum(1 for it, ok in zip(items, pred) if ok and not it["gold"])
    summ = {"claims": len(items), "correct": sum(v[0] for v in by.values()), "false_accepts": false_acc,
            "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in by.items()}}
    (HERE / "logic_results.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    for it, ok in zip(items, pred):
        if ok != it["gold"]:
            print("  ошибка:", it["kind"], it["level"], it["text"][:90], "→", ok)


if __name__ == "__main__":
    main()
