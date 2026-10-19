"""Шаг 4. Сборка интерактивного приложения → out/provable.html (один файл, работает без интернета:
Plotly встроен). Все числа берутся из out/findings.json."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
TEMPLATE = Path(__file__).resolve().parent / "app_template.html"


def main():
    f = json.loads((OUT / "findings.json").read_text())
    keep = ["a", "b", "n", "raw", "raw_ci", "partial", "partial_ci", "p_perm", "null95", "household_valid",
            "form_a", "form_b", "reg", "reg_ci", "reg_robust", "level"]
    f["pairs"] = [{k: p.get(k) for k in keep} for p in f["pairs"]]
    import engine  # ключевые слова каталога берутся из актуального engine.py
    for k, (label, form, kw) in engine.CATALOG.items():
        f["catalog"][k]["kw"] = kw
    trap = [p for p in f["pairs"] if p["level"] == "REGIONAL" and "D002" in p["form_a"] + p["form_b"]]
    f["trap"] = sorted(trap, key=lambda p: -abs(p["reg"]["2024"]))[:8]
    # результаты оценки качества ИИ (eval/run_eval.py), если они посчитаны
    ev = Path(__file__).resolve().parent / "eval"
    if (ev / "results_test2.json").exists() and (ev / "results_test2_v2.json").exists():
        f["eval"] = {k: json.loads((ev / fn).read_text())["summary"]
                     for k, fn in [("v1", "results_test2_v2.json"), ("v2", "results_test2.json")]}
    if (ev / "results_team_v3.json").exists():
        f["team"] = {"independent": json.loads((ev / "results_team_v3.json").read_text())["summary"],
                     "after": json.loads((ev / "results_team.json").read_text())["summary"]}
    if (ev / "results_test3.json").exists() and (ev / "results_test3_llm_merge.json").exists():
        f["llm_eval"] = {"rules": json.loads((ev / "results_test3.json").read_text())["summary"],
                         "llm": json.loads((ev / "results_test3_llm_merge.json").read_text())["summary"]}
    if (ev / "explain_results.json").exists():
        f["explain_eval"] = json.loads((ev / "explain_results.json").read_text())["summary"]
    if (ev / "planted_results.json").exists():
        f["planted"] = json.loads((ev / "planted_results.json").read_text())["summary"]
    # средние по 20 регионам для каждого показателя — для графика-доказательства в ответе
    import engine
    hh = OUT / "households.csv"
    if hh.exists():
        d = engine.load(hh)
        g = d.groupby(["year", "region"])[list(engine.CATALOG)].mean().round(4)
        f["regmeans"] = {str(y): {str(r): row.dropna().to_dict() for (yy, r), row in g.iterrows() if yy == y}
                         for y in sorted(d.year.unique())}
        f["regnames"] = {str(k): v for k, v in engine.REGIONS.items()}
    import plotly  # plotly.js встраивается в файл: демо работает без интернета
    pjs = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text()
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(f, ensure_ascii=False))
    html = html.replace("/*__PLOTLY__*/", pjs.replace("</script", "<\\/script"))
    (OUT / "provable.html").write_text(html)
    print("written", OUT / "provable.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
