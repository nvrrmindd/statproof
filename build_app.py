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
    trap = [p for p in f["pairs"] if p["level"] == "REGIONAL" and "D002" in p["form_a"] + p["form_b"]]
    f["trap"] = sorted(trap, key=lambda p: -abs(p["reg"]["2024"]))[:8]
    import plotly  # plotly.js встраивается в файл: демо работает без интернета
    pjs = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text()
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(f, ensure_ascii=False))
    html = html.replace("/*__PLOTLY__*/", pjs.replace("</script", "<\\/script"))
    (OUT / "provable.html").write_text(html)
    print("written", OUT / "provable.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
