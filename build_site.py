"""Публичный сайт Provable (GitHub Pages): главная, анализ любой таблицы в браузере, отчёт по обследованию.

    python3 qol_ai/build_site.py      → qol_ai/site/

Без сервера: таблицу считает движок audit_engine.js прямо в браузере, данные никуда не отправляются.
Языковая модель на публичном сайте не подключается (ключи не публикуются): вопросы понимает словарь.
Производные от микроданных организаторов примеры (домохозяйства, доходы людей) на сайт не попадают —
только выдуманная таблица предприятий с известным ответом.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
if SITE.exists():
    shutil.rmtree(SITE)
(SITE / "demo").mkdir(parents=True)

home = (HERE / "home.html").read_text()
home = home.replace('href="audit"', 'href="audit.html"').replace('href="report"', 'href="report.html"')
(SITE / "index.html").write_text(home)

audit = (HERE / "audit_app.html").read_text().replace('href="report"', 'href="report.html"')
(SITE / "audit.html").write_text(audit)

shutil.copy(HERE / "out" / "provable.html", SITE / "report.html")
shutil.copy(HERE / "audit_engine.js", SITE / "audit_engine.js")
shutil.copy(HERE / "demo_data" / "предприятия_синтетика.csv", SITE / "demo" / "предприятия_синтетика.csv")
(SITE / ".nojekyll").write_text("")
print("готово:", sorted(str(p.relative_to(SITE)) for p in SITE.rglob("*") if p.is_file()))
