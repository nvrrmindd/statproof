"""Сервер Provable.

    python3 qol_ai/serve.py
    http://127.0.0.1:8799/audit   — аудит любой таблицы: загрузите CSV, получите отчёт
    http://127.0.0.1:8799/        — готовый отчёт по обследованию домохозяйств (ИИ-режим включится сам)

Ключи — в qol_ai/.env (см. llm_api.py). В модель уходит только текст вопроса, данные домохозяйств не отправляются.
Утверждения и записки по-прежнему проверяет только детерминированная логика.
Без сервера или без интернета приложение работает как обычно, на словаре.
"""
import io
import json
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

import generic
import llm_api

HERE = Path(__file__).resolve().parent
APP = HERE / "out" / "provable.html"
AUDIT_PAGE = HERE / "audit_app.html"
DEMOS = HERE / "demo_data"
TABLES: dict[str, pd.DataFrame] = {}      # загруженные таблицы, только в памяти этого компьютера
MAX_ROWS = 200_000
GROUP_NAME = re.compile(r"(регион|region|^te$|город|urban|^k$|год|year|^god|квартал|quarter|^kv|отрасл|oked|kato|район|district|область|пол$|^pol$)", re.I)


def read_csv_text(text: str) -> pd.DataFrame:
    head = text.split("\n", 1)[0]
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)


def describe(df: pd.DataFrame, preset: dict | None = None) -> dict:
    """Роли столбцов: показатель, группа или не использовать — по типу, числу значений и названию."""
    sampled = len(df) > MAX_ROWS
    if sampled:
        df = df.sample(n=MAX_ROWS, random_state=1)
    token = uuid.uuid4().hex
    TABLES[token] = df
    while len(TABLES) > 5:
        TABLES.pop(next(iter(TABLES)))
    cols, n_var = [], 0
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        is_num = num.notna().mean() >= 0.5
        u = int(s.nunique())
        if preset is not None:
            role = preset.get(c, "skip")
        elif u >= 0.9 * len(df) and u > 50:
            role = "skip"                                   # идентификатор
        elif GROUP_NAME.search(str(c)) and u <= 100:
            role = "group"
        elif not is_num:
            role = "group" if u <= 50 else "skip"
        elif u >= 3 and n_var < 12:
            role = "var"
        else:
            role = "skip"
        n_var += role == "var"
        ex = ", ".join(map(str, s.dropna().unique()[:4]))
        cols.append({"name": str(c), "role": role, "unique": u, "example": ex[:60],
                     "suggest_compare": bool(re.search(r"(регион|region|^te$|kato|область)", str(c), re.I)),
                     "suggest_repeat": bool(re.search(r"(год|year|^god)", str(c), re.I))})
    return {"token": token, "rows": int(len(df)), "sampled": sampled, "columns": cols}


def load_demo(name: str) -> dict:
    path = DEMOS / Path(name).name
    if not path.exists():
        raise FileNotFoundError("демо-набор не создан: запустите python3 qol_ai/make_demo.py")
    if path.name == "домохозяйства.csv":            # подписи показателей из каталога и разумные роли по умолчанию
        import engine
        df = engine.load(path)                       # добавляет производные столбцы (число лишений и др.)
        keep = ["income_pc", "dep_n", "life_sat", "fin_sat", "fies_raw", "dep_ritual", "n_children", "hh_size", "total_area", "borrowed"]
        ren = {k: engine.CATALOG[k][0] for k in keep}
        ren.update({"region": "регион", "urban": "город_село", "year": "год"})
        df = df.rename(columns=ren)[list(ren.values())]
        preset = {v: "var" for k, v in ren.items() if k in keep}
        preset.update({"регион": "group", "город_село": "group", "год": "group"})
        return describe(df, preset)
    return describe(pd.read_csv(path, low_memory=False))
PORT = 8799


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")     # страница может быть открыта как файл
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, APP.read_bytes(), "text/html; charset=utf-8")
        elif self.path in ("/audit", "/audit/"):
            self._send(200, AUDIT_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            ch = llm_api.chain()
            self._send(200, {"ok": bool(ch), "model": ch[0][1] if ch else None})
        else:
            self._send(404, {"error": "не найдено"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            if self.path == "/api/upload":                # CSV — тело запроса
                return self._send(200, describe(read_csv_text(raw.decode("utf-8-sig", errors="replace"))))
            if self.path == "/api/demo":
                return self._send(200, load_demo(json.loads(raw or b"{}").get("name", "")))
            if self.path == "/api/audit":
                q = json.loads(raw or b"{}")
                df = TABLES.get(q.get("token"))
                if df is None:
                    return self._send(400, {"error": "таблица не найдена, загрузите её снова"})
                rep = generic.audit_table(df, q["variables"][:15], q["cell_groups"], q["compare_group"], q.get("repeat_by"))
                return self._send(200, json.loads(json.dumps(rep, default=float)))
        except Exception as e:
            return self._send(400, {"error": str(e)[:300]})
        body = json.loads(raw or b"{}")
        if self.path == "/api/understand":
            self._send(200, llm_api.understand(str(body.get("text", ""))[:500]) or {})
        elif self.path == "/api/explain":   # пересказ вердикта; числа проверяются, при расхождении текст блокируется
            self._send(200, llm_api.explain(body.get("facts", {}), body.get("lang", "ru")))
        else:
            self._send(404, {"error": "не найдено"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ch = llm_api.chain()
    print(f"Provable: аудит любой таблицы — http://127.0.0.1:{PORT}/audit")
    print(f"          отчёт по домохозяйствам — http://127.0.0.1:{PORT}/  (модели: {', '.join(p + ':' + m for p, m, _ in ch) or 'нет ключей, работает словарь'})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
