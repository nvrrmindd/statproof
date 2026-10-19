"""Provable с ИИ-режимом: облачная модель (Gemini/Groq, бесплатные лимиты) понимает вопросы.

    python3 qol_ai/serve.py        # затем откройте out/provable.html (или http://127.0.0.1:8765/)

Ключи — в qol_ai/.env (см. llm_api.py). В модель уходит только текст вопроса, данные домохозяйств не отправляются.
Утверждения и записки по-прежнему проверяет только детерминированная логика.
Без сервера или без интернета приложение работает как обычно, на словаре.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import llm_api

APP = Path(__file__).resolve().parent / "out" / "provable.html"
PORT = 8765


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
        elif self.path == "/api/health":
            ch = llm_api.chain()
            self._send(200, {"ok": bool(ch), "model": ch[0][1] if ch else None})
        else:
            self._send(404, {"error": "не найдено"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
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
    print(f"Provable ИИ-режим: http://127.0.0.1:{PORT}/  модели: {', '.join(p + ':' + m for p, m, _ in ch) or 'ключи не найдены'}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
