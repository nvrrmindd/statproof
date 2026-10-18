"""Локальный сервер Provable: открывает приложение и подключает локальную языковую модель.

    ollama serve                  # в отдельном окне (модель: llm.MODEL)
    python3 qol_ai/serve.py       # затем откройте http://127.0.0.1:8765/

Без сервера приложение out/provable.html работает как обычный файл, только со словарём.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import llm

APP = Path(__file__).resolve().parent / "out" / "provable.html"
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, APP.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, {"ok": llm.available(), "model": llm.MODEL})
        else:
            self._send(404, {"error": "не найдено"})

    def do_POST(self):
        if self.path != "/api/understand":
            return self._send(404, {"error": "не найдено"})
        n = int(self.headers.get("Content-Length", 0))
        text = json.loads(self.rfile.read(n) or b"{}").get("text", "")[:1000]
        self._send(200, llm.understand(text) or {})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Provable: http://127.0.0.1:{PORT}/  (модель {llm.MODEL}: {'подключена' if llm.available() else 'не найдена, работает словарь'})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
