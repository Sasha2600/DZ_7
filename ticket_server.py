"""Локальная тикет-система для DZ_7, кейс 1 (мок внешней интеграции).

Предоставляет тикет-функцию «создать тикет» по HTTP (JSON), чтобы агент
мог вызывать инструмент create_ticket «через протокол» — как внешнюю
систему, а не функцию в своём процессе. Сервер — тонкая прослойка над
файлом data/tickets.json: append-лог тикетов, атомарная запись
(.tmp + os.replace).

Только stdlib (http.server), без внешних зависимостей.

Запуск:
    .venv/bin/python ticket_server.py
Эндпоинты:
    POST /tickets  {"title", "priority", "body"} -> {"ticket_id", "status"}
    GET  /health   -> {"status": "ok", "tickets": N}
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    """Относительный путь — от корня проекта (где лежит ticket_server.py)."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


# Путь к файлу тикетов. Резолвится от каталога модуля при импорте —
# сервер не зависит от CWD (ловушка из DZ_3). Мутация
# ticket_server.TICKETS_PATH (selftest) учитывается: при каждом обращении
# читается текущее значение глобального имени.
TICKETS_PATH = _resolve("data/tickets.json")

# Защита concurrent-append в ThreadingHTTPServer.
_WRITE_LOCK = threading.Lock()


def load_tickets(path: Optional[str] = None) -> list[dict]:
    """Прочитать тикеты из data/tickets.json (нет файла = пустой список)."""
    path = path or TICKETS_PATH
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tickets", [])


def append_ticket(ticket: dict, path: Optional[str] = None) -> None:
    """Дописать тикет в data/tickets.json. Атомарно: .tmp + os.replace."""
    path = path or TICKETS_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with _WRITE_LOCK:
        tickets = load_tickets(path)
        tickets.append(ticket)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"tickets": tickets}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> tuple[Optional[dict], Optional[str]]:
        """Читает и разбирает JSON-тело запроса. (obj, ошибка)."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return None, "тело запроса пустое"
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return None, f"тело запроса не является корректным JSON: {e}"
        if not isinstance(obj, dict):
            return None, "тело запроса должно быть JSON-объектом"
        return obj, None

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/tickets":
            self._send_json(404, {"status": "error",
                                  "error": f"неизвестный путь {path}"})
            return
        body, err = self._read_json_body()
        if err:
            self._send_json(400, {"status": "error", "error": err})
            return
        missing = [k for k in ("title", "priority", "body") if not body.get(k)]
        if missing:
            self._send_json(400, {"status": "error",
                                  "error": f"нет обязательных полей: {', '.join(missing)}"})
            return
        ticket = {
            "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "status": "open",
            "title": body["title"],
            "priority": body["priority"],
            "body": body["body"],
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        append_ticket(ticket)
        self._send_json(201, {"ticket_id": ticket["ticket_id"], "status": ticket["status"]})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok", "tickets": len(load_tickets())})
        else:
            self._send_json(404, {"status": "error",
                                  "error": f"неизвестный путь {path}"})

    def log_message(self, *args) -> None:
        # Тихий сервер: не шумим в консоли при каждом запросе.
        pass


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """Собрать HTTP-сервер (используется и в main, и в selftest in-process)."""
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    host = os.environ.get("TICKET_API_HOST", "127.0.0.1")
    port = int(os.environ.get("TICKET_API_PORT", "8766"))
    server = make_server(host, port)
    print(f"Тикет-сервис: http://{host}:{port} | файл: {TICKETS_PATH}")
    print("Эндпоинты: POST /tickets, GET /health. Остановка — Ctrl-C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
