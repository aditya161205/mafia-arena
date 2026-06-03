"""A tiny dependency-free web server for watching games in the browser.

Uses only the Python standard library. It serves a single-page UI and a JSON
API that runs a fresh game on demand with the engine:

    GET /                       -> the viewer (static HTML)
    GET /api/game?players=6&seed=3&rounds=2&provider=mock
                                -> a full game as JSON (events + roles)

Start it with:  python -m mafia serve   (then open http://localhost:8000)
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .arena import run_game
from .engine import GameConfig
from .logging_util import game_to_payload

STATIC_DIR = Path(__file__).parent / "static"


def _build_game(params: dict[str, list[str]]) -> dict:
    def get_int(key: str, default: int) -> int:
        try:
            return int(params.get(key, [str(default)])[0])
        except (ValueError, TypeError):
            return default

    provider = params.get("provider", ["mock"])[0]
    model = params.get("model", [None])[0]
    seed_raw = params.get("seed", [None])[0]
    seed = None
    if seed_raw not in (None, "", "random"):
        try:
            seed = int(seed_raw)
        except ValueError:
            seed = None

    cfg = GameConfig(
        num_players=max(5, min(get_int("players", 6), 12)),
        discussion_rounds=max(0, min(get_int("rounds", 2), 5)),
        memory_limit=None,
        seed=seed,
    )
    state, metrics = run_game(cfg, provider=provider, model=model)
    payload = game_to_payload(state)
    payload["metrics"] = metrics.to_dict()
    payload["seed"] = seed
    return payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quieter console
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")

        if parsed.path == "/api/game":
            try:
                payload = _build_game(parse_qs(parsed.query))
                body = json.dumps(payload).encode()
                return self._send(200, body, "application/json")
            except Exception as exc:  # surface engine errors to the UI
                body = json.dumps({"error": str(exc)}).encode()
                return self._send(500, body, "application/json")

        self._send(404, b"not found", "text/plain")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"Mafia Arena UI running at {url}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()
