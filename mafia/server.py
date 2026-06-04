"""A dependency-free web server for watching and playing games in the browser.

Uses only the Python standard library. It serves a single-page UI plus a JSON
API with two modes:

Watch (autonomous AI vs AI):
    GET  /api/game?players=6&seed=3&rounds=2        -> a full game as JSON

Play (human in one seat against the agents):
    POST /api/newgame  {players,rounds,seed,role}   -> create session, first view
    POST /api/act      {session, action}            -> submit your action, new view
    GET  /api/view?session=...                      -> current view

Start it with:  python -m mafia serve   (then open http://localhost:8000)
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .arena import run_game
from .engine import GameConfig
from .logging_util import game_to_payload
from .session import GameSession

STATIC_DIR = Path(__file__).parent / "static"

_SESSIONS: dict[str, GameSession] = {}
_LOCK = threading.Lock()
_MAX_SESSIONS = 200


def _int(params: dict, key: str, default: int) -> int:
    try:
        return int(params.get(key, [str(default)])[0])
    except (ValueError, TypeError, IndexError):
        return default


def _parse_seed(raw) -> int | None:
    if raw in (None, "", "random"):
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _config(players: int, rounds: int, seed) -> GameConfig:
    return GameConfig(
        num_players=max(5, min(players, 12)),
        discussion_rounds=max(0, min(rounds, 5)),
        seed=_parse_seed(seed),
    )


def _build_watch_game(params: dict) -> dict:
    cfg = _config(_int(params, "players", 6), _int(params, "rounds", 2),
                  params.get("seed", [None])[0])
    provider = params.get("provider", ["mock"])[0]
    model = params.get("model", [None])[0]
    state, metrics = run_game(cfg, provider=provider, model=model)
    payload = game_to_payload(state)
    payload["metrics"] = metrics.to_dict()
    return payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quieter console
        pass

    # ---------------------------------------------------------------- plumbing
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            return {}

    # -------------------------------------------------------------------- GET
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            return self._send(200, (STATIC_DIR / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        params = parse_qs(parsed.query)
        if parsed.path == "/api/game":
            try:
                return self._json(_build_watch_game(params))
            except Exception as exc:  # surface engine errors to the UI
                return self._json({"error": str(exc)}, 500)
        if parsed.path == "/api/view":
            sid = params.get("session", [""])[0]
            with _LOCK:
                sess = _SESSIONS.get(sid)
            if not sess:
                return self._json({"error": "unknown session"}, 404)
            return self._json(sess.view())
        self._send(404, b"not found", "text/plain")

    # -------------------------------------------------------------------- POST
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_json()
        try:
            if parsed.path == "/api/newgame":
                cfg = _config(int(body.get("players", 7)), int(body.get("rounds", 2)),
                              body.get("seed"))
                sess = GameSession(cfg, human_role=body.get("role"),
                                   provider=body.get("provider", "mock"),
                                   model=body.get("model"))
                with _LOCK:
                    if len(_SESSIONS) >= _MAX_SESSIONS:
                        _SESSIONS.clear()
                    _SESSIONS[sess.id] = sess
                return self._json(sess.view())
            if parsed.path == "/api/act":
                sid = body.get("session", "")
                with _LOCK:
                    sess = _SESSIONS.get(sid)
                if not sess:
                    return self._json({"error": "unknown session"}, 404)
                sess.submit(body.get("action", {}))
                return self._json(sess.view())
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)
        self._send(404, b"not found", "text/plain")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Mafia Arena UI running at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()
