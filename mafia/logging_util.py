"""Structured logging, persistence, and replay.

Every game can be saved to a single JSON file containing the full (un-filtered)
event log plus the role assignments. That file is enough to (a) recompute every
metric and (b) replay the game as a human-readable transcript, including the
agents' private chain-of-thought if desired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .state import Event, EventType, GameState

# ANSI colours for the live/replay transcript (degrade gracefully if unused).
_C = {
    "dim": "\033[2m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m", "bold": "\033[1m",
    "reset": "\033[0m",
}


def _c(text: str, *styles: str, color: bool) -> str:
    if not color:
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


@dataclass
class GameLogger:
    """Collects events; optionally streams a transcript as the game runs."""

    verbose: bool = False
    show_reasoning: bool = False
    color: bool = True

    def __call__(self, event: Event) -> None:
        if self.verbose:
            line = format_event(event, show_reasoning=self.show_reasoning, color=self.color)
            if line:
                print(line)


def game_to_payload(state: GameState) -> dict:
    """Serialise a finished game to a plain dict (shared by file-save and the UI)."""
    return {
        "config": state.config,
        "winner": state.winner.value if state.winner else None,
        "players": [
            {"name": p.name, "role": p.role.value, "model": p.model,
             "partners": p.partners, "alive": p.alive}
            for p in state.players
        ],
        "events": [e.to_dict() for e in state.events],
        "beliefs": state.beliefs,
    }


def save_game(state: GameState, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(game_to_payload(state), indent=2))
    return path


def load_events(path: str | Path) -> tuple[dict, list[Event]]:
    data = json.loads(Path(path).read_text())
    events = [Event.from_dict(e) for e in data["events"]]
    return data, events


def format_event(event: Event, *, show_reasoning: bool, color: bool = True) -> str | None:
    t = event.type
    if t is EventType.REASONING:
        if not show_reasoning:
            return None
        return _c(f"    ↳ ({event.actor} thinks) {event.content}", "dim", color=color)
    if t is EventType.GAME_START:
        return _c(f"\n=== {event.content} ===", "bold", "cyan", color=color)
    if t is EventType.NIGHT_START:
        return _c(f"\n🌙 Night {event.day}", "bold", "blue", color=color)
    if t is EventType.DAY_START:
        return _c(f"\n☀️  Day {event.day} — {event.content}", "bold", "yellow", color=color)
    if t is EventType.INVESTIGATION:
        return _c(f"   🔎 {event.actor} investigated {event.target}: "
                  f"{event.meta.get('faction','?').upper()}", "magenta", color=color)
    if t is EventType.NIGHT_ACTION:
        kind = event.meta.get("kind", "action")
        return _c(f"   • {event.actor} ({kind}) → {event.target}", "dim", color=color)
    if t is EventType.KILL:
        return _c(f"   💀 {event.target} was killed in the night "
                  f"(was {event.meta.get('role','?')}).", "red", color=color)
    if t is EventType.SAVE:
        return _c("   🛡️  The mafia struck, but the doctor saved the target!", "green", color=color)
    if t is EventType.SPEECH:
        return f"   {_c(event.actor + ':', 'bold', color=color)} {event.content}"
    if t is EventType.VOTE:
        line = _c(f"   🗳️  {event.actor} votes for {event.target}.", "cyan", color=color)
        cites = (event.meta or {}).get("citations", [])
        for c in cites:
            line += "\n" + _c(f"        ↳ {c}", "dim", color=color)
        return line
    if t is EventType.ELIMINATION:
        return _c(f"   ⚖️  {event.target} was voted out (was {event.meta.get('role','?')}).",
                  "red", color=color)
    if t is EventType.GAME_OVER:
        return _c(f"\n🏁 {event.content}", "bold", "green", color=color)
    return None


def replay(path: str | Path, *, show_reasoning: bool = False, color: bool = True) -> None:
    """Print a saved game back as a transcript."""
    data, events = load_events(path)
    roles = {p["name"]: p["role"] for p in data["players"]}
    print(_c("Roles: " + ", ".join(f"{n}={r}" for n, r in roles.items()), "dim", color=color))
    for e in events:
        line = format_event(e, show_reasoning=show_reasoning, color=color)
        if line is not None:
            print(line)
