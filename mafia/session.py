"""Interactive (human-in-the-loop) game sessions.

A :class:`GameSession` lets one human play a single seat against LLM (or mock)
agents. It owns an :class:`~mafia.engine.Engine` and drives its
``interactive_flow`` generator, pausing whenever the human must act and exposing
a fully *information-filtered* view of the game (the human only ever sees what
their role is entitled to see).
"""

from __future__ import annotations

import random
import uuid

from .engine import Engine, GameConfig
from .evaluation import evaluate_game
from .llm import LLMClient, build_client
from .roles import Role
from .state import EventType


class GameSession:
    def __init__(self, config: GameConfig, *, human_role: str | None = None,
                 provider: str = "mock", model: str | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.config = config

        def client_factory(name: str) -> LLMClient:
            seed = None if config.seed is None else hash((config.seed, name)) & 0xFFFFFFFF
            return build_client(provider, model=model, seed=seed)

        self.engine = Engine.setup(config, client_factory)
        self.human = self._choose_seat(human_role)
        self.gen = self.engine.interactive_flow(self.human)
        self.pending: dict | None = None
        self.done = False
        self._advance(None)

    # -------------------------------------------------------------- seat choice
    def _choose_seat(self, human_role: str | None) -> str:
        players = self.engine.state.players
        if human_role:
            want = human_role.lower()
            matches = [p for p in players if p.role.value == want]
            if matches:
                return matches[0].name
        rng = random.Random(self.config.seed)
        return rng.choice(players).name

    # ------------------------------------------------------------------ driving
    def _advance(self, send_value: dict | None) -> None:
        try:
            self.pending = self.gen.send(send_value)
        except StopIteration:
            self.pending = None
            self.done = True

    def submit(self, action: dict) -> None:
        if self.done or self.pending is None:
            raise RuntimeError("No pending action to submit.")
        self._advance(action)

    # --------------------------------------------------------------------- view
    def view(self) -> dict:
        state = self.engine.state
        human = self.human
        me = state.player(human)
        game_over = state.winner is not None

        # information-filtered transcript (drop other players' private events and
        # our own raw reasoning echoes to keep the human's feed clean)
        events = [
            e.to_dict() for e in state.visible_events(human)
            if e.type is not EventType.REASONING
        ]

        roster = []
        for p in state.players:
            reveal = (not p.alive) or p.name == human or game_over
            roster.append({
                "name": p.name,
                "alive": p.alive,
                "role": p.role.value if reveal else None,
                "is_you": p.name == human,
            })

        known = {}
        for e in state.events:
            if e.type is EventType.INVESTIGATION and e.actor == human and e.target:
                known[e.target] = e.meta.get("faction")

        you = {
            "name": human,
            "role": me.role.value,
            "faction": me.faction.value,
            "alive": me.alive,
            "partners": list(me.partners),
            "known_factions": known,
            "ability": me.role.public_description,
        }

        out = {
            "session": self.id,
            "you": you,
            "roster": roster,
            "events": events,
            "day": state.day,
            "phase": state.phase.value,
            "pending": None if self.done else self.pending,
            "done": self.done,
            "winner": state.winner.value if state.winner else None,
            "living": state.living_names(),
        }
        if game_over:
            out["metrics"] = evaluate_game(state).to_dict()
            out["reveal_roles"] = {p.name: p.role.value for p in state.players}
        return out
