"""The game engine: phase logic, action resolution, and win detection."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .agents import Action, Agent
from .llm import LLMClient
from .roles import Faction, Role, role_setup
from .state import Event, EventType, GameState, Phase, Player

DEFAULT_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank",
    "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Niaj",
]


@dataclass
class GameConfig:
    num_players: int = 6
    discussion_rounds: int = 2
    memory_limit: int | None = None
    max_days: int = 20
    seed: int | None = None
    names: list[str] = field(default_factory=lambda: list(DEFAULT_NAMES))


class Engine:
    def __init__(
        self,
        state: GameState,
        agents: dict[str, Agent],
        config: GameConfig,
        rng: random.Random,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.state = state
        self.agents = agents
        self.config = config
        self.rng = rng
        self.on_event = on_event

    # ----------------------------------------------------------- construction
    @classmethod
    def setup(
        cls,
        config: GameConfig,
        client_factory: Callable[[str], LLMClient],
        on_event: Callable[[Event], None] | None = None,
    ) -> "Engine":
        """Deal roles, wire up agents, and emit the opening event."""
        rng = random.Random(config.seed)
        roles = role_setup(config.num_players)
        rng.shuffle(roles)
        names = list(config.names)[: config.num_players]
        if len(names) < config.num_players:
            raise ValueError("Not enough player names for the requested table size.")

        players = [Player(name=n, role=r) for n, r in zip(names, roles)]
        mafia_names = [p.name for p in players if p.faction is Faction.MAFIA]
        for p in players:
            if p.faction is Faction.MAFIA:
                p.partners = [m for m in mafia_names if m != p.name]

        state = GameState(players=players, day=0, phase=Phase.NIGHT, config=vars(config).copy())
        agents = {
            p.name: Agent(p, client_factory(p.name), memory_limit=config.memory_limit)
            for p in players
        }
        engine = cls(state, agents, config, rng, on_event=on_event)
        engine._emit(Event(
            type=EventType.GAME_START, day=0, phase=Phase.NIGHT,
            content=(
                f"A game of Mafia begins with {len(players)} players: "
                f"{', '.join(p.name for p in players)}."
            ),
            meta={"roles": {p.name: p.role.value for p in players}},
        ))
        return engine

    # ------------------------------------------------------------------ events
    def _emit(self, event: Event) -> Event:
        self.state.log(event)
        if self.on_event:
            self.on_event(event)
        return event

    def _living_with_role(self, role: Role) -> list[Player]:
        return [p for p in self.state.living if p.role is role]

    def _valid_target(self, action: Action, allowed: list[str]) -> str | None:
        if action.target and action.target in allowed:
            return action.target
        # Fuzzy: case-insensitive / substring match before giving up.
        if action.target:
            low = action.target.lower()
            for n in allowed:
                if n.lower() == low or low in n.lower():
                    return n
        return self.rng.choice(allowed) if allowed else None

    # ------------------------------------------------------------------- night
    def run_night(self) -> None:
        self.state.day += 1
        self.state.phase = Phase.NIGHT
        self._emit(Event(type=EventType.NIGHT_START, day=self.state.day, phase=Phase.NIGHT,
                         content="Night falls. The town sleeps."))

        protected: str | None = None
        kill_votes: Counter[str] = Counter()

        # Detective and doctor act first so their info exists; order is private anyway.
        for det in self._living_with_role(Role.DETECTIVE):
            agent = self.agents[det.name]
            allowed = [n for n in self.state.living_names() if n != det.name]
            act = agent.night_action(self.state)
            target = self._valid_target(act, allowed)
            if target:
                faction = self.state.player(target).faction.value
                self._record_reasoning(det.name, act)
                self._emit(Event(
                    type=EventType.INVESTIGATION, day=self.state.day, phase=Phase.NIGHT,
                    actor=det.name, target=target, private=True, visible_to=[det.name],
                    meta={"faction": faction},
                ))

        for doc in self._living_with_role(Role.DOCTOR):
            agent = self.agents[doc.name]
            act = agent.night_action(self.state)
            target = self._valid_target(act, self.state.living_names())
            self._record_reasoning(doc.name, act)
            if target:
                protected = target
                self._emit(Event(
                    type=EventType.NIGHT_ACTION, day=self.state.day, phase=Phase.NIGHT,
                    actor=doc.name, target=target, private=True, visible_to=[doc.name],
                    meta={"kind": "protect"},
                ))

        # Mafia vote on a kill target.
        mafia = [p for p in self.state.living if p.faction is Faction.MAFIA]
        for m in mafia:
            agent = self.agents[m.name]
            allowed = [n for n in self.state.living_names()
                       if n != m.name and n not in m.partners] or \
                      [n for n in self.state.living_names() if n != m.name]
            act = agent.night_action(self.state)
            target = self._valid_target(act, allowed)
            self._record_reasoning(m.name, act)
            if target:
                kill_votes[target] += 1
                self._emit(Event(
                    type=EventType.NIGHT_ACTION, day=self.state.day, phase=Phase.NIGHT,
                    actor=m.name, target=target, private=True,
                    visible_to=[p.name for p in mafia], meta={"kind": "kill_vote"},
                ))

        victim = self._top_choice(kill_votes)
        if victim and victim == protected:
            self._emit(Event(type=EventType.SAVE, day=self.state.day, phase=Phase.NIGHT,
                             target=victim, meta={"saved": True}))
        elif victim:
            self.state.player(victim).alive = False
            self._emit(Event(
                type=EventType.KILL, day=self.state.day, phase=Phase.NIGHT, target=victim,
                content=f"{victim} was killed during the night.",
                meta={"role": self.state.player(victim).role.value},
            ))
        self._snapshot_beliefs("night resolved")

    # --------------------------------------------------------------------- day
    def run_day(self) -> None:
        self.state.phase = Phase.DAY_DISCUSSION
        alive = self.state.living_names()
        self._emit(Event(
            type=EventType.DAY_START, day=self.state.day, phase=Phase.DAY_DISCUSSION,
            content=f"The town wakes. Living players: {', '.join(alive)}.",
        ))

        # Discussion: several rounds, each living player speaks once per round.
        for r in range(self.config.discussion_rounds):
            for name in self.state.living_names():
                agent = self.agents[name]
                act = agent.speak(self.state, r, self.config.discussion_rounds)
                self._record_reasoning(name, act)
                msg = act.message.strip() or "(stays quiet)"
                self._emit(Event(
                    type=EventType.SPEECH, day=self.state.day, phase=Phase.DAY_DISCUSSION,
                    actor=name, target=act.target, content=msg,
                ))
                self._snapshot_beliefs(f"{name} spoke")

        self._run_vote()

    def _run_vote(self) -> None:
        self.state.phase = Phase.DAY_VOTE
        tally: Counter[str] = Counter()
        for name in self.state.living_names():
            agent = self.agents[name]
            allowed = [n for n in self.state.living_names() if n != name]
            act = agent.vote(self.state)
            target = self._valid_target(act, allowed)
            self._record_reasoning(name, act)
            if target:
                tally[target] += 1
                self._emit(Event(
                    type=EventType.VOTE, day=self.state.day, phase=Phase.DAY_VOTE,
                    actor=name, target=target, content=act.message.strip(),
                    meta={"citations": list(act.citations)},
                ))
                self._snapshot_beliefs(f"{name} voted")

        eliminated = self._top_choice(tally)
        if eliminated:
            self.state.player(eliminated).alive = False
            self._emit(Event(
                type=EventType.ELIMINATION, day=self.state.day, phase=Phase.DAY_VOTE,
                target=eliminated, content=f"{eliminated} was voted out by the town.",
                meta={"role": self.state.player(eliminated).role.value,
                      "votes": dict(tally)},
            ))
        self._snapshot_beliefs("vote resolved")

    # ----------------------------------------------------------------- helpers
    def _snapshot_beliefs(self, trigger: str) -> None:
        """Record every living agent's current suspicion vector over others.

        Keyed to the current event count so analysis/UI can align beliefs with the
        moment they were held. Negative scores (e.g. detective-cleared players) are
        kept; consumers can clamp for display.
        """
        beliefs: dict[str, dict[str, float]] = {}
        for name in self.state.living_names():
            scores = self.agents[name].suspicion_scores(self.state)
            beliefs[name] = {k: round(v, 2) for k, v in scores.items() if k != name}
        self.state.beliefs.append({
            "i": len(self.state.events),
            "day": self.state.day,
            "phase": self.state.phase.value,
            "trigger": trigger,
            "beliefs": beliefs,
        })

    def _record_reasoning(self, name: str, act: Action) -> None:
        if act.reasoning:
            self._emit(Event(
                type=EventType.REASONING, day=self.state.day, phase=self.state.phase,
                actor=name, content=act.reasoning, private=True, visible_to=[name],
            ))

    def _top_choice(self, tally: Counter[str]) -> str | None:
        if not tally:
            return None
        top = max(tally.values())
        leaders = [name for name, c in tally.items() if c == top]
        return self.rng.choice(leaders)  # ties broken randomly

    # -------------------------------------------------------------------- loop
    def play(self) -> GameState:
        while self.state.winner is None and self.state.day < self.config.max_days:
            self.run_night()
            if self._finish_if_over():
                break
            self.run_day()
            if self._finish_if_over():
                break
        if self.state.winner is None:
            # Hit the day cap — call it for whoever leads on numbers (mafia goal).
            self._declare(self.state.check_winner() or Faction.TOWN)
        return self.state

    def _finish_if_over(self) -> bool:
        winner = self.state.check_winner()
        if winner is not None:
            self._declare(winner)
            return True
        return False

    def _declare(self, winner: Faction) -> None:
        self.state.winner = winner
        self.state.phase = Phase.GAME_OVER
        survivors = ", ".join(self.state.living_names()) or "no one"
        self._emit(Event(
            type=EventType.GAME_OVER, day=self.state.day, phase=Phase.GAME_OVER,
            content=f"The {winner.value.upper()} win! Survivors: {survivors}.",
            meta={"winner": winner.value,
                  "roles": {p.name: p.role.value for p in self.state.players},
                  "survivors": self.state.living_names()},
        ))
