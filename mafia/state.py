"""Game state, players, and the public event log.

The entire game is driven by an append-only list of :class:`Event` objects.
Public events are visible to every agent (and form the basis of each agent's
memory); private events (night actions, investigation results) are filtered per
recipient. This makes the game fully serialisable and replayable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .roles import Faction, Role


class Phase(str, Enum):
    NIGHT = "night"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTE = "day_vote"
    GAME_OVER = "game_over"


@dataclass
class Player:
    name: str
    role: Role
    model: str = "mock"
    alive: bool = True
    # Set for mafia players: the names of their partners.
    partners: list[str] = field(default_factory=list)

    @property
    def faction(self) -> Faction:
        return self.role.faction


class EventType(str, Enum):
    GAME_START = "game_start"
    NIGHT_START = "night_start"
    NIGHT_ACTION = "night_action"      # private: a role's night submission
    INVESTIGATION = "investigation"    # private: detective result
    KILL = "kill"                      # public: someone died in the night
    SAVE = "save"                      # public-ish: doctor prevented a kill
    DAY_START = "day_start"
    SPEECH = "speech"                  # public: a daytime statement
    VOTE = "vote"                      # public: a vote during the voting phase
    ELIMINATION = "elimination"        # public: voted out during the day
    GAME_OVER = "game_over"
    REASONING = "reasoning"            # private: an agent's chain-of-thought


@dataclass
class Event:
    type: EventType
    day: int
    phase: Phase
    actor: str | None = None           # who produced the event
    target: str | None = None          # who it was directed at
    content: str = ""                  # public text (speech / announcement)
    private: bool = False              # if True, only `visible_to` may see it
    visible_to: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def is_visible_to(self, player_name: str) -> bool:
        if not self.private:
            return True
        return player_name in self.visible_to

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "day": self.day,
            "phase": self.phase.value,
            "actor": self.actor,
            "target": self.target,
            "content": self.content,
            "private": self.private,
            "visible_to": list(self.visible_to),
            "meta": self.meta,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            type=EventType(d["type"]),
            day=d["day"],
            phase=Phase(d["phase"]),
            actor=d.get("actor"),
            target=d.get("target"),
            content=d.get("content", ""),
            private=d.get("private", False),
            visible_to=list(d.get("visible_to", [])),
            meta=d.get("meta", {}),
            ts=d.get("ts", 0.0),
        )


@dataclass
class GameState:
    players: list[Player]
    day: int = 0
    phase: Phase = Phase.NIGHT
    events: list[Event] = field(default_factory=list)
    winner: Faction | None = None
    config: dict = field(default_factory=dict)
    # Belief snapshots: each entry records every living agent's suspicion vector
    # at a point in the game, keyed to the event index so the UI/analysis can
    # replay how beliefs evolved. Populated by the engine; not part of agent memory.
    beliefs: list[dict] = field(default_factory=list)

    # ----------------------------------------------------------------- queries
    def player(self, name: str) -> Player:
        for p in self.players:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def living(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    def living_names(self) -> list[str]:
        return [p.name for p in self.living]

    def living_faction(self, faction: Faction) -> list[Player]:
        return [p for p in self.living if p.faction is faction]

    def mafia_players(self) -> list[Player]:
        return [p for p in self.players if p.faction is Faction.MAFIA]

    # ------------------------------------------------------------------ events
    def log(self, event: Event) -> Event:
        self.events.append(event)
        return event

    def visible_events(self, player_name: str) -> list[Event]:
        """Events the given player is allowed to have in memory."""
        return [e for e in self.events if e.is_visible_to(player_name)]

    # -------------------------------------------------------------- win checks
    def check_winner(self) -> Faction | None:
        mafia = len(self.living_faction(Faction.MAFIA))
        town = len(self.living_faction(Faction.TOWN))
        if mafia == 0:
            return Faction.TOWN
        # Mafia win once they reach parity with the town (they can't be out-voted).
        if mafia >= town:
            return Faction.MAFIA
        return None
