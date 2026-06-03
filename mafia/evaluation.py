"""Evaluation harness.

Social deduction has no scalar reward, so we score the things that actually
matter for the research question:

* **Detection** — how well the town's *votes* and *eliminations* land on real
  mafia (town competence at finding wolves).
* **Deception** — how well mafia evade town suspicion and survive (wolf skill at
  blending in). ``deception_index`` is the share of town suspicion that mafia
  *dodged*; a high value means convincing alibis.
* **Outcomes** — win rates and game length across many games (the bottom line).

Metrics are computed purely from a finished :class:`GameState` (or a reloaded
event log), so they are reproducible from a saved game file.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from .roles import Faction, Role
from .state import EventType, GameState


@dataclass
class GameMetrics:
    winner: str
    days: int
    num_players: int
    num_mafia: int
    town_votes_cast: int
    town_vote_accuracy: float | None      # town day-votes that hit a mafia
    eliminations: int
    elimination_accuracy: float | None    # day-eliminations that were mafia
    mafia_avg_lifespan: float             # mean day a mafia was removed (or game end)
    deception_index: float | None         # 1 - town_vote_accuracy
    detective_value: float | None         # share of detective findings the town acted on

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_game(state: GameState) -> GameMetrics:
    faction_of = {p.name: p.faction for p in state.players}
    role_of = {p.name: p.role for p in state.players}
    num_mafia = len(state.mafia_players())

    # --- town voting accuracy ------------------------------------------------
    town_votes = 0
    town_votes_on_mafia = 0
    for e in state.events:
        if e.type is EventType.VOTE and e.actor and e.target:
            if faction_of.get(e.actor) is Faction.TOWN:
                town_votes += 1
                if faction_of.get(e.target) is Faction.MAFIA:
                    town_votes_on_mafia += 1
    town_acc = (town_votes_on_mafia / town_votes) if town_votes else None

    # --- elimination accuracy ------------------------------------------------
    elims = [e for e in state.events if e.type is EventType.ELIMINATION and e.target]
    elim_mafia = sum(1 for e in elims if faction_of.get(e.target) is Faction.MAFIA)
    elim_acc = (elim_mafia / len(elims)) if elims else None

    # --- mafia lifespan ------------------------------------------------------
    removed_day: dict[str, int] = {}
    for e in state.events:
        if e.type in (EventType.ELIMINATION, EventType.KILL) and e.target:
            removed_day.setdefault(e.target, e.day)
    lifespans = []
    for p in state.mafia_players():
        lifespans.append(removed_day.get(p.name, state.day))
    mafia_lifespan = statistics.mean(lifespans) if lifespans else 0.0

    # --- detective value: did the town eliminate mafia the detective found? --
    found_mafia: set[str] = set()
    for e in state.events:
        if e.type is EventType.INVESTIGATION and e.meta.get("faction") == Faction.MAFIA.value:
            if e.target:
                found_mafia.add(e.target)
    if found_mafia:
        acted = sum(1 for e in elims if e.target in found_mafia)
        detective_value = acted / len(found_mafia)
    else:
        detective_value = None

    return GameMetrics(
        winner=state.winner.value if state.winner else "none",
        days=state.day,
        num_players=len(state.players),
        num_mafia=num_mafia,
        town_votes_cast=town_votes,
        town_vote_accuracy=town_acc,
        eliminations=len(elims),
        elimination_accuracy=elim_acc,
        mafia_avg_lifespan=mafia_lifespan,
        deception_index=(1 - town_acc) if town_acc is not None else None,
        detective_value=detective_value,
    )


@dataclass
class AggregateMetrics:
    games: int
    town_win_rate: float
    mafia_win_rate: float
    mean_days: float
    mean_town_vote_accuracy: float | None
    mean_elimination_accuracy: float | None
    mean_deception_index: float | None
    mean_mafia_lifespan: float
    mean_detective_value: float | None

    def to_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{100 * x:5.1f}%"
        return (
            f"Games played:            {self.games}\n"
            f"Town win rate:           {pct(self.town_win_rate)}\n"
            f"Mafia win rate:          {pct(self.mafia_win_rate)}\n"
            f"Mean game length:        {self.mean_days:.2f} days\n"
            f"Town vote accuracy:      {pct(self.mean_town_vote_accuracy)}\n"
            f"Elimination accuracy:    {pct(self.mean_elimination_accuracy)}\n"
            f"Mafia deception index:   {pct(self.mean_deception_index)}\n"
            f"Mean mafia lifespan:     {self.mean_mafia_lifespan:.2f} days\n"
            f"Detective value:         {pct(self.mean_detective_value)}"
        )


def _mean_opt(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return statistics.mean(nums) if nums else None


def aggregate(metrics: list[GameMetrics]) -> AggregateMetrics:
    n = len(metrics)
    if n == 0:
        raise ValueError("No games to aggregate.")
    town_wins = sum(1 for m in metrics if m.winner == Faction.TOWN.value)
    mafia_wins = sum(1 for m in metrics if m.winner == Faction.MAFIA.value)
    return AggregateMetrics(
        games=n,
        town_win_rate=town_wins / n,
        mafia_win_rate=mafia_wins / n,
        mean_days=statistics.mean(m.days for m in metrics),
        mean_town_vote_accuracy=_mean_opt([m.town_vote_accuracy for m in metrics]),
        mean_elimination_accuracy=_mean_opt([m.elimination_accuracy for m in metrics]),
        mean_deception_index=_mean_opt([m.deception_index for m in metrics]),
        mean_mafia_lifespan=statistics.mean(m.mafia_avg_lifespan for m in metrics),
        mean_detective_value=_mean_opt([m.detective_value for m in metrics]),
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a win-rate proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))
