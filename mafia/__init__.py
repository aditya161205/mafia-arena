"""Multi-Agent Mafia / Werewolf Arena.

A fully autonomous game of Mafia in which every player is an LLM agent. See the
README for the design and the evaluation methodology.
"""

from .arena import run_ablation, run_game, run_many
from .engine import Engine, GameConfig
from .evaluation import AggregateMetrics, GameMetrics, aggregate, evaluate_game
from .roles import Faction, Role
from .state import GameState

__version__ = "0.1.0"

__all__ = [
    "run_game",
    "run_many",
    "run_ablation",
    "Engine",
    "GameConfig",
    "GameState",
    "GameMetrics",
    "AggregateMetrics",
    "evaluate_game",
    "aggregate",
    "Role",
    "Faction",
    "__version__",
]
