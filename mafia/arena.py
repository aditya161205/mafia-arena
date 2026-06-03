"""High-level orchestration: run one game, a batch, or an ablation sweep."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .engine import Engine, GameConfig
from .evaluation import AggregateMetrics, GameMetrics, aggregate, evaluate_game
from .llm import LLMClient, build_client
from .logging_util import GameLogger
from .state import GameState


def run_game(
    config: GameConfig,
    *,
    provider: str = "mock",
    model: str | None = None,
    logger: GameLogger | None = None,
) -> tuple[GameState, GameMetrics]:
    """Play a single game end-to-end and score it."""

    def client_factory(_player_name: str) -> LLMClient:
        # One client per agent. The mock client is seeded per-agent so different
        # players make different (but reproducible) choices.
        seed = None if config.seed is None else hash((config.seed, _player_name)) & 0xFFFFFFFF
        return build_client(provider, model=model, seed=seed)

    engine = Engine.setup(config, client_factory, on_event=logger)
    state = engine.play()
    return state, evaluate_game(state)


def run_many(
    num_games: int,
    config: GameConfig,
    *,
    provider: str = "mock",
    model: str | None = None,
    progress: bool = False,
) -> tuple[list[GameMetrics], AggregateMetrics]:
    """Play a batch of games (varying the seed) and aggregate the metrics."""
    metrics: list[GameMetrics] = []
    base_seed = config.seed if config.seed is not None else 0
    for i in range(num_games):
        cfg = replace(config, seed=base_seed + i)
        _, m = run_game(cfg, provider=provider, model=model)
        metrics.append(m)
        if progress:
            print(f"  game {i + 1}/{num_games}: {m.winner} wins in {m.days} days")
    return metrics, aggregate(metrics)


@dataclass
class AblationResult:
    label: str
    overrides: dict
    aggregate: AggregateMetrics


def run_ablation(
    num_games: int,
    base_config: GameConfig,
    variants: dict[str, dict],
    *,
    provider: str = "mock",
    model: str | None = None,
) -> list[AblationResult]:
    """Run the same batch under several config variants for comparison.

    ``variants`` maps a human label to a dict of :class:`GameConfig` overrides,
    e.g. ``{"shallow-memory": {"memory_limit": 6}, "deep-memory": {}}``.
    """
    results: list[AblationResult] = []
    for label, overrides in variants.items():
        cfg = replace(base_config, **overrides)
        _, agg = run_many(num_games, cfg, provider=provider, model=model)
        results.append(AblationResult(label=label, overrides=overrides, aggregate=agg))
    return results
