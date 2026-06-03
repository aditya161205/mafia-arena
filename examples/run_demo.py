"""A tiny end-to-end demo: play one narrated game, then benchmark a batch.

Run it with:  python examples/run_demo.py
"""

from mafia.arena import run_game, run_many
from mafia.engine import GameConfig
from mafia.logging_util import GameLogger, save_game

if __name__ == "__main__":
    print("### One narrated game (mock agents) ###")
    cfg = GameConfig(num_players=6, discussion_rounds=2, seed=11)
    logger = GameLogger(verbose=True, show_reasoning=True, color=True)
    state, metrics = run_game(cfg, provider="mock", logger=logger)
    save_game(state, "games/demo.json")
    print("\nSaved to games/demo.json")

    print("\n### Benchmark over 50 games ###")
    _, agg = run_many(50, GameConfig(num_players=6, seed=0), provider="mock")
    print(agg.report())
