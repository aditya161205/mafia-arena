"""Command-line interface.

Examples
--------
Play one game with the offline mock agents and watch it unfold::

    python -m mafia play --players 6 --verbose --reasoning

Play a live game with Claude agents (needs ANTHROPIC_API_KEY)::

    python -m mafia play --provider anthropic --model claude-opus-4-8 --verbose

Benchmark over many games and print aggregate metrics::

    python -m mafia eval --games 50 --players 7

Run an ablation over memory depth and discussion length::

    python -m mafia ablate --games 30

Replay a saved game::

    python -m mafia replay games/game.json --reasoning
"""

from __future__ import annotations

import argparse
import json
import sys

from .arena import run_ablation, run_game, run_many
from .engine import GameConfig
from .evaluation import wilson_interval
from .logging_util import GameLogger, replay, save_game


def _config_from_args(args: argparse.Namespace) -> GameConfig:
    return GameConfig(
        num_players=args.players,
        discussion_rounds=args.rounds,
        memory_limit=args.memory_limit,
        seed=args.seed,
    )


def cmd_play(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    logger = GameLogger(verbose=True, show_reasoning=args.reasoning, color=not args.no_color)
    state, metrics = run_game(cfg, provider=args.provider, model=args.model, logger=logger)
    print("\n--- metrics ---")
    print(json.dumps(metrics.to_dict(), indent=2))
    if args.save:
        path = save_game(state, args.save)
        print(f"\nSaved game to {path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    metrics, agg = run_many(
        args.games, cfg, provider=args.provider, model=args.model, progress=args.verbose
    )
    print("\n=== Aggregate over", args.games, "games ===")
    print(agg.report())
    lo, hi = wilson_interval(round(agg.town_win_rate * args.games), args.games)
    print(f"Town win-rate 95% CI:    [{100*lo:.1f}%, {100*hi:.1f}%]")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"aggregate": agg.to_dict(), "games": [m.to_dict() for m in metrics]},
                f, indent=2,
            )
        print(f"\nWrote per-game metrics to {args.json}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    variants = {
        "baseline": {},
        "shallow-memory(6)": {"memory_limit": 6},
        "no-discussion": {"discussion_rounds": 0},
        "long-discussion(4)": {"discussion_rounds": 4},
        "big-table(9)": {"num_players": 9},
    }
    results = run_ablation(args.games, cfg, variants, provider=args.provider, model=args.model)
    print(f"\n=== Ablation ({args.games} games per variant) ===\n")
    header = f"{'variant':<22} {'town win':>9} {'vote acc':>9} {'decept':>8} {'days':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        a = r.aggregate
        def p(x): return "  n/a" if x is None else f"{100*x:6.1f}%"
        print(f"{r.label:<22} {p(a.town_win_rate):>9} {p(a.mean_town_vote_accuracy):>9} "
              f"{p(a.mean_deception_index):>8} {a.mean_days:>6.2f}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    replay(args.path, show_reasoning=args.reasoning, color=not args.no_color)
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--players", type=int, default=6, help="number of players (>=5)")
    p.add_argument("--rounds", type=int, default=2, help="discussion rounds per day")
    p.add_argument("--memory-limit", type=int, default=None,
                   help="cap each agent's memory to the last N events")
    p.add_argument("--seed", type=int, default=None, help="base RNG seed")
    p.add_argument("--provider", default="mock", choices=["mock", "anthropic", "claude"])
    p.add_argument("--model", default=None, help="model id for the anthropic provider")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mafia", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_play = sub.add_parser("play", help="play and narrate a single game")
    _add_common(p_play)
    p_play.add_argument("--verbose", action="store_true", default=True)
    p_play.add_argument("--reasoning", action="store_true", help="show private chain-of-thought")
    p_play.add_argument("--no-color", action="store_true")
    p_play.add_argument("--save", default=None, help="path to save the game JSON")
    p_play.set_defaults(func=cmd_play)

    p_eval = sub.add_parser("eval", help="benchmark over many games")
    _add_common(p_eval)
    p_eval.add_argument("--games", type=int, default=20)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.add_argument("--json", default=None, help="path to dump per-game metrics")
    p_eval.set_defaults(func=cmd_eval)

    p_abl = sub.add_parser("ablate", help="compare config variants")
    _add_common(p_abl)
    p_abl.add_argument("--games", type=int, default=20)
    p_abl.set_defaults(func=cmd_ablate)

    p_rep = sub.add_parser("replay", help="replay a saved game JSON")
    p_rep.add_argument("path")
    p_rep.add_argument("--reasoning", action="store_true")
    p_rep.add_argument("--no-color", action="store_true")
    p_rep.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
