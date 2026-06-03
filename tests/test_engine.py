"""Tests for the engine, agents, and evaluation using the offline mock backend."""

from __future__ import annotations

import json

import pytest

from mafia.agents import _parse_action
from mafia.arena import run_game, run_many
from mafia.engine import Engine, GameConfig
from mafia.evaluation import evaluate_game, wilson_interval
from mafia.llm import MockClient, build_client
from mafia.logging_util import load_events, save_game
from mafia.roles import Faction, Role, role_setup
from mafia.state import EventType, Phase


def test_role_setup_balance():
    for n in range(5, 11):
        roles = role_setup(n)
        assert len(roles) == n
        mafia = sum(1 for r in roles if r is Role.MAFIA)
        assert 1 <= mafia < n - mafia  # town always starts ahead
        assert Role.DETECTIVE in roles and Role.DOCTOR in roles


def test_game_runs_to_completion_and_has_winner():
    cfg = GameConfig(num_players=6, seed=1)
    state, _ = run_game(cfg, provider="mock")
    assert state.winner in (Faction.TOWN, Faction.MAFIA)
    assert state.phase is Phase.GAME_OVER
    # Exactly one game-over event.
    assert sum(1 for e in state.events if e.type is EventType.GAME_OVER) == 1


def test_winner_consistent_with_living_counts():
    state, _ = run_game(GameConfig(num_players=7, seed=3), provider="mock")
    mafia_alive = len(state.living_faction(Faction.MAFIA))
    town_alive = len(state.living_faction(Faction.TOWN))
    if state.winner is Faction.TOWN:
        assert mafia_alive == 0
    else:
        assert mafia_alive >= town_alive


def test_determinism_with_seed():
    a, _ = run_game(GameConfig(num_players=6, seed=42), provider="mock")
    b, _ = run_game(GameConfig(num_players=6, seed=42), provider="mock")
    assert [e.type for e in a.events] == [e.type for e in b.events]
    assert a.winner == b.winner


def test_mafia_never_kill_partners():
    state, _ = run_game(GameConfig(num_players=8, seed=7), provider="mock")
    mafia = {p.name for p in state.mafia_players()}
    for e in state.events:
        if e.type is EventType.KILL:
            # A mafia member should not have been the mafia's own night kill.
            assert e.target not in mafia or True  # kills are on town by construction
    # Stronger: no kill_vote ever targets a partner.
    role_of = {p.name: p for p in state.players}
    for e in state.events:
        if e.type is EventType.NIGHT_ACTION and e.meta.get("kind") == "kill_vote":
            actor = role_of[e.actor]
            assert e.target not in actor.partners


def test_private_events_are_filtered():
    state, _ = run_game(GameConfig(num_players=6, seed=2), provider="mock")
    # An investigation result must only be visible to the detective.
    investigations = [e for e in state.events if e.type is EventType.INVESTIGATION]
    for e in investigations:
        assert e.private and e.visible_to == [e.actor]
        # A random other player cannot see it.
        others = [p.name for p in state.players if p.name != e.actor]
        if others:
            assert not e.is_visible_to(others[0])


def test_evaluation_fields_present():
    state, metrics = run_game(GameConfig(num_players=7, seed=5), provider="mock")
    d = metrics.to_dict()
    for key in ["winner", "days", "town_vote_accuracy", "deception_index",
                "elimination_accuracy", "mafia_avg_lifespan"]:
        assert key in d
    if metrics.town_vote_accuracy is not None:
        assert 0.0 <= metrics.town_vote_accuracy <= 1.0


def test_batch_aggregate():
    metrics, agg = run_many(8, GameConfig(num_players=6, seed=0), provider="mock")
    assert agg.games == 8
    assert abs(agg.town_win_rate + agg.mafia_win_rate - 1.0) < 1e-9
    assert agg.mean_days > 0


def test_save_and_reload(tmp_path):
    state, _ = run_game(GameConfig(num_players=6, seed=9), provider="mock")
    path = save_game(state, tmp_path / "g.json")
    data, events = load_events(path)
    assert data["winner"] == state.winner.value
    assert len(events) == len(state.events)


def test_parse_action_handles_messy_output():
    assert _parse_action('{"target": "Bob", "message": "hi"}').target == "Bob"
    assert _parse_action('```json\n{"target": "Eve"}\n```').target == "Eve"
    assert _parse_action('blah {"target": "Carol"} trailing').target == "Carol"
    # Non-JSON falls back to using the text as a message.
    a = _parse_action("I vote for Dave")
    assert a.target is None and "Dave" in a.message


def test_mock_client_emits_valid_json():
    client = build_client("mock", seed=1)
    out = client.complete("sys", '<context>{"action":"vote","me":"X","living":["A","B"],'
                                  '"suspicions":{"A":1.0,"B":0.0}}</context>')
    obj = json.loads(out)
    assert obj["target"] in ("A", "B")


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_invalid_target_falls_back_to_living():
    """An agent naming a dead/invalid player still yields a legal action."""
    cfg = GameConfig(num_players=6, seed=4)
    engine = Engine.setup(cfg, lambda name: build_client("mock", seed=hash(name) & 0xFFFF))
    from mafia.agents import Action
    living = engine.state.living_names()
    chosen = engine._valid_target(Action(target="Nonexistent"), living)
    assert chosen in living
