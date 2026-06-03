# Mafia / Werewolf Arena — Multi-Agent Social Deduction for LLMs

A fully autonomous game of **Mafia** (a.k.a. Werewolf) in which **every player is an
LLM agent**. Mafia agents must deceive convincingly and coordinate secretly;
town agents must reason about inconsistencies, detect lies, and build a coalition
to vote out the wolves. The game is self-running — you observe, evaluate, and tune
the emergent social dynamics.

The whole game state lives in natural language, which makes it a clean testbed for
the hard parts of multi-agent AI: **theory-of-mind**, **strategic deception**, and
**real-time belief updating from text**. This repo ships a complete game engine, a
per-agent memory/agent framework, an LLM abstraction (Claude or an offline mock),
a **deception/detection evaluation harness**, and structured logging + replay.

> **Runs with zero dependencies and zero API keys.** The engine, agents, evaluation,
> and tests all work out-of-the-box on a deterministic, rule-based *mock* backend.
> Plug in the Anthropic backend for live Claude-vs-Claude games.

---

## Quick start

```bash
git clone <this-repo> && cd mafia-arena

# 1) Play & narrate a single game with offline mock agents (no install needed):
python -m mafia play --players 6 --seed 11

# 2) See the agents' private chain-of-thought too:
python -m mafia play --players 6 --reasoning

# 3) Benchmark deception/detection over many games:
python -m mafia eval --games 60 --players 7

# 4) Run an ablation sweep over memory depth & discussion length:
python -m mafia ablate --games 40

# 5) Save a game and replay it later:
python -m mafia play --save games/g.json
python -m mafia replay games/g.json --reasoning
```

### Live games with Claude agents

```bash
pip install anthropic           # only needed for the live backend
export ANTHROPIC_API_KEY=sk-...
python -m mafia play --provider anthropic --model claude-opus-4-8 --reasoning
```

### Running the tests

```bash
pip install pytest
pytest -q          # 13 tests, all on the offline backend, run in <0.1s
```

---

## The game

Classic Mafia with hidden-information town roles:

| Role | Faction | Night ability |
|------|---------|---------------|
| **Mafia** | Mafia | Collectively choose one player to eliminate. Know each other. |
| **Detective** | Town | Privately learn one player's true faction. |
| **Doctor** | Town | Privately protect one player from being killed. |
| **Villager** | Town | No ability — pure reasoning. |

**Loop:** `Night` (mafia kill, detective investigates, doctor protects) →
`Day discussion` (every living agent speaks, for *N* rounds) →
`Day vote` (majority eliminates one player; ties broken at random).

**Win conditions:** Town wins when all mafia are eliminated. Mafia win on reaching
numerical parity with the town. Default role tables are defined in
[`mafia/roles.py`](mafia/roles.py) for 5–10 players.

---

## Architecture

```
mafia/
  roles.py          Roles, factions, abilities, and balanced role tables
  state.py          Players, the append-only Event log, win detection
  llm.py            Provider abstraction: AnthropicClient + offline MockClient
  prompts.py        Per-role system prompts (mafia = deceive; town = infer) + CoT contract
  agents.py         Agent: filtered memory, belief/suspicion model, action parsing
  engine.py         Phase logic, night/day resolution, voting, the game loop
  evaluation.py     Deception & detection metrics, aggregation, Wilson CIs
  logging_util.py   Structured JSON persistence + colourised replay
  arena.py          run_game / run_many / run_ablation orchestration
  cli.py            `python -m mafia {play,eval,ablate,replay}`
tests/              13 tests covering engine invariants, evaluation, parsing
examples/run_demo.py
```

### Design notes

- **Hidden information is enforced by the event log, not by trust.** Every action
  produces an `Event`; private events (night moves, investigation results,
  chain-of-thought) carry a `visible_to` whitelist, and each agent's memory is the
  *filtered* projection of the log. A villager literally cannot see the detective's
  findings — see `GameState.visible_events` in [`state.py`](mafia/state.py).
- **Per-role prompting drives the social dynamics.** Mafia prompts grant secret
  partner knowledge and an explicit mandate to build alibis and stay internally
  consistent; town prompts push contradiction-hunting and coalition building. See
  [`prompts.py`](mafia/prompts.py).
- **Explicit theory-of-mind & chain-of-thought.** Every turn, agents reason
  privately in a `reasoning` field before committing a public `message`/`target`.
  Reasoning is logged privately so you can audit *why* an agent acted (replay with
  `--reasoning`).
- **Robust action parsing.** Agents return JSON; the parser tolerates markdown
  fences, surrounding prose, and bad target names (illegal targets fall back to a
  legal choice), so a single malformed generation never crashes a game.
- **The mock backend is a real (if simple) player.** It reads the structured
  `<context>` block in each prompt and plays by suspicion heuristics — mafia avoid
  killing partners, the detective acts on confirmed findings, town tracks votes.
  This makes the engine and metrics fully testable offline and gives a sane
  baseline to compare LLM agents against.

---

## Evaluation harness

Social deduction has no scalar reward, so we measure the things the research
question is actually about. All metrics are computed purely from a finished (or
reloaded) game log, so they're reproducible from a saved file.

| Metric | What it captures |
|--------|------------------|
| **Town vote accuracy** | Share of town *day-votes* cast at real mafia — town's raw lie-detection signal. |
| **Elimination accuracy** | Share of *day-eliminations* that were actually mafia — did the coalition convert? |
| **Deception index** | `1 − town vote accuracy` — the share of town suspicion the mafia *dodged*. High = convincing alibis. |
| **Mafia lifespan** | Mean number of days a mafia member survived. |
| **Detective value** | Share of the detective's confirmed-mafia findings the town acted on. |
| **Win rates + Wilson 95% CI** | The bottom line, with proper confidence intervals over a batch. |

Example over 60 games of the offline baseline (7 players, 2 mafia):

```
Town win rate:            15.0%        Town vote accuracy:   37.3%
Mafia win rate:           85.0%        Elimination accuracy: 21.8%
Mean game length:         2.42 days    Mafia deception index:62.7%
Town win-rate 95% CI:    [8.1%, 26.1%] Detective value:      35.5%
```

### Ablations

`mafia ablate` runs the same batch under several config variants and tabulates the
effect on town win-rate, vote accuracy, deception, and game length:

```
variant                 town win  vote acc   decept   days
----------------------------------------------------------
baseline                   57.5%     42.6%    57.4%   2.00
no-discussion              57.5%     39.2%    60.8%   1.98
long-discussion(4)         50.0%     36.8%    63.2%   2.00
big-table(9)               45.0%     44.5%    55.5%   3.35
```

The orchestration in [`arena.py`](mafia/arena.py) makes it trivial to sweep any
`GameConfig` field — memory depth, discussion rounds, table size, provider/model —
which is exactly the knob-set the project brief calls for: *vary LLM, memory depth,
reasoning prompt style, and number of agents to understand what drives performance.*

---

## Extending it

- **New roles** (e.g. Jester, Vigilante): add to `Role` in `roles.py`, give it a
  night branch in `Agent.night_action` and resolution in `Engine.run_night`.
- **New backends** (OpenAI, local models): implement the three-line `LLMClient`
  protocol in `llm.py` and register it in `build_client`.
- **New metrics** (argument consistency, alibi survival): read the event log in
  `evaluation.py` — every speech, vote, and private thought is there.
- **Mixed tables** (Claude mafia vs. mock town, or model-vs-model): the per-agent
  `client_factory` in `arena.run_game` already lets each seat use a different
  client; branch on player name or role to assign models.

## License

MIT.
