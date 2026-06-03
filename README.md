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
a **deception / detection / reasoning-quality evaluation harness**, a **browser UI
with a live belief dashboard**, and structured logging + replay.

> **Runs with zero dependencies and zero API keys.** The engine, agents, evaluation,
> UI, and tests all work out-of-the-box on a deterministic, rule-based *mock* backend.
> Plug in the Anthropic backend for live Claude-vs-Claude games.

### Why this is interesting (for AI / ML work)

Most agent demos stop at "it runs." The hard, interview-worthy part of social
deduction is **measuring whether the reasoning is any good**, and this project is
built around that:

- **Belief tracking** — every agent's suspicion vector over every other player is
  snapshotted after each speech, vote, and night result, so you can watch beliefs
  *update from natural-language evidence* over time (and render it live).
- **Alibi / lie consistency** — mafia agents are scored on whether their public
  story holds together: contradictory role-claims and unjustified accusation flips
  are detected and penalised. A direct, automatable proxy for *deception quality*.
- **Evidence-based voting** — agents must **cite specific prior statements/votes**
  before voting; citations are checked against the actual event log, so we separate
  *grounded reasoning* from confident hallucination — judging **reasoning quality,
  not just win rate**.

Together these turn an entertaining game into a reproducible **multi-agent
evaluation benchmark** with theory-of-mind, deception, and grounded-reasoning
metrics, ablations, and confidence intervals.

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

### Watch it in your browser (no API key needed)

```bash
python -m mafia serve          # then open http://localhost:8000
```

A self-contained web UI (Python stdlib server + one HTML file, **no JS build, no
npm**) lets you watch games unfold:

- a **New Game** button that runs the engine live (set players / rounds / seed),
- a player roster with avatars and live **alive/dead** status,
- **day/night theming** and animated, color-coded chat bubbles for every speech,
- **grounded citations shown under each vote** (the evidence the agent cited),
- a **live Belief Dashboard** (right pane): pick any agent (or "Town consensus")
  and watch their suspicion of everyone else update bar-by-bar through the game,
- a **Game-metrics card** (detection / deception / reasoning-quality) at game end,
- play / pause / step / speed controls (and `Space` / `←` / `→` shortcuts),
- a **👁 God mode** toggle that reveals roles, secret night actions, and each
  agent's private chain-of-thought — turn it off to spectate blind like the town.

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
  evaluation.py     Detection, deception (alibi consistency) & reasoning-quality
                    (evidence-grounded voting) metrics, aggregation, Wilson CIs
  logging_util.py   Structured JSON persistence + colourised replay
  arena.py          run_game / run_many / run_ablation orchestration
  cli.py            `python -m mafia {play,eval,ablate,replay,serve}`
  server.py         stdlib HTTP server: runs games on demand for the web UI
  static/index.html dependency-free single-page viewer (the browser UI)
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

| Metric | Pillar | What it captures |
|--------|--------|------------------|
| **Town vote accuracy** | Detection | Share of town *day-votes* cast at real mafia — town's raw lie-detection signal. |
| **Elimination accuracy** | Detection | Share of *day-eliminations* that were actually mafia — did the coalition convert? |
| **Detective value** | Detection | Share of the detective's confirmed-mafia findings the town acted on. |
| **Deception index** | Deception | `1 − town vote accuracy` — the share of town suspicion the mafia *dodged*. High = convincing alibis. |
| **Alibi consistency** | Deception | Mafia's internal story coherence: penalises contradictory role-claims and unjustified accusation flips. High = a coherent, hard-to-catch liar. |
| **Mafia lifespan** | Deception | Mean number of days a mafia member survived. |
| **Evidence-based votes** | Reasoning | Share of votes that cite the *voted player's real prior behaviour* (grounded justification, right target). |
| **Citation grounding** | Reasoning | Share of *all* citations that reference a real prior action in the log, vs. hallucinated references. |
| **Win rates + Wilson 95% CI** | Outcome | The bottom line, with proper confidence intervals over a batch. |

#### How each new metric is computed

- **Belief tracking** (`GameState.beliefs`) — after every public event the engine
  snapshots `agent.suspicion_scores()` for each living agent, keyed to the event
  index. The browser UI replays these as animated bars ("how did Bob's suspicion
  of Carol move after that speech?"); analysis code can diff snapshots directly.
- **Alibi consistency** (`evaluation.consistency_scores`) — each player's role
  self-claims are regex-extracted from speech; >1 distinct claim is a contradiction.
  Accusation *flips* (changing your named suspect while the previous one is still
  alive — i.e. not because they were removed) are counted as unjustified. Score =
  `target_consistency × (0.5 if role-contradiction else 1.0)`, averaged over mafia.
- **Evidence-based voting** (`evaluation.evidence_vote_metrics`) — agents return a
  `citations` list with each vote. A citation is *grounded* iff it names a player
  who actually acted earlier in the log; a vote is *evidence-based* iff a grounded
  citation also names the player being voted for. This catches the failure mode of
  fluent-but-fabricated justifications.

Example over 40 games of the offline baseline (7 players, 2 mafia):

```
Town win rate:            27.5%        -- Deception (mafia) --
Mafia win rate:           72.5%        Mafia deception index:    60.2%
Mean game length:         2.38 days    Mafia alibi consistency:  96.0%
-- Detection (town) --                 Mean mafia lifespan:      2.09 days
Town vote accuracy:       39.8%        -- Reasoning quality --
Elimination accuracy:     27.5%        Evidence-based votes:    100.0%
Detective value:          40.0%        Citation grounding:      100.0%
Town win-rate 95% CI:    [16.1%, 42.8%]
```

> The rule-based mock agents are *honest by construction*, so they ceiling the
> reasoning-quality metrics (100%) and stay highly consistent (96%) — exactly the
> baseline you want. The metrics earn their keep on **real LLM agents**, where
> grounding and consistency drop and start to *discriminate* models, prompts, and
> memory depths. That separation is the point.

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
