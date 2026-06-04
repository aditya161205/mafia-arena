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
npm**) has two modes:

**Watch** — autonomous AI vs AI:

- a setup screen to pick players / discussion rounds / seed,
- a player roster with avatars and live **alive/dead** status,
- **day/night phases** and animated, color-coded chat bubbles for every speech,
- **grounded citations shown under each vote** (the evidence the agent cited),
- a **live Belief Dashboard** (right pane): pick any agent (or "Town consensus")
  and watch their suspicion of everyone else update bar-by-bar through the game,
- a **Game-metrics card** (detection / deception / reasoning-quality) at game end,
- play / pause / step / speed controls (and `Space` / left / right shortcuts),
- a **God mode** toggle that reveals roles, secret night actions, and each
  agent's private chain-of-thought — turn it off to spectate blind like the town.

**Play** — you take one seat against the agents:

- choose your role (or random); a **role card** shows your ability, your secret
  mafia partners, and your detective investigation results as you learn them,
- act through phase-aware composers: **speak** to the table, **vote** (with an
  evidence/citation box that feeds the reasoning-quality metric), and use your
  **night ability** (kill / investigate / protect),
- strict **information filtering** — you only ever see what your role is entitled
  to see; the engine rules are identical to the autonomous game,
- AI replies **roll in one message at a time** so the table reads like a real
  conversation, and a built-in **How to play** guide explains the full rules.

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
  engine.py         Phase logic, night/day resolution, voting, the game loop,
                    plus a generator-based interactive_flow for human play
  evaluation.py     Detection, deception (alibi consistency) & reasoning-quality
                    (evidence-grounded voting) metrics, aggregation, Wilson CIs
  logging_util.py   Structured JSON persistence + colourised replay
  arena.py          run_game / run_many / run_ablation orchestration
  cli.py            `python -m mafia {play,eval,ablate,replay,serve}`
  session.py        interactive human-in-the-loop sessions (generator-driven,
                    information-filtered per the human's role)
  server.py         stdlib HTTP server: watch (/api/game) + play (/api/newgame,
                    /api/act) endpoints for the web UI
  static/index.html dependency-free single-page app: Watch + Play modes
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

## How it's built — a full tour

This section walks through *what* was built, *how* the pieces fit, and the design
decisions behind them.

### 1. The whole system at a glance

```
                         ┌──────────────────────────────────────────────┐
                         │                 mafia/ (engine)              │
   CLI  ─┐               │                                              │
  (cli)  │   GameConfig  │  roles ─► state (Players + Event log)         │
         ├──────────────►│     │                                        │
  Web ───┤               │     ▼                                        │
 server  │   client      │  Engine ──uses──► Agent ──calls──► LLMClient  │
 +session│   factory     │   (phases,         (memory,        (Anthropic │
         │               │    voting,          beliefs,        or Mock)  │
         └──────────────►│    resolution)      CoT, parsing)             │
                         │     │                                        │
                         │     ▼                                        │
                         │  GameState.events ─► evaluation ─► metrics    │
                         │                    └► logging  ─► JSON/replay │
                         └──────────────────────────────────────────────┘
```

Everything is driven by one **append-only `Event` log**. Public events are visible
to all; private events (night moves, investigation results, chain-of-thought)
carry a `visible_to` whitelist. An agent's "memory" is simply the **filtered
projection** of that log for its name — which is how hidden information is
*enforced structurally* rather than by trusting prompts. The same log is the input
to evaluation, to the replay/transcript, and to the web UI.

### 2. Lifecycle of a single turn (where the AI reasoning happens)

1. The **Engine** reaches a decision point (e.g. it's a player's turn to speak).
2. It calls that player's **Agent**, which:
   - builds its **memory** by filtering the event log to what it may see,
   - computes structured context — a **suspicion vector** over other players,
     **evidence** it could cite, who has accused it, the latest death, the
     detective's findings (if any),
   - assembles a **role-specific system prompt** (mafia are told to deceive and
     protect partners; town to hunt contradictions) plus a task prompt that asks
     for **explicit private chain-of-thought first, then a structured action**,
   - sends it to the **LLMClient** and **parses the JSON** reply defensively
     (tolerates markdown fences, stray prose, and illegal targets).
3. The Engine validates the action, **emits events** (public speech/vote + private
   reasoning), updates state, and **snapshots every agent's beliefs**.
4. Win conditions are checked; the loop continues.

The autonomous game (`play()`) runs this straight through. The **interactive game**
(`interactive_flow`, a Python *generator*) runs the identical logic but `yield`s
control whenever it's the human's turn, so a person can occupy any seat with the
exact same rules.

### 3. The evaluation harness (the research core)

Social deduction has no scalar reward, so the project scores the things that
actually matter, all recomputable from a saved game:

- **Detection** — do town votes/eliminations land on real mafia? does the town act
  on the detective's findings?
- **Deception** — how much town suspicion did mafia dodge (`deception_index`), and
  is their alibi internally **consistent** (no contradictory role-claims, no
  unjustified accusation flips)?
- **Reasoning quality** — agents must **cite prior events** before voting;
  citations are checked against the real log to separate grounded reasoning from
  fluent hallucination.
- Aggregated across many games with **win rates and Wilson confidence intervals**,
  plus **ablation sweeps** (memory depth, discussion length, table size, model).

### 4. Key design decisions (and why)

| Decision | Why |
|---|---|
| **Event-sourced state** | One immutable log → trivial hidden-info filtering, replay, and metrics from a single source of truth. |
| **Provider abstraction + a real mock backend** | The whole system (engine, eval, UI, tests) runs offline at zero cost; swapping in Claude is a one-liner. The mock isn't a stub — it plays a valid heuristic game so baselines and CI are meaningful. |
| **Structured JSON actions + defensive parsing** | Keeps free-form LLM output machine-checkable without crashing on a single bad generation. |
| **Generator-based interactive flow** | Lets a human share the *exact* engine path as the AI — no duplicated rules to drift out of sync. |
| **Metrics computed from the log, not in the loop** | Evaluation is reproducible and auditable; you can re-score any saved game. |
| **Stdlib-only web server + single HTML file** | No build step, no npm, no framework — clone and run. |

## Skills & techniques demonstrated

Mapped to the kind of work this exercises — useful as a quick read of what the
project shows.

**Core ML / agentic**
- Multi-agent orchestration with **per-agent state, memory, and hidden information**.
- **Theory-of-mind prompting** (modelling what others believe about you) and
  **role-conditioned strategic deception** vs. inference.
- **Explicit chain-of-thought** before action; structured tool-style JSON outputs.
- **LLM-agent evaluation design**: turning fuzzy notions (deception quality,
  reasoning quality) into automatable, log-grounded metrics.
- **Belief-state tracking** and updating from natural-language signals over time.
- **Ablation methodology** and reporting with confidence intervals.

**Engineering**
- A clean **game engine** (phase state machine, action resolution, win logic).
- **Event-sourced architecture** with serialization, persistence, and replay.
- A **provider-abstraction layer** (Anthropic + offline mock) behind one interface.
- **Generator/coroutine** design for human-in-the-loop interactivity.
- A dependency-free **HTTP API + session management** and a hand-written **SPA**
  (state management, animation, a live data-viz dashboard) — no frameworks.
- **Robust parsing** of unreliable model output; **deterministic seeding** for
  reproducibility; a **21-test** suite covering invariants, metrics, and play.

**Domain**
- Mafia/Werewolf rules, role balance, and optimal-ish strategy encoded into both
  the prompts and the heuristic baseline.
- How to measure **argument consistency** and **evidence grounding** in text.

## How it was made (process)

1. **Modelled the domain first** — roles, factions, abilities, and balanced role
   tables (`roles.py`), then an event-sourced `GameState` so hidden information had
   a single enforcement point.
2. **Built the engine** as a phase state machine (night → discussion → vote →
   resolve → win-check) with random-tie-breaking and validated targets.
3. **Added the agent layer** — memory projection, suspicion/evidence context,
   role-specific prompts with mandatory chain-of-thought, and defensive JSON
   parsing — behind an **LLM provider interface** with a genuinely playable
   **mock backend** so everything runs without an API key.
4. **Wrote the evaluation harness** — detection, deception/consistency, and
   evidence-grounded-reasoning metrics, batch aggregation, Wilson intervals, and
   ablation sweeps — all computed from the saved log.
5. **Layered on tooling** — CLI (`play`/`eval`/`ablate`/`replay`/`serve`),
   structured logging + colourised replay, and per-game JSON save/load.
6. **Made it observable and interactive** — a stdlib web server, then a single-page
   UI with a **Watch** mode (belief dashboard + metrics) and a **Play** mode
   (human takes a seat, AI messages stream in, in-app rules guide).
7. **Tested throughout** — invariants (no self-kills, hidden-info filtering,
   determinism), metric correctness (grounded vs. hallucinated citations,
   consistency penalties), and a full scripted human session.

Built iteratively, validated at each step by running real games and inspecting the
transcripts, metrics, and UI — not just unit tests.

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
