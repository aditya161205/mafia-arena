"""LLM provider abstraction.

Two backends are supported:

* :class:`AnthropicClient` — calls the real Claude API (requires the ``anthropic``
  package and an ``ANTHROPIC_API_KEY``). Used for live games.
* :class:`MockClient` — a deterministic, dependency-free heuristic "brain" that
  lets the full engine, evaluation harness, and tests run offline. It plays a
  surprisingly coherent game of Mafia using simple rules, which is invaluable for
  CI and for sanity-checking the engine without spending tokens.

Every backend returns a JSON string describing the agent's action; the parsing
and fallbacks live in :mod:`mafia.agents`.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        """Return the model's raw text completion for a single-turn request."""
        ...


# --------------------------------------------------------------------------- API
class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, model: str = "claude-opus-4-8", temperature: float = 1.0) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic backend. "
                "Install it with `pip install anthropic`, or use the 'mock' provider."
            ) from exc
        from anthropic import Anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or run with --provider mock."
            )
        self._anthropic = Anthropic()
        self.model = model
        self.name = model
        self.temperature = temperature

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        msg = self._anthropic.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in msg.content if getattr(block, "type", "") == "text"]
        return "\n".join(parts)


# -------------------------------------------------------------------------- Mock
class MockClient:
    """Rule-based stand-in that emits the same JSON contract as a real model.

    It reads the structured context block that :mod:`mafia.prompts` embeds in the
    prompt (a ``<context>...</context>`` JSON island) and chooses a plausible
    action. Mafia avoid killing themselves; the detective investigates the most
    suspicious living player; town votes track simple suspicion heuristics.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.name = "mock"
        self._rng = random.Random(seed)

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        ctx = self._extract_context(prompt)
        action = ctx.get("action", "speech")
        me = ctx.get("me", "")
        role = ctx.get("role", "villager")
        living = [n for n in ctx.get("living", []) if n != me]
        partners = ctx.get("partners", [])
        suspicions = ctx.get("suspicions", {})  # name -> score from prior events
        known = ctx.get("known_factions", {})    # name -> "mafia"/"town" (detective)

        candidates = living
        if role == "mafia":
            candidates = [n for n in living if n not in partners] or living

        def pick_suspect() -> str | None:
            pool = [n for n in candidates]
            if not pool:
                return None
            # Prefer anyone the detective has confirmed mafia.
            confirmed = [n for n in pool if known.get(n) == "mafia"]
            if confirmed:
                return self._rng.choice(confirmed)
            ranked = sorted(pool, key=lambda n: (-suspicions.get(n, 0.0), self._rng.random()))
            return ranked[0]

        def pick_protect() -> str | None:
            # Doctor: protect a likely-town target, often a vocal claimer or self.
            pool = ctx.get("living", [])
            if not pool:
                return None
            cleared = [n for n in pool if known.get(n) == "town"]
            return self._rng.choice(cleared) if cleared else self._rng.choice(pool)

        if action == "night_kill":
            target = pick_suspect()
            return json.dumps({
                "reasoning": f"As mafia I quietly remove a dangerous towner ({target}).",
                "target": target,
            })
        if action == "night_investigate":
            target = pick_suspect()
            return json.dumps({
                "reasoning": f"Investigating {target}, who has been acting suspiciously.",
                "target": target,
            })
        if action == "night_protect":
            target = pick_protect()
            return json.dumps({
                "reasoning": f"Protecting {target} from a likely mafia hit.",
                "target": target,
            })
        if action == "vote":
            target = pick_suspect()
            line = self._vote_line(target, ctx)
            return json.dumps({
                "reasoning": self._vote_reasoning(target, role, ctx),
                "target": target,
                "message": line,
            })
        # default: a daytime speech
        target = pick_suspect()
        line, reasoning = self._speech(target, role, ctx)
        return json.dumps({"reasoning": reasoning, "message": line, "target": target})

    # ---------------------------------------------------------- dialogue model
    def _speech(self, target, role, ctx) -> tuple[str, str]:
        """Produce a varied, context-aware day statement and private reasoning."""
        me = ctx.get("me", "")
        day = ctx.get("day", 1)
        accusers = [a for a in ctx.get("accused_by", []) if a]
        last_dead = ctx.get("last_dead")
        known = ctx.get("known_factions", {})
        pick = self._rng.choice

        opener = ""
        if last_dead and day > 1:
            opener = pick([
                f"Losing {last_dead} hurts. ",
                f"Whoever killed {last_dead} made a calculated choice — ",
                f"{last_dead}'s death tells us something. ",
                "",
            ])

        # A detective with a confirmed wolf should consider hard-claiming.
        if role == "detective" and known.get(target) == "mafia":
            line = pick([
                f"I'll say it plainly: I'm the detective, and I investigated {target} — they are mafia. Vote {target}.",
                f"I've been holding this, but it's time: {target} came back guilty when I checked them. They're our wolf.",
                f"Trust me or don't, but I have a read on {target} I'd stake my life on. {target} is mafia.",
            ])
            return line, f"I have {target} confirmed as mafia; claiming now to swing the vote before I'm killed."

        # Being accused → defend, then redirect.
        if accusers:
            accuser = accusers[-1]
            defense = pick([
                f"{accuser}, pinning this on me is convenient — too convenient. ",
                f"I find it telling that {accuser} is so eager to point at me. ",
                f"Look, {accuser} wants you all staring at me instead of the real threat. ",
                f"I'm town, {accuser}, and wasting a day on me is exactly what the mafia want. ",
            ])
            redirect = pick([
                f"Ask yourself who benefits if I'm gone. I think it's {target}.",
                f"Meanwhile {target} has skated by without a hard question all game.",
                f"If you want a real suspect, watch {target} — their votes don't line up.",
            ])
            line = (opener + defense + redirect).strip()
            return line, f"{accuser} is heat on me; I'll deflect and steer the table onto {target}."

        # Mafia: sound reasonable, quietly bus a townie.
        if role == "mafia":
            line = opener + pick([
                f"I've been watching the votes, and {target} keeps drifting toward whoever's safe. That reads wolf to me.",
                f"Honestly? {target} has contributed nothing but vibes. Quiet players win games for the mafia.",
                f"I could be wrong, but {target}'s reasoning earlier didn't hold together. I'd want them explaining themselves.",
                f"Let's be disciplined. {target} is my lean — not certain, but the cleanest case we've got.",
            ])
            return line.strip(), f"I'll build a calm, plausible case against the townie {target} without exposing myself."

        # Doctor: town-aligned, a touch protective.
        if role == "doctor":
            line = opener + pick([
                f"I want to slow down before we lynch — but if pushed, {target} worries me most.",
                f"Keep your eyes on {target}. And whoever's looking strong to the town, stay alive for us.",
                f"My gut says {target}. Let's pressure-test them rather than rushing.",
            ])
            return line.strip(), f"Play like a villager, lean on {target}, and keep key town alive at night."

        # Plain villager.
        line = opener + pick([
            f"Nothing's certain yet, but {target}'s story has a seam in it. Anyone else feel that?",
            f"I keep coming back to {target}. Their reads are always one step behind the room.",
            f"I'll throw a soft vote on {target} and see who rushes to defend them — that'll be informative.",
            f"{target} is my lean. Convince me I'm wrong before we vote.",
        ])
        return line.strip(), f"{target} looks most suspicious to me; I'll voice it and watch the reactions."

    def _vote_line(self, target, ctx) -> str:
        if not target:
            return "I'll abstain — nothing's clear enough."
        return self._rng.choice([
            f"My vote is {target}. The case is as good as it'll get today.",
            f"I'm locking in {target}. Their story never added up for me.",
            f"Voting {target} — and if I'm wrong, the flip tells us plenty.",
            f"{target}. I've heard enough deflection from them.",
        ])

    def _vote_reasoning(self, target, role, ctx) -> str:
        if role == "mafia":
            return f"Voting {target} keeps me looking town and removes a real threat."
        if role == "detective" and ctx.get("known_factions", {}).get(target) == "mafia":
            return f"I confirmed {target} as mafia; this vote is correct."
        return f"{target} is the strongest suspect on my read; committing the vote."

    @staticmethod
    def _extract_context(prompt: str) -> dict:
        m = re.search(r"<context>(.*?)</context>", prompt, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return {}


def build_client(provider: str, *, model: str | None = None, seed: int | None = None) -> LLMClient:
    provider = provider.lower()
    if provider == "mock":
        return MockClient(seed=seed)
    if provider in ("anthropic", "claude"):
        return AnthropicClient(model=model or "claude-opus-4-8")
    raise ValueError(f"Unknown provider: {provider!r} (use 'mock' or 'anthropic').")
