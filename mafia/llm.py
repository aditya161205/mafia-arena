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
            return json.dumps({
                "reasoning": f"My read says {target} is the most likely wolf.",
                "target": target,
                "message": f"I'm voting {target} — their story doesn't add up.",
            })
        # default: a daytime speech
        target = pick_suspect()
        if role == "mafia":
            line = (
                f"I've been paying close attention. {target} has been awfully quiet "
                f"and quick to deflect — I think we should keep an eye on them."
            )
        elif role == "detective" and known.get(target) == "mafia":
            line = f"I have strong reason to believe {target} is mafia. We should vote them."
        else:
            line = (
                f"Nothing concrete yet, but {target}'s reasoning felt off to me. "
                f"What does everyone else think?"
            )
        return json.dumps({
            "reasoning": f"I'll steer suspicion toward {target} while sounding reasonable.",
            "message": line,
            "target": target,
        })

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
