"""The agent: wraps a player, an LLM client, memory, and action parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLMClient
from .prompts import role_system_prompt, task_prompt
from .roles import Faction, Role
from .state import Event, EventType, GameState, Player


@dataclass
class Action:
    reasoning: str = ""
    message: str = ""
    target: str | None = None
    raw: str = ""


def _render_event(e: Event, me: str) -> str | None:
    """Turn an event into a memory line from `me`'s perspective."""
    t = e.type
    if t is EventType.GAME_START:
        return f"[Setup] {e.content}"
    if t is EventType.NIGHT_START:
        return f"--- Night {e.day} ---"
    if t is EventType.DAY_START:
        return f"--- Day {e.day} ---  {e.content}".rstrip()
    if t is EventType.KILL:
        return f"[Night {e.day}] {e.target} was found dead. (was {e.meta.get('role','?')})"
    if t is EventType.SAVE:
        return f"[Night {e.day}] The mafia attacked, but no one died — the doctor saved a life."
    if t is EventType.SPEECH:
        who = "You" if e.actor == me else e.actor
        return f"{who}: \"{e.content}\""
    if t is EventType.VOTE:
        who = "You" if e.actor == me else e.actor
        return f"[Vote] {who} voted for {e.target}."
    if t is EventType.ELIMINATION:
        return f"[Day {e.day}] {e.target} was voted out. (was {e.meta.get('role','?')})"
    if t is EventType.INVESTIGATION:
        return f"[Investigation] You learned that {e.target} is {e.meta.get('faction','?').upper()}."
    if t is EventType.NIGHT_ACTION and e.actor == me:
        return f"[Your night action] target: {e.target}"
    if t is EventType.GAME_OVER:
        return f"[Game over] {e.content}"
    return None


class Agent:
    def __init__(self, player: Player, client: LLMClient, *, memory_limit: int | None = None,
                 max_tokens: int = 800) -> None:
        self.player = player
        self.client = client
        self.memory_limit = memory_limit
        self.max_tokens = max_tokens
        self.last_reasoning: str = ""

    @property
    def name(self) -> str:
        return self.player.name

    # ------------------------------------------------------------- perception
    def memory_lines(self, state: GameState) -> list[str]:
        lines: list[str] = []
        for e in state.visible_events(self.name):
            rendered = _render_event(e, self.name)
            if rendered:
                lines.append(rendered)
        return lines

    def known_factions(self, state: GameState) -> dict[str, str]:
        """Detective's accumulated investigation results."""
        known: dict[str, str] = {}
        for e in state.events:
            if e.type is EventType.INVESTIGATION and e.actor == self.name and e.target:
                known[e.target] = e.meta.get("faction", "")
        return known

    def suspicion_scores(self, state: GameState) -> dict[str, float]:
        """A cheap heuristic suspicion map used by the mock backend and metrics.

        Players accrue suspicion for receiving votes and for being named as a
        target in others' speeches.
        """
        scores: dict[str, float] = {p.name: 0.0 for p in state.living}
        for e in state.visible_events(self.name):
            if e.type is EventType.VOTE and e.target in scores and e.actor != self.name:
                scores[e.target] += 1.0
            if e.type is EventType.SPEECH and e.target in scores and e.actor != self.name:
                scores[e.target] += 0.3
        # The detective trusts its own findings most of all.
        for name, faction in self.known_factions(state).items():
            if name in scores:
                scores[name] += 5.0 if faction == Faction.MAFIA.value else -5.0
        return scores

    # ------------------------------------------------------------------ acting
    def _context(self, state: GameState, action: str) -> dict:
        return {
            "me": self.name,
            "role": self.player.role.value,
            "living": state.living_names(),
            "partners": list(self.player.partners),
            "suspicions": self.suspicion_scores(state),
            "known_factions": self.known_factions(state),
            "day": state.day,
        }

    def _act(self, state: GameState, action: str, instruction: str) -> Action:
        system = role_system_prompt(self.player.role, self.name, self.player.partners)
        prompt = task_prompt(
            action=action,
            instruction=instruction,
            memory_lines=self.memory_lines(state),
            context=self._context(state, action),
            memory_limit=self.memory_limit,
        )
        raw = self.client.complete(system, prompt, max_tokens=self.max_tokens)
        act = _parse_action(raw)
        self.last_reasoning = act.reasoning
        return act

    def night_action(self, state: GameState) -> Action:
        role = self.player.role
        living = [n for n in state.living_names() if n != self.name]
        if role is Role.MAFIA:
            living = [n for n in living if n not in self.player.partners] or living
            instr = (
                "It is night. Choose one living player for the mafia to eliminate. "
                f"Valid targets: {', '.join(living)}."
            )
            return self._act(state, "night_kill", instr)
        if role is Role.DETECTIVE:
            instr = (
                "It is night. Choose one living player to investigate; you will learn "
                f"their true alignment. Valid targets: {', '.join(living)}."
            )
            return self._act(state, "night_investigate", instr)
        if role is Role.DOCTOR:
            allnames = state.living_names()
            instr = (
                "It is night. Choose one living player to protect from being killed "
                f"(you may protect yourself). Valid targets: {', '.join(allnames)}."
            )
            return self._act(state, "night_protect", instr)
        return Action()  # villagers do nothing at night

    def speak(self, state: GameState, round_idx: int, total_rounds: int) -> Action:
        instr = (
            f"It is day {state.day}, discussion round {round_idx + 1} of {total_rounds}. "
            "Make a public statement to the group: share reads, defend yourself, ask "
            "questions, or build a case. Optionally name a player you find suspicious "
            "in 'target'. Keep it to a few sentences."
        )
        return self._act(state, "speech", instr)

    def vote(self, state: GameState) -> Action:
        living = [n for n in state.living_names() if n != self.name]
        instr = (
            "Voting time. Choose exactly one living player to eliminate. "
            f"Valid targets: {', '.join(living)}. Put the name in 'target'."
        )
        return self._act(state, "vote", instr)


# --------------------------------------------------------------------- parsing
def _parse_action(raw: str) -> Action:
    obj = _loads_lenient(raw)
    if obj is None:
        return Action(reasoning="", message=raw.strip()[:400], target=None, raw=raw)
    target = obj.get("target")
    if isinstance(target, str):
        target = target.strip() or None
    return Action(
        reasoning=str(obj.get("reasoning", "")),
        message=str(obj.get("message", "")),
        target=target if isinstance(target, str) else None,
        raw=raw,
    )


def _loads_lenient(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip markdown fences if a model added them.
    fence = re.match(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Last resort: grab the first {...} block.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
