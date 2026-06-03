"""System prompts and prompt construction.

Each agent receives:

* a **role system prompt** establishing identity, goals, and (for mafia) secret
  partner knowledge plus an explicit instruction to deceive convincingly;
* a **task prompt** containing the agent's full memory (a filtered transcript),
  the current request, and a machine-readable ``<context>`` block that the mock
  backend reads and that real models can ignore.

Agents are always asked to reason step-by-step privately before producing their
public action, and to return a single JSON object so the engine can parse it.
"""

from __future__ import annotations

import json

from .roles import Role

_COMMON_RULES = """\
GAME RULES
- This is a game of Mafia. The town and the mafia are in conflict.
- Each NIGHT, the mafia secretly choose one player to eliminate. The detective
  investigates one player's true alignment. The doctor protects one player.
- Each DAY, the survivors discuss, then vote. The player with the most votes is
  eliminated and their role is revealed.
- The TOWN wins when every mafia member is eliminated.
- The MAFIA win when they reach numerical parity with the town.

HOW TO PLAY WELL
- Track who said what, who voted for whom, and look for inconsistencies.
- Build a model of what others believe about YOU (theory of mind).
- Reason explicitly and privately before you act, then commit to one clear action.
"""

_OUTPUT_CONTRACT = """\
OUTPUT FORMAT
Respond with a SINGLE JSON object and nothing else. Fields:
  "reasoning": a short private chain-of-thought (never shown to other players).
  "message":   your public statement, when one is requested (omit otherwise).
  "target":    the exact name of the player you are voting for / acting on, or null.
Do not wrap the JSON in markdown fences. Use only names of LIVING players.
"""


def role_system_prompt(role: Role, name: str, partners: list[str]) -> str:
    persona = f"You are {name}, playing a high-stakes game of social deduction.\n\n"

    if role is Role.MAFIA:
        partner_txt = (
            f"Your secret mafia partner(s): {', '.join(partners)}. "
            "Protect them subtly; never out them."
            if partners else
            "You are the lone mafia member."
        )
        special = f"""\
YOUR ROLE: MAFIA (secret).
{partner_txt}
Your goal is to eliminate the town without being discovered. During the day you
must pass as an innocent villager: build a credible alibi, cast plausible
suspicion on town players, agree with the crowd when it's safe, and never
contradict yourself. Coordinate kills to remove the town's most dangerous
players (likely the detective or sharp reasoners). Deceive convincingly — your
survival depends on it — but stay internally consistent, because contradictions
are how you get caught.
"""
    elif role is Role.DETECTIVE:
        special = """\
YOUR ROLE: DETECTIVE (secret town power).
Each night you learn the TRUE alignment of one player. This is your edge. Decide
carefully when to reveal what you know: claiming too early paints a target on
your back for the mafia's next kill, but staying silent too long wastes
information. Use your investigations to anchor the town's votes on real mafia.
"""
    elif role is Role.DOCTOR:
        special = """\
YOUR ROLE: DOCTOR (secret town power).
Each night you protect one player from being killed. You may protect yourself.
Think about who the mafia most want dead — often a revealed or suspected
detective — and guard them. During the day, reason like a villager and help vote
out the mafia.
"""
    else:
        special = """\
YOUR ROLE: VILLAGER.
You have no special power — only your reasoning. Pay close attention to
contradictions, voting patterns, and who benefits from each death. Form
coalitions with players you trust and push the vote toward the real mafia.
"""

    return persona + special + "\n" + _COMMON_RULES + "\n" + _OUTPUT_CONTRACT


def _transcript(memory_lines: list[str], limit: int | None) -> str:
    lines = memory_lines if limit is None else memory_lines[-limit:]
    return "\n".join(lines) if lines else "(no events yet)"


def task_prompt(
    *,
    action: str,
    instruction: str,
    memory_lines: list[str],
    context: dict,
    memory_limit: int | None,
) -> str:
    """Assemble the per-turn user prompt."""
    ctx_json = json.dumps({"action": action, **context}, ensure_ascii=False)
    return f"""\
GAME SO FAR (your private memory):
{_transcript(memory_lines, memory_limit)}

CURRENT REQUEST:
{instruction}

<context>{ctx_json}</context>

Remember: reason privately in "reasoning", then give your action. Return ONLY the JSON object.
"""
