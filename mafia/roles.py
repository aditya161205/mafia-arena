"""Role definitions for the Mafia / Werewolf arena.

The game is a classic social-deduction setup. Two factions:

* **Mafia** (a.k.a. werewolves) — know each other, act together at night to
  eliminate a villager, and must blend in during the day.
* **Town** (villagers + special roles) — do not know who the mafia are and must
  deduce it from discussion and voting behaviour.

Special town roles add hidden information that makes deduction non-trivial:

* **Detective** — each night privately learns the true faction of one player.
* **Doctor** — each night privately protects one player from being killed.
"""

from __future__ import annotations

from enum import Enum


class Faction(str, Enum):
    MAFIA = "mafia"
    TOWN = "town"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class Role(str, Enum):
    MAFIA = "mafia"
    VILLAGER = "villager"
    DETECTIVE = "detective"
    DOCTOR = "doctor"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value

    @property
    def faction(self) -> Faction:
        return Faction.MAFIA if self is Role.MAFIA else Faction.TOWN

    @property
    def acts_at_night(self) -> bool:
        """Whether this role submits a private action during the night phase."""
        return self in (Role.MAFIA, Role.DETECTIVE, Role.DOCTOR)

    @property
    def public_description(self) -> str:
        """A short description every player could plausibly claim to be."""
        return {
            Role.MAFIA: "a member of the mafia",
            Role.VILLAGER: "an ordinary villager with no special ability",
            Role.DETECTIVE: "the detective, able to investigate one player each night",
            Role.DOCTOR: "the doctor, able to protect one player each night",
        }[self]


# Default role distributions keyed by player count. Each entry is the multiset
# of roles dealt for that table size. Town always outnumbers mafia at the start.
DEFAULT_SETUPS: dict[int, list[Role]] = {
    5: [Role.MAFIA, Role.DETECTIVE, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER],
    6: [Role.MAFIA, Role.DETECTIVE, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER],
    7: [Role.MAFIA, Role.MAFIA, Role.DETECTIVE, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER],
    8: [Role.MAFIA, Role.MAFIA, Role.DETECTIVE, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER],
    9: [Role.MAFIA, Role.MAFIA, Role.DETECTIVE, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER],
    10: [Role.MAFIA, Role.MAFIA, Role.MAFIA, Role.DETECTIVE, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER],
}


def role_setup(num_players: int) -> list[Role]:
    """Return the list of roles to deal for ``num_players`` players."""
    if num_players in DEFAULT_SETUPS:
        return list(DEFAULT_SETUPS[num_players])
    if num_players < 5:
        raise ValueError("Need at least 5 players for a meaningful game.")
    # Generic fallback: ~1 mafia per 3 players, one detective, one doctor.
    num_mafia = max(1, round(num_players / 3.5))
    roles = [Role.MAFIA] * num_mafia + [Role.DETECTIVE, Role.DOCTOR]
    roles += [Role.VILLAGER] * (num_players - len(roles))
    return roles
