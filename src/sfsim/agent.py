"""The Agent data model — an immutable SF resident."""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Agent:
    """A single simulated San Francisco resident.

    Immutable: use :meth:`with_updates` to derive a modified copy rather than
    mutating in place. ``ocean`` maps each trait name (openness, conscientiousness,
    extraversion, agreeableness, neuroticism) to a 1-10 integer score.
    """

    id: int
    name: str
    age: int
    neighborhood: str
    occupation: str
    income_bracket: str
    sex: str
    education: str
    tenure: str  # "owner" | "renter"
    ocean: dict[str, int]
    profile: str
    memories: tuple[str, ...] = field(default_factory=tuple)

    def with_updates(self, **changes: object) -> "Agent":
        """Return a new Agent with the given fields replaced."""
        return replace(self, **changes)
