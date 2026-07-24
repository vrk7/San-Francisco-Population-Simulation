"""Agent memory: three past delivery-app experiences per agent (Part 4).

Each agent gets three concrete, first-person memories generated *deterministically
from their demographics* — one through a money/budget lens, one through their
occupation, one through their age/usage. Grounding the vote in lived experience is
the principled fix for the LLM's pro-social bias: a budget-tight, delivery-reliant
agent now carries real reasons to worry about their own costs (and defect to No),
while a restaurant-connected or comfortable agent carries reasons to sympathize.

Memories are injected into the voting prompt (see ``scenario.build_prompt``). The
memory-on run is compared against the memory-off 100%-Yes baseline.
"""

import random

from sfsim.agent import Agent
from sfsim.constants import RANDOM_SEED

MEMORY_SEED_OFFSET = 2000  # distinct from OCEAN (×1000) and name (×1) seed streams

# Occupations that experience delivery apps from the *inside* of a restaurant.
_RESTAURANT_INSIDER_OCCUPATIONS = frozenset({"Food Service Worker"})
# Occupations of gig/delivery drivers themselves.
_DRIVER_OCCUPATIONS = frozenset({"Transportation & Delivery Worker"})
# Busy, higher-earning office jobs that tend to order in a lot.
_BUSY_PROFESSIONAL_OCCUPATIONS = frozenset(
    {"Software & Tech Worker", "Manager", "Business & Finance Specialist",
     "Engineer / Architect", "Legal Professional"}
)

_BUDGET_TIGHT_BRACKETS = frozenset({"<$30k", "$30k-$75k"})
_SENIOR_AGE = 65
_YOUNG_AGE = 35

# --- Slot A: money / budget lens -------------------------------------------------
_BUDGET_TIGHT_MEMORIES = (
    "Money's tight, and I'm scared a cap just pushes the apps to charge me a bigger delivery fee instead.",
    "I already cancel orders when the service fee spikes — I can't risk delivery getting any pricier for me.",
    "I've heard the apps say fee caps make them raise customer charges, and on my budget that really worries me.",
)
_COMFORTABLE_MEMORIES = (
    "I order delivery without thinking twice about the fees — convenience is worth it to me.",
    "The service fees annoy me, but they don't really change what I can afford to order.",
    "I'd happily pay a little more per order if it meant my neighborhood spots stayed open.",
)

# --- Slot B: occupation lens -----------------------------------------------------
_RESTAURANT_INSIDER_MEMORIES = (
    "At work I watched a 30% app commission eat the profit on every online order we filled.",
    "My manager once said the delivery apps take a bigger cut than our rent some months.",
)
_DRIVER_MEMORIES = (
    "As a driver I've seen how the apps squeeze both the restaurants and my own pay.",
    "The apps change their fee math constantly, and it's usually the little guys who lose.",
)
_BUSY_PROFESSIONAL_MEMORIES = (
    "After long workdays I order in most nights, so anything that slows or raises delivery hits me.",
    "I rely on fast cheap delivery to get through crunch weeks — I don't want it disrupted.",
)
_GENERAL_CUSTOMER_MEMORIES = (
    "A local cafe I loved closed, and the owner blamed the delivery apps' commissions.",
    "A friend who runs a restaurant showed me the huge cut the apps take on each order.",
)

# --- Slot C: age / usage lens ----------------------------------------------------
_SENIOR_USAGE_MEMORIES = (
    "Honestly I rarely use these apps — I usually cook at home or eat out in person.",
    "I've only used delivery apps a handful of times, mostly when I wasn't feeling well.",
)
_YOUNG_USAGE_MEMORIES = (
    "Delivery apps are just part of my routine — I order several times a week.",
    "Most of my meals during the week come off a delivery app, honestly.",
)
_MID_USAGE_MEMORIES = (
    "I order delivery now and then, usually on busy weeknights.",
    "Delivery is an occasional treat for me, not an everyday thing.",
)


def _budget_pool(agent: Agent) -> tuple[str, ...]:
    if agent.income_bracket in _BUDGET_TIGHT_BRACKETS:
        return _BUDGET_TIGHT_MEMORIES
    return _COMFORTABLE_MEMORIES


def _occupation_pool(agent: Agent) -> tuple[str, ...]:
    if agent.occupation in _RESTAURANT_INSIDER_OCCUPATIONS:
        return _RESTAURANT_INSIDER_MEMORIES
    if agent.occupation in _DRIVER_OCCUPATIONS:
        return _DRIVER_MEMORIES
    if agent.occupation in _BUSY_PROFESSIONAL_OCCUPATIONS:
        return _BUSY_PROFESSIONAL_MEMORIES
    return _GENERAL_CUSTOMER_MEMORIES


def _usage_pool(agent: Agent) -> tuple[str, ...]:
    if agent.age >= _SENIOR_AGE:
        return _SENIOR_USAGE_MEMORIES
    if agent.age < _YOUNG_AGE:
        return _YOUNG_USAGE_MEMORIES
    return _MID_USAGE_MEMORIES


def generate_memories(agent: Agent) -> tuple[str, str, str]:
    """Return three demographic-grounded memories (budget, occupation, usage).

    Deterministic per agent: the same agent always gets the same memories.
    """
    rng = random.Random(RANDOM_SEED * MEMORY_SEED_OFFSET + agent.id)
    return (
        rng.choice(_budget_pool(agent)),
        rng.choice(_occupation_pool(agent)),
        rng.choice(_usage_pool(agent)),
    )


def with_memories(agent: Agent) -> Agent:
    """Return a copy of ``agent`` carrying its generated memories."""
    return agent.with_updates(memories=generate_memories(agent))


def attach_memories(agents: list[Agent]) -> list[Agent]:
    """Return copies of every agent with demographic-grounded memories attached."""
    return [with_memories(a) for a in agents]
