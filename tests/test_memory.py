"""Phase 5.1 tests: demographic-grounded agent memories and prompt injection."""

from sfsim.agent import Agent
from sfsim.memory import (
    _BUDGET_TIGHT_MEMORIES,
    _RESTAURANT_INSIDER_MEMORIES,
    _SENIOR_USAGE_MEMORIES,
    attach_memories,
    generate_memories,
    with_memories,
)
from sfsim.scenario import build_prompt

OCEAN = {"openness": 5, "conscientiousness": 5, "extraversion": 5, "agreeableness": 5, "neuroticism": 5}


def _agent(agent_id=1, *, age=40, income="$75k-$150k", occupation="Sales Worker") -> Agent:
    return Agent(
        id=agent_id, name="Test Person", age=age, neighborhood="Mission",
        occupation=occupation, income_bracket=income, sex="female",
        education="Bachelor's degree", tenure="renter", ocean=OCEAN,
        profile="A profile. Two sentences.",
    )


def test_generates_exactly_three_memories():
    mems = generate_memories(_agent())
    assert len(mems) == 3
    assert all(isinstance(m, str) and m for m in mems)


def test_memories_are_deterministic():
    assert generate_memories(_agent(7)) == generate_memories(_agent(7))


def test_budget_tight_agent_gets_a_cost_memory():
    mems = generate_memories(_agent(income="<$30k"))
    assert mems[0] in _BUDGET_TIGHT_MEMORIES  # slot A is the money lens


def test_restaurant_worker_gets_insider_memory():
    mems = generate_memories(_agent(occupation="Food Service Worker"))
    assert mems[1] in _RESTAURANT_INSIDER_MEMORIES  # slot B is the occupation lens


def test_senior_gets_low_usage_memory():
    mems = generate_memories(_agent(age=72))
    assert mems[2] in _SENIOR_USAGE_MEMORIES  # slot C is the usage lens


def test_with_memories_is_immutable():
    original = _agent()
    updated = with_memories(original)
    assert original.memories == ()  # original untouched
    assert len(updated.memories) == 3
    assert updated.id == original.id


def test_attach_memories_covers_all_agents():
    agents = [_agent(1), _agent(2)]
    withmem = attach_memories(agents)
    assert all(len(a.memories) == 3 for a in withmem)


def test_prompt_includes_memories_only_when_present():
    plain = build_prompt(_agent())
    assert "RECENT DELIVERY-APP EXPERIENCES" not in plain
    withmem = build_prompt(with_memories(_agent()))
    assert "RECENT DELIVERY-APP EXPERIENCES" in withmem
