"""Phase 5.3 tests: reflection trait choice, OCEAN nudge, and storage (no API)."""

from sfsim.agent import Agent
from sfsim.reflection import (
    apply_reflection,
    build_reflection_prompt,
    choose_trait,
    generate_reflection,
    run_reflection,
)
from sfsim.scenario import VOTE_NO, VOTE_YES, Vote


def _agent(agent_id: int = 1, **ocean_overrides) -> Agent:
    ocean = {"openness": 5, "conscientiousness": 5, "extraversion": 5,
             "agreeableness": 5, "neuroticism": 5}
    ocean.update(ocean_overrides)
    return Agent(
        id=agent_id, name="Test Person", age=40, neighborhood="Mission",
        occupation="Sales Worker", income_bracket="$30k-$75k", sex="female",
        education="Bachelor's degree", tenure="renter", ocean=ocean,
        profile="A profile. Two sentences.", memories=("existing memory",),
    )


def _vote(vote: str = VOTE_YES, reason: str = "because") -> Vote:
    return Vote(1, "Test Person", vote, reason)


def test_choose_trait_from_reflection_wording():
    assert choose_trait("I have to watch my budget on these costs", VOTE_NO) == "neuroticism"
    assert choose_trait("I want to support my community", VOTE_YES) == "agreeableness"
    assert choose_trait("I'm open to trying a new policy", VOTE_YES) == "openness"
    assert choose_trait("I should weigh the long-term consequences", VOTE_YES) == "conscientiousness"


def test_choose_trait_falls_back_to_vote():
    assert choose_trait("hmm not sure", VOTE_YES) == "agreeableness"
    assert choose_trait("hmm not sure", VOTE_NO) == "neuroticism"


def test_apply_reflection_nudges_trait_and_stores_memory():
    agent = _agent(agreeableness=5)
    updated = apply_reflection(agent, "I want to help my community", VOTE_YES)
    assert updated.ocean["agreeableness"] == 6  # +1
    assert updated.memories[-1] == "Reflection: I want to help my community"
    assert agent.ocean["agreeableness"] == 5  # original untouched
    assert agent.memories == ("existing memory",)


def test_apply_reflection_clamps_at_ten():
    agent = _agent(agreeableness=10)
    updated = apply_reflection(agent, "community and support", VOTE_YES)
    assert updated.ocean["agreeableness"] == 10  # already max, stays


def test_build_prompt_includes_vote_and_reason():
    prompt = build_reflection_prompt(_agent(), _vote(VOTE_NO, "it costs too much"))
    assert "You voted No" in prompt
    assert "it costs too much" in prompt


def test_generate_reflection_takes_first_nonempty_line():
    fake = lambda p: '\n"I care about my neighbors."\nextra junk'  # noqa: E731
    assert generate_reflection(_agent(), _vote(), fake) == "I care about my neighbors."


def test_run_reflection_updates_every_agent():
    agents = [_agent(1), _agent(2)]
    votes = [_vote(VOTE_YES), Vote(2, "Test Person", VOTE_NO, "cost")]
    fake = lambda p: "I want to support local restaurants."  # noqa: E731
    updated = run_reflection(agents, votes, call_fn=fake)
    assert all(len(a.memories) == 2 for a in updated)  # existing + reflection
    assert all(a.memories[-1].startswith("Reflection:") for a in updated)
