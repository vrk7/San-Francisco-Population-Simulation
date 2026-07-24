"""Phase 3 tests: prompt building, vote parsing, and the scenario loop (no real API)."""

import pytest

from sfsim.agent import Agent
from sfsim.scenario import (
    VOTE_ERROR,
    VOTE_NO,
    VOTE_YES,
    Vote,
    VoteParseError,
    build_prompt,
    parse_vote,
    run_scenario,
)

OCEAN = {
    "openness": 8,
    "conscientiousness": 5,
    "extraversion": 6,
    "agreeableness": 3,
    "neuroticism": 4,
}


def _agent(agent_id: int = 1) -> Agent:
    return Agent(
        id=agent_id,
        name="Test Person",
        age=34,
        neighborhood="Mission",
        occupation="Sales Worker",
        income_bracket="$30k-$75k",
        sex="female",
        education="Bachelor's degree",
        tenure="renter",
        ocean=OCEAN,
        profile="A profile sentence. Another one.",
    )


def test_prompt_embeds_neighborhood_and_ocean():
    prompt = build_prompt(_agent())
    assert "Mission" in prompt
    assert "Openness 8/10 (high)" in prompt
    assert "Agreeableness 3/10 (low)" in prompt
    assert "15%" in prompt  # the fee cap from the challenge question


def test_parse_strict_format():
    vote, reason = parse_vote("VOTE: Yes\nREASON: Restaurants deserve fair fees.")
    assert vote == VOTE_YES
    assert reason == "Restaurants deserve fair fees."


def test_parse_handles_messy_casing_and_extra_text():
    text = "Hmm, let me think.\nvote - no\nreason - I want cheap delivery, honestly."
    vote, reason = parse_vote(text)
    assert vote == VOTE_NO
    assert reason == "I want cheap delivery, honestly."


def test_parse_falls_back_to_bare_yes_no():
    vote, reason = parse_vote("Yes, absolutely — small restaurants need this.")
    assert vote == VOTE_YES
    assert reason  # some reason text is captured


def test_parse_raises_when_no_vote():
    with pytest.raises(VoteParseError):
        parse_vote("I am undecided and cannot make up my mind.")


def test_run_scenario_flows_end_to_end_with_fake_llm():
    agents = [_agent(1), _agent(2)]
    fake = lambda prompt: "VOTE: Yes\nREASON: Fairer for restaurant workers."  # noqa: E731
    votes = run_scenario(agents, call_fn=fake, write=False)
    assert [v.vote for v in votes] == [VOTE_YES, VOTE_YES]
    assert all(isinstance(v, Vote) for v in votes)
    assert votes[0].agent_id == 1


def test_run_scenario_retries_then_records_error():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        return "I really can't decide either way."  # never parseable

    votes = run_scenario([_agent(1)], call_fn=flaky, write=False)
    assert votes[0].vote == VOTE_ERROR
    assert calls["n"] == 2  # original attempt + one retry


def test_run_scenario_retry_recovers_on_second_try():
    responses = iter(["mumble mumble", "VOTE: No\nREASON: Prices will rise."])

    def recovering(prompt: str) -> str:
        return next(responses)

    votes = run_scenario([_agent(1)], call_fn=recovering, write=False)
    assert votes[0].vote == VOTE_NO
