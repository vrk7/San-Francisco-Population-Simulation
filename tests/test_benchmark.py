"""Phase 4 tests: split, delta, reason clustering, and standout pick (no API)."""

from sfsim.agent import Agent
from sfsim.benchmark import benchmark, pick_standout, top_reasons
from sfsim.constants import REAL_YES_PCT
from sfsim.scenario import VOTE_ERROR, VOTE_NO, VOTE_YES, Vote


def _agent(agent_id: int, *, agreeableness: int = 5, openness: int = 5, neuroticism: int = 5) -> Agent:
    return Agent(
        id=agent_id,
        name=f"Agent {agent_id}",
        age=40,
        neighborhood="Mission",
        occupation="Sales Worker",
        income_bracket="$30k-$75k",
        sex="female",
        education="Bachelor's degree",
        tenure="renter",
        ocean={
            "openness": openness,
            "conscientiousness": 5,
            "extraversion": 5,
            "agreeableness": agreeableness,
            "neuroticism": neuroticism,
        },
        profile="A profile. Two sentences.",
    )


def _vote(agent_id: int, vote: str, reason: str) -> Vote:
    return Vote(agent_id, f"Agent {agent_id}", vote, reason)


def test_split_and_delta():
    votes = [
        _vote(1, VOTE_YES, "fair to restaurants"),
        _vote(2, VOTE_YES, "support local restaurants"),
        _vote(3, VOTE_NO, "it will cost me more"),
        _vote(4, VOTE_ERROR, "garbled"),
    ]
    agents = [_agent(i) for i in range(1, 5)]
    result = benchmark(votes, agents)
    assert (result.yes, result.no, result.errors, result.total) == (2, 1, 1, 4)
    # yes_pct is over valid votes only: 2 of 3.
    assert round(result.yes_pct, 1) == 66.7
    assert round(result.delta, 1) == round(66.7 - REAL_YES_PCT, 1)


def test_top_reasons_cluster_by_theme_and_are_deterministic():
    from sfsim.benchmark import YES_THEMES

    votes = [
        _vote(1, VOTE_YES, "It's only fair, these apps gouge restaurants."),
        _vote(2, VOTE_YES, "This is unfair price-gouging of small eateries."),
        _vote(3, VOTE_YES, "I want to support my local restaurants and cafes."),
    ]
    top = top_reasons(votes, VOTE_YES, YES_THEMES)
    assert top[0] == ("Fairness / curb price-gouging", 2)
    assert top_reasons(votes, VOTE_YES, YES_THEMES) == top  # deterministic


def test_empty_side_returns_no_reasons():
    from sfsim.benchmark import NO_THEMES

    votes = [_vote(1, VOTE_YES, "fair")]
    assert top_reasons(votes, VOTE_NO, NO_THEMES) == []


def test_standout_is_the_biggest_personality_vote_mismatch():
    # Agent 2 is very disagreeable (expects No) but votes Yes -> the standout.
    agents = {
        1: _agent(1, agreeableness=8),  # warm, votes Yes (consistent)
        2: _agent(2, agreeableness=1, neuroticism=9),  # cold+anxious, expects No
    }
    votes = [
        _vote(1, VOTE_YES, "restaurants deserve fairness"),
        _vote(2, VOTE_YES, "surprisingly, I think it's fair"),
    ]
    pick = pick_standout(votes, agents)
    assert pick is not None
    assert pick.agent_id == 2
    assert pick.vote == VOTE_YES


def test_standout_falls_back_when_no_contradiction():
    agents = {1: _agent(1, agreeableness=8)}
    votes = [_vote(1, VOTE_YES, "consistent yes")]
    pick = pick_standout(votes, agents)
    assert pick is not None and pick.agent_id == 1


def test_benchmark_and_report_render():
    from sfsim.benchmark import render_report

    votes = [_vote(1, VOTE_YES, "fair to local restaurants"), _vote(2, VOTE_NO, "costs me more")]
    agents = [_agent(1, agreeableness=8), _agent(2, agreeableness=2)]
    result = benchmark(votes, agents)
    report = render_report(result, votes)
    assert "50.0% Yes" in report
    assert f"{REAL_YES_PCT:.1f}% Yes" in report
    assert "Most interesting response" in report
