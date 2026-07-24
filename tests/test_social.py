"""Phase 5.2 tests: neighbor tally, social prompt, re-vote, and delta (no API)."""

from sfsim.agent import Agent
from sfsim.scenario import VOTE_NO, VOTE_YES, Vote
from sfsim.social import (
    build_social_prompt,
    district_of,
    format_summary,
    neighbor_tally,
    run_social_round,
    social_delta,
)

OCEAN = {"openness": 5, "conscientiousness": 5, "extraversion": 5, "agreeableness": 5, "neuroticism": 5}


def _agent(agent_id: int, neighborhood: str) -> Agent:
    return Agent(
        id=agent_id, name=f"Agent {agent_id}", age=40, neighborhood=neighborhood,
        occupation="Sales Worker", income_bracket="$30k-$75k", sex="female",
        education="Bachelor's degree", tenure="renter", ocean=OCEAN,
        profile="A profile. Two sentences.",
    )


def _vote(agent_id: int, vote: str) -> Vote:
    return Vote(agent_id, f"Agent {agent_id}", vote, "because")


def test_district_lookup_maps_neighborhoods_to_puma():
    # SoMa and Mission are both in PUMA 07509 per the crosswalk.
    assert district_of(_agent(1, "SoMa")) == district_of(_agent(2, "Mission"))
    assert district_of(_agent(3, "Richmond")) != district_of(_agent(1, "SoMa"))


def test_neighbor_tally_counts_same_district_excluding_self():
    agents = [_agent(1, "SoMa"), _agent(2, "Mission"), _agent(3, "Richmond")]
    by_id = {a.id: a for a in agents}
    votes = [_vote(1, VOTE_YES), _vote(2, VOTE_NO), _vote(3, VOTE_YES)]
    # Agent 1 (SoMa): neighbor 2 (Mission, same PUMA) counts; 3 (Richmond) does not; self excluded.
    assert neighbor_tally(agents[0], votes, by_id) == (0, 1)


def test_format_summary_wording():
    assert "already voted" in format_summary(3, 1)
    assert "yet" in format_summary(0, 0)


def test_social_prompt_includes_prior_vote_and_neighbors():
    prompt = build_social_prompt(_agent(1, "Mission"), _vote(1, VOTE_YES), 4, 1)
    assert "You answered Yes" in prompt
    assert "4 voted Yes and 1 voted No" in prompt


def test_run_social_round_flows_with_fake_llm():
    agents = [_agent(1, "SoMa"), _agent(2, "Mission")]
    round1 = [_vote(1, VOTE_YES), _vote(2, VOTE_YES)]
    fake = lambda prompt: "VOTE: No\nREASON: My neighbors changed my mind... actually no."  # noqa: E731
    round2 = run_social_round(agents, round1, call_fn=fake, write=False)
    assert [v.vote for v in round2] == [VOTE_NO, VOTE_NO]


def test_social_delta_reports_shift_and_changes():
    round1 = [_vote(1, VOTE_YES), _vote(2, VOTE_NO), _vote(3, VOTE_YES)]
    round2 = [_vote(1, VOTE_YES), _vote(2, VOTE_YES), _vote(3, VOTE_YES)]
    delta = social_delta(round1, round2)
    assert delta["round1_yes_pct"] == 66.7
    assert delta["round2_yes_pct"] == 100.0
    assert delta["changed_votes"] == 1
