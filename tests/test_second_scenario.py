"""Phase 5.5 tests: the $5-credit second scenario (no API)."""

from sfsim.agent import Agent
from sfsim.scenario import VOTE_NO, VOTE_YES, Vote
from sfsim.second_scenario import CREDIT_USD, build_incentive_prompt, run_second_scenario
from sfsim.social import social_delta

OCEAN = {"openness": 5, "conscientiousness": 5, "extraversion": 5, "agreeableness": 5, "neuroticism": 5}


def _agent(agent_id: int = 1) -> Agent:
    return Agent(
        id=agent_id, name=f"Agent {agent_id}", age=40, neighborhood="Mission",
        occupation="Sales Worker", income_bracket="$30k-$75k", sex="female",
        education="Bachelor's degree", tenure="renter", ocean=OCEAN,
        profile="A profile. Two sentences.",
    )


def _vote(agent_id: int, vote: str) -> Vote:
    return Vote(agent_id, f"Agent {agent_id}", vote, "because")


def test_prompt_mentions_prior_vote_and_the_credit():
    prompt = build_incentive_prompt(_agent(1), _vote(1, VOTE_YES))
    assert "You voted Yes" in prompt
    assert f"${CREDIT_USD} credit" in prompt


def test_run_flows_and_can_flip_votes():
    agents = [_agent(1), _agent(2)]
    round1 = [_vote(1, VOTE_YES), _vote(2, VOTE_YES)]
    take_deal = lambda p: "VOTE: No\nREASON: Five bucks is five bucks."  # noqa: E731
    round2 = run_second_scenario(agents, round1, call_fn=take_deal, write=False)
    assert [v.vote for v in round2] == [VOTE_NO, VOTE_NO]


def test_behavioral_delta_is_measurable():
    agents = [_agent(1), _agent(2), _agent(3)]
    round1 = [_vote(1, VOTE_YES), _vote(2, VOTE_YES), _vote(3, VOTE_NO)]
    # Agent 1 takes the deal, others hold.
    def selective(prompt: str) -> str:
        return "VOTE: No\nREASON: ok" if "Agent 1" in prompt else "VOTE: Yes\nREASON: principle"

    round2 = run_second_scenario(agents, round1, call_fn=selective, write=False)
    delta = social_delta(round1, round2)
    assert delta["changed_votes"] == 2  # agent 1 (Y->N) and agent 3 (N->Y)


def test_skips_agents_without_a_prior_vote():
    agents = [_agent(1), _agent(2)]
    round1 = [_vote(1, VOTE_YES)]  # no vote for agent 2
    round2 = run_second_scenario(agents, round1, call_fn=lambda p: "VOTE: No\nREASON: x", write=False)
    assert [v.agent_id for v in round2] == [1]
