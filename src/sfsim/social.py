"""Social influence: agents see how their neighbors voted, then re-vote.

After round 1, each agent is shown the Yes/No tally of their *same-district*
neighbors (the PUMA the agent's neighborhood belongs to — a real geographic unit
with several agents each, unlike the sparse fine-grained neighborhoods) and asked
for a final vote. The round-1 vs round-2 split is the emergent-behavior delta
(Part 4): does social proof make agents conform to the local majority?
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from sfsim.agent import Agent
from sfsim.geography import SF_PUMA_INFO
from sfsim.llm import complete
from sfsim.population import RESULTS_DIR
from sfsim.scenario import (
    DECISION_RULES,
    SCENARIO_QUESTION,
    VOTE_NO,
    VOTE_YES,
    Vote,
    cast_vote,
)

CallFn = Callable[[str], str]

# neighborhood name -> its PUMA district code (built once from the crosswalk).
_NEIGHBORHOOD_TO_DISTRICT: dict[str, str] = {
    neighborhood: code
    for code, info in SF_PUMA_INFO.items()
    for neighborhood in info.neighborhoods
}


def district_of(agent: Agent) -> str | None:
    """The PUMA district code for an agent's neighborhood, or None if unknown."""
    return _NEIGHBORHOOD_TO_DISTRICT.get(agent.neighborhood)


def neighbor_tally(
    agent: Agent, votes: Sequence[Vote], agents_by_id: dict[int, Agent]
) -> tuple[int, int]:
    """(yes, no) counts among the agent's same-district neighbors, excluding self."""
    district = district_of(agent)
    yes = no = 0
    for vote in votes:
        if vote.agent_id == agent.id:
            continue
        neighbor = agents_by_id.get(vote.agent_id)
        if neighbor is None or district_of(neighbor) != district:
            continue
        if vote.vote == VOTE_YES:
            yes += 1
        elif vote.vote == VOTE_NO:
            no += 1
    return yes, no


def format_summary(yes: int, no: int) -> str:
    """Plain-language description of how the neighbors voted."""
    total = yes + no
    if total == 0:
        return "None of your neighbors have shared their vote yet."
    return (
        f"Among your neighbors nearby who have already voted: "
        f"{yes} voted Yes and {no} voted No."
    )


def build_social_prompt(agent: Agent, prior: Vote, yes: int, no: int) -> str:
    """Round-2 prompt: remind the agent of their vote, show neighbors, ask final."""
    return (
        f"You are {agent.name}, a {agent.age}-year-old {agent.occupation.lower()} in "
        f"{agent.neighborhood}, San Francisco.\n\n"
        f"Earlier you were asked: {SCENARIO_QUESTION}\n"
        f"You answered {prior.vote} — \"{prior.reason}\"\n\n"
        f"Now you've heard how your community is leaning.\n"
        f"{format_summary(yes, no)}\n\n"
        f"Knowing how your neighbors voted, what is your FINAL vote? You may keep your "
        f"vote or change it — real people are sometimes swayed by their community and "
        f"sometimes hold firm. {DECISION_RULES}\n\n"
        f"Answer in EXACTLY this format, nothing else:\n"
        f"VOTE: Yes or No\n"
        f"REASON: <one sentence in your own voice>"
    )


def run_social_round(
    agents: Sequence[Agent],
    round1_votes: Sequence[Vote],
    *,
    call_fn: CallFn = complete,
    write: bool = True,
) -> list[Vote]:
    """Run the round-2 re-vote after showing each agent their neighbors' tally."""
    agents_by_id = {a.id: a for a in agents}
    votes_by_id = {v.agent_id: v for v in round1_votes}
    round2: list[Vote] = []
    for agent in agents:
        prior = votes_by_id.get(agent.id)
        if prior is None:
            continue
        yes, no = neighbor_tally(agent, round1_votes, agents_by_id)
        prompt = build_social_prompt(agent, prior, yes, no)
        round2.append(cast_vote(agent, prompt, call_fn))
    if write:
        _write_social(round2)
    return round2


def social_delta(round1: Sequence[Vote], round2: Sequence[Vote]) -> dict[str, object]:
    """Summarize the shift: Yes% before/after and how many agents changed vote."""
    def yes_pct(votes: Sequence[Vote]) -> float:
        valid = [v for v in votes if v.vote in (VOTE_YES, VOTE_NO)]
        return 100.0 * sum(v.vote == VOTE_YES for v in valid) / len(valid) if valid else 0.0

    before, after = {v.agent_id: v.vote for v in round1}, {v.agent_id: v.vote for v in round2}
    changed = sum(
        1 for aid in before if aid in after and before[aid] != after[aid]
    )
    return {
        "round1_yes_pct": round(yes_pct(round1), 1),
        "round2_yes_pct": round(yes_pct(round2), 1),
        "changed_votes": changed,
    }


def _write_social(votes: Sequence[Vote], path: Path | None = None) -> None:
    """Persist round-2 votes to results/votes_social.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    target = path or (RESULTS_DIR / "votes_social.json")
    target.write_text(json.dumps([asdict(v) for v in votes], indent=2))
