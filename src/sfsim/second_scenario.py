"""Second scenario: DoorDash offers a $5 credit to vote No, and we measure the shift.

After the first vote, each agent is personally offered a small self-interested
incentive (a $5 credit) to switch to No. Re-voting under the offer measures how
mercenary vs. principled the population is: how many agents a tiny personal
benefit can pull away from their original choice (the behavioral delta).
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from sfsim.agent import Agent
from sfsim.llm import complete
from sfsim.population import RESULTS_DIR
from sfsim.scenario import SCENARIO_QUESTION, Vote, cast_vote

CallFn = Callable[[str], str]

CREDIT_USD = 5  # the personal incentive offered to vote No


def build_incentive_prompt(agent: Agent, prior: Vote) -> str:
    """Round-2 prompt: remind the agent of its vote, then dangle the $5 credit."""
    return (
        f"You are {agent.name}, a {agent.age}-year-old {agent.occupation.lower()} in "
        f"{agent.neighborhood}, San Francisco.\n\n"
        f"Earlier you were asked: {SCENARIO_QUESTION}\n"
        f"You voted {prior.vote} — \"{prior.reason}\"\n\n"
        f"Now DoorDash makes you a personal offer: a ${CREDIT_USD} credit on your next "
        f"order if you vote No on the measure.\n\n"
        f"Knowing this offer, what is your FINAL vote? Decide as this person genuinely "
        f"would — some people take the deal, others refuse on principle, and a small "
        f"perk may or may not outweigh what they cared about. Do not just take it "
        f"automatically.\n\n"
        f"Answer in EXACTLY this format, nothing else:\n"
        f"VOTE: Yes or No\n"
        f"REASON: <one sentence in your own voice>"
    )


def run_second_scenario(
    agents: Sequence[Agent],
    round1_votes: Sequence[Vote],
    *,
    call_fn: CallFn = complete,
    write: bool = True,
) -> list[Vote]:
    """Re-vote after offering each agent the $5-credit incentive to vote No."""
    votes_by_id = {v.agent_id: v for v in round1_votes}
    result: list[Vote] = []
    for agent in agents:
        prior = votes_by_id.get(agent.id)
        if prior is None:
            continue
        result.append(cast_vote(agent, build_incentive_prompt(agent, prior), call_fn))
    if write:
        _write_bribe(result)
    return result


def _write_bribe(votes: Sequence[Vote], path: Path | None = None) -> None:
    """Persist incentive-round votes to results/votes_bribe.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    target = path or (RESULTS_DIR / "votes_bribe.json")
    target.write_text(json.dumps([asdict(v) for v in votes], indent=2))
