"""Proposition X voting scenario: build the prompt, call Groq, parse the vote.

Each agent answers Yes/No with a one-sentence reason, grounded in their
demographics and OCEAN profile. Parsing is defensive (the model may not follow
the format exactly) and failures are retried once, then recorded — never
silently dropped.
"""

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from sfsim.agent import Agent
from sfsim.constants import FEE_CAP_PERCENT
from sfsim.llm import complete
from sfsim.population import RESULTS_DIR

# The exact challenge question, with the fee cap injected from constants.
SCENARIO_QUESTION = (
    f"San Francisco is voting on a measure that would cap food delivery app fees "
    f"(DoorDash, Uber Eats) at {FEE_CAP_PERCENT}%. As a resident, would you vote Yes "
    f"or No? Give your single most important reason in one sentence."
)

VOTE_YES = "Yes"
VOTE_NO = "No"
VOTE_ERROR = "Error"  # recorded when a response can't be parsed after a retry

# OCEAN score bands, for rendering the persona in plain language.
_HIGH_BAND = 7
_LOW_BAND = 4

CallFn = Callable[[str], str]


@dataclass(frozen=True)
class Vote:
    """One agent's parsed ballot on Proposition X."""

    agent_id: int
    agent_name: str
    vote: str  # VOTE_YES | VOTE_NO | VOTE_ERROR
    reason: str


class VoteParseError(ValueError):
    """Raised when a model response has no discernible Yes/No vote."""


def _band(score: int) -> str:
    if score >= _HIGH_BAND:
        return "high"
    if score <= _LOW_BAND:
        return "low"
    return "moderate"


def describe_ocean(ocean: dict[str, int]) -> str:
    """Render OCEAN scores as a plain-language line for the prompt."""
    return ", ".join(
        f"{trait.capitalize()} {score}/10 ({_band(score)})"
        for trait, score in ocean.items()
    )


# Both sides of the real Prop F debate, so "No" is a genuine option and the model
# isn't nudged toward a one-sided "support local restaurants" answer.
STAKES = (
    "WHAT'S AT STAKE (both sides are real)\n"
    "- YES supporters: the apps charge restaurants commissions as high as 30%, and a "
    "cap protects struggling neighborhood eateries and the people who work in them.\n"
    "- NO opponents: the delivery apps warn that a cap pushes them to raise the "
    "delivery and service fees YOU pay as a customer, trim driver pay and availability, "
    "and pull back service — some people just want cheap, fast delivery and less "
    "government meddling in prices."
)

# Neutral decision instruction: vote as this individual. No benchmark is leaked and
# neither side is favored — the two competing considerations are named symmetrically
# (own wallet/convenience vs. supporting local businesses) and personality tips it.
DECISION_RULES = (
    "Before answering, genuinely weigh BOTH sides for this person: on one hand their "
    "own wallet and convenience (a cap may raise the fees they personally pay), on the "
    "other their feelings about local restaurants and workers. A cost-focused, skeptical, "
    "or self-reliant person may well vote No; a warm, community-minded one may vote Yes. "
    "Let their budget and personality tip the balance — pick what truly fits them, not "
    "the answer that sounds nicest. Both Yes and No are common, legitimate choices."
)


def build_prompt(agent: Agent) -> str:
    """Compose the in-character voting prompt for a single agent."""
    return (
        f"You are role-playing a real San Francisco resident. Stay fully in character "
        f"and answer as this specific person would.\n\n"
        f"WHO YOU ARE\n"
        f"- Name: {agent.name}\n"
        f"- Age: {agent.age}, {agent.sex}\n"
        f"- Neighborhood: {agent.neighborhood}\n"
        f"- Occupation: {agent.occupation}\n"
        f"- Household income: {agent.income_bracket}\n"
        f"- Housing: {agent.tenure}\n"
        f"- Education: {agent.education}\n"
        f"- Personality (Big Five, 1-10): {describe_ocean(agent.ocean)}\n"
        f"- Profile: {agent.profile}\n\n"
        f"{STAKES}\n\n"
        f"THE QUESTION\n{SCENARIO_QUESTION}\n\n"
        f"{DECISION_RULES}\n\n"
        f"Answer in EXACTLY this format, nothing else:\n"
        f"VOTE: Yes or No\n"
        f"REASON: <one sentence in your own voice>"
    )


def _extract_vote(text: str) -> str | None:
    """Find the Yes/No vote, preferring an explicit VOTE: line."""
    for line in text.splitlines():
        match = re.match(r"\s*vote\s*[:\-]\s*(yes|no)\b", line, re.IGNORECASE)
        if match:
            return VOTE_YES if match.group(1).lower() == "yes" else VOTE_NO
    # Fallback: first standalone yes/no anywhere in the response.
    match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if match:
        return VOTE_YES if match.group(1).lower() == "yes" else VOTE_NO
    return None


def _extract_reason(text: str) -> str:
    """Pull the one-sentence reason, preferring an explicit REASON: line."""
    for line in text.splitlines():
        match = re.match(r"\s*reason\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # Fallback: the first non-empty line that isn't the VOTE line.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not re.match(r"\s*vote\s*[:\-]", stripped, re.IGNORECASE):
            return stripped
    return text.strip()


def parse_vote(text: str) -> tuple[str, str]:
    """Parse a model response into a (vote, reason) pair.

    Raises :class:`VoteParseError` if no Yes/No vote can be found.
    """
    vote = _extract_vote(text)
    if vote is None:
        raise VoteParseError(f"No Yes/No vote found in response: {text!r}")
    return vote, _extract_reason(text)


def _vote_once(agent: Agent, call_fn: CallFn) -> Vote:
    """Ask one agent to vote, retrying a single time on a parse failure."""
    prompt = build_prompt(agent)
    for attempt in range(2):
        response = call_fn(prompt)
        try:
            vote, reason = parse_vote(response)
            return Vote(agent.id, agent.name, vote, reason)
        except VoteParseError:
            if attempt == 0:
                continue
            print(f"  ! Could not parse vote for agent {agent.id} ({agent.name}); recording Error.")
            return Vote(agent.id, agent.name, VOTE_ERROR, response.strip())
    raise AssertionError("unreachable")  # pragma: no cover


def run_scenario(
    agents: Sequence[Agent],
    *,
    call_fn: CallFn = complete,
    pause_seconds: float = 0.0,
    write: bool = True,
) -> list[Vote]:
    """Collect a Yes/No vote from every agent on Proposition X.

    ``call_fn`` is injectable so tests can supply a fake LLM. ``pause_seconds``
    adds a courtesy delay between live calls for rate limiting.
    """
    votes: list[Vote] = []
    for agent in agents:
        votes.append(_vote_once(agent, call_fn))
        if pause_seconds:
            time.sleep(pause_seconds)
    if write:
        _write_votes(votes)
    return votes


def _write_votes(votes: Sequence[Vote], path: Path | None = None) -> None:
    """Persist votes to results/votes.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    target = path or (RESULTS_DIR / "votes.json")
    target.write_text(json.dumps([asdict(v) for v in votes], indent=2))
