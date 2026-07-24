"""Reflection step: after voting, each agent writes a one-sentence reflection.

Following the paper's reflection idea, the agent derives a short self-observation
from how it just voted. That reflection then (a) nudges *one* OCEAN parameter by
+1 (clamped to 1-10) — the trait is chosen from the reflection's own wording, so
the update is grounded in what the agent said — and (b) is appended to the agent's
memory so it can shape future scenarios. Immutable: a new Agent is returned.
"""

from collections.abc import Callable, Sequence

from sfsim.agent import Agent
from sfsim.llm import complete
from sfsim.ocean import SCORE_MAX, SCORE_MIN
from sfsim.scenario import SCENARIO_QUESTION, VOTE_YES, Vote

CallFn = Callable[[str], str]

# Ordered (keywords -> trait): the first group the reflection text hits wins.
# Rationale ties to the OCEAN->vote mapping: self-cost worry -> Neuroticism;
# prosocial/community language -> Agreeableness; openness to a new policy -> Openness;
# weighing long-term economics -> Conscientiousness.
_REFLECTION_TRAIT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("cost", "afford", "budget", "wallet", "my own", "pay more", "expensive", "price"),
     "neuroticism"),
    (("fair", "community", "help", "support", "neighbor", "worker", "local", "together"),
     "agreeableness"),
    (("new", "change", "try", "policy", "idea", "different", "open"),
     "openness"),
    (("long-term", "long term", "future", "responsib", "careful", "weigh", "consequence"),
     "conscientiousness"),
)

NUDGE = 1  # a reflection strengthens the identified trait by one point


def choose_trait(reflection: str, vote: str) -> str:
    """Pick which OCEAN trait the reflection reinforces.

    Chosen from the reflection's wording; falls back to the vote (a Yes leans
    prosocial -> Agreeableness, a No leans self-protective -> Neuroticism).
    """
    text = reflection.lower()
    for keywords, trait in _REFLECTION_TRAIT_RULES:
        if any(kw in text for kw in keywords):
            return trait
    return "agreeableness" if vote == VOTE_YES else "neuroticism"


def _clamp(value: int) -> int:
    return min(SCORE_MAX, max(SCORE_MIN, value))


def apply_reflection(agent: Agent, reflection: str, vote: str) -> Agent:
    """Return a new Agent with the nudged OCEAN score and the reflection stored."""
    trait = choose_trait(reflection, vote)
    new_ocean = {**agent.ocean, trait: _clamp(agent.ocean[trait] + NUDGE)}
    return agent.with_updates(
        ocean=new_ocean,
        memories=(*agent.memories, f"Reflection: {reflection}"),
    )


def build_reflection_prompt(agent: Agent, vote: Vote) -> str:
    """Ask the agent for a one-sentence reflection on its own vote."""
    return (
        f"You are {agent.name}. You were asked: {SCENARIO_QUESTION}\n"
        f"You voted {vote.vote}, saying: \"{vote.reason}\"\n\n"
        f"In ONE sentence, reflect on what this choice says about you or what you'd "
        f"want to remember about yourself next time. Write only that single sentence."
    )


def generate_reflection(agent: Agent, vote: Vote, call_fn: CallFn) -> str:
    """Get a one-sentence reflection from the model (first non-empty line)."""
    response = call_fn(build_reflection_prompt(agent, vote))
    for line in response.splitlines():
        stripped = line.strip().strip('"')
        if stripped:
            return stripped
    return f"I voted {vote.vote} and want to stay true to what mattered to me."


def run_reflection(
    agents: Sequence[Agent],
    votes: Sequence[Vote],
    *,
    call_fn: CallFn = complete,
) -> list[Agent]:
    """Generate a reflection per agent and return agents with nudged OCEAN + memory."""
    votes_by_id = {v.agent_id: v for v in votes}
    updated: list[Agent] = []
    for agent in agents:
        vote = votes_by_id.get(agent.id)
        if vote is None:
            updated.append(agent)
            continue
        reflection = generate_reflection(agent, vote, call_fn)
        updated.append(apply_reflection(agent, reflection, vote.vote))
    return updated
