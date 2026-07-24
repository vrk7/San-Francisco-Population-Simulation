"""End-to-end pipeline: population -> votes -> extras -> benchmarked report.

Ties Phases 2-5 into a single run and writes everything to ``results/``. Stages
are flag-guarded so the base Parts 1-3 run works without the extras.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from sfsim.benchmark import benchmark, write_report
from sfsim.constants import REAL_YES_PCT
from sfsim.memory import attach_memories
from sfsim.population import RESULTS_DIR, build
from sfsim.reflection import run_reflection
from sfsim.scenario import VOTE_NO, VOTE_YES, Vote, run_scenario
from sfsim.social import run_social_round, social_delta


def _yes_pct(votes: Sequence[Vote]) -> float:
    valid = [v for v in votes if v.vote in (VOTE_YES, VOTE_NO)]
    return 100.0 * sum(v.vote == VOTE_YES for v in valid) / len(valid) if valid else 0.0


def run(*, extras: bool = True, compare: bool = True, limit: int = 0, pause: float = 0.0) -> None:
    """Run the simulation end to end and write results/.

    ``compare`` also runs a memory-off baseline; ``extras`` adds the social and
    reflection stages; ``limit`` (>0) caps the agent count for a quick demo.
    """
    agents = build(write=True)
    if limit:
        agents = agents[:limit]
    print(f"Built {len(agents)} agents -> results/agents.md")

    mem_agents = attach_memories(agents)
    extra_lines: list[str] = ["## Extras — experiments", ""]

    base_yes: float | None = None
    if compare:
        print("Round 0: base votes (memory OFF)...")
        base = run_scenario(agents, pause_seconds=pause, write=False)
        base_yes = _yes_pct(base)

    print("Round 1: votes with memory ON...")
    votes = run_scenario(mem_agents, pause_seconds=pause, write=True)
    mem_yes = _yes_pct(votes)

    if compare:
        extra_lines += [
            "### Memory (5.1): grounding votes in lived experience",
            "",
            f"- Memory OFF: {base_yes:.1f}% Yes",
            f"- Memory ON: {mem_yes:.1f}% Yes ({mem_yes - base_yes:+.1f} pp)",
            "",
        ]

    if extras:
        print("Round 2: social re-vote (saw neighbors)...")
        social = run_social_round(mem_agents, votes, write=True)
        delta = social_delta(votes, social)
        extra_lines += [
            "### Social influence (5.2): round-1 vs round-2",
            "",
            f"- Round 1 (private): {delta['round1_yes_pct']}% Yes",
            f"- Round 2 (saw neighbors): {delta['round2_yes_pct']}% Yes",
            f"- Agents who changed their vote: {delta['changed_votes']}/{len(mem_agents)}",
            "",
        ]

        print("Reflection: each agent reflects on its vote...")
        reflected = run_reflection(mem_agents, votes)
        _write_reflections(reflected)
        extra_lines += ["### Reflection (5.3): sample self-observations", ""]
        for agent in reflected[:5]:
            reflection = agent.memories[-1].removeprefix("Reflection: ")
            extra_lines.append(f"- **{agent.name}**: {reflection}")
        extra_lines.append("")

    result = benchmark(votes, mem_agents)
    report_path = write_report(result, votes, extra_sections=extra_lines)

    _print_summary(result, base_yes, mem_yes, report_path)


def _write_reflections(agents: Sequence, path: Path | None = None) -> None:
    """Persist each agent's reflection to results/reflections.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    target = path or (RESULTS_DIR / "reflections.json")
    data = [
        {"agent_id": a.id, "name": a.name, "reflection": a.memories[-1].removeprefix("Reflection: ")}
        for a in agents
    ]
    target.write_text(json.dumps(data, indent=2))


def _print_summary(result, base_yes: float | None, mem_yes: float, report_path: Path) -> None:
    print("\n=== SUMMARY ===")
    if base_yes is not None:
        print(f"  Base (no memory):     {base_yes:.1f}% Yes")
    print(f"  Final (memory-on):    {result.yes_pct:.1f}% Yes")
    print(f"  Real SF Prop F:       {REAL_YES_PCT:.1f}% Yes")
    print(f"  Delta vs reality:     {result.delta:+.1f} pp")
    print(f"  Report written to:    {report_path}")
