"""Aggregate voting results and benchmark against reality.

Computes the Yes/No split, the delta versus the real 60.8% Prop F result, the top
three *themes* per side (deterministic keyword clustering — no extra LLM call so the
report is reproducible), and picks the single most interesting agent. Writes the
required Part 2 + Part 3 output to ``results/report.md``.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sfsim.agent import Agent
from sfsim.constants import REAL_YES_PCT
from sfsim.population import INCOME_BRACKETS, RESULTS_DIR
from sfsim.scenario import VOTE_NO, VOTE_YES, Vote

# Income bracket labels in their natural low->high order, for breakdown tables.
_INCOME_ORDER: tuple[str, ...] = tuple(label for _, label in INCOME_BRACKETS)

# One breakdown row: (group label, yes count, no count, Yes % of valid).
BreakdownRow = tuple[str, int, int, float]

# Theme -> substrings that signal it. Reasons are matched against themes in this
# order; the first theme with any keyword hit wins, else "Other reasons".
YES_THEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Fairness / curb price-gouging",
     ("fair", "price-goug", "gouge", "rip off", "ripped", "take advantage",
      "advantage of", "exorbitant", "greedy", "too big of a")),
    ("Protect restaurant workers & drivers",
     ("worker", "driver", "employee", "livelihood", "wages", "jobs", "margin")),
    ("Support local restaurants / small business",
     ("restaurant", "eateries", "eatery", "small business", "small businesses",
      "local business", "cafe", "cafes")),
    ("Community & neighborhood",
     ("community", "neighborhood", "neighbors", "friends", "family", "character")),
    ("Willing to pay a bit more to help",
     ("pay a bit more", "worth", "willing to pay", "potentially paying")),
)
NO_THEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("My own delivery costs would rise",
     ("cost me", "costing me", "my fees", "fees go up", "fees will go up", "pass",
      "higher fees", "pay more", "my wallet", "hit my wallet", "raise")),
    ("Tight budget / fixed income",
     ("tight budget", "fixed income", "budget", "on a budget", "strain")),
    ("Value convenience / order often",
     ("convenience", "order delivery", "order in", "order takeout", "busy",
      "after a long", "late night")),
    ("Skeptical of regulation",
     ("regulation", "free market", "government", "meddling", "interfere", "rock the boat")),
)
OTHER_THEME = "Other reasons"
TOP_N = 3

# Weights turning an agent's OCEAN into an expected Yes-lean (per CLAUDE.md:
# high Agreeableness -> sympathy -> Yes; high Openness -> receptive -> Yes;
# high Neuroticism -> worried about own prices -> No). Used only to find the
# agent whose actual vote most *contradicts* their profile (the standout).
_TRAIT_MIDPOINT = 5.5
_AGREEABLENESS_W = 1.0
_OPENNESS_W = 0.5
_NEUROTICISM_W = 0.5


@dataclass(frozen=True)
class StandoutPick:
    """The single most interesting agent response and why it stands out."""

    agent_id: int
    agent_name: str
    vote: str
    reason: str
    why: str


@dataclass(frozen=True)
class BenchmarkResult:
    """The full benchmark of a voting run against the real Prop F outcome."""

    total: int
    yes: int
    no: int
    errors: int
    yes_pct: float  # share of *valid* (Yes+No) votes that are Yes
    delta: float  # yes_pct - REAL_YES_PCT (signed, percentage points)
    top_yes_reasons: list[tuple[str, int]]
    top_no_reasons: list[tuple[str, int]]
    standout: StandoutPick | None
    by_neighborhood: list[BreakdownRow]
    by_income: list[BreakdownRow]


def _classify(reason: str, themes: Sequence[tuple[str, tuple[str, ...]]]) -> str:
    """Assign one reason to its first matching theme, else OTHER_THEME."""
    text = reason.lower()
    for theme, keywords in themes:
        if any(kw in text for kw in keywords):
            return theme
    return OTHER_THEME


def top_reasons(
    votes: Sequence[Vote], side: str, themes: Sequence[tuple[str, tuple[str, ...]]]
) -> list[tuple[str, int]]:
    """Top themes (theme, count) among votes on ``side``, most common first.

    Ties break alphabetically by theme name so the output is deterministic.
    """
    counts = Counter(
        _classify(v.reason, themes) for v in votes if v.vote == side
    )
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:TOP_N]


def _expected_yes_lean(agent: Agent) -> float:
    """Signed OCEAN-based lean: positive expects Yes, negative expects No."""
    o = agent.ocean
    return (
        _AGREEABLENESS_W * (o["agreeableness"] - _TRAIT_MIDPOINT)
        + _OPENNESS_W * (o["openness"] - _TRAIT_MIDPOINT)
        - _NEUROTICISM_W * (o["neuroticism"] - _TRAIT_MIDPOINT)
    )


def pick_standout(
    votes: Sequence[Vote], agents_by_id: dict[int, Agent]
) -> StandoutPick | None:
    """Pick the agent whose vote most contradicts their OCEAN-expected lean.

    A Yes vote from a profile that leans No (or vice-versa) is the most
    interesting; among ties the largest contradiction wins. If nothing
    contradicts, fall back to the strongest-leaning valid vote.
    """
    best: tuple[float, StandoutPick] | None = None
    fallback: tuple[float, StandoutPick] | None = None
    for vote in votes:
        if vote.vote not in (VOTE_YES, VOTE_NO):
            continue
        agent = agents_by_id.get(vote.agent_id)
        if agent is None:
            continue
        lean = _expected_yes_lean(agent)
        voted_yes = vote.vote == VOTE_YES
        contradicts = (voted_yes and lean < 0) or (not voted_yes and lean > 0)
        direction = "warm/receptive" if lean > 0 else "cost-focused/skeptical"
        why = (
            f"Their profile leans {direction} (expected-Yes score {lean:+.1f}), yet "
            f"they voted {vote.vote} — the sharpest personality-vote mismatch in the run."
        )
        pick = StandoutPick(agent.id, agent.name, vote.vote, vote.reason, why)
        if contradicts and (best is None or abs(lean) > best[0]):
            best = (abs(lean), pick)
        if fallback is None or abs(lean) > fallback[0]:
            fallback_why = (
                f"Strongest-leaning voter (expected-Yes score {lean:+.1f}); their "
                f"{vote.vote} vote is the clearest read on how profile drives the choice."
            )
            fallback = (abs(lean), StandoutPick(agent.id, agent.name, vote.vote, vote.reason, fallback_why))
    if best is not None:
        return best[1]
    return fallback[1] if fallback else None


def _tally_by(
    votes: Sequence[Vote],
    agents_by_id: dict[int, Agent],
    key: Callable[[Agent], str],
) -> dict[str, tuple[int, int]]:
    """(yes, no) counts grouped by ``key(agent)`` over valid votes."""
    acc: dict[str, list[int]] = {}
    for vote in votes:
        if vote.vote not in (VOTE_YES, VOTE_NO):
            continue
        agent = agents_by_id.get(vote.agent_id)
        if agent is None:
            continue
        pair = acc.setdefault(key(agent), [0, 0])
        pair[0 if vote.vote == VOTE_YES else 1] += 1
    return {group: (y, n) for group, (y, n) in acc.items()}


def _rows(
    tally: dict[str, tuple[int, int]], order: Sequence[str] | None = None
) -> list[BreakdownRow]:
    """Turn grouped tallies into sorted (group, yes, no, yes%) rows.

    With ``order`` given (e.g. income brackets), rows follow it; otherwise they
    sort by Yes% descending, then group name.
    """
    rows: list[BreakdownRow] = []
    for group, (yes, no) in tally.items():
        total = yes + no
        rows.append((group, yes, no, round(100.0 * yes / total, 1) if total else 0.0))
    if order is not None:
        rank = {label: i for i, label in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r[0], len(order)))
    else:
        rows.sort(key=lambda r: (-r[3], r[0]))
    return rows


def breakdown_by_neighborhood(
    votes: Sequence[Vote], agents: Sequence[Agent]
) -> list[BreakdownRow]:
    """Yes/No breakdown per neighborhood, highest Yes% first."""
    by_id = {a.id: a for a in agents}
    return _rows(_tally_by(votes, by_id, lambda a: a.neighborhood))


def breakdown_by_income(
    votes: Sequence[Vote], agents: Sequence[Agent]
) -> list[BreakdownRow]:
    """Yes/No breakdown per income bracket, in low->high order."""
    by_id = {a.id: a for a in agents}
    return _rows(_tally_by(votes, by_id, lambda a: a.income_bracket), order=_INCOME_ORDER)


def benchmark(votes: Sequence[Vote], agents: Sequence[Agent]) -> BenchmarkResult:
    """Compute the full benchmark of a voting run."""
    yes = sum(1 for v in votes if v.vote == VOTE_YES)
    no = sum(1 for v in votes if v.vote == VOTE_NO)
    errors = sum(1 for v in votes if v.vote not in (VOTE_YES, VOTE_NO))
    valid = yes + no
    yes_pct = (100.0 * yes / valid) if valid else 0.0
    agents_by_id = {a.id: a for a in agents}
    return BenchmarkResult(
        total=len(votes),
        yes=yes,
        no=no,
        errors=errors,
        yes_pct=yes_pct,
        delta=yes_pct - REAL_YES_PCT,
        top_yes_reasons=top_reasons(votes, VOTE_YES, YES_THEMES),
        top_no_reasons=top_reasons(votes, VOTE_NO, NO_THEMES),
        standout=pick_standout(votes, agents_by_id),
        by_neighborhood=breakdown_by_neighborhood(votes, agents),
        by_income=breakdown_by_income(votes, agents),
    )


def _render_reasons(reasons: list[tuple[str, int]]) -> list[str]:
    if not reasons:
        return ["- _(none — no votes on this side in this run)_"]
    return [f"{i}. **{theme}** — {count} agent(s)" for i, (theme, count) in enumerate(reasons, 1)]


def _render_breakdown(title: str, label: str, rows: list[BreakdownRow]) -> list[str]:
    """Render one breakdown table (e.g. Yes% by neighborhood or income)."""
    lines = [f"## {title}", "", f"| {label} | Yes | No | Yes % |", "| --- | --- | --- | --- |"]
    if not rows:
        lines.append("| _(no valid votes)_ | | | |")
    for group, yes, no, pct in rows:
        lines.append(f"| {group} | {yes} | {no} | {pct:.1f}% |")
    return lines


def render_report(
    result: BenchmarkResult,
    votes: Sequence[Vote],
    extra_sections: Sequence[str] | None = None,
) -> str:
    """Render the required Part 2 + Part 3 output as Markdown.

    ``extra_sections`` (pre-rendered Markdown lines, e.g. the Phase-5 experiment
    summaries) are inserted just before the full vote list.
    """
    lines = [
        "# Proposition X — Simulation Report",
        "",
        "Simulated ballot: cap third-party delivery-app fees (DoorDash, Uber Eats) at 15%.",
        "",
        "## Overall result",
        "",
        f"- **{result.yes_pct:.1f}% Yes** ({result.yes} Yes / {result.no} No"
        + (f" / {result.errors} unparseable" if result.errors else "")
        + f", {result.total} agents)",
        f"- Real SF Prop F (2021): **{REAL_YES_PCT:.1f}% Yes**",
        f"- **Delta: {result.delta:+.1f} percentage points** vs. reality",
        "",
        "## Top reasons to vote YES",
        "",
        *_render_reasons(result.top_yes_reasons),
        "",
        "## Top reasons to vote NO",
        "",
        *_render_reasons(result.top_no_reasons),
        "",
        "## Most interesting response",
        "",
    ]
    if result.standout:
        s = result.standout
        lines += [
            f"**{s.agent_name}** (agent {s.agent_id}) voted **{s.vote}**:",
            "",
            f"> {s.reason}",
            "",
            f"_Why it stands out:_ {s.why}",
        ]
    else:
        lines.append("_(no valid votes to choose from)_")
    lines += [
        "",
        *_render_breakdown("Yes % by neighborhood", "Neighborhood", result.by_neighborhood),
        "",
        *_render_breakdown("Yes % by income bracket", "Income", result.by_income),
    ]
    if extra_sections:
        lines += ["", *extra_sections]
    lines += ["", "## All votes", ""]
    for v in votes:
        lines.append(f"- **{v.agent_id}. {v.agent_name}** — _{v.vote}_: {v.reason}")
    return "\n".join(lines) + "\n"


def write_report(
    result: BenchmarkResult,
    votes: Sequence[Vote],
    path: Path | None = None,
    extra_sections: Sequence[str] | None = None,
) -> Path:
    """Write the report to results/report.md and return its path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    target = path or (RESULTS_DIR / "report.md")
    target.write_text(render_report(result, votes, extra_sections))
    return target
