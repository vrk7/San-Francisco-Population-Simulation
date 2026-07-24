"""Build 30 SF residents by weighted-sampling PUMS microdata.

Sampling rows in proportion to the person weight (``PWGTP``) reproduces the real
*joint* distribution of age, income, occupation, and neighborhood — so the
population automatically reflects SF's mix (tech workers *and* service workers,
renters *and* owners, a range of incomes) instead of 30 software engineers. The
sample is seeded, so the same 30 residents come out every run.
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sfsim.agent import Agent
from sfsim.constants import NUM_AGENTS, RANDOM_SEED
from sfsim.geography import SF_PUMA_INFO
from sfsim.occupation import NOT_EMPLOYED_LABEL, is_customer_facing, occupation_label
from sfsim.ocean import derive_ocean
from sfsim.pums import household_adults, load_sf_pums

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Cosmetic name pools — a plausible label only; demographics carry the real signal.
FIRST_NAMES_FEMALE = (
    "Maria", "Grace", "Amara", "Priya", "Sofia", "Wei", "Elena", "Aisha",
    "Rosa", "Hana", "Nadia", "Linda", "Carmen", "Yuki", "Fatima",
)
FIRST_NAMES_MALE = (
    "James", "David", "Miguel", "Raj", "Chen", "Omar", "Andre", "Tomas",
    "Kevin", "Diego", "Sam", "Luis", "Hiro", "Marcus", "Ahmed",
)
LAST_NAMES = (
    "Nguyen", "Garcia", "Chen", "Johnson", "Patel", "Kim", "Silva", "Okafor",
    "Rossi", "Martinez", "Lee", "Brown", "Tanaka", "Hassan", "Wong",
    "Rivera", "Cohen", "Singh", "Murphy", "Diaz",
)

# Household-income bracket edges (HINCP dollars) and their labels.
INCOME_BRACKETS: tuple[tuple[float, str], ...] = (
    (30_000, "<$30k"),
    (75_000, "$30k-$75k"),
    (150_000, "$75k-$150k"),
    (float("inf"), "$150k+"),
)

# PUMS TEN tenure codes that mean owner-occupied (1/2 = owned; 3/4 = rented / no rent).
TENURE_OWNER_CODES = frozenset({1, 2})


def income_bracket(hincp: float) -> str:
    """Bucket a raw household income (HINCP) into a labeled bracket."""
    for upper, label in INCOME_BRACKETS:
        if hincp < upper:
            return label
    return INCOME_BRACKETS[-1][1]


def education_label(schl: float) -> str:
    """Readable educational attainment from the PUMS SCHL code."""
    schl = int(schl)
    if schl >= 22:
        return "Graduate degree"
    if schl >= 21:
        return "Bachelor's degree"
    if schl >= 20:
        return "Associate degree"
    if schl >= 18:
        return "Some college"
    if schl >= 16:
        return "High school graduate"
    return "No high school diploma"


def tenure_label(ten: float) -> str:
    """Owner vs renter from the PUMS TEN code."""
    return "owner" if int(ten) in TENURE_OWNER_CODES else "renter"


# --- Personality descriptors for the behavioral profile ------------------------

HIGH_TRAIT_PHRASES = {
    "openness": "open to new ideas",
    "conscientiousness": "organized and dependable",
    "extraversion": "outgoing and social",
    "agreeableness": "warm and cooperative",
    "neuroticism": "prone to worry",
}
LOW_TRAIT_PHRASES = {
    "openness": "practical and routine-minded",
    "conscientiousness": "spontaneous and easygoing",
    "extraversion": "reserved and private",
    "agreeableness": "blunt and self-reliant",
    "neuroticism": "calm and even-keeled",
}
TRAIT_MIDPOINT = 5.5


def _dominant_descriptors(ocean: dict[str, int]) -> tuple[str, str]:
    """The two most-defining trait phrases, by distance from the neutral midpoint."""
    ranked = sorted(ocean, key=lambda t: abs(ocean[t] - TRAIT_MIDPOINT), reverse=True)
    phrases = []
    for trait in ranked[:2]:
        table = HIGH_TRAIT_PHRASES if ocean[trait] >= TRAIT_MIDPOINT else LOW_TRAIT_PHRASES
        phrases.append(table[trait])
    return phrases[0], phrases[1]


def _article(word: str) -> str:
    """Choose 'a' or 'an' — spoken-form aware for the ages we render (8x, 11, 18)."""
    return "an" if word[:2] in ("8-", "11", "18") or word[:1] == "8" else "a"


def build_profile(
    *,
    name: str,
    age: int,
    occupation: str,
    neighborhood: str,
    tenure: str,
    income_bracket_label: str,
    ocean: dict[str, int],
) -> str:
    """A 2-sentence behavioral profile grounded in demographics + OCEAN."""
    tenure_phrase = "who owns their home" if tenure == "owner" else "who rents"
    age_word = f"{age}-year-old"
    if occupation == NOT_EMPLOYED_LABEL:
        role = f"{_article(age_word)} {age_word} resident, not currently employed,"
    else:
        role = f"{_article(age_word)} {age_word} {occupation.lower()}"
    first = (
        f"{name} is {role} in {neighborhood}, "
        f"{tenure_phrase} with a household income of {income_bracket_label}."
    )
    desc1, desc2 = _dominant_descriptors(ocean)
    second = f"They come across as {desc1} and {desc2}."
    return f"{first} {second}"


def _build_agent(agent_id: int, row: pd.Series) -> Agent:
    """Assemble one immutable Agent from a sampled PUMS person record."""
    name_rng = random.Random(RANDOM_SEED + agent_id)
    sex_code = int(row["SEX"])
    first_pool = FIRST_NAMES_FEMALE if sex_code == 2 else FIRST_NAMES_MALE
    name = f"{name_rng.choice(first_pool)} {name_rng.choice(LAST_NAMES)}"

    age = int(row["AGEP"])
    neighborhood = name_rng.choice(SF_PUMA_INFO[row["PUMA"]].neighborhoods)
    occupation = occupation_label(row["SOCP"])
    bracket = income_bracket(row["HINCP"])
    education = education_label(row["SCHL"])
    tenure = tenure_label(row["TEN"])

    ocean = derive_ocean(
        age=age,
        sex=sex_code,
        schl=float(row["SCHL"]),
        hincp=float(row["HINCP"]),
        customer_facing=is_customer_facing(row["SOCP"]),
        seed=RANDOM_SEED * 1000 + agent_id,
    )
    profile = build_profile(
        name=name,
        age=age,
        occupation=occupation,
        neighborhood=neighborhood,
        tenure=tenure,
        income_bracket_label=bracket,
        ocean=ocean,
    )
    return Agent(
        id=agent_id,
        name=name,
        age=age,
        neighborhood=neighborhood,
        occupation=occupation,
        income_bracket=bracket,
        sex="female" if sex_code == 2 else "male",
        education=education,
        tenure=tenure,
        ocean=ocean,
        profile=profile,
    )


def build(*, write: bool = True) -> list[Agent]:
    """Sample and return ``NUM_AGENTS`` SF residents; optionally write results/.

    Weighted, seeded sampling makes the run reproducible. When ``write`` is set,
    the population is saved to ``results/agents.json`` and ``results/agents.md``.
    """
    adults = household_adults(load_sf_pums())
    sample = adults.sample(
        n=NUM_AGENTS, weights="PWGTP", random_state=RANDOM_SEED
    ).reset_index(drop=True)

    agents = [_build_agent(i + 1, row) for i, row in sample.iterrows()]

    if write:
        _write_results(agents)
    return agents


def _write_results(agents: list[Agent]) -> None:
    """Persist the population as machine-readable JSON and human-readable Markdown."""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "agents.json").write_text(
        json.dumps([asdict(a) for a in agents], indent=2)
    )
    (RESULTS_DIR / "agents.md").write_text(_render_markdown(agents))


def _render_markdown(agents: list[Agent]) -> str:
    """Render the population as a readable Markdown document."""
    lines = [
        "# SF Simulated Population (30 residents)",
        "",
        "Sampled from 2022 ACS PUMS microdata (person weights) for San Francisco.",
        "OCEAN scores are derived from demographics — see `src/sfsim/ocean.py`.",
        "",
        "| # | Name | Age | Sex | Neighborhood | Occupation | Income | Tenure | O | C | E | A | N |",
        "| - | ---- | --- | --- | ------------ | ---------- | ------ | ------ | - | - | - | - | - |",
    ]
    for a in agents:
        o = a.ocean
        lines.append(
            f"| {a.id} | {a.name} | {a.age} | {a.sex} | {a.neighborhood} | "
            f"{a.occupation} | {a.income_bracket} | {a.tenure} | "
            f"{o['openness']} | {o['conscientiousness']} | {o['extraversion']} | "
            f"{o['agreeableness']} | {o['neuroticism']} |"
        )
    lines += ["", "## Behavioral profiles", ""]
    for a in agents:
        lines.append(f"**{a.id}. {a.name}** — {a.profile}")
        lines.append("")
    return "\n".join(lines)
