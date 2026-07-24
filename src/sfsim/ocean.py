"""Derive OCEAN (Big Five) personality scores from demographics — never random.

Each trait starts at a neutral baseline and is nudged by *documented*
population-level correlates of personality, then perturbed by a small,
per-agent *seeded* amount so no two agents are identical. Final scores are
clamped to the challenge's 1-10 range. Every rule below is intentional and is
mirrored in the README for defensibility.

Correlates used (population tendencies, not absolutes):
- Younger -> higher Openness; higher education & income -> higher Openness.
- Older -> higher Conscientiousness; older -> lower Neuroticism.
- Women (population average) -> higher Agreeableness and Neuroticism.
- People-facing occupations -> higher Extraversion.
- Higher household income -> slightly lower Neuroticism (financial security).

Sources: Big Five age trends (Roberts et al. 2006, "maturity principle");
gender differences (Schmitt et al. 2008); openness-education link (McCrae 1994).
These are directional nudges, deliberately modest, not deterministic labels.
"""

import random

TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

# All traits start here (midpoint of the 1-10 scale); correlates nudge from it.
BASELINE = 5.5

# Reference age around which the age-based nudges pivot.
PIVOT_AGE = 35

# --- Per-year age slopes (points of trait change per year away from PIVOT_AGE) ---
OPENNESS_YOUTH_SLOPE = 0.06  # younger -> higher openness
CONSCIENTIOUSNESS_AGE_SLOPE = 0.05  # older -> higher conscientiousness
NEUROTICISM_YOUTH_SLOPE = 0.04  # younger -> higher neuroticism

# Cap any single age term so extreme ages don't dominate the score.
MAX_AGE_EFFECT = 2.5

# --- Education (SCHL) thresholds and nudges ---
SCHL_BACHELORS = 21  # PUMS SCHL: 21 = bachelor's, higher = graduate degrees
SCHL_SOME_COLLEGE = 18
OPENNESS_DEGREE_BONUS = 1.2
OPENNESS_SOME_COLLEGE_BONUS = 0.4
CONSCIENTIOUSNESS_DEGREE_BONUS = 0.3

# --- Household income (HINCP) thresholds and nudges ---
INCOME_HIGH = 150_000
INCOME_MID = 75_000
OPENNESS_HIGH_INCOME_BONUS = 0.6
OPENNESS_MID_INCOME_BONUS = 0.3
NEUROTICISM_HIGH_INCOME_RELIEF = 0.4

# --- Sex nudges (population averages) ---
FEMALE_SEX_CODE = 2  # PUMS SEX: 1 = male, 2 = female
AGREEABLENESS_FEMALE_BONUS = 0.7
NEUROTICISM_FEMALE_BONUS = 0.7

# --- Occupation nudge ---
EXTRAVERSION_CUSTOMER_FACING_BONUS = 1.0
EXTRAVERSION_NON_FACING_PENALTY = 0.2

# --- Per-agent seeded noise so agents aren't identical ---
NOISE_MAGNITUDE = 0.8

SCORE_MIN = 1
SCORE_MAX = 10


def _clamp_round(value: float) -> int:
    """Clamp a raw trait value into [1, 10] and round to an integer score."""
    return int(round(min(SCORE_MAX, max(SCORE_MIN, value))))


def _age_effect(age: int, slope: float, *, older_positive: bool) -> float:
    """Signed, capped age contribution relative to PIVOT_AGE.

    ``older_positive`` True means older ages raise the trait; False means
    younger ages raise it.
    """
    delta = (age - PIVOT_AGE) if older_positive else (PIVOT_AGE - age)
    effect = delta * slope
    return max(-MAX_AGE_EFFECT, min(MAX_AGE_EFFECT, effect))


def derive_ocean(
    *,
    age: int,
    sex: int,
    schl: float,
    hincp: float,
    customer_facing: bool,
    seed: int,
) -> dict[str, int]:
    """Return 1-10 OCEAN scores derived from one resident's demographics.

    Deterministic given the same inputs and ``seed``: the only randomness is a
    small seeded jitter per trait. ``customer_facing`` (whether the resident's
    occupation is people-facing) drives the Extraversion nudge; the caller
    computes it from occupation so this module stays demographics-only.
    """
    is_female = int(sex) == FEMALE_SEX_CODE
    facing = bool(customer_facing)

    openness = (
        BASELINE
        + _age_effect(age, OPENNESS_YOUTH_SLOPE, older_positive=False)
        + _education_openness(schl)
        + _income_openness(hincp)
    )
    conscientiousness = (
        BASELINE
        + _age_effect(age, CONSCIENTIOUSNESS_AGE_SLOPE, older_positive=True)
        + (CONSCIENTIOUSNESS_DEGREE_BONUS if schl >= SCHL_BACHELORS else 0.0)
    )
    extraversion = BASELINE + (
        EXTRAVERSION_CUSTOMER_FACING_BONUS if facing else -EXTRAVERSION_NON_FACING_PENALTY
    )
    agreeableness = BASELINE + (AGREEABLENESS_FEMALE_BONUS if is_female else 0.0)
    neuroticism = (
        BASELINE
        + _age_effect(age, NEUROTICISM_YOUTH_SLOPE, older_positive=False)
        + (NEUROTICISM_FEMALE_BONUS if is_female else 0.0)
        - (NEUROTICISM_HIGH_INCOME_RELIEF if hincp >= INCOME_HIGH else 0.0)
    )

    raw = {
        "openness": openness,
        "conscientiousness": conscientiousness,
        "extraversion": extraversion,
        "agreeableness": agreeableness,
        "neuroticism": neuroticism,
    }

    rng = random.Random(seed)
    return {
        trait: _clamp_round(raw[trait] + rng.uniform(-NOISE_MAGNITUDE, NOISE_MAGNITUDE))
        for trait in TRAITS
    }


def _education_openness(schl: float) -> float:
    """Openness bonus from educational attainment."""
    if schl >= SCHL_BACHELORS:
        return OPENNESS_DEGREE_BONUS
    if schl >= SCHL_SOME_COLLEGE:
        return OPENNESS_SOME_COLLEGE_BONUS
    return 0.0


def _income_openness(hincp: float) -> float:
    """Openness bonus from household income."""
    if hincp >= INCOME_HIGH:
        return OPENNESS_HIGH_INCOME_BONUS
    if hincp >= INCOME_MID:
        return OPENNESS_MID_INCOME_BONUS
    return 0.0
