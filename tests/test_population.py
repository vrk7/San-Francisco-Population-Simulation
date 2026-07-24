"""Phase 2 tests: population sampling, brackets, and agent assembly."""

from sfsim.constants import NUM_AGENTS
from sfsim.occupation import NOT_EMPLOYED_LABEL, occupation_label
from sfsim.population import build, education_label, income_bracket, tenure_label


def test_income_brackets_bucket_correctly():
    assert income_bracket(-5_000) == "<$30k"
    assert income_bracket(29_999) == "<$30k"
    assert income_bracket(30_000) == "$30k-$75k"
    assert income_bracket(120_000) == "$75k-$150k"
    assert income_bracket(150_000) == "$150k+"
    assert income_bracket(2_000_000) == "$150k+"


def test_education_and_tenure_labels():
    assert education_label(24) == "Graduate degree"
    assert education_label(21) == "Bachelor's degree"
    assert education_label(16) == "High school graduate"
    assert education_label(10) == "No high school diploma"
    assert tenure_label(1) == "owner"
    assert tenure_label(2) == "owner"
    assert tenure_label(3) == "renter"
    assert tenure_label(4) == "renter"


def test_occupation_label_handles_missing_and_unknown():
    assert occupation_label("151256") == "Software & Tech Worker"
    assert occupation_label(None) == NOT_EMPLOYED_LABEL
    assert occupation_label(float("nan")) == NOT_EMPLOYED_LABEL
    assert occupation_label("990000") == NOT_EMPLOYED_LABEL


def test_build_returns_thirty_valid_agents():
    agents = build(write=False)
    assert len(agents) == NUM_AGENTS
    for a in agents:
        assert 18 <= a.age <= 120
        assert a.neighborhood
        assert a.sex in ("male", "female")
        assert a.tenure in ("owner", "renter")
        assert set(a.ocean) == {
            "openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism",
        }
        assert all(1 <= v <= 10 for v in a.ocean.values())
        assert a.profile.count(".") >= 2  # two-sentence profile


def test_sampling_is_reproducible():
    first = build(write=False)
    second = build(write=False)
    assert [a.name for a in first] == [a.name for a in second]
    assert [a.ocean for a in first] == [a.ocean for a in second]


def test_population_is_diverse_not_all_engineers():
    agents = build(write=False)
    occupations = {a.occupation for a in agents}
    assert len(occupations) >= 6  # a real mix, not 30 software engineers
    tech = sum(1 for a in agents if a.occupation == "Software & Tech Worker")
    assert tech < NUM_AGENTS // 2
