"""Phase 2 tests: OCEAN derivation from demographics."""

from sfsim.ocean import TRAITS, derive_ocean


def _scores(**overrides):
    base = dict(age=40, sex=1, schl=16, hincp=60_000, customer_facing=False, seed=1)
    base.update(overrides)
    return derive_ocean(**base)


def test_all_traits_present_and_in_range():
    scores = _scores()
    assert set(scores) == set(TRAITS)
    for value in scores.values():
        assert 1 <= value <= 10


def test_scores_clamp_at_extremes():
    # A profile pushing every nudge toward its bound must still land in [1, 10].
    young_rich = _scores(age=18, sex=2, schl=24, hincp=2_000_000, customer_facing=True)
    old_poor = _scores(age=90, sex=1, schl=1, hincp=-5_000, customer_facing=False)
    for scores in (young_rich, old_poor):
        for value in scores.values():
            assert 1 <= value <= 10


def test_young_educated_scores_higher_openness_than_old_uneducated():
    young_grad = _scores(age=24, schl=22, hincp=150_000, seed=7)
    old_no_degree = _scores(age=68, schl=12, hincp=25_000, seed=7)
    assert young_grad["openness"] > old_no_degree["openness"]


def test_older_scores_higher_conscientiousness_and_lower_neuroticism():
    younger = _scores(age=25, seed=3)
    older = _scores(age=65, seed=3)
    assert older["conscientiousness"] > younger["conscientiousness"]
    assert older["neuroticism"] < younger["neuroticism"]


def test_customer_facing_raises_extraversion():
    facing = _scores(customer_facing=True, seed=5)
    not_facing = _scores(customer_facing=False, seed=5)
    assert facing["extraversion"] > not_facing["extraversion"]


def test_derivation_is_deterministic():
    assert _scores(seed=42) == _scores(seed=42)
