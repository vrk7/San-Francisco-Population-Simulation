"""Phase 1 tests: SF PUMS loader, validation, and geography crosswalk."""

import pandas as pd
import pytest

from sfsim.geography import SF_PUMA_CODES, SF_PUMA_INFO
from sfsim.pums import PumsDataError, household_adults, load_sf_pums


def test_geography_has_eight_sf_pumas():
    assert len(SF_PUMA_CODES) == 8
    assert set(SF_PUMA_CODES) == {"0750" + str(n) for n in range(7, 10)} | {
        "0751" + str(n) for n in range(0, 5)
    }


def test_every_puma_has_official_name_and_neighborhoods():
    for code in SF_PUMA_CODES:
        info = SF_PUMA_INFO[code]
        assert info.official_name.startswith("San Francisco County")
        assert len(info.neighborhoods) >= 1


def test_challenge_key_neighborhoods_are_covered():
    all_neighborhoods = {n for info in SF_PUMA_INFO.values() for n in info.neighborhoods}
    for expected in ("Mission", "SoMa", "Richmond", "Sunset", "Castro", "Bayview", "Marina"):
        assert expected in all_neighborhoods


def test_load_sf_pums_is_valid_and_sf_only():
    df = load_sf_pums()
    assert not df.empty
    assert set(df["PUMA"].unique()) <= set(SF_PUMA_CODES)
    assert df["AGEP"].between(0, 120).all()


def test_household_adults_are_18plus_with_income_and_tenure():
    adults = household_adults(load_sf_pums())
    assert (adults["AGEP"] >= 18).all()
    assert adults["HINCP"].notna().all()
    assert adults["TEN"].notna().all()
    assert len(adults) > 1000  # SF has thousands of household adults in the sample


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(PumsDataError):
        load_sf_pums(tmp_path / "nope.csv")


def test_load_rejects_non_sf_puma(tmp_path):
    bad = pd.DataFrame(
        {
            "SERIALNO": ["x"], "PUMA": ["09999"], "PWGTP": [1], "AGEP": [30], "SEX": [1],
            "SCHL": [21], "OCCP": ["1010"], "SOCP": ["111021"], "HINCP": [100000],
            "TEN": [3], "NP": [2],
        }
    )
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(PumsDataError, match="non-SF PUMA"):
        load_sf_pums(path)
