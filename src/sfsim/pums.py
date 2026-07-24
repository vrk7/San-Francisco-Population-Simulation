"""Load and validate the cached SF PUMS table produced by scripts/build_sf_pums.py."""

from pathlib import Path

import pandas as pd

from sfsim.geography import SF_PUMA_CODES

SF_PUMS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "sf_pums.csv"

REQUIRED_COLUMNS = frozenset(
    {"SERIALNO", "PUMA", "PWGTP", "AGEP", "SEX", "SCHL", "OCCP", "SOCP", "HINCP", "TEN", "NP"}
)
STR_COLS = {"SERIALNO": str, "PUMA": str, "OCCP": str, "SOCP": str}
MAX_PLAUSIBLE_AGE = 120


class PumsDataError(RuntimeError):
    """Raised when the cached PUMS table is missing or malformed."""


def load_sf_pums(path: Path = SF_PUMS_FILE) -> pd.DataFrame:
    """Return the validated SF PUMS DataFrame.

    Raises :class:`PumsDataError` if the cache is missing, empty, missing required
    columns, contains non-SF PUMAs, or has out-of-range ages.
    """
    if not path.exists():
        raise PumsDataError(
            f"Cached SF PUMS not found at {path}. Run: uv run python scripts/build_sf_pums.py "
            "(after downloading the PUMS source files — see README)."
        )

    df = pd.read_csv(path, dtype=STR_COLS, low_memory=False)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise PumsDataError(f"SF PUMS is missing required columns: {sorted(missing)}")
    if df.empty:
        raise PumsDataError("SF PUMS table is empty.")

    stray = set(df["PUMA"].unique()) - set(SF_PUMA_CODES)
    if stray:
        raise PumsDataError(f"SF PUMS contains non-SF PUMA codes: {sorted(stray)}")

    if not df["AGEP"].between(0, MAX_PLAUSIBLE_AGE).all():
        raise PumsDataError("SF PUMS contains out-of-range ages.")

    return df


def household_adults(df: pd.DataFrame) -> pd.DataFrame:
    """Voting-age (18+) residents living in households with known income and tenure.

    Excludes group-quarters residents (no HINCP/TEN), who lack a household income
    bracket or owner/renter status.
    """
    adults = df[(df["AGEP"] >= 18) & df["HINCP"].notna() & df["TEN"].notna()]
    return adults.reset_index(drop=True)
