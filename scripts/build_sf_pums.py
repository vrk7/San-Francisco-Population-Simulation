"""One-off: build a slim, cached SF person-level PUMS table.

Reads the 2022 ACS 1-year California PUMS person and housing files, filters to the
eight San Francisco PUMAs, joins household income (HINCP) and tenure (TEN) from the
housing record onto each person, and writes ``data/sf_pums.csv``.

Prereqs (downloaded into ``data/`` — see README):
  data/psam_p06.csv   (person records, from csv_pca.zip)
  data/psam_h06.csv   (housing records, from csv_hca.zip)

Run: uv run python scripts/build_sf_pums.py
"""

from pathlib import Path

import pandas as pd

from sfsim.geography import SF_PUMA_CODES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERSON_FILE = DATA_DIR / "psam_p06.csv"
HOUSING_FILE = DATA_DIR / "psam_h06.csv"
OUTPUT_FILE = DATA_DIR / "sf_pums.csv"

PERSON_COLS = ["SERIALNO", "PUMA", "PWGTP", "AGEP", "SEX", "SCHL", "OCCP", "SOCP", "PINCP", "WAGP"]
HOUSING_COLS = ["SERIALNO", "PUMA", "HINCP", "TEN", "NP"]
# Preserve leading zeros / alphanumerics as strings; leave the rest to pandas.
STR_COLS = {"SERIALNO": str, "PUMA": str, "OCCP": str, "SOCP": str}


def _read_sf(path: Path, cols: list[str]) -> pd.DataFrame:
    """Read a large PUMS CSV in chunks, keeping only SF-PUMA rows."""
    sf_codes = set(SF_PUMA_CODES)
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=cols, dtype=STR_COLS, chunksize=200_000, low_memory=False):
        frames.append(chunk[chunk["PUMA"].isin(sf_codes)])
    return pd.concat(frames, ignore_index=True)


def build() -> pd.DataFrame:
    if not PERSON_FILE.exists() or not HOUSING_FILE.exists():
        raise FileNotFoundError(
            f"Missing PUMS source files in {DATA_DIR}. Download csv_pca.zip and "
            "csv_hca.zip (2022 1-Year CA) and extract psam_p06.csv / psam_h06.csv. "
            "See README for the exact URLs."
        )

    persons = _read_sf(PERSON_FILE, PERSON_COLS)
    housing = _read_sf(HOUSING_FILE, HOUSING_COLS)[["SERIALNO", "HINCP", "TEN", "NP"]]

    merged = persons.merge(housing, on="SERIALNO", how="left")

    if merged.empty:
        raise ValueError("No SF PUMS records found after filtering — check PUMA codes.")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(merged):,} SF person records -> {OUTPUT_FILE}")
    print("PUMA record counts:")
    print(merged["PUMA"].value_counts().sort_index().to_string())
    return merged


if __name__ == "__main__":
    build()
