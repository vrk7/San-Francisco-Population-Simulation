"""San Francisco PUMA geography and the PUMA -> neighborhood crosswalk.

San Francisco County (FIPS 06075) is fully partitioned into eight 2020-vintage
PUMAs (codes 07507-07514), used by the 2022 ACS 1-year PUMS. A PUMA is coarse
(~90k+ residents spanning several neighborhoods), so mapping a PUMA to a single
recognizable neighborhood is necessarily an approximation. Official PUMA names are
from the Census Bureau (via censusreporter.org); the representative neighborhood
lists are a documented approximation used to give agents recognizable SF locations.
"""

from typing import NamedTuple


class PumaInfo(NamedTuple):
    """One SF PUMA: its official name and representative neighborhoods."""

    official_name: str
    neighborhoods: tuple[str, ...]


# PUMA code (as stored in PUMS, zero-padded 5-char string) -> info.
SF_PUMA_INFO: dict[str, PumaInfo] = {
    "07507": PumaInfo(
        "San Francisco County (South Central)--Bayview & Hunters Point",
        ("Bayview", "Hunters Point", "Potrero Hill"),
    ),
    "07508": PumaInfo(
        "San Francisco County (Northeast)--Chinatown, North Beach & Russian Hill",
        ("Chinatown", "North Beach", "Russian Hill", "Nob Hill"),
    ),
    "07509": PumaInfo(
        "San Francisco County (Southeast)--South of Market & Mission",
        ("SoMa", "Mission", "Tenderloin", "Mission Bay"),
    ),
    "07510": PumaInfo(
        "San Francisco County (Southwest)--Central & Bernal Heights",
        ("Noe Valley", "Castro", "Bernal Heights", "Glen Park"),
    ),
    "07511": PumaInfo(
        "San Francisco County (Southwest)--Outer Sunset & Inner Sunset",
        ("Sunset", "Inner Sunset", "Parkside"),
    ),
    "07512": PumaInfo(
        "San Francisco County (Southwest)--Ingleside & South Central",
        ("Excelsior", "Ingleside", "Outer Mission"),
    ),
    "07513": PumaInfo(
        "San Francisco County (Northwest)--Western Addition & Marina",
        ("Marina", "Pacific Heights", "Western Addition", "Haight"),
    ),
    "07514": PumaInfo(
        "San Francisco County (Northwest)--Richmond, Western Addition & Presidio",
        ("Richmond", "Inner Richmond", "Presidio"),
    ),
}

# The eight SF PUMA codes, as a tuple for filtering.
SF_PUMA_CODES: tuple[str, ...] = tuple(SF_PUMA_INFO.keys())
