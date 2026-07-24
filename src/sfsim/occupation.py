"""Map PUMS SOC occupation codes to readable labels.

PUMS stores an occupation as ``SOCP`` — a 6-character SOC code whose first two
digits are the SOC *major group* (e.g. ``15`` = Computer & Mathematical). Mapping
to the major group gives a recognizable, defensible occupation label without a
several-hundred-row code table. ``99`` marks people who are unemployed or have
never worked, and a missing code marks people not in the labor force; both render
as "Not currently employed".
"""

import math

# SOC major group (first two SOCP digits) -> readable occupation label.
SOC_MAJOR_GROUPS: dict[str, str] = {
    "11": "Manager",
    "13": "Business & Finance Specialist",
    "15": "Software & Tech Worker",
    "17": "Engineer / Architect",
    "19": "Scientist",
    "21": "Social & Community Worker",
    "23": "Legal Professional",
    "25": "Educator",
    "27": "Arts & Media Worker",
    "29": "Healthcare Practitioner",
    "31": "Healthcare Support Worker",
    "33": "Protective Service Worker",
    "35": "Food Service Worker",
    "37": "Building & Grounds Worker",
    "39": "Personal Care Worker",
    "41": "Sales Worker",
    "43": "Office & Admin Worker",
    "45": "Farming & Fishing Worker",
    "47": "Construction Worker",
    "49": "Repair & Maintenance Worker",
    "51": "Production Worker",
    "53": "Transportation & Delivery Worker",
    "55": "Military",
}

NOT_EMPLOYED_LABEL = "Not currently employed"

# Major groups whose jobs are customer- or people-facing; used as a mild upward
# nudge on Extraversion in the OCEAN derivation.
CUSTOMER_FACING_GROUPS: frozenset[str] = frozenset(
    {"21", "25", "27", "31", "33", "35", "39", "41"}
)


def _major_group(socp: object) -> str | None:
    """Return the two-digit SOC major group for a raw SOCP value, or None.

    Handles the several shapes SOCP arrives in: a zero-padded string, a float
    read from CSV, or NaN/None for people with no occupation.
    """
    if socp is None:
        return None
    if isinstance(socp, float) and math.isnan(socp):
        return None
    text = str(socp).strip()
    if not text or text.lower() == "nan":
        return None
    return text[:2]


def occupation_label(socp: object) -> str:
    """Return a readable occupation label for a raw PUMS SOCP value."""
    group = _major_group(socp)
    if group is None:
        return NOT_EMPLOYED_LABEL
    return SOC_MAJOR_GROUPS.get(group, NOT_EMPLOYED_LABEL)


def is_customer_facing(socp: object) -> bool:
    """Whether the occupation is people-facing (nudges Extraversion upward)."""
    group = _major_group(socp)
    return group in CUSTOMER_FACING_GROUPS
