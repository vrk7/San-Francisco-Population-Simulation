"""Project-wide constants for the SF population simulation."""

# Simulation size
NUM_AGENTS: int = 30

# Scenario: Proposition X — cap third-party delivery app fees
FEE_CAP_PERCENT: int = 15

# Real-world benchmark: SF Prop F (2021) passed with 60.8% Yes
REAL_YES_PCT: float = 60.8

# Reproducibility
RANDOM_SEED: int = 42

# LLM provider: "groq" (hosted, free tier) or "ollama" (local, no rate limits).
# Overridable at runtime via the LLM_PROVIDER env var.
DEFAULT_PROVIDER: str = "groq"

# Groq model used for agent reasoning. llama3-70b-8192 was decommissioned by Groq;
# llama-3.3-70b-versatile is its current 70B-Llama successor.
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# Default local Ollama model (override with the OLLAMA_MODEL env var). 8B runs on
# most laptops; pull it once with `ollama pull llama3.1:8b`.
OLLAMA_MODEL: str = "llama3.1:8b"

# SF geography (state 06 = California, county 075 = San Francisco)
STATE_FIPS: str = "06"
COUNTY_FIPS: str = "075"
