# San Francisco Population Simulation

A mini "generative agents" simulation: build **30 AI agents grounded in real US Census
(ACS PUMS) data** for San Francisco, run a real behavioral scenario against them — a
simulated ballot on **Proposition X** (cap third-party delivery-app fees at 15%) — and
**benchmark the result against reality**: the real SF **Prop F (2021) passed with 60.8% Yes**.

Inspired by *Generative Agents: Interactive Simulacra of Human Behavior* (Park et al., 2023).
The twist this project explores: **what changes when your agents are real humans with real
demographic context instead of invented characters?**

**Headline result (local Llama-3.1-8B, one representative run):**

| Stage | Yes % |
| --- | --- |
| Base run (private, no memory) | **100.0%** |
| + Agent memory (5.1) | 90.0% |
| + Social-influence re-vote (5.2) | 90.0% |
| **Real SF Prop F (2021)** | **60.8%** |

The base run votes **Yes for nearly all 30 agents (≈90–100% across runs)** — an artifact of the
LLM's pro-social bias, not the demographics. The realism mechanisms then pull it toward reality,
but the effects are **noisy on this small 8B model and vary run to run**: in the run above,
memory dropped it 10 pp and the social re-vote held steady; in other runs the social re-vote
removed ~10 pp while memory moved less. The direction is consistent (toward reality), the exact
per-mechanism size is not. The *reasoning about that gap* is the point (see the answers below).

The one **consistently large** effect is the **second scenario (5.5):** after voting, each agent
is personally offered a **$5 credit to vote No**. On the 8B model this flips **most agents (24 of
30 in the run above)** from Yes to No, collapsing support from 90% to 23% — a tiny self-interested
nudge easily overrides the model's stated principles, which is itself a finding about how shallow
the "conviction" behind these votes is.

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and a free LLM — either **Groq** (hosted) or
**Ollama** (local, no rate limits).

```bash
uv sync                      # install deps from uv.lock
cp .env.example .env         # then edit .env (see below)
uv run main.py               # run the full pipeline -> results/
```

### Choose your LLM in `.env`

```ini
# Option A — Groq (hosted, fast, free tier is rate-limited to 100k tokens/day)
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here          # free key: https://console.groq.com

# Option B — Ollama (local, no token cap; needs ~5 GB for the 8B model)
LLM_PROVIDER=ollama
# install from https://ollama.com, then: ollama serve && ollama pull llama3.1:8b
# OLLAMA_MODEL=llama3.1:8b
```

### Useful flags

```bash
uv run main.py --no-extras     # Parts 1-3 only (skip social + reflection)
uv run main.py --no-compare    # skip the memory-off baseline
uv run main.py --limit 5       # quick demo on 5 agents
uv run main.py --pause 0.5     # delay between calls (courtesy for hosted providers)
uv run pytest                  # 62 tests
```

Outputs land in `results/`: `agents.{json,md}` (the population), `votes.json`,
`votes_social.json`, `reflections.json`, and `report.md` (the full benchmark).

---

## How it works

```
PUMS microdata  ->  30 agents (demographics + OCEAN + profile + memories)
   (cached CSV)          |
                    vote on Prop X (Groq or Ollama, in character)
                         |
     memory -> social re-vote -> reflection      benchmark vs 60.8%
                         |                              |
                    results/*.json               results/report.md
```

| Module | Role |
| --- | --- |
| `pums.py`, `geography.py` | Load cached SF PUMS; map PUMAs -> neighborhoods |
| `ocean.py` | Derive Big-Five scores from demographic correlates (never random) |
| `population.py` | Weighted-sample 30 agents; build profiles |
| `memory.py` (5.1) | 3 demographic-grounded past delivery experiences per agent |
| `scenario.py` | Balanced Prop X prompt, in-character vote, defensive parsing |
| `social.py` (5.2) | Show same-district neighbors' votes -> round-2 re-vote |
| `reflection.py` (5.3) | Post-vote reflection nudges one OCEAN trait |
| `second_scenario.py` (5.5) | Offer a $5 credit to vote No; measure who is swayed |
| `benchmark.py` | Split, delta vs 60.8%, top reasons, standout, breakdowns (5.4) |
| `pipeline.py` / `main.py` | Orchestrate the whole run |

### Data sources

- **US Census ACS PUMS** (2022, 1-year), person + housing records for San Francisco County
  (state 06, county 075). We download the keyless bulk CSVs and filter to the **eight SF
  PUMAs (07507–07514)** in `scripts/build_sf_pums.py`; the slim `data/sf_pums.csv` (~578 KB,
  8,536 residents) is committed so the repo runs without the ~360 MB download.
- **PUMA → neighborhood crosswalk** (`geography.py`): a PUMA spans several neighborhoods, so
  mapping it to one recognizable name (Mission, SoMa, Richmond, Sunset, Castro, Bayview,
  Marina, …) is a **documented approximation**.
- **Benchmark:** SF **Prop F (2021) = 60.8% Yes** (real delivery-fee-cap measure).

### How OCEAN is derived (defensibility)

Scores are **not random**. Each trait starts at a neutral 5.5 baseline and is nudged by
*documented* population-level correlates, then given a small **seeded** per-agent jitter and
clamped to 1–10 (`ocean.py`, every coefficient a named constant):

- Younger → higher **Openness**; higher education & income → higher Openness.
- Older → higher **Conscientiousness**; older → lower **Neuroticism**.
- Women (population average) → higher **Agreeableness** and **Neuroticism**.
- People-facing occupations → higher **Extraversion**.
- Higher income → slightly lower **Neuroticism** (financial security).

Sources: Big-Five age trends (Roberts et al. 2006), gender differences (Schmitt et al. 2008),
openness–education link (McCrae 1994). These are deliberately modest, directional nudges.

---

## Sample output

Population is demographically diverse — **not 30 software engineers** (only ~2 tech workers;
service, care, delivery, retired, managers; incomes from `<$30k` to `$150k+`; renters & owners):

| # | Name | Age | Neighborhood | Occupation | Income | O C E A N |
| - | ---- | --- | ------------ | ---------- | ------ | --------- |
| 1 | James Martinez | 26 | Potrero Hill | Software & Tech Worker | $150k+ | 8 5 5 6 5 |
| 3 | Chen Hassan | 49 | Mission Bay | Transportation & Delivery Worker | <$30k | 5 6 6 5 6 |
| 5 | Wei Chen | 23 | Ingleside | Personal Care Worker | $150k+ | 8 6 7 6 6 |
| 6 | Rosa Lee | 71 | Pacific Heights | Not currently employed | $150k+ | 4 8 6 6 4 |

Full population, votes, reflections, and the benchmarked report (split, delta, top-3 reasons
per side, standout agent, and Yes% breakdowns by neighborhood & income) are written to
`results/` on every run.

---

## Written answers

### 1. Biggest difference between simulating fictional Smallville characters vs. real SF residents?

Fictional characters are free. You can give them any backstory you want, and no one can say you
got it wrong. Real SF residents are tied to real data, and that changes the whole job. Our 30
agents are drawn from US Census ACS PUMS microdata, which is a table of real anonymized people.
We sample 30 of them in proportion to the person weight (PWGTP), so the real joint mix comes out
on its own: rich and poor, tech workers and service workers, renters and owners, young and old.
Their Big Five (OCEAN) personality scores are also not invented. Each score starts at a neutral
5.5 and is nudged by documented demographic correlates (for example younger means higher
Openness, older means higher Conscientiousness and lower Neuroticism), then given a small seeded
amount of noise so agents are not identical. On top of that, there is a real answer to check
against. The real SF Prop F vote was 60.8% Yes. So the task is no longer believable storytelling.
It is a calibrated prediction with a measurable error (our delta versus 60.8%). When the delta is
large, it is diagnostic: our 100% Yes base run exposed a real bias in the model itself, which a
made up world would have hidden completely.

### 2. Single biggest change you would make with 2 more weeks?

Model who actually votes, not everyone. Right now all 30 agents vote with equal weight. In real
life turnout is skewed: older people, richer people, and homeowners vote much more often. I would
compute a turnout probability per agent from the PUMS demographics and weight each vote by it, so
the reported Yes share reflects the real electorate. That alone shifts the result before any
prompt changes. Second, replace the hand tuned demographic to OCEAN nudges with a mapping fit to
real personality and voting data, and condition each vote on the real ballot context: DoorDash
funded a large No campaign, and voters saw those arguments, but our agents never do. Third,
upgrade memory from three fixed templates to the full memory stream from the paper: store every
experience as a timestamped note and retrieve the most relevant ones by a weighted score of
recency, importance, and embedding relevance to the question. The common thread is simple. Close
the loop between the agents and the real outcome with data, not with clever prompt wording.

### 3. You got X% vs. real 60.8%. What is causing the gap, and the first thing you would fix?

Our base result was about 97% to 100% Yes. The real answer was 60.8% Yes, so the delta is roughly
plus 36 to plus 39 percentage points. The main cause is the model itself. Instruction tuned
models (trained with RLHF) are pushed to be agreeable and to side with the underdog. Prop X is
framed as "protect small local restaurants from big delivery apps," so the model votes Yes almost
no matter which persona it is given. The demographics and OCEAN scores change how each agent
talks (the reasons name the right neighborhood and job), but not how it votes. We confirmed the
effect is framing driven, not persona driven: when we reframed the same question around personal
cost, the model swung all the way to 27% Yes. So it snaps to whichever side the framing licenses
and has no natural middle. Secondary causes: no turnout model, a coarse demographic to OCEAN to
vote mapping that the model mostly ignores in favor of its prior, and no exposure to the real No
campaign. First fix: stop relying on the model to decide on its own, and inject the real decision
context for both sides, the personal cost side and the community side. That is exactly what moved
our number down when we added agent memory and same neighborhood social influence. Both of those
moved the result toward reality, though the exact size was noisy on the small 8B model (each was
worth roughly 0 to 10 points depending on the run, since a weak reasoner does not always let a
"this could cost me" memory beat its default sympathy). The one large and steady effect was the
second scenario: a $5 credit to vote No flipped most agents, which shows how shallow the model's
conviction is. The next highest leverage fix is turnout weighting.
One important note. We did not tune the prompt to hit 60.8% on purpose, because that would leak
the answer and defeat the benchmark.

---

## Reproducibility

Randomness is seeded (`RANDOM_SEED = 42`) and PUMS data is cached, so the population and its
OCEAN scores are identical every run. LLM outputs vary slightly (temperature ~0.5), so exact
vote counts move a little run-to-run; the qualitative story (100% base, memory & social pulling
it down) is stable.
