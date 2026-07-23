# Track A Challenge

**Time:** 4–5 hours of actual work. You decide your deadline — reply with how many days you need and that becomes your commitment.

**Free LLM API options (don't pay from your own pocket):**

- **Groq** (groq.com) — completely free, no credit card needed, runs Llama3-70B very fast. Best option.
- **Google Gemini API** (aistudio.google.com) — free tier, 1M tokens/day, easy signup
- **Ollama** (ollama.com) — runs models fully locally, zero cost, no account, no internet needed
- You do NOT need OpenAI or Anthropic paid access for this challenge.

---

## Read This First (15 minutes)

Read the abstract and Sections 2–4 of:

**Generative Agents: Interactive Simulacra of Human Behavior** — Park et al., 2023

arxiv.org/abs/2304.03442

This is the foundational paper in our field. They simulate 25 fictional characters in a small town called Smallville. Your job: build something similar but using **real people in a real US city**.

The question you'll answer by doing this: *what changes when your agents are real humans with real demographic context instead of fictional characters?*

---

## The Challenge — San Francisco Population Simulation

Sapiaverse's core use case: companies test decisions on a simulated population before deploying them in reality. Your task is to build a working mini-simulation of SF residents and run a real, measurable behavioral scenario against it.

SF is your population: 880K people, 41 neighborhoods, rich Census data, and publicly measurable real-world behavioral benchmarks.

---

## Part 1 — Generate your population (required)

Using **US Census Bureau data** (free, no key needed for basic queries), generate **30 San Francisco residents**.

Each agent must have:

- Name, age, SF neighborhood (Mission, SoMa, Richmond, Sunset, Castro, Tenderloin, Noe Valley, Bayview, Marina, Haight, Excelsior, etc.)
- Occupation and household income bracket (pull realistic distributions from ACS data)
- OCEAN personality scores (1–10 each) derived from demographic correlates — don't assign randomly
- A 2-sentence behavioral profile grounded in their OCEAN scores and life context

**Data sources:**

- Census ACS API: api.census.gov/data/2022/acs/acs5 with state=06, county=075
    - Get api key: https://api.census.gov/data/key_signup.html (use "personal use" and your email address)
- Census QuickFacts SF: census.gov/quickfacts/sanfranciscocitycountycalifornia — use for realistic demographic distributions [Census QuickFacts may be under maintenance — PUMS or any other Census data source works fine]
- PUMS microdata: census.gov/programs-surveys/acs/microdata/access.html — individual-level records to sample from directly
- SF Open Data (optional): data.sfgov.org — neighborhood-level income, housing, demographics

Your population should reflect how SF actually looks: significant income inequality, large tech worker population but also service workers, renters vs. owners, ethnic diversity. Don't make 30 software engineers.

---

## Part 2 — Run a real behavioral scenario (required)

San Francisco is voting on **Proposition X: Cap on third-party delivery app fees at 15%** (modeled on real SF Prop F, 2021).

Ask each agent:

*"San Francisco is voting on a measure that would cap food delivery app fees (DoorDash, Uber Eats) at 15%. As a resident, would you vote Yes or No? Give your single most important reason in one sentence."*

Each agent's response must reflect their specific demographic context and OCEAN profile. A 28-year-old software engineer in SoMa should reason differently from a 54-year-old restaurant owner in the Mission or a 67-year-old retiree in the Sunset.

**Output to show:**

- Overall Yes/No split across your 30 agents
- Top 3 reasons for Yes, top 3 reasons for No
- The single most interesting individual agent response, and one sentence on why it's interesting

---

## Part 3 — Benchmark against reality (required)

The **real SF Prop F (2021)** passed with **60.8% Yes**.

How close did your simulation get? Calculate the delta. Then explain in 3–5 sentences: what do you think caused the gap? What would you change first to close it?

This is the hardest part. Don't skip it even if your number is wildly off. The reasoning matters more than the accuracy.

---

## Part 4 — Go further (optional — this is where you stand out)

Pick one or more:

- Show Yes/No splits broken down by neighborhood or income bracket
- Add agent memory: each agent has 3 past experiences with delivery apps that influence their vote
- Run a second scenario: *"DoorDash offers you a $5 credit if you vote No. Does your answer change?"* Show the behavioral delta
- Add social influence: agents in the same neighborhood see how their neighbors voted before making their final decision. How does collective behavior change the outcome?
- Implement a reflection step (from the Stanford paper): after voting, each agent writes a 1-sentence reflection on why they voted the way they did, and this updates one of their behavioral parameters for the next scenario

---

## Deliverables

1. **GitHub repo** (public or share with prajit@sapiaverse.com)
    - Working code
    - README: setup instructions, data sources used, how to run
    - Sample output: full agent list + voting results pasted in the README or a results file
2. **Three written answers** (1 paragraph each — in the README or email):
    - What's the biggest difference between simulating fictional Smallville characters vs. real SF residents?
    - What's the single biggest change you'd make to your approach with 2 more weeks?
    - You got X% vs. the real 60.8%. What's causing the gap and what's the first thing you'd fix?

---

## What we're evaluating

| Criterion | What great looks like |
| --- | --- |
| It runs | no setup hell |
| Agents feel real | Demographic diversity reflects actual SF, responses reflect personality + context |
| Benchmark reasoning | Thoughtful analysis of why the gap exists — not just stating the number |
| Creative depth | At least one insight we didn't ask for |
| Reflection quality | The 3 answers show how you think about hard problems, not just hard code |

---

## Timeline

Reply with how many days you need. That's your commitment. Delivering early is a positive signal. Missing your own deadline is a red flag.

## Submission

Send GitHub link + written answers to **prajit@sapiaverse.com** — subject: **[Challenge A] Your Name**
