# Agent Plan — Mock Market Basket Data (5,000 transactions) + Association Rule Mining

## Objective
Generate a realistic mock cafe transaction dataset of **5,000 transactions**, then run **Association Rule Mining** on it. The dataset must contain *designed* associations (ground truth) so that the discovered `support` / `confidence` / `lift` values can be verified against what was intentionally built in. This mirrors the 5-row sample: Coffee↔Butter bonded, Tea↔Jam bonded, Coffee↔Jam repelling.

**Non-negotiable rule:** Do NOT generate purely random baskets. Random independent items destroy all association structure (every `lift` collapses to ≈ 1). Use the archetype method below.

---

## Decisions to Lock Before Coding
The agent MUST confirm these with the user (or accept the stated defaults) before writing any generation code. Do not silently change them later.

| # | Decision | Options | Default |
|---|----------|---------|---------|
| 1 | Item catalog | keep 5 / expand to ~10 | Keep 5: Coffee, Bread, Butter, Tea, Jam |
| 2 | Number of customer archetypes | 2 / 3 / 4 | 3 (Coffee-crowd, Tea-crowd, Mixed) |
| 3 | Target associations | which pairs must be strong | Coffee↔Butter (high lift), Tea↔Jam (high lift), Coffee↔Jam (lift < 1) |
| 4 | Basket size | fixed / random | Random, Poisson (λ ≈ 2.5) |
| 5 | Thresholds | min_support / min_confidence | support=0.02, confidence=0.5, lift>1 |
| 6 | Random seed | fixed? | Fixed (42) for reproducibility |

---

## Data Generation Method
Use the **archetype / latent-class method** — the approach used in real simulation work, and more robust than naive random generation because associations emerge naturally from customer behavior rather than being forced pair-by-pair.

| Method | Principle | Ground truth? | Realism | Verdict |
|--------|-----------|---------------|---------|---------|
| Naive random | each item sampled independently | No (lift ≈ 1) | Low | Do not use |
| Pairwise boost | directly raise P(Butter\|Coffee) | Yes, but hard to control at scale | Medium | Acceptable |
| Archetype / latent-class | pick a customer type first, each type has its own item probabilities | Yes, clean control | High | **Use this** |

### Example archetypes (adjust after Decision #2)
- **Coffee-crowd (~45%):** P(Coffee)=0.95, P(Butter)=0.90, P(Bread)=0.60, P(Jam)=0.05, P(Tea)=0.02
- **Tea-crowd (~35%):** P(Tea)=0.90, P(Jam)=0.85, P(Bread)=0.50, P(Coffee)=0.05, P(Butter)=0.03
- **Mixed (~20%):** all items around 0.30–0.40, prevents unrealistically clean data

Coffee and Jam live in separate archetypes → Coffee↔Jam lift < 1 (repel) emerges automatically.

---

## Phased Work Plan

### Phase 1 — Setup
- Create project structure and `requirements.txt` (pandas, numpy, mlxtend).
- Fix the random seed.
- Declare ALL configuration (catalog, archetypes, thresholds) as top-level constants in one place — no scattered hardcoding.
- **Output:** `config.py`

### Phase 2 — Generate
- For each of 5,000 transactions: pick an archetype by weight, then sample each item by that archetype's probability.
- Guarantee no empty baskets (resample if empty).
- TxID starts at 101 (continue the sample's numbering).
- Save both long format (TxID, Item) and wide one-hot format.
- **Output:** `transactions.csv`

### Phase 3 — Verify Ground Truth (critical)
- Print the support / confidence / lift for every *designed* pair.
- `assert` they match intended direction, e.g.:
  - `Coffee → Butter` lift > 1.3
  - `Tea → Jam` lift > 1.3
  - `Coffee → Jam` lift < 1.0
- If any assertion fails, STOP and report — do not proceed silently.
- **Output:** verification log

### Phase 4 — Analyze
- `TransactionEncoder` → one-hot matrix.
- Run **FP-Growth** (`mlxtend.frequent_patterns.fpgrowth`) as primary.
- Optionally also run `apriori` and assert identical results (cross-check + matches what students do by hand).
- `association_rules(metric="lift")`, sort by lift.
- **Output:** `rules.csv`

### Phase 5 — Report
- Table comparing **designed vs discovered** lift for each target pair.
- Top-10 rules by lift.
- Shelf-layout recommendation grounded in the numbers (co-locate high-lift pairs, separate lift<1 pairs).
- **Output:** summary

---

## Tech Stack
- **Python + pandas** — data handling
- **mlxtend** — `fpgrowth` primary (faster than apriori on large data; Han, Pei & Yin, 2000); `apriori` kept as an optional cross-check
- **numpy** — seeded random sampling

---

## Guardrails (anti-hallucination)
1. Never introduce items outside the declared catalog — ask first.
2. Never hardcode result numbers — every value is computed from the generated data.
3. Print all design parameters before running.
4. Keep the Phase 3 assertions — if ground truth doesn't match, fail loud, not silent.
5. Fix the seed on every random operation for reproducibility.
6. Report every threshold used; never change one silently.

---

## Deliverables
- `config.py` — all parameters
- `generate.py` — data generator
- `analyze.py` — mining + report
- `transactions.csv` — 5,000 rows
- `rules.csv` — discovered rules
- Verification log + final summary
