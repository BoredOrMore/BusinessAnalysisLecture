# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Teaching material for a **Business Analysis / Business Data Analytics** course. It is not an
application — there is no build, no test suite, no package manifest. Each deliverable is a
self-contained weekly case study: a Python analysis script (+ often a narrative notebook), the
dataset it reads, the `.png` figures it writes, and an executive-summary `.md` report that
converts the statistics into business impact.

Two documents on `main` are the authoritative standards and should be read before doing
substantive work:

- `DATA_SCIENCE_PIPELINE.md` — the analytical standard: CRISP-DM phases mapped to a modern
  Python stack (uv, Polars/DuckDB, sklearn Pipelines, MLflow, Pandera, SHAP), plus the
  reproducibility and project-layout rules. Note it aspires to a stack the weekly work does not
  yet use (see "Reality vs. the pipeline doc" below).
- `GIT_WORKFLOW.md` — the branching and folder standard (summarized next).

## Branching model (important — content lives on branches, not `main`)

`main` is the **foundational baseline only**: `.gitignore`, the two standard docs, and `Slide/`
lecture PDFs. All weekly case work lives on its own branch and is merged to `main` by PR when
finished.

- Branch per week: `WeekXX-Descriptive-Topic-Name` (existing:
  `Week03-Data-Exploratory-Analysis-for-Business-Intelligence`, `Week04`, `W5`).
- Start a week from a freshly pulled `main`; never commit half-finished weekly experiments
  directly to `main`.
- Folder convention per `GIT_WORKFLOW.md` is `wX_TopicName/` (Week 03 follows this:
  `w3_Churn/`, `w3_Brazillian/`, `w3_datasauRus/`). Week 04 drifted to a numbered form
  (`01-Cafe-Tx-log/`, `02-Large-scal-retail-mining/`); when adding to an existing week, match
  that week's existing folder style rather than "fixing" it.
- Data files live in `<week-folder>/data/`. Keep tracked datasets under GitHub's 50 MB
  recommendation — compress, use git-lfs, or gitignore the raw file while keeping scripts and
  summary tables tracked.

Because `main` is nearly empty, use `git show <branch>:<path>` or check out the week branch to
read prior case studies for reference.

## Running the analyses

There is no virtualenv, lockfile, or requirements file. Scripts run against the system
interpreter (`python3`, currently 3.14) with pandas/numpy/scipy/matplotlib/seaborn/mlxtend
installed globally.

```bash
# Run from inside the week folder — data paths are resolved relative to the script file
python3 churn_analysis.py
python3 retail_market_basket_mining.py --min_support 0.015 --min_confidence 0.30 --margin 0.38

# Two-stage cases run generate first, then analyze (config.py holds all constants)
python3 generate.py && python3 analyze.py
```

Scripts print their full report to stdout and write `.png` figures next to themselves; the
committed `.md` report is a hand-written narrative of that output.

## Conventions every analysis script follows

- Shebang + module docstring naming the **business context and stakeholder** (CEO/CMO/CFO), the
  dataset, and the analytical tasks — the docstring frames the business question, not the code.
- `def main():` entrypoint under `if __name__ == "__main__":`; `argparse` for
  business-assumption knobs (conversion rate, gross margin, thresholds) so scenarios are
  re-runnable without editing code.
- Data located via `os.path.dirname(os.path.abspath(__file__))` + `data/`, so the script works
  from any cwd.
- Output is a printed executive report: `"=" * 80` section banners, aligned f-string tables,
  currency/percentage formatting (`f"${x:,.2f}"`, `f"{x:.2f}%"`).
- Plotting style is set once at module level — either `plt.style.use('default')` +
  `sns.set_theme(style="whitegrid", palette="deep")`, or `seaborn-v0_8-whitegrid` with explicit
  `rcParams`. No emoji or non-ASCII glyphs in figures or console output.
- Where a project has a `plan.md` (see `Week04:01-Cafe-Tx-log/plan.md`), it locks parameter
  decisions and anti-hallucination guardrails before coding: never hardcode result numbers,
  print every threshold used, fix `RANDOM_SEED = 42`, and keep ground-truth `assert`s that fail
  loudly rather than degrading silently. Follow the plan; don't silently change a locked
  decision.

## Conventions every report follows

Reports (`*_REPORT.md`) are written for an executive, not a reviewer of code: an Executive
Summary with the headline number first, then numbered sections with markdown tables of the
cross-tabs/rules, statistical evidence quoted inline (χ², p-values, lift), and every finding
tied to a financial lever — EBITDA, working capital, annual profit run-rate, retained revenue.
Numbers in the report must come from an actual script run.

## Reality vs. the pipeline doc

The weekly work is pandas + scipy + matplotlib/seaborn + mlxtend, with no uv project, no MLflow,
no Pandera. `DATA_SCIENCE_PIPELINE.md` describes the target stack for larger/modeling projects.
For an ordinary weekly EDA or association-mining case, match the existing weekly scripts; reach
for the pipeline doc's tooling when a case actually involves trained models, tuning, or serving.
