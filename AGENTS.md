# Repository Guidelines

## Project Structure & Module Organization

This repository contains weekly Business Data Analytics teaching materials. Keep each case self-contained in a folder such as `w5_CustomerSegmentation/`, with datasets under `data/`, scripts and notebooks at the case root, `.png` figures, and `*_REPORT.md` executive reports. `Slide/` stores lecture PDFs. Read `DATA_SCIENCE_PIPELINE.md` for analytical standards and `GIT_WORKFLOW.md` for branching. Preserve each week's naming style; older cases may use numbered directories such as `01-Cafe-Tx-log/`.

## Branch-Aware Agent Instructions

Weekly content may exist only on feature branches. Before assuming a case or file is absent, inspect every local and remote branch with `git branch --all`; use `git ls-tree` or `git show <branch>:<path>` to review content without switching. Make changes on the user-requested branch only, and never switch branches when uncommitted work could be disturbed.

## Change Discipline

- Keep diffs minimal: touch only the files the change needs. No unrelated refactors, renames, or reformatting; they hide the real change from review. If you spot an unrelated problem (another week's script, a stale report number), report it instead of fixing it.
- Match the neighbouring files of the same week: naming, idioms, comment density, and how work is split across `generate.py`, `analyze.py`, `config.py`, notebooks, and `*_REPORT.md`.
- Write the simplest code that solves the current case. Keep a function next to its only caller; extract a shared module (for example, one week-level module used by several labs) on the third use, not before. Never create generic `helpers`/`utils` files.
- Fail fast on internal invariants: `assert` row counts, keys, and value ranges, and never add fallbacks or defaults for states that should not happen. Validate at the boundaries: raw CSV ingestion, `argparse` inputs, and downloaded data. A fallback the case brief explicitly declares (for example, a cold-start rule) is a business rule, not a silent default: implement it as specified and print when it fires.
- Every function gets a docstring saying what it does or returns, not how.
- Every number that sets behaviour gets an `UPPER_CASE` name at the top of its module. When a case has a `config.py`, put the value there; expose business assumptions (margins, thresholds, conversion rates) through `argparse`.
- Ask before adding a new dependency beyond the stack the weekly cases already use (pandas, numpy, scipy, matplotlib, seaborn, mlxtend).
- Never bypass git hooks (`--no-verify`), and never weaken or delete an assertion or validation check to make a script run; fix the cause.
- One logical change per commit. Push only when the user asks.

## Build, Test, and Development Commands

There is no repository-wide build, dependency manifest, or test suite. Run analysis scripts with the Python environment required by that case:

```bash
python3 path/to/analysis.py
python3 path/to/analysis.py --help
python3 -m py_compile path/to/analysis.py
git status --short
```

These commands regenerate findings, inspect parameters, catch syntax errors, and reveal unintended outputs. For two-stage cases, run generation before analysis: `python3 generate.py && python3 analyze.py`.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 naming: `snake_case` for files, functions, and variables; `UPPER_CASE` for constants such as `RANDOM_SEED = 42`. Scripts need a business-focused module docstring, `main()`, and an `if __name__ == "__main__":` guard. Resolve data paths relative to `__file__`. Expose business assumptions through `argparse`, format currency and percentages clearly, and avoid non-ASCII glyphs in figures and console output.

## Testing & Analytical Validation

No formal coverage target exists. Run the complete workflow and confirm every report number is reproduced by script output. Retain data-invariant assertions, use deterministic seeds, print thresholds, and inspect charts. Never hardcode analytical results. Do not commit caches, virtual environments, secrets, or large raw datasets; `.gitignore` excludes common Python artifacts and `.env` files.

## Commit & Pull Request Guidelines

Name weekly branches `WeekXX-Descriptive-Topic-Name`; keep unfinished work off `main`. Use short, imperative commit subjects, preferably specific: `Add Week 05 clustering analysis and report`. Pull requests should explain the business question, list deliverables, document data sources and assumptions, and report validation commands. Link issues and include screenshots for chart or slide changes. Exclude caches and datasets over 50 MB.
