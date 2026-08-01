# Repository Guidelines

## Project Structure & Module Organization

This repository contains weekly Business Data Analytics teaching materials. Keep each case self-contained in a folder such as `w5_CustomerSegmentation/`, with datasets under `data/`, scripts and notebooks at the case root, `.png` figures, and `*_REPORT.md` executive reports. `Slide/` stores lecture PDFs. Read `DATA_SCIENCE_PIPELINE.md` for analytical standards and `GIT_WORKFLOW.md` for branching. Preserve each week's naming style; older cases may use numbered directories such as `01-Cafe-Tx-log/`.

## Branch-Aware Agent Instructions

Weekly content may exist only on feature branches. Before assuming a case or file is absent, inspect every local and remote branch with `git branch --all`; use `git ls-tree` or `git show <branch>:<path>` to review content without switching. Make changes on the user-requested branch only, and never switch branches when uncommitted work could be disturbed.

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
