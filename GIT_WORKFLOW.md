# Business Analysis Lecture: Git Workflow & Repository Architecture

This document establishes the official Git branching and folder structure workflow for the **Business Analysis Lecture** repository (`BusinessAnalysisLecture`). It ensures clean version control, separation of weekly business intelligence cases, alignment with our analytical rigor standards ([`DATA_SCIENCE_PIPELINE.md`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/DATA_SCIENCE_PIPELINE.md)), and seamless collaboration.

---

## 1. Repository Architecture & Branching Strategy

Our repository uses a **Foundational Main + Weekly Feature Branching Model**:

```
main (Foundational Branch)
 ├── .gitignore
 ├── DATA_SCIENCE_PIPELINE.md  (Core analytical & reporting standards)
 └── GIT_WORKFLOW.md           (This workflow guide)
      │
      ├── [Branched Weekly] -> Week01-Topic-Name
      │                         └── w1_TopicFolder/
      ├── [Branched Weekly] -> Week02-Topic-Name
      │                         └── w2_TopicFolder/
      └── [Branched Weekly] -> Week03-Data-Exploratory-Analysis-for-Business-Intelligence
                                ├── w3_Brazillian/
                                ├── w3_Churn/
                                └── w3_datasauRus/
```

### Core Rules:
1. **`main` Branch is the Baseline:** `main` contains foundational repository setup, configuration files (`.gitignore`, `DATA_SCIENCE_PIPELINE.md`, `GIT_WORKFLOW.md`), and historical weekly folders once merged via Pull Request. **Never commit half-finished weekly experiments directly to `main`.**
2. **Weekly Branches:** Every new week of lectures, case studies, or assignments gets its own dedicated Git branch named using the convention `WeekXX-Descriptive-Topic-Name` (e.g., `Week03-Data-Exploratory-Analysis-for-Business-Intelligence`, `Week04-Predictive-Modeling-and-Regression`).
3. **Weekly Folder Structure:** Inside each weekly branch, code, datasets, scripts, and markdown reports must reside inside organized weekly folders prefixed with `wX_` (e.g., `w3_Brazillian/`, `w3_Churn/`).

---

## 2. Step-by-Step Weekly Git Workflow

### Step 1: Start the New Week from a Clean `main`
Always begin by checking out the `main` branch and pulling the latest updates to ensure your local baseline is synced with GitHub:
```bash
git checkout main
git pull origin main
```

### Step 2: Create Your New Weekly Branch
Create and switch (`checkout -b`) to a descriptive branch for the upcoming week:
```bash
# Replace XX with the week number and Topic-Name with the actual topic
git checkout -b Week04-Customer-Segmentation-and-Clustering
```

### Step 3: Organize Your Folder & Build Deliverables
Create your weekly folder following our standard naming (`wX_TopicName/`) and build out your analytical deliverables adhering to [`DATA_SCIENCE_PIPELINE.md`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/DATA_SCIENCE_PIPELINE.md):
- **Scripts:** Reproducible `.py` scripts (`main()` entrypoint, professional matplotlib/seaborn styling).
- **Notebooks:** Narrative `.ipynb` Jupyter notebooks formatted with clear Markdown headings and storytelling blocks.
- **Reports:** Executive summary `.md` reports mapping analytical findings to business impact (`EBITDA`, `Working Capital`, `Churn`).
- **Data & Figures:** Raw data inside `data/` and generated visualization images saved as `.png`.

```bash
mkdir -p w4_CustomerSegmentation/data
```

### Step 4: Check Status, Stage, and Commit Your Work
Throughout your work session, stage and commit logical units of progress with clear commit messages:
```bash
# Check what files are modified or untracked
git status

# Stage your weekly folder
git add w4_CustomerSegmentation/

# Commit with a clear, descriptive message
git commit -m "Add Week 04 customer segmentation K-Means clustering script and executive report"
```

### Step 5: Push the Weekly Branch to GitHub
Push your local weekly branch to remote `origin` on GitHub:
```bash
# For the first push of the branch (sets upstream tracking)
git push -u origin Week04-Customer-Segmentation-and-Clustering

# For subsequent pushes during the same week
git push
```

---

## 3. Creating Pull Requests & Merging to `main`

Once all analytical tasks for the week are completed, verified, and reviewed:
1. Navigate to the repository on GitHub: [BoredOrMore/BusinessAnalysisLecture](https://github.com/BoredOrMore/BusinessAnalysisLecture).
2. Click the green **Compare & pull request** button next to your weekly branch (`WeekXX-Topic-Name`).
3. Title the Pull Request clearly (e.g., *"Week 04: Customer Segmentation & K-Means Clustering Case"*).
4. Review the file diffs to ensure no temporary files (`.DS_Store`, `__pycache__`, or massive uncompressed `.csv` files exceeding GitHub limits) are included.
5. Merge the Pull Request into `main`.
6. Locally, switch back to `main` and pull the freshly merged code:
   ```bash
   git checkout main
   git pull origin main
   ```

---

## 4. Best Practices & Dataset Handling (`.gitignore` & Large Files)

- **GitHub File Size Limits:** Standard Git repositories have a strict file size recommendation of **`50 MB`** and a hard limit of **`100 MB`**.
- **Large Dataset Handling:** If a weekly case involves massive datasets (`> 50 MB`, such as geolocation or high-frequency transaction logs):
  1. Ensure the raw data is documented in the project report.
  2. If the file exceeds `50 MB`, either compress it (`.tar.gz` / `.zip`), use Git Large File Storage (`git-lfs`), or exclude the specific multi-hundred-megabyte raw file in `.gitignore` while keeping the sample summary tables and analytical scripts fully tracked.
- **System & Cache Files:** `.DS_Store`, `.venv/`, and `__pycache__/` directories are permanently excluded via the root `.gitignore`.

---

## 5. Quick Reference Cheat Sheet

| Task | Git Command |
| :--- | :--- |
| **Switch to baseline `main`** | `git checkout main` |
| **Update local `main` from GitHub** | `git pull origin main` |
| **Start a new week's branch** | `git checkout -b WeekXX-Topic-Name` |
| **Check current branch and untracked files** | `git status` |
| **Stage all changes in a weekly folder** | `git add wX_Folder/` |
| **Commit changes locally** | `git commit -m "Description of analytical changes"` |
| **Push new branch to remote GitHub** | `git push -u origin WeekXX-Topic-Name` |
| **Sync branch if `main` was updated** | `git pull origin main` *(while on weekly branch)* |
