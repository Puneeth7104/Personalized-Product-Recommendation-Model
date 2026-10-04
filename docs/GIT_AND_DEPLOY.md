# From zero to live: Git, GitHub and Streamlit deployment (Windows / PowerShell)

Every command below is meant to be pasted into **PowerShell**, one block at a time.
Replace nothing unless it says `<...>`. Your GitHub username is assumed to be `Puneeth7104`.

## 0. One-time prerequisites

```powershell
git --version        # need Git; if missing:  winget install --id Git.Git -e
python --version     # need 3.11 or 3.12; if missing:  winget install --id Python.Python.3.12 -e
```

Close and reopen PowerShell after installing anything. Tell Git who you are (once per machine):

```powershell
git config --global user.name  "Your Name"
git config --global user.email "your-github-email@example.com"
```

Use the email attached to your GitHub account so commits show up on your profile.

## 1. Put the project on your machine

Unzip `product-recommendation-system.zip` somewhere without spaces in the path, then:

```powershell
cd C:\Projects\product-recommendation-system
dir
```

You should see `streamlit_app.py`, `requirements.txt`, `src`, `tests`, and so on.

## 2. Create the environment and verify everything locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell says scripts are disabled, run this once and retry the activate line:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/run_pipeline.py
streamlit run streamlit_app.py
```

Expected: pytest ends with `31 passed`; the pipeline prints result tables in about 10 seconds;
Streamlit opens http://localhost:8501. Click through all five tabs, switch models and shoppers.
Stop the app with `Ctrl+C`.

> The pipeline rewrites the files in `reports/`. Numbers may differ slightly from the README if
> your library versions differ from mine. If so, update the README table before publishing.

## 3. Create the GitHub repository (no conflicts if you do this exactly)

In the browser: https://github.com/new

- Repository name: `product-recommendation-system`
- Public (needed for the simplest Streamlit deployment)
- **Do not** tick "Add a README", "Add .gitignore" or "Choose a license". The repo must be empty,
  because the project already contains those files. An initialised repo is the usual cause of
  push conflicts.

Click **Create repository**. This is a new repository, separate from your
`Personalized-Product-Recommendation-Model` one.

## 4. First commit and push

```powershell
git init -b main
git status
```

Check the `git status` list: you should **not** see `.venv`, `__pycache__`, `data/` or `models/`.
They are ignored by `.gitignore`. Then:

```powershell
git add .
git commit -m "Add personalised product recommendation system"
git remote add origin https://github.com/Puneeth7104/product-recommendation-system.git
git push -u origin main
```

A browser window asks you to sign in to GitHub the first time (Git Credential Manager). Refresh the
repo page to confirm the files are there. Warnings like `LF will be replaced by CRLF` are harmless.

### Optional: a commit history that tells the story

If you prefer several commits instead of one (nicer on a portfolio), replace the single
`git add . / git commit` pair above with this sequence, then push as above:

```powershell
git add .gitignore requirements.txt requirements-dev.txt pyproject.toml .streamlit
git commit -m "Set up project scaffolding and dependencies"

git add src/recsys/__init__.py src/recsys/data.py src/recsys/features.py src/recsys/interactions.py
git commit -m "Add synthetic data generator, interaction table and time-based split"

git add src/recsys/models.py
git commit -m "Add popularity, matrix factorisation, content-based and hybrid models"

git add src/recsys/metrics.py src/recsys/evaluate.py src/recsys/segments.py src/recsys/pipeline.py scripts reports
git commit -m "Add evaluation, segment analysis and pipeline"

git add tests .github
git commit -m "Add tests and CI workflow"

git add streamlit_app.py
git commit -m "Add Streamlit dashboard"

git add README.md docs
git commit -m "Add README and guides"

git status          # should say: nothing to commit, working tree clean
```

### If `git push` is rejected

Message like `! [rejected] main -> main (fetch first)` means the GitHub repo was not empty
(you ticked a box in step 3). Fix without losing anything:

```powershell
git pull origin main --allow-unrelated-histories --no-rebase
```

If Git reports a conflict in `README.md` (or `.gitignore`), keep **your** version and finish:

```powershell
git checkout --ours README.md
git add README.md
git commit --no-edit
git push -u origin main
```

(For `.gitignore` do the same with `.gitignore`.)

## 5. Confirm CI is green

On GitHub open the repo, **Actions** tab, workflow **CI**. It installs dependencies and runs
`pytest` on Python 3.11 and 3.12 on every push. Wait for the green tick. If it fails, open the run,
read the failing step and fix locally; see step 7 for how to push fixes.

## 6. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub (authorise access when asked).
2. Click **Create app** (top right), then choose **"Yup, I have an app"**.
3. Fill in:
   - **Repository:** `Puneeth7104/product-recommendation-system`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL** (optional): pick a short custom subdomain, e.g. `product-recommender`
4. Click **Advanced settings** and choose **Python 3.12** (or 3.11), then **Save**.
5. Click **Deploy**. The first start installs packages and then trains the models, so allow a few
   minutes. Dependencies come from `requirements.txt` in the repo root automatically.

Notes from Streamlit's docs: if several dependency files exist, only the first by their priority
order is used, and `requirements.txt` outranks `pyproject.toml`, so this repo's pytest-only
`pyproject.toml` is ignored by the deployer. `requirements-dev.txt` is not a recognised name and is
ignored too.

Add the live link to the README, then push the change:

```powershell
# edit README.md: replace "_add your Streamlit URL here after deploying_" with your URL
git add README.md
git commit -m "Add live demo link"
git push
```

The deployed app redeploys automatically whenever you push to `main`.

## 7. Making changes later without conflicts

You are the only contributor, so conflicts are rare. These habits keep it that way:

```powershell
git switch main
git pull                          # always start from the latest main
git switch -c fix/short-name      # work on a branch
# ...edit files...
python -m pytest -q               # never push red tests
git add -A
git commit -m "Describe the change"
git push -u origin fix/short-name
```

Then on GitHub click **Compare & pull request**, wait for CI to go green, **Merge**, and:

```powershell
git switch main
git pull
git branch -d fix/short-name
```

If you edit on GitHub's website *and* locally, run `git pull` before every `git push`.

## 8. Final checklist before sharing

- [ ] `python -m pytest -q` shows 31 passed locally and CI is green on GitHub
- [ ] The README results table matches what `python scripts/run_pipeline.py` prints on your machine
- [ ] The live app opens in a private/incognito window and every tab loads
- [ ] The README has your live URL and no placeholder text left
- [ ] `git status` is clean and no `.venv`, secrets or large data files are in the repo
- [ ] You can explain the points in `docs/INTERVIEW_GUIDE.md` without reading them
