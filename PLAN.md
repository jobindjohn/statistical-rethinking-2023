# Plan: Jupyter → marimo conversion

Goal: convert the 19 lecture notebooks to marimo so they can carry interactive widgets that improve the learning experience.

Work happens on the `marimo` branch. The `jupyter` branch keeps the original `.ipynb` notebooks.

## Decisions

| Topic | Decision |
|---|---|
| Old `.ipynb` files | Delete each one on the `marimo` branch once its `.py` replaces it |
| Rollout | Pilot on Lecture 03, then the rest in batches |
| Phasing | Pass 1 is a faithful port; pass 2 adds widgets |
| Sampling | `mo.persistent_cache` on every fit, plus a `mo.ui.run_button` gate on very slow fits (e.g. the Lecture 19 ODE model) |
| Naming | Rename reused variables by hand: throwaway ones become cell-local (`_fig`, `_ax`), meaningful ones get descriptive names |
| Setup | `init_notebook.py` becomes an importable module; each notebook imports explicitly in its `app.setup` block |
| Publishing | Local only for now (WASM is not an option: PyMC and nutpie don't run in Pyodide) |
| Plotting | Matplotlib only |
| Widgets | `mo.ui` by default; a custom anywidget only with explicit approval each time |
| Code visibility | `hide_code=True` on plotting and layout boilerplate; model and analysis code stays visible |
| Filenames | Keep current names, e.g. `Lecture 03 - Geocentric Models.py` |
| Validation | Automated checks on every lecture; side-by-side review of the pilot, then only flagged differences |
| Commits | One commit per approved batch, made only after approval |
| Environment | At the end, drop jupyterlab/ipykernel, add marimo tasks, update README and CHANGELOG |

## Phase 0: shared foundation

- [ ] Rewrite `init_notebook.py` as a module with a `setup()` function that applies:
  - the `statistical-rethinking-2023.mplstyle` style;
  - ArviZ `stats.ci_kind = "hdi"` and `stats.ci_prob = 0.89`;
  - the warning filters.
- [ ] Turn the watermark into a function the notebooks call in a footer cell. Drop the colorama import printing, because explicit imports make it redundant.
- [ ] Add one sampling helper built on `mo.persistent_cache`, so every lecture caches fits the same way. The cache lives in `__marimo__/`, which is already gitignored.
- [ ] Write a validation script (kept out of the repo) that:
  - runs a converted notebook headlessly;
  - extracts the `az.summary` and model-comparison tables;
  - compares them with the text outputs stored in the matching `.ipynb`, within MCMC noise.

## Phase 1: pilot on Lecture 03 (pass 1)

- [ ] Run `marimo convert`.
- [ ] Rename variables by hand, write the `app.setup` block and add caching.
- [ ] Hide the plotting cells, and make each figure the cell's last expression.
- [ ] Run `marimo check`, a headless run and the numeric comparison.
- [ ] **Gate:** compare the original and converted notebooks side by side, and adjust conventions here before they spread to the other lectures.
- [ ] After approval, delete the `.ipynb` and commit.

## Phase 2: pass 1 for the remaining lectures

Each batch gets the automated checks, a short report of differences, approval, then a commit.

- [ ] 02 + 04
- [ ] 05 – 07
- [ ] 08 – 11
- [ ] 12 – 14
- [ ] 15 – 17
- [ ] 18 – 20

Spots that need extra care:

- **`global` statements** in Lectures 09, 10, 13 and 15. They need restructuring, because marimo's dependency graph can't follow them.
- **graphviz DAGs** in Lectures 07, 10, 11 and 16. Check that they render as cell outputs.
- **Lecture 19:**
  - It has no `%run init_notebook.py`, so its setup differs.
  - Its ODE fit takes about 10 minutes, so it gets a run button.
  - It has an intermittent `init="adapt_diag"` issue.
- **Lecture 18:** it reuses the most names (32), so expect the most renaming.
- **Cache invalidation:** make sure that editing model code or data invalidates the cached fits.

## Phase 3: pass 2, widgets

- [ ] For each lecture, propose a short widget list and get approval before building anything.
- [ ] Build the approved widgets, check and review them in batches as in pass 1, and commit.

Candidate ideas, none of them approved yet:

- **02:** a slider for the toss sequence and grid resolution, with a live posterior.
- **03:** sliders for the prior parameters, with the prior predictive lines updating.
- **07:** a slider for polynomial degree, showing in-sample vs out-of-sample error.
- **08:** a step-by-step Metropolis run and an HMC step-size slider. An HMC path animation may need an anywidget, which needs approval.
- **12:** a slider for σ (the spread between groups) that shows shrinkage.

## Phase 4: cleanup

- [ ] `pixi.toml`:
  - remove `jupyterlab` and `ipykernel`, and the `lab` task;
  - add marimo tasks, e.g. `pixi run lecture 03`;
  - regenerate `pixi.lock`.
- [ ] Update the README with the setup and usage for marimo.
- [ ] Add a CHANGELOG entry for the migration.
- [ ] Clear stale items from `TODO.md`, if wanted.

## Decision gates still ahead

1. Approve this plan before any conversion starts.
2. Review the Lecture 03 pilot and approve the conventions.
3. Approve each pass-1 batch before its commit.
4. Approve the widget list for each lecture, plus any anywidget.
5. Approve each pass-2 batch before its commit.
6. Approve the environment and docs cleanup.
