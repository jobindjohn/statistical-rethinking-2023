# Plan: Jupyter → marimo conversion

Goal: convert the 19 lecture notebooks to marimo so they can carry interactive widgets that improve the learning experience.

Work happens on the `marimo` branch. The `jupyter` branch keeps the original `.ipynb` notebooks.

## Decisions

| Topic | Decision |
| --- | --- |
| Old `.ipynb` files | Delete each one on the `marimo` branch once its `.py` replaces it |
| Rollout | Pilot on Lecture 03, then the rest in batches |
| Phasing | Pass 1 is a faithful port; pass 2 adds widgets |
| Sampling | `cached_sample` (in `init_notebook.py`) in place of `pm.sample`, plus a `mo.ui.run_button` gate on very slow fits (e.g. the Lecture 19 ODE model). Originally `mo.persistent_cache`; see Phase 0 |
| Naming | Rename reused variables by hand: throwaway ones become cell-local (`_fig`, `_ax`), meaningful ones get descriptive names |
| Setup | `init_notebook.py` becomes an importable module; each notebook imports explicitly in its `app.setup` block |
| Publishing | Local only for now (WASM is not an option: PyMC and nutpie don't run in Pyodide) |
| Plotting | Matplotlib only |
| Widgets | `mo.ui` by default; a custom anywidget only with explicit approval each time |
| Code visibility | `hide_code=True` on plotting and layout boilerplate; model and analysis code stays visible (confirmed on the pilot) |
| Seeding | Every notebook seeds any simulated data that feeds a fit, inside the simulating cell, so the cached fits are reused across sessions |
| Typos | Pass 1 leaves the markdown text as it is; typos are fixed in pass 2 |
| Filenames | Keep current names, e.g. `Lecture 03 - Geocentric Models.py` |
| Validation | Automated checks on every lecture; side-by-side review of the pilot, then only flagged differences |
| Commits | One commit per approved batch, made only after approval |
| Environment | At the end, drop jupyterlab/ipykernel, add marimo tasks, update README and CHANGELOG |

## Phase 0: shared foundation

- [x] Rewrite `init_notebook.py` as a module with a `setup()` function that applies:
  - the `statistical-rethinking-2023.mplstyle` style;
  - ArviZ `stats.ci_kind = "hdi"` and `stats.ci_prob = 0.89`;
  - the warning filters.
- [x] Turn the watermark into a function the notebooks call in a footer cell. Drop the colorama import printing, because explicit imports make it redundant.
- [x] Add one sampling helper, `cached_sample`, so every lecture caches fits the same way. The cache lives in `__marimo__/cache/samples/`, which is gitignored.
  - `mo.persistent_cache` can't do this job. PyMC models don't pickle, and when only the trace is cached, marimo misses changes to data upstream of the model and returns stale fits.
  - `cached_sample` keys the cache on a hash of the model graph, every data and constant value in it, the coords, the `pm.sample` arguments and the PyMC version.
  - Tested: editing the prior, the upstream data or the sampling arguments each trigger a refit; an unchanged model loads from disk.
  - Simulated data that feeds a fit must be seeded, otherwise every session simulates new data and refits.
- [x] Write a validation script (kept out of the repo) that:
  - runs a converted notebook headlessly;
  - extracts the `az.summary` and model-comparison tables;
  - compares them with the text outputs stored in the matching `.ipynb`, within MCMC noise.

## Phase 1: pilot on Lecture 03 (pass 1)

- [x] Run `marimo convert`.
- [x] Rename variables by hand, write the `app.setup` block and add caching.
- [x] Hide the plotting cells, and make each figure the cell's last expression.
- [x] Run `marimo check`, a headless run and the numeric comparison.
- [x] **Gate:** compare the original and converted notebooks side by side, and adjust conventions here before they spread to the other lectures.
- [x] After approval, delete the `.ipynb` and commit.

## Phase 2: pass 1 for the remaining lectures

Each batch gets the automated checks, a short report of differences, approval, then a commit.

- [x] 02 + 04
- [x] 05 – 07
- [x] 08 – 11
- [x] 12 – 14
- [x] 15 – 17
- [x] 18 – 20

Spots that need extra care:

- ~~**`global` statements** in Lectures 09, 10, 13 and 15.~~ None exist: the matches were the word "global" in docstrings, names and output (e.g. `global_intercept_model`).
- **graphviz DAGs** in Lectures 07, 10, 11 and 16. Check that they render as cell outputs.
- **Lecture 19:**
  - It has no `%run init_notebook.py`, so its setup differs.
  - Its ODE fit takes about 10 minutes, so it gets a run button.
  - It has an intermittent `init="adapt_diag"` issue.
- **Lecture 18:** it reuses the most names (32), so expect the most renaming.
- **Cache invalidation:** make sure that editing model code or data invalidates the cached fits.

## Phase 3: pass 2, widgets

Pass 2 goes one notebook at a time, in an order you choose, with a commit after each approved lecture. Each lecture gets the same steps:

- [ ] Propose a short widget list and get approval before building anything.
- [ ] Fix typos in the markdown text, and the bugs carried over from the original notebooks (listed below).
- [ ] Give every non-markdown cell a descriptive name, so it can be referred to; widget cells use `<topic>_controls` and `<topic>_explorer` (or `_display` / `_plot`).
- [ ] Build the approved widgets, validate (marimo check, headless run, tables), review, and commit.

Progress:

- [x] **02:** 4 widgets (garden explorer, estimator test, toss-the-globe updater, misclassification explorer); `simulate_noisy_globe_toss` now misclassifies both ways.
- [x] **03:** 4 widgets (summed steps to Normal, grid learning with +1/+8 buttons, prior explorer, estimator recovery), and a bonus section on vectorizing the grid posterior (about 4 s down to about 40 ms) with a slider version of the grid demo.
- [x] **04:** 7 widgets (synthetic people, quadratic Bayesian updating, random polynomials, fitted polynomial order, spline basis playground, spline prior, cherry blossom refit with a run button); fixed the unused boosted sample, the missing `alpha` in the Full Luxury Bayes contrasts, the `Hbar` override in the height-wise contrast, polynomial standardization and axis labels.
- [x] **05:** 6 widgets (prior predictive, breaking the fork, intervention size, confound playground, selection threshold, descendant strength); fixed the cross-tab labels, the no-op Loc/South shuffle and all-zero marriage rate in `simulate_divorce`, the true-value markers in the posterior plot, missing seeds, and equations that didn't match the code. The β priors are now Normal(0, 0.5) as the text says, and the simulation test now has a fork (`beta_age_marriage=1`).
- [x] **06:** 4 widgets (backdoor path finder with d-separation, backdoor simulation, good & bad controls lab, bias amplification sliders); fixed the bias-amplification `beta_XY`, the backdoor-path text that called Z a collider on two pipe paths, the A-perspective text, DAG edge highlights on missing edges, missing seeds, and the "correct/wrong" legends on the ancestor plots.
- [x] **09:** 5 widgets (admissions simulator with direct and indirect parts, logistic updating with a vectorized grid posterior, logit prior explorer, post-stratification population explorer, censoring explorer with Kaplan-Meier curves); fixed the reversed censoring in the cat adoption model, the post-stratification on the log-odds scale without weights, the shifted Gamma plot, the prior-predictive window, axis labels and equations that didn't match the code.
- [x] **12:** 3 widgets (σ and shrinkage explorer over the grid-search fits, hierarchical prior explorer, pooling simulator with an exact grid posterior); fixed the shifted `stats.expon(1)` σ prior in the prior plot, the single-chain σ overlay, `alph_bar`, the τ prior and latent Mundlak equations, and unused names.
- [x] **13:** 3 widgets (shrinkage explorer with an exact grid posterior at a fixed σ, district inspector, centered vs non-centered fit with a run button); fixed the district indexing that plotted districts 53 and 54 in each other's place, the τ and collider text, the rural-vs-urban scatter, and the sample size.
- [x] **14:** 4 widgets (MVN prior explorer with the Cholesky transform, LKJ η explorer, correlated pooling demo with an exact grid posterior, devil's funnel explorer with a run button); fixed the district indexing (as in Lecture 13), the 95% IPR quantile, the 89% ellipse radius, the `stats.expon(.1)` standard deviations, the DAG highlights, and the IPR and logit text.

All fixes made so far are listed in [FIXES.md](FIXES.md).

Bugs carried over from the original notebooks (pass 1 keeps them as they are):

- **Lecture 04:** under "Simulate some more people", the sample with the 10 kg boost for males is never used. The direct-effect fit on synthetic data uses the earlier `simulated_people` instead, so its plots don't show the 10 kg gap the text describes. Fit on the new sample instead.
- **Lecture 06:** the bias-amplification cell sets an unused `beta_ZY = 0` and passes `beta_XY` from the earlier simulation (which happens to be 0). Set `beta_XY = 0` locally and drop `beta_ZY`.
- **Lecture 15:** `plot_dyadic_ties` is never called, and its first line overwrote its argument with an undefined `wealth_gifts_inference`. Pass 1 dropped that line so the cell runs without error. Use the function or delete it.
- **Lecture 16:** two cells read the `eta` and `sigma` left over from the second GP-prior loop (pass 1 keeps them as globals of that cell):
    - The GP posterior plot's kernel title shows σ=0.5, but the kernel uses `sigma_kernel = 0.75`. Use `eta_kernel` and `sigma_kernel`.
    - `population_model` uses `eta` (= 1) as the Exponential rate of its priors instead of `ETA = 5.1`, which it defines and never uses. Use `ETA`; this changes the fit.
- **Lecture 16:** `distance_model` mixes poorly on `rho_squared`. The original was borderline (r̂ 1.01, 271 bulk ESS); the cached pass-1 run is worse (r̂ 1.03, 139 bulk ESS). Consider more tuning or draws, or a tighter prior.
- **Lecture 16:** the posterior covariance-matrix plots are titled "Prior Covariance / Intercept-only phylogenetic Model", including the one for the full model. Fix the titles.
- **Lecture 18:** `MeanBodyMassSocialGroupSize`, `PRIMATE_ID`, `PRIMATE` and the first `coords` are used only by the commented-out `gp.Marginal` models. Delete them, or restore the `gp.Marginal` versions.
- **Lecture 19:** the "Prior for $k$" plot evaluates the pdf on `ps` (0 to 1) instead of `ks` (0 to 5), so the curve doesn't match its x-axis. Use `ks`.
- **Lecture 19:** both ODE fits mix poorly. They give r̂ > 1.01 and under 100 ESS per chain, and the simulated-data fit has divergences (106 in the original, 43 in pass 1). Consider more tuning, or a gradient-based ODE solver so the ODE parameters can use NUTS instead of Slice.

Candidate ideas, none of them approved yet:

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
