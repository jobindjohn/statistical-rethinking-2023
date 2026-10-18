# Fixes to the original notebooks

This file lists the corrections made during pass 2 of the marimo conversion (see [PLAN.md](PLAN.md)). It covers bugs in the code, and text or equations that didn't match the code. Each lecture also had typos fixed; they aren't listed one by one. The Notes under each lecture record things that look surprising but were left as they are.

Bugs that are known but not fixed yet are listed under "Bugs carried over from the original notebooks" in [PLAN.md](PLAN.md).

## Lecture 02 - The Garden of Forking Data

Code:

- `simulate_noisy_globe_toss` only ever turned W into L. It now misclassifies in both directions, as the measurement-error model assumes.

Text that didn't match the code:

- The workflow step "Test (3) using (2)" now reads "Test (3) using (1)".
- The analytic solution said $R$ is "the number of possible globes"; it is the number of sides of the globe.
- The Beta posterior formula was written as $p = \dots$; it is now $\Pr(p \mid W, L) = \dots$.

Notes:

- Because the noisy toss simulation changed, its printed sample differs from the original notebook's.

## Lecture 03 - Geocentric Models

Code:

- `linear_regression_posterior` flattened `slope_grid` whenever `np.ndim(slope_grid)` was non-zero, which includes 1-D grids. It now checks `np.ndim(slope_grid) > 1`.

Text that didn't match the code:

- The weight equation used $\sim$ for a deterministic relationship: $W_i = \beta H_i + U_i$.
- The height range said 120 to 170 cm; the simulation uses $\text{Uniform}(130, 170)$.
- The posterior was written $p(\alpha, \beta, \sigma)$; it is now $p(\alpha, \beta, \sigma \mid W_i)$.
- The prior for $\beta$ was written $\text{Normal}(0, 1)$; the prior predictive code uses $\text{Uniform}(0, 1)$.

Notes:

- `linear_regression_posterior` loops in Python over all 10,000 grid points and builds a `stats.norm` object at each one, so a grid posterior takes about 4 s. The main text keeps the loop; a bonus section at the end adds a vectorized version (about 40 ms, identical results), which the grid-learning widget uses. The bonus comparison runs the slow loop once, which adds about 4 s to opening the notebook.
- The estimator-recovery widget can miss the true β by a couple of standard deviations at small N. That is sampling noise, and part of the lesson.

## Lecture 04 - Categories & Curves

Code:

- The simulated sample with a 10 kg boost for males was never used. The direct-effect fit on synthetic data used the earlier sample instead, so its plots didn't show the gap the text describes. The fit now uses the boosted sample.
- The Full Luxury Bayes contrasts left out the intercepts $\alpha$ when simulating weights, and simulated male heights with the female $\tau$.
- The height-wise contrast overrode `Hbar` with the mean of the prediction grid. It now keeps the training-data $\bar H$.
- The polynomial predictions standardized heights with `std(ddof=1)`, while the fit used `utils.standardize` (`ddof=0`).
- The weight axis label said "km"; it is "kg".

Text that didn't match the code:

- The summary of the direct effect said Sex has little effect on *height*; it is the effect on *weight*.
- The polynomial equations had wrong indices: $\sum_{d=1}^D \beta_d x^d$, and $\beta_1 x + \beta_2 x^2$.
- The log-weight model used $=$ for a distribution: $\log W_i \sim \text{Normal}(\mu_i, \sigma)$.
- The spline equation ended in $\alpha_S B_K$; it is now $\alpha_K B_K$.
- "Height / Width data" now reads "Height / Weight data".

Notes:

- The direct-effect fit on the boosted sample estimates the male boost at about 7.2 kg against a true 10 kg, with slopes of about 0.6 against 0.5. The gap is sampling noise from this seed.
- Adding the intercepts to the Full Luxury Bayes contrasts barely moves the result (7.09 to 6.99 kg), because α_M ≈ α_F in the Howell data.
- The quadratic-learning widget still uses the original loop posterior, so each step takes about 5 s.

## Lecture 05 - Elemental Confounds

Code:

- `summarize_discrete_confound_simulation` (spelled "counfound" before) passed `labels=['X', 'Y']` to its cross-tabs. That renamed the 0/1 levels to "X" and "Y", so the tables read as X vs Y on both axes. It now names the axes X and Y and keeps the 0/1 levels. For the descendant example, it now says "conditioned on A" instead of Z.
- `simulate_divorce`:
  - The marriage rate was all zeros unless marriage affected divorce. It is now always simulated.
  - The Loc/South shuffle did nothing, because pandas realigned the shuffled values on the index. It now shuffles with `.to_numpy()`.
- `plot_divorce_model_posterior` drew every true value as a vertical line across all rows. It now marks each true value on its own parameter's row.
- The β priors in the divorce models were $\text{Normal}(0, 1)$. They are now $\text{Normal}(0, 0.5)$, as the text says. On the real data, β_M moves from −0.10 to −0.06 and β_A from −0.66 to −0.61.
- The simulation test set `beta_age_marriage=0`, so it had no fork to recover. It is now 1.
- Added seeds to the simulations.
- The counterfactual plots labeled their y axis "causal effect"; they now read "divorce rate".

Text that didn't match the code:

- The Greek Χ used for X in several equations is now a Latin X.
- The equations now match the simulations:
  - the continuous fork: $\mu^*$;
  - the age model: $D = f(A)$;
  - the marriage probability in the age–happiness simulation;
  - the restaurant survival model: $\beta_F F$;
  - the collider descendant: $q^*$.
- The pipe-descendant list of what the simulation demonstrates now matches its output.

Notes:

- In the simulated fork test, the true values sit at the edges of the 89% intervals (β_M 0.79 and β_A −0.76 against 1 and −1). With β_AM = 1, M and A are strongly correlated, and the Normal(0, 0.5) priors pull both slopes toward 0. A note in the notebook says so, and suggests `beta_age_marriage=0.5` for a closer recovery.

## Lecture 06 - Good & Bad Controls

Code:

- The bias-amplification cell set an unused `beta_ZY = 0` and passed `beta_XY` from an earlier simulation. It now sets `beta_XY` locally, and both coefficients are sliders.
- Two DAG plots highlighted edges that don't exist in the graph. They now use ("Z", "X"). A duplicate dictionary key is removed.
- Added seeds to six simulation cells.
- `n_samples_per_simulation` was misspelled.
- The ancestor plots' legends said "correct" and "wrong". They now say "(ignore Z)" and "(stratify by Z)", and the x axis reads "estimated effect of X on Y".

Text that didn't match the code:

- On two backdoor paths the text called Z a collider, but Z is a pipe on those paths. The text now says they can be blocked by Z (or A/B), and that the path is already closed where that applies.
- The do-calculus text now names the confound U, and a missing parenthesis is restored.
- The text on the perspective of A is rewritten for $\beta_A$.

Notes:

- The collider mistake was in the notebook's own notes, not in McElreath's lecture.
- The widgets use a vectorized least-squares helper (`ols_first_slopes`) so they update quickly; the static simulation cells keep their original `smf.ols` code.

## Lecture 12 - Multilevel Models

Code:

- The prior plot drew σ from `stats.expon(1)`, which shifts the *location* to 1 (mean 2) instead of setting the rate. The implied prior on $\alpha_j$ came out too wide as a result. It now uses `stats.expon(scale=1)`.
- The σ posterior overlaid on the cross-validation plot used only chain 0. It now uses all chains.
- The latent Mundlak model named its intercept `alph_bar`; it is now `alpha_bar` (this refits the model once).
- Removed `N_TROLLEY_RESPONSES` and `N_RESPONSE_CATEGORIES`, which were never used.
- Renamed `plot_trolly_response_distribution` to `plot_trolley_response_distribution`.
- The legend of the σ comparison said only "mST" and "mSTP". It now reads "without predators (mST)" and "with predators (mSTP)".
- The group-size histogram has axis labels.

Text that didn't match the code:

- The multilevel and Mundlak equations gave $\tau \sim \text{Exponential}(10)$; the code uses $\text{Exponential}(1)$.
- "Baby's first multi-level model" still wrote $\sigma_?$ after giving σ a prior.
- Latent Mundlak:
  - The X sub-model was missing the $\beta_{GX} \sim \text{Exponential}(1)$ prior, and now uses $\mu_{X_i}$.
  - The Y sub-model was missing the $\beta_{ZY} Z_{G[i]}$ term, and called its intercept both $\alpha_{Y,G[i]}$ and $\alpha_{GY}$. It is now $\alpha_{Y,j}$.
- The random-confound equations now index the observations ($Y_i$, $p_i$, $X_i$).
- "Can't include group-level predictors with Mixed-effect model" now reads "with a fixed-effects model".
- The caption over the group histogram said the groups "aren't randomly sampled". The histogram shows that group sizes are imbalanced, and the caption now says so.
- The text on the fixed-effects results now says what they estimate better: the individual-level effect $\beta_{XY}$.
- The link to Lecture 11 pointed to the old `.ipynb` file.

Notes:

- With the default seed, the pooling simulator shows partial pooling doing slightly worse than no pooling for the 5-tadpole tanks (mean error 0.125 vs 0.113). Other seeds (the "new simulation" button) show the usual pattern. The seed was left as it is rather than picked for the result.
- The pooling simulator computes the partial-pooling posterior exactly on a grid instead of with MCMC. On the frog data it matches the MCMC fit to within 0.002 in survival probability.
- The latent Mundlak refit recovers β_XY ≈ 1.17 (true 1) and β_ZY ≈ −0.40 (true −0.5).
- There is no widget for the random-confound section: each setting would refit five MCMC models, which takes minutes.
