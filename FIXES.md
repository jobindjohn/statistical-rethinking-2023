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

## Lecture 09 - Modeling Events

Code:

- The cat adoption model had its censoring reversed. It coded adoption with `pd.factorize`, whose order depends on the first row of the data (a transfer), so the 11,351 adopted cats were treated as censored and every other cat as adopted. Adoption is now read directly from `out_event == "Adoption"`, and adopted cats get an infinite censoring bound. The refit gives an average wait of 57.6 days for black cats and 48.4 for other cats, against 54.8 and 50.6 before. These match the exact estimate (total days ÷ number of adoptions).
- The post-stratification multiplied each department's log-odds by its share of applications and pooled the six results as if they were draws, which is neither a weighted average nor on the probability scale. It now takes the male − female contrast in admission probability per department, and draws each posterior sample's department in proportion to its applications. The weighted average is marked (−0.036 ± 0.018).
- The Gamma plot used `loc=10`, which shifted both curves 10 days to the right, and its label called the scale β. It now uses shape 10 with scales 2 and 5.
- The prior-predictive plot of α and β showed ±1 sd, so most sampled points fell outside the frame. It now shows ±3 sd.
- `logistic_regression_posterior` had an unused `likelihood_prior_std` argument, and `if np.ndim(beta_grid):` was missing `> 1`.
- Removed a cell that only built an unused `partial(gaussian_2d_pdf)` (its plot call was commented out).
- `_log_adoption_rate` held the rate, not its log; it is now `_adoption_rate`.
- Axis labels:
  - The logistic curve was labeled $\frac{1}{e^{-(\alpha + \beta X)}}$, missing the $1 +$.
  - The CDF and CCDF were labeled without λ.
  - The prior-predictive curves were labeled invlogit(x) instead of invlogit(α + βx).
  - The post-stratification title was missing "Effect".
- Added seeds to the logit prior plots and the prior-predictive draws.

Text that didn't match the code:

- The dataset has 4526 applications, not 4562.
- The logit was given as $\frac{p}{1-p}$, missing the log.
- The simulation text said gender 0 has a lower overall *rejection* rate; it is the acceptance rate.
- The cat model's equations used $\log \mu_i = \alpha$, while the code puts the Exponential prior on $\mu_i = \alpha$ directly. They now match the code, with a note on McElreath's log link. The censored line read $D_i | A_i = 1$ twice.
- The direct-effect α matrix used commas instead of `&`, so it rendered as one column.
- The direct-effect model's docstring said "total effect".
- γ was called the Exponential prior's mean; it is the rate, and the prior mean is 1/γ = 50 days.
- The text after the post-stratification said the unweighted contrast gives "similar results". It now explains the equal-weight comparison (average −0.029 against −0.036) and why the population matters.
- There are 22,356 cats, not "20k".
- Typos: ocurring, Bernouilli, Logisitic, Bayeisan, continous, unqual, deparment, estiamte, Deptartment, historgram, probablity, kittes, Estimands(s).

Notes:

- The conclusion about black cats holds, and the gap between black and other cats grows from about 4 days to 9.
- The censoring explorer computes the posterior of μ exactly on a grid, which matches the MCMC fit. Its Kaplan-Meier curves follow the censored model and sit above the model that drops the non-adopted cats.
- The logistic updating explorer uses a vectorized grid posterior on its own simulated data; the 8-figure demo keeps the original loop.

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

## Lecture 13 - Multilevel Adventures

Code:

- The district models numbered districts with `pd.factorize`, which skips district 54 (it has no data), so districts 55 to 61 used the parameters labeled 54 to 60. The plotting helper moved the no-data estimate back to position 52 instead of 53, so the plots showed the no-data estimate at district 53 and district 53's estimate at 54. The models now index by district number (`FERTILITY.district.values - 1`), and the reordering step is gone. The author's note about an indexing bug in the error bars is removed.
- The district-urban model recorded `p_C`, an exact copy of `p_C_urban`. It is removed, and `p_C_rural` and `p_C_urban` now have the `district` dimension.
- Renamed `plot_survival_posterior` (left over from the frogs) to `plot_district_posterior`.
- The rural-vs-urban scatter drew lines at 0.5, which can't show whether urban use is higher. It now draws the urban = rural diagonal.
- The plot of the standard deviations also shows $\sqrt{\sigma^2 + \tau^2}$, the spread of the urban log-odds.

Text that didn't match the code:

- The survey has 1934 women, not 1924; district 54 has no data.
- The district without data is 54, not 49 (district 49 has 4 women).
- $\tau$ was called the standard deviation for urban areas. It is the spread of the urban *difference* $\beta_j$; the urban log-odds $\alpha_j + \beta_j$ vary with $\sqrt{\sigma^2 + \tau^2}$.
- "A majority of points are to the left of the vertical line" only showed that rural use is low. The text now says all but one district lie above the diagonal.
- $K$ was called a collider that opens the confound through $D$, but $D$ is in the model. The text now says $K$ is a mediator of $U$, and a collider on $U \rightarrow K \leftarrow F \rightarrow C$ through the unobserved family traits $F$, which is added to the DAG.
- A heading said "posterior predictive" over a plot of posterior means.

Notes:

- The refit after the indexing fix barely changes the estimates (σ 0.52, τ 0.54, correlation of rural and urban means 0.75).
- The shrinkage explorer holds σ fixed, so at σ ≈ 0.52 its estimates are close to, but not the same as, the MCMC fit, which also averages over σ.
- The centered version of the district-urban model (behind a run button) samples badly on purpose: 771 divergences, an ESS of 7 for τ and r̂ 1.56, against none, 396 and 1.01 for the non-centered model.

## Lecture 14 - Correlated Features

Code:

- The same district indexing bug as in Lecture 13: `pd.factorize` skipped district 54, so districts 55 to 61 used the parameters labeled 54 to 60, and the per-district plots were shifted from district 54 on. Both models now index by district number (`FERTILITY.district.values - 1`) and were refit.
- The "95% IPR" bars on the outcome-scale plot used the 25% and 97.5% quantiles. They now use 2.5% and 97.5%.
- `plot_2d_guassian_ci` (now `plot_2d_gaussian_ci`) used `norm.ppf(0.89)` = 1.23 as the radius of the "89%" ellipse, which holds only about 53% of a 2D Gaussian. The radius is now $\sqrt{\chi^2_2(0.89)}$ = 2.10. The axes of the α–β plots are wider to fit the larger ellipses.
- `sample_LKJCorr_prior` drew the standard deviations from `stats.expon(.1)`, which shifts the location instead of setting the scale. It now uses `stats.expon(scale=1)`, the model's Exponential(1) prior.
- Both models recorded `p_C`, an exact copy of `p_C_urban`. It is removed, and the uncorrelated model's `p_C_rural` and `p_C_urban` now have the `district` dimension.
- The DAG of the previous model highlighted U → K and K → C, but the model has no K. It now highlights D → C and U → C.
- Added seeds to the prior-predictive and LKJ draws.

Text that didn't match the code:

- The correlated model's equation gave $p_i = \alpha_{D[i]} + \beta_{D[i]} U_i$; it is now $\text{logit}(p_i)$.
- The MVNormal prior used $=$ for a distribution, and the Cholesky transform used $\sim$ for a calculation.
- The text on the outcome-scale plot read the spread of the district means as posterior uncertainty, and a ⚠️ note said the result was the opposite of the lecture. The text now says what the bars show: the correlated model pulls the urban means together (standard deviation across districts 0.071 against 0.090), while the posterior uncertainty of each district barely changes (0.121 against 0.123). The note is removed.
- The correlation described in "Including feature correlations" is between rural use $\alpha_j$ and the urban difference $\beta_j$.
- "Demonstrate sampling random correlation matrix" now reads "sampling correlated data", which is what the cell does.
- The non-centered funnel has no divergences at all three values of $\sigma_v$ (against 4, 108 and 405 for the centered funnel); the text said "consistently reduced".
- The links to Lecture 13 pointed to the old `.ipynb` file.

Notes:

- The refit barely changes the estimates: the correlation between $\alpha$ and $\beta$ is −0.54 (−0.55 before), with standard deviations 0.55 and 0.74.
- The correlated pooling demo holds $\bar\alpha$, $\bar\beta$, $\sigma$ and $\tau$ at the correlated model's posterior means and computes one district's posterior exactly on a grid, so its estimates are close to, but not the same as, the MCMC fit.
- The funnel explorer's fits use `float` values of $\sigma_v$, so $\sigma_v = 1.0$ is a separate fit from the static cell's integer `1` (76 against 108 centered divergences). The funnel models are sampled without a seed.
- The static LKJ ellipse plot calls `pm.draw` once per ellipse (80 calls), which takes about 30 s when the notebook opens. The LKJ explorer draws all its samples in one call (about 1 s); a comment in `sample_LKJCorr_prior` shows how the static plot could do the same.
