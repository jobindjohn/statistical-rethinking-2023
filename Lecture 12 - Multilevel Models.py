import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import logging

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    from scipy import stats
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 12 - Multilevel Models](https://www.youtube.com/watch?v=iwVqiiXYeC4)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Repeat observations
    At the end of [Lecture 11 on Ordered Categories](<Lecture 11 - Ordered Categories.py>), McElreath alluded to leveraging repeat observations of stories and participants in the Trolley dataset to improve estimation

    Using repeat observations can lead to better estimators
    """)
    return


@app.cell
def trolley_data():
    TROLLEY = utils.load_data("Trolley")
    TROLLEY.head()
    return (TROLLEY,)


@app.cell
def trolley_response_plot_helper(TROLLEY):
    def plot_trolley_response_distribution(variable, n_display=50, error_kwargs={}):

        gb = TROLLEY[[variable, 'response']].groupby(variable)
        plot_data = gb.mean()
        plot_data.reset_index(inplace=True)
        plot_data = plot_data.iloc[:n_display]

        # IQR
        plot_data.loc[:, 'error_lower'] = plot_data['response'] - gb.quantile(.25).reset_index()['response']
        plot_data.loc[:, 'error_upper'] = gb.quantile(.75).reset_index()['response'] - plot_data['response']

        utils.plot_scatter(plot_data.index, plot_data.response, color='C0')
        utils.plot_errorbar(
            plot_data.index,
            plot_data.response,
            plot_data.error_lower.abs(),
            plot_data.error_upper.abs(), colors='C0',
            **error_kwargs
        )
        plt.ylim(1, 7)
        plt.xlabel(f"{variable} index")
        plt.ylabel('response')
        plt.title(f"{variable} response distribution")

    return (plot_trolley_response_distribution,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Variability in Story Responses
    ### Story repeats
    12 stories
    """)
    return


@app.cell
def story_counts(TROLLEY):
    TROLLEY.groupby('story').count()['case']
    return


@app.cell(hide_code=True)
def story_response_plot(plot_trolley_response_distribution):
    plot_trolley_response_distribution('story')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Variability in Participant Responses
    331 individuals
    """)
    return


@app.cell
def participant_counts(TROLLEY):
    TROLLEY.groupby('id').count()['case']
    return


@app.cell(hide_code=True)
def participant_response_plot(plot_trolley_response_distribution):
    plot_trolley_response_distribution('id', error_kwargs={'error_width': 6})
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Ways of modeling variability across observations

    #### Complete Pooling

    $$
    \begin{align}
    R_i &\sim \text{OrderedLogit}(\phi_i, \alpha) \\
    \phi_i &= \beta
    \end{align}
    $$

    - global $\beta$ parameter
    - treat all unique categories (e.g. story and/or participant) as the same
    - **underfits data** because model isn't flexible enough

    #### No Pooling

    $$
    \begin{align}
    R_i &\sim \text{OrderedLogit}(\phi_i, \alpha) \\
    \phi_i &= \beta_{S[i]}
    \end{align}
    $$

    - treat all unique categories as independent; each category gets its own $\beta_{S}$
    - model has "anterograde amnesia"
      - doesn't share information across consecutive observations
      - inefficient
    - **overfits data** because model too flexible, and fits to individual noise

    #### Partial Pooling (Multi-level Models)

    $$
    \begin{align}
    R_i &\sim \text{OrderedLogit}(\phi_i, \alpha) \\
    \phi_i &= \beta_{S[i]} \\
    \beta_i &\sim \text{Prior}_\beta(\eta) \\
    \eta &\sim \text{PopulationPrior}(\theta)
    \end{align}
    $$

    - parameters are drawn from a global distribution that is shared across the population
    - allows flexibility without overfitting
    - shares information across observations
      - has "memory"
      - more efficient
        - learns faster
      - get adaptive regularization for free
        - **resists overfitting**
      - improves estimates for imbalanced sampling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Case Study: Reed Frog Survival
    - 48 groups; "tanks"
    - Treatment: density, size, and predation level
    - Outcome: survival rate
    """)
    return


@app.cell
def frogs_data():
    FROGS = utils.load_data("reedfrogs")
    N_TANKS = len(FROGS)
    FROGS.head()
    return FROGS, N_TANKS


@app.cell(hide_code=True)
def frogs_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("T", "S"),
            ("D", "S"),
            ("G", "S"),
            ("P", "S")
        ],
        node_props={
            "T": {"label": "tank, T"},
            "S": {"label": "survival, S", "color": "red"},
            "D": {"label": "density, D"},
            "G": {"label": "size, G"},
            "P": {"label": "predators, P"},
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - $T$: tank ID
    - $D$: tank density - the number of tadpoles in each tank - counts
    - $G$: tank size - categorical (large/small)
    - $P$: presence/absence of predators - categorical
    - $S$: survival, the number of tadpoles that survived - counts
    - propsurv: survival rate $\frac{S}{D}$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Plot average survival rates for all tanks
    """)
    return


@app.cell(hide_code=True)
def tank_survival_plot(FROGS, N_TANKS):
    plt.scatter(FROGS.index, FROGS.propsurv, color='k')

    # Plot average survival rate across tanks
    _global_mean = FROGS.propsurv.mean()
    plt.axhline(_global_mean, color='k', linestyle='--', label=f"average tank survival: {_global_mean:1.2}")

    # Highlight different densities
    _density_change = FROGS[FROGS.density.diff() > 0].index.tolist()
    _density_change.append(N_TANKS-1)
    _start = 0
    _density_labels = ['low', 'medium', 'high']
    for _ii in range(3):
        _end = _density_change[_ii]
        plt.axvspan(_start, _end, alpha=.1, label=f"{_density_labels[_ii]} density", color=f'C{_ii+2}')
        _start = _end

    plt.xlabel("tank")
    plt.xlim([-5, N_TANKS])
    plt.ylabel("proportion survival")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Let's build a (multi-level) model

    $$
    \begin{align*}
    S_i &\sim \text{Binomial}(D_i, p_i) \\
    \text{logit}(p_i) &= \alpha_{T[i]} \\
    \alpha_j &\sim \text{Normal}(\bar \alpha, \sigma_?) \\
    \bar \alpha &\sim \text{Normal}(0, 1.5) \\
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What about the prior variance $\sigma$?

    For now, let's try setting $\sigma$ manually, using cross-validation to see the effect on the multi-level model (we'll estimate it later)
    """)
    return


@app.cell
def frogs_model_data(FROGS):
    # Set up data / coords

    N_SURVIVED = FROGS['surv'].astype(float)
    N_TRIALS = FROGS['density'].values.astype(float)
    PREDATOR_ID, PREDATORS = pd.factorize(FROGS['pred'], sort=True)
    return N_SURVIVED, N_TRIALS, PREDATOR_ID


@app.cell
def sigma_grid_search(N_SURVIVED, N_TANKS, N_TRIALS):
    # Run grid search over sigmas

    # Turn off logging for simulation
    _pymc_logger = logging.getLogger("pymc")
    _pymc_logger.setLevel(logging.CRITICAL)

    n_sigmas = 20
    sigmas = np.linspace(0.1, 5, n_sigmas)
    inferences = []
    print("Running grid search...")
    for _sigma in sigmas:
        print("\r", f"sigma={_sigma:1.2}")
        with pm.Model() as _m:
            # Prior
            _alpha_bar = pm.Normal('alpha_bar', 0, 1.5)
            _alpha = pm.Normal('alpha', _alpha_bar, _sigma, shape=N_TANKS)

            # Likelihood -- record p_survived for visualization
            _p_survived = pm.Deterministic('p_survived', pm.math.invlogit(_alpha))
            _S = pm.Binomial('survived', n=N_TRIALS, p=_p_survived, observed=N_SURVIVED)
            _inference = cached_sample(progressbar=False)

            # log-likelihood for LOOCV scores and model comparison
            _inference = pm.compute_log_likelihood(_inference, progressbar=False)

        inferences.append(_inference)

    # Turn logging back on
    _pymc_logger.setLevel(logging.INFO)
    return inferences, sigmas


@app.cell
def survival_posterior_plot_helper(FROGS):
    def plot_survival_posterior(inference, sigma=None, color='C0', hdi_prob=.89):

        plt.figure()

        # Plot observations
        plt.scatter(x=FROGS.index, y=FROGS.propsurv, color='k', s=50, zorder=3)

        # Posterior per-tank mean survival probability
        posterior_mean = inference.posterior.mean(dim=('chain', 'draw'))['p_survived']

        utils.plot_scatter(FROGS.index, posterior_mean, color=color, zorder=50, alpha=1)

        # Posterior HDI error bars
        hdis = az.hdi(
            inference.posterior,
            var_names='p_survived',
            prob=hdi_prob
        )['p_survived']


        error_upper = hdis.sel(ci_bound='upper').values - posterior_mean
        error_lower = posterior_mean - hdis.sel(ci_bound='lower').values
        xs = np.arange(len(posterior_mean))
        utils.plot_errorbar(
            xs=xs,
            ys=posterior_mean,
            error_lower=error_lower,
            error_upper=error_upper,
            colors=color,
            error_width=8
        )

        # Add mean indicators
        empirical_mean = FROGS.propsurv.mean()
        plt.axhline(y=empirical_mean, c='k', linestyle="--", label='Global Mean')

        # Posterior mean
        global_posterior_mean = utils.invlogit(inference.posterior.mean(dim=('chain', 'draw'))['alpha_bar'])
        plt.axhline(global_posterior_mean, linestyle="--", label='Posterior Mean')

        # Add tank density indicators
        plt.axvline(15.5, color='k', alpha=.25)
        plt.axvline(31.5, color='k', alpha=.25)

        if sigma is not None:
            plt.title(f'$\\sigma={sigma:1.2}$')

        plt.ylim([0, 1.05])
        plt.xlabel("tank #")
        plt.ylabel("proportion survival")

        plt.legend()

    return (plot_survival_posterior,)


@app.cell(hide_code=True)
def sigma_extremes_plot(inferences, plot_survival_posterior, sigmas):
    # Show extremes of parameter grid
    _figs = []
    for _idx in [0, -1]:
        plot_survival_posterior(inferences[_idx], sigmas[_idx])
        _figs.append(plt.gcf())
    mo.vstack(_figs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Identifying the optimal $\sigma$ via Cross-validation

    - $\sigma=0.1$ underfits the data, the model isn't flexible enough to capture the variability of the different tanks
    - $\sigma=5.0$ overfits the data, the model is too flexible and the posterior centers around each datapoint
    - $\sigma_{optimal}=?$ We can compare the LOOCV scores for models fit with each $\sigma$ value in the parameter grid, and identify the model with the highest score (ELPD).
    """)
    return


@app.cell
def sigma_model_comparison(inferences, sigmas):
    def plot_model_comparisons(sigmas, inferences, multilevel_posterior=None):
        plt.subplots(figsize=(10, 4))
        comparisons = az.compare(dict(zip(sigmas, inferences)), round_to='none')
        comparisons.sort_index(inplace=True)

        utils.plot_scatter(
            xs=sigmas,
            ys=comparisons['elpd'],
            color='C0',
            label='Cross-validation score'
        )
        utils.plot_errorbar(
            xs=sigmas,
            ys=comparisons['elpd'],
            error_lower=comparisons['se'],
            error_upper=comparisons['se'],
            colors='C0'
        )

        # Highlight the optimal sigma
        best = comparisons[comparisons['rank'] == 0]
        plt.scatter(
            x=best.index,
            y=best['elpd'],
            s=300, marker='*',
            color='C1',
            label="$\\sigma_{LOOCV}^*$:" + f" {best.index[0]:1.2}",
            zorder=20
        )

        # If provided, overlay the multilevel posterior (or, a scaled/offset version of it)
        if multilevel_posterior is not None:
            # KDE over the draws of all chains
            kde = stats.gaussian_kde(multilevel_posterior['sigma'].values.ravel())
            sigma_grid = np.linspace(0.1, 5, 100)
            sigma_posterior = kde(sigma_grid)
            max_sigma_posterior = sigma_posterior.max()
            elpd_range = comparisons['elpd'].max() - comparisons['elpd'].min()
            multiplier = 0.7 * elpd_range / max_sigma_posterior
            offset = comparisons['elpd'].min()
            plot_posterior = sigma_posterior * multiplier + offset

            plt.plot(
                sigma_grid,
                plot_posterior,
                color='k', linewidth=3,
                label='Multi-level Posterior'
            )

        plt.xlim([0, 4])
        plt.xlabel("$\\sigma$")
        plt.ylabel("LOOCV score (ELPD)")
        plt.legend()

        # Output optimal parameters
        optimal_sigma = best.index.values[0]
        optimal_sigma_idx = sigmas.tolist().index(optimal_sigma)

        return optimal_sigma_idx, optimal_sigma

    optimal_sigma_idx, optimal_sigma = plot_model_comparisons(sigmas, inferences)
    plt.gcf()
    return optimal_sigma_idx, plot_model_comparisons


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model with the optimal $\sigma$ identified through cross-validation
    """)
    return


@app.cell(hide_code=True)
def optimal_sigma_plot(
    inferences,
    optimal_sigma_idx,
    plot_survival_posterior,
    sigmas,
):
    plot_survival_posterior(inferences[optimal_sigma_idx], sigmas[optimal_sigma_idx])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The optimal model

    - is regularized, trading off bias and variance
    - **demonstrates shrinkage**:
      - posterior means do not hover over datapoints, but are "pulled" toward the global mean.
      - the amount of "pull" is stronger for datapoints further away from the mean
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore: how $\sigma$ controls shrinkage

    Drag the slider through the 20 values of $\sigma$ from the grid search. Gray lines connect each tank's observed survival rate to its posterior mean: the longer the line, the more that tank is pulled toward the population mean.

    - Small $\sigma$ pools all tanks together, so the model underfits.
    - Large $\sigma$ leaves each tank on its own, so the model overfits.
    - The right panel shows where the chosen $\sigma$ sits on the cross-validation curve.
    """)
    return


@app.cell(hide_code=True)
def sigma_grid_summary(inferences, sigmas):
    # Per-tank posterior summaries and LOOCV scores for every sigma in the grid search
    sigma_elpd = az.compare(dict(zip(sigmas, inferences)), round_to='none').sort_index()
    sigma_grid_posteriors = []
    for _inference in inferences:
        _hdi = az.hdi(_inference.posterior, var_names='p_survived', prob=.89)['p_survived']
        sigma_grid_posteriors.append({
            'mean': _inference.posterior['p_survived'].mean(dim=('chain', 'draw')).values,
            'lower': _hdi.sel(ci_bound='lower').values,
            'upper': _hdi.sel(ci_bound='upper').values,
            'alpha_bar': float(utils.invlogit(_inference.posterior['alpha_bar'].mean())),
        })
    return sigma_elpd, sigma_grid_posteriors


@app.cell(hide_code=True)
def sigma_controls(sigmas):
    sigma_steps = [round(float(_s), 2) for _s in sigmas]
    sigma_choice = mo.ui.slider(
        steps=sigma_steps,
        value=sigma_steps[0],
        label="fixed $\\sigma$",
        show_value=True,
        full_width=True,
    )
    sigma_choice
    return sigma_choice, sigma_steps


@app.cell(hide_code=True)
def sigma_explorer(
    FROGS,
    N_TANKS,
    sigma_choice,
    sigma_elpd,
    sigma_grid_posteriors,
    sigma_steps,
    sigmas,
):
    _idx = sigma_steps.index(sigma_choice.value)
    _post = sigma_grid_posteriors[_idx]
    _tanks = np.arange(N_TANKS)
    _observed = FROGS.propsurv.values

    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [2.3, 1]})

    # Tank posteriors: shrinkage lines, 89% HDIs, observed and posterior means
    _ax0.vlines(_tanks, _observed, _post['mean'], color='gray', lw=1.5, alpha=.7)
    _ax0.vlines(_tanks, _post['lower'], _post['upper'], color='C0', lw=5, alpha=.25)
    _ax0.scatter(_tanks, _observed, color='k', s=30, zorder=3, label='observed')
    _ax0.scatter(_tanks, _post['mean'], color='C0', s=30, zorder=4, label='posterior mean')
    _ax0.axhline(_observed.mean(), color='k', linestyle='--', label='Global Mean')
    _ax0.axhline(_post['alpha_bar'], color='C0', linestyle='--', label='Posterior Mean')
    for _x in (15.5, 31.5):
        _ax0.axvline(_x, color='k', alpha=.25)
    _shrinkage = np.abs(_observed - _post['mean']).mean()
    _ax0.set_title(f"$\\sigma={sigma_choice.value}$: average shrinkage {_shrinkage:.3f}")
    _ax0.set_xlabel("tank #")
    _ax0.set_ylabel("proportion survival")
    _ax0.set_ylim([0, 1.05])
    _ax0.legend(loc='lower right', ncol=2, fontsize=9)

    # Cross-validation curve with the chosen sigma
    _best = sigma_elpd['elpd'].idxmax()
    _ax1.errorbar(sigma_elpd.index, sigma_elpd['elpd'], yerr=sigma_elpd['se'], fmt='o', color='C0', markersize=4, alpha=.5)
    _ax1.scatter(_best, sigma_elpd.loc[_best, 'elpd'], marker='*', s=300, color='C1', zorder=5, label=f"$\\sigma_{{LOOCV}}^*$: {_best:1.2}")
    _ax1.scatter(sigmas[_idx], sigma_elpd['elpd'].iloc[_idx], s=180, facecolor='none', edgecolor='k', linewidth=2, zorder=6, label='chosen $\\sigma$')
    _ax1.set_xlabel("$\\sigma$")
    _ax1.set_ylabel("LOOCV score (ELPD)")
    _ax1.legend(loc='lower right')

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Building a Multilevel (Hierarchical) Model

    ### Automatic regularization
    In principle, we can estimate optimal hyperparameter values using cross-validation like we did above, but we don't need to. We can simply learn the optimal $\sigma$ by using a hierarchical model structure, putting a prior distribution on that variance parameter.

    ### Baby's first multi-level model

    $$
    \begin{align*}
    S_i &\sim \text{Binomial}(D_i, p_i)\\
    \text{logit}(p_i) &= \alpha_{T[i]} \\
    \alpha_j &\sim \text{Normal}(\bar \alpha, \sigma) \\
    \bar \alpha &\sim \text{Normal}(0, 1.5) \\
    \sigma &\sim \text{Exponential}(1)
    \end{align*}
    $$

    Note that this prior is shared amongst all groups, allowing it to perform partial pooling.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Plot the priors for baby's first multi-level model
    """)
    return


@app.cell(hide_code=True)
def multilevel_prior_plot():
    _n_samples = 100_000
    _fig, _axs = plt.subplots(1, 3, figsize=(10, 3))

    # rate 1, i.e. scale 1 (`stats.expon(1)` would shift the location to 1)
    _sigmas = stats.expon(scale=1).rvs(size=_n_samples)
    utils.plot_dist(_sigmas, ax=_axs[0])
    _axs[0].set_title('$\\sigma \\sim Exponential(1)$')
    _axs[0].set_xlim([0, 5])
    _axs[0].set_ylim([0, 1])


    _alpha_bars = stats.norm(0, 1.5).rvs(size=_n_samples)
    utils.plot_dist(_alpha_bars, ax=_axs[1])
    _axs[1].set_xlim([-6, 6])
    _axs[1].set_title('$\\bar \\alpha \\sim Normal(0, 1.5)$')


    _alphas = stats.norm.rvs(_alpha_bars, _sigmas)
    utils.plot_dist(_alphas, ax=_axs[2])
    _axs[2].set_xlim([-6, 6])
    _axs[2].set_title('$\\alpha \\sim Normal(\\bar \\alpha, \\sigma)$')
    plt.suptitle("Priors", fontsize=18)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note the prior distribution for $a_j$ (far right plot above) is **a mixture of Normal distributions** with different means (sampled from the $\bar a$ prior distribution) and variances (sampled from the $\sigma$ prior distribution). Therefore it isn't a Normal, but a thicker-tailed distribution, more akin to a student-t
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore: the implied priors

    The prior on each tank's $\alpha_j$ comes from the priors on $\bar\alpha$ and $\sigma$. Change the rate $\lambda$ of the $\sigma$ prior and the spread of $\bar\alpha$ to see how they shape the prior on $\alpha_j$ and, through the logit link, the prior on each tank's survival probability $p_j$.

    - The dashed curve is a Normal with the same standard deviation as the $\alpha_j$ prior. The mixture has thicker tails than that Normal.
    - A wide prior on the log-odds scale piles up probability near 0 and 1 on the probability scale.
    """)
    return


@app.cell(hide_code=True)
def hierarchical_prior_controls():
    prior_sigma_rate = mo.ui.slider(0.1, 5, step=0.1, value=1, label="rate $\\lambda$ of $\\sigma \\sim \\text{Exponential}(\\lambda)$", show_value=True)
    prior_alpha_bar_sd = mo.ui.slider(0.1, 5, step=0.1, value=1.5, label="sd of $\\bar\\alpha \\sim \\text{Normal}(0, \\text{sd})$", show_value=True)
    mo.hstack([prior_sigma_rate, prior_alpha_bar_sd], justify='start', gap=2)
    return prior_alpha_bar_sd, prior_sigma_rate


@app.cell(hide_code=True)
def hierarchical_prior_explorer(prior_alpha_bar_sd, prior_sigma_rate):
    _rng = np.random.default_rng(12)
    _n_samples = 100_000
    _sigmas = _rng.exponential(1 / prior_sigma_rate.value, size=_n_samples)
    _alpha_bars = _rng.normal(0, prior_alpha_bar_sd.value, size=_n_samples)
    _alphas = _rng.normal(_alpha_bars, _sigmas)
    _ps = utils.invlogit(_alphas)

    _fig, _axs = plt.subplots(1, 4, figsize=(16, 3.4))
    _hist_kwargs = dict(density=True, histtype='stepfilled', alpha=.4, color='C0')

    _axs[0].hist(_sigmas, bins=np.linspace(0, 5, 60), **_hist_kwargs)
    _axs[0].set_title(f"$\\sigma \\sim Exponential({prior_sigma_rate.value:g})$")
    _axs[0].set_xlabel("$\\sigma$")

    _axs[1].hist(_alpha_bars, bins=np.linspace(-10, 10, 80), **_hist_kwargs)
    _axs[1].set_title(f"$\\bar \\alpha \\sim Normal(0, {prior_alpha_bar_sd.value:g})$")
    _axs[1].set_xlabel("$\\bar \\alpha$")

    _alpha_xs = np.linspace(-10, 10, 400)
    _axs[2].hist(_alphas, bins=_alpha_xs[::5], **_hist_kwargs)
    _axs[2].plot(_alpha_xs, stats.norm(0, _alphas.std()).pdf(_alpha_xs), 'k--', label='Normal, same sd')
    _axs[2].set_title(f"$\\alpha_j$ (excess kurtosis {stats.kurtosis(_alphas):.1f})")
    _axs[2].set_xlabel("$\\alpha_j$")
    _axs[2].legend(loc='upper left', fontsize=9)

    _extreme = np.mean((_ps < .05) | (_ps > .95))
    _axs[3].hist(_ps, bins=np.linspace(0, 1, 41), **_hist_kwargs)
    _axs[3].set_title(f"$p_j$: {_extreme:.0%} below .05 or above .95")
    _axs[3].set_xlabel("$p_j = $ logit$^{-1}(\\alpha_j)$")

    for _ax in _axs:
        _ax.set_yticks([])
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the multi-level model
    """)
    return


@app.cell
def multilevel_fit(N_SURVIVED, N_TANKS, N_TRIALS):
    with pm.Model() as multilevel_model:

        # Priors
        _sigma = pm.Exponential("sigma", 1)
        _alpha_bar = pm.Normal("alpha_bar", 0, 1.5)
        _alpha = pm.Normal("alpha", _alpha_bar, _sigma, shape=N_TANKS)

        # Likelihood (record log p_survived for visualization)
        _p_survived = pm.Deterministic("p_survived", pm.math.invlogit(_alpha))
        _S = pm.Binomial('survived', n=N_TRIALS, p=_p_survived, observed=N_SURVIVED)

        multilevel_inference = cached_sample()

        # Log-likelihood for model comparison
        multilevel_inference = pm.compute_log_likelihood(multilevel_inference)
    return (multilevel_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Summarize the multi-level model posterior
    """)
    return


@app.cell
def multilevel_summary(multilevel_inference):
    az.summary(multilevel_inference, var_names=['alpha_bar', 'sigma'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### The multi-level model's posterior HDI overlaps the optimal value found via cross-validation
    """)
    return


@app.cell(hide_code=True)
def multilevel_vs_cv_plot(
    inferences,
    multilevel_inference,
    plot_model_comparisons,
    sigmas,
):
    plot_model_comparisons(sigmas, inferences, multilevel_inference.posterior)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - multi-level models learn population variation automatically, efficiently
    - get regularization for free

    ### Comparing multi-level and fixed-sigma model

    #### Fixed sigma model
    $$
    \begin{align*}
    S_i &\sim \text{Binomial}(D_i, p_i)\\
    \text{logit}(p_i) &= \alpha_{T[i]} \\
    \alpha_j &\sim \text{Normal}(\bar \alpha, 1) \\
    \bar \alpha &\sim \text{Normal}(0, 1.5) \\
    \end{align*}
    $$
    """)
    return


@app.cell
def fixed_sigma_fit(N_SURVIVED, N_TANKS, N_TRIALS):
    with pm.Model() as fixed_sigma_model:
        # Sigma is fixed globally (no prior, and thus fewer defined params)
        _sigma = 1

        # Prior
        _alpha_bar = pm.Normal('alpha_bar', 0, 1.5)
        _alpha = pm.Normal('alpha', _alpha_bar, _sigma, shape=N_TANKS)

        # Likelihood -- record p_survived for visualization
        _p_survived = pm.Deterministic('p_survived', pm.math.invlogit(_alpha))
        _S = pm.Binomial('survived', n=N_TRIALS, p=_p_survived, observed=N_SURVIVED)

        fixed_sigma_inference = cached_sample()
        fixed_sigma_inference = pm.compute_log_likelihood(fixed_sigma_inference)
    return (fixed_sigma_inference,)


@app.cell
def multilevel_vs_fixed_sigma_comparison(
    fixed_sigma_inference,
    multilevel_inference,
):
    _comparison = az.compare(
        {
            "multi-level": multilevel_inference,
            "fixed-sigma": fixed_sigma_inference
        }
    )

    # Multi-level model has fewer effective parameters, despite having more defined parameters
    assert _comparison.loc["multi-level", "p"] < _comparison.loc["fixed-sigma", "p"]
    _comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - multi-level model performs better in terms of LOO score (no surprise)
    - multi-level model **has fewer effective parameters** despite having more programmed parameters
      - this is because the model is more efficient, and shares information across parameters

    #### Demonstrating Posterior Uncertainty due to sample size
    """)
    return


@app.cell(hide_code=True)
def multilevel_posterior_plot(multilevel_inference, plot_survival_posterior):
    plot_survival_posterior(multilevel_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Small tanks (on the far left of the above figure) have fewer observations, and thus
      - wider posterior variances
      - larger amount of shrinkage toward the global mean
    - Large tanks (on the far right of the above figure) have more observations, and thus
      - tighter posterior variances
      - less shrinkage toward the global mean
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore: why partial pooling helps small tanks

    This simulation follows Section 13.2 of *Statistical Rethinking*. We draw tanks from a known population (the true $\bar\alpha$ and $\sigma$), simulate the survivors in tanks of 5, 10, 25 and 35 tadpoles, and compare three estimates of each tank's survival probability with the true value:

    - **no pooling**: each tank's own survival rate;
    - **complete pooling**: one survival rate for all tanks;
    - **partial pooling**: the posterior mean of the multi-level model above.

    To keep the widget instant, the partial-pooling posterior is computed exactly on a grid (over each $\alpha_j$, and over $\bar\alpha$ and $\sigma$) instead of with MCMC. On the frog data, the grid gives the same tank estimates as the MCMC fit to within 0.002.

    A single simulation is noisy, so click **new simulation** a few times. On average, partial pooling wins most clearly for the small tanks. Complete pooling only does well when $\sigma$ is small, that is when the tanks really are alike.
    """)
    return


@app.cell
def pooling_simulation_helpers():
    def grid_partial_pooling(
        survived,
        n,
        alpha_grid=np.linspace(-10, 10, 401),
        alpha_bar_grid=np.linspace(-4, 4, 60),
        sigma_grid=np.linspace(0.02, 5, 60),
    ):
        """Posterior mean survival probability of each tank under the multi-level model

            alpha_j ~ Normal(alpha_bar, sigma), alpha_bar ~ Normal(0, 1.5), sigma ~ Exponential(1)

        computed exactly on a grid instead of with MCMC: each tank's alpha_j is
        integrated over `alpha_grid`, and (alpha_bar, sigma) over a 2-D grid.
        """
        survived, n = np.asarray(survived), np.asarray(n)
        p = utils.invlogit(alpha_grid)

        # Likelihood of each tank at each alpha, scaled per tank (the scale cancels below)
        log_lik = survived[:, None] * np.log(p) + (n - survived)[:, None] * np.log1p(-p)
        lik = np.exp(log_lik - log_lik.max(axis=1, keepdims=True))  # (tanks, alphas)

        # Prior density of alpha_j for every (alpha_bar, sigma) pair, times the grid step
        alpha_bars, sigmas_ = (g.ravel() for g in np.meshgrid(alpha_bar_grid, sigma_grid, indexing="ij"))
        prior = stats.norm.pdf(alpha_grid[:, None], alpha_bars, sigmas_) * (alpha_grid[1] - alpha_grid[0])

        # Integrate out each alpha_j: p(S_j | alpha_bar, sigma) and E[p_j * likelihood]
        marginal = lik @ prior  # (tanks, hyperparameter pairs)
        weighted = (lik * p) @ prior

        # Posterior over (alpha_bar, sigma)
        log_post = (
            stats.norm.logpdf(alpha_bars, 0, 1.5)
            + stats.expon.logpdf(sigmas_)
            + np.log(np.maximum(marginal, 1e-300)).sum(axis=0)
        )
        weights = np.exp(log_post - log_post.max())
        weights /= weights.sum()

        p_tank = np.divide(weighted, marginal, out=np.zeros_like(weighted), where=marginal > 0)
        return p_tank @ weights, weights @ alpha_bars, weights @ sigmas_


    def simulate_pooling(alpha_bar, sigma, n_tanks_per_size, seed, tank_sizes=(5, 10, 25, 35)):
        """Simulate tanks from a known population, then estimate each tank's survival
        probability with no pooling, complete pooling and partial pooling.

        Returns a DataFrame of absolute errors per tank, plus the posterior means of
        alpha_bar and sigma from the partial-pooling model.
        """
        rng = np.random.default_rng(seed)
        n = np.repeat(tank_sizes, n_tanks_per_size)
        p_true = utils.invlogit(rng.normal(alpha_bar, sigma, size=len(n)))
        survived = rng.binomial(n, p_true)

        partial_pooling, alpha_bar_hat, sigma_hat = grid_partial_pooling(survived, n)
        estimates = {
            "no pooling": survived / n,
            "partial pooling": partial_pooling,
            "complete pooling": np.full(len(n), survived.sum() / n.sum()),
        }
        errors = pd.DataFrame({name: np.abs(est - p_true) for name, est in estimates.items()})
        errors.insert(0, "tank_size", n)
        return errors, alpha_bar_hat, sigma_hat

    return (simulate_pooling,)


@app.cell(hide_code=True)
def pooling_controls():
    pooling_alpha_bar = mo.ui.slider(-2, 2, step=0.1, value=1.5, label="true $\\bar\\alpha$", show_value=True)
    pooling_sigma = mo.ui.slider(0.1, 3, step=0.1, value=1.5, label="true $\\sigma$", show_value=True)
    pooling_n_tanks = mo.ui.slider(5, 20, step=1, value=15, label="tanks per size", show_value=True)
    pooling_resample = mo.ui.button(value=0, on_click=lambda count: count + 1, label="new simulation")
    mo.hstack([pooling_alpha_bar, pooling_sigma, pooling_n_tanks, pooling_resample], justify='start', gap=2, align='center')
    return pooling_alpha_bar, pooling_n_tanks, pooling_resample, pooling_sigma


@app.cell(hide_code=True)
def pooling_explorer(
    pooling_alpha_bar,
    pooling_n_tanks,
    pooling_resample,
    pooling_sigma,
    simulate_pooling,
):
    _errors, _alpha_bar_hat, _sigma_hat = simulate_pooling(
        pooling_alpha_bar.value,
        pooling_sigma.value,
        pooling_n_tanks.value,
        seed=12 + pooling_resample.value,
    )
    _styles = {'no pooling': 'C1', 'partial pooling': 'C0', 'complete pooling': 'k'}
    _xs = np.arange(len(_errors))

    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [2.3, 1]})

    # Per-tank errors, with the average error per tank size as horizontal lines
    _ax0.scatter(_xs, _errors['no pooling'], facecolor='none', edgecolor='C1', s=30, label='no pooling')
    _ax0.scatter(_xs, _errors['partial pooling'], color='C0', s=20, label='partial pooling')
    for _size, _group in _errors.groupby('tank_size'):
        _start, _end = _group.index.min() - .4, _group.index.max() + .4
        for _method, _color in _styles.items():
            _linestyle = ':' if _method == 'complete pooling' else '-'
            _ax0.hlines(_group[_method].mean(), _start, _end, color=_color, linestyle=_linestyle, linewidth=2)
        _ax0.text((_start + _end) / 2, 1.02, f"{_size} tadpoles", ha='center', va='bottom', transform=_ax0.get_xaxis_transform())
        if _start > 0:
            _ax0.axvline(_start, color='k', alpha=.25)
    _ax0.plot([], [], color='k', linestyle=':', label='complete pooling (average)')
    _ax0.set_xlabel("tank")
    _ax0.set_ylabel("absolute error in p(survive)")
    _ax0.legend(loc='upper right', fontsize=9)

    # Average error by tank size
    _averages = _errors.groupby('tank_size').mean()
    _width = .27
    for _ii, (_method, _color) in enumerate(_styles.items()):
        _ax1.bar(np.arange(len(_averages)) + (_ii - 1) * _width, _averages[_method], _width, color=_color, alpha=.7, label=_method)
    _ax1.set_xticks(np.arange(len(_averages)), _averages.index)
    _ax1.set_xlabel("tadpoles per tank")
    _ax1.set_ylabel("average absolute error")
    _ax1.legend(fontsize=9)

    _fig.suptitle(
        f"Partial-pooling posterior means: $\\bar\\alpha$ = {_alpha_bar_hat:.2f} (true {pooling_alpha_bar.value:g}), "
        f"$\\sigma$ = {_sigma_hat:.2f} (true {pooling_sigma.value:g})",
        y=1.04,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Including the presence of predators
    """)
    return


@app.cell
def predator_colored_plot(
    PREDATOR_ID,
    multilevel_inference,
    plot_survival_posterior,
):
    PREDATOR_COLORS = ["C0" if p == 1 else "C1" for p in PREDATOR_ID]

    # Multi-level model without predators (same plot as above, but with different color-coding)
    plot_survival_posterior(multilevel_inference, color=PREDATOR_COLORS)
    plt.title("Predators absent (blue) or present (red)")
    plt.gcf()
    return (PREDATOR_COLORS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Highlighting the tanks with predators shows that the presence of predators (red) decreases survival rates
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Multilevel model with predator effects
    """)
    return


@app.cell(hide_code=True)
def predator_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("T", "S"),
            ("D", "S"),
            ("G", "S"),
            ("P", "S")
        ],
        node_props={
            "T": {"label": "tank, T"},
            "S": {"label": "survival, S", "color": "red"},
            "D": {"label": "density, D"},
            "G": {"label": "size, G"},
            "P": {"label": "predators, P"},
        },
        edge_props={
            ("T", "S"): {"color": "red"},
            ("P", "S"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    S_i &\sim \text{Binomial}(D_i, p_i)\\
    \text{logit}(p_i) &= \alpha_{T[i]} + \beta_P P_i\\
    \beta_P &\sim \text{Normal}(0, .5) \\
    \alpha_j &\sim \text{Normal}(\bar \alpha, \sigma) \\
    \bar \alpha &\sim \text{Normal}(0, 1.5) \\
    \sigma &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell
def predator_fit(N_SURVIVED, N_TANKS, N_TRIALS, PREDATOR_ID):
    TANK_ID = np.arange(N_TANKS)
    with pm.Model() as predator_model:

        # Global Priors
        _sigma = pm.Exponential("sigma", 1)
        _alpha_bar = pm.Normal("alpha_bar", 0, 1.5)

        # Predator-specific prior
        _beta_predator = pm.Normal("beta_predator", 0, 0.5)

        # Tank-specific prior
        _alpha = pm.Normal("alpha", _alpha_bar, _sigma, shape=N_TANKS)

        # Record p_survived for visualization
        _p_survived = pm.math.invlogit(_alpha[TANK_ID] + _beta_predator * PREDATOR_ID)
        _p_survived = pm.Deterministic('p_survived', _p_survived)

        # Likelihood
        _S = pm.Binomial("survived", n=N_TRIALS, p=_p_survived, observed=N_SURVIVED)

        predator_inference = cached_sample()
    return (predator_inference,)


@app.cell(hide_code=True)
def predator_posterior_plot(
    PREDATOR_COLORS,
    plot_survival_posterior,
    predator_inference,
):
    # Predator model
    plot_survival_posterior(predator_inference, color=PREDATOR_COLORS)
    plt.gcf()
    return


@app.cell(hide_code=True)
def predator_effect_plot(predator_inference):
    utils.plot_dist(predator_inference.posterior["beta_predator"])
    plt.xlabel("$\\beta_P$ (effect of predators)")
    plt.ylabel("density")
    plt.title("Predators have negative effect on survival")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Predictions vs Inference
    """)
    return


@app.cell(hide_code=True)
def predictions_vs_sigma_plot(
    PREDATOR_COLORS,
    multilevel_inference,
    predator_inference,
):
    # Should we probably be using the posterior predictive here, rather than the posterior?
    _multilevel_posterior_mean = multilevel_inference.posterior.mean(dim=('chain', 'draw'))['p_survived']
    _predator_posterior_mean = predator_inference.posterior.mean(dim=('chain', 'draw'))['p_survived']


    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))

    plt.sca(_axs[0])
    for _x, _y, _c in zip(_multilevel_posterior_mean, _predator_posterior_mean, PREDATOR_COLORS):
        utils.plot_scatter(xs=_x, ys=_y, color=_c, alpha=.8)

    plt.plot((0, 1), (0, 1), 'k--')
    plt.xlabel("p(survive) - model without predators")
    plt.ylabel("p(survive) - model with predators")

    # Legend
    _legend_kwargs = dict(marker='o', color='none', markersize=8, alpha=.8)
    _legend_elements = [
        Line2D([0], [0], markerfacecolor='C0', label='Predators', **_legend_kwargs),
        Line2D([0], [0], markerfacecolor='C1', label='No Predators', **_legend_kwargs),
    ]
    plt.legend(handles=_legend_elements, loc='lower right')
    plt.title("extremely similar predictions")
    plt.axis('square')

    plt.sca(_axs[1])
    utils.plot_dist(multilevel_inference.posterior['sigma'], ax=_axs[1], color='C1', label='without predators (mST)', plot_kwargs={"lw": 3})
    utils.plot_dist(predator_inference.posterior['sigma'], ax=_axs[1], color='C0', label='with predators (mSTP)', plot_kwargs={"lw": 3})
    plt.xlim([0, 2.5])
    plt.xlabel("$\\sigma$")
    plt.title("very different $\\sigma$ values")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Model without predators can predict survival just as well as the model with predators
    - This is because the multi-level model can still capture the tank-level variability through the individual $\alpha_T$
    - However, adding predators "explains away" a lot more of the tank-level variance. This is demonstrated by the $\sigma$ values for tank-level variability being smaller in the predator model. Thus the predator model has to consider less tank-based variability in order to capture the variability in survival.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Varying Effect Superstitions
    - ~~Partial pooling requires random sampling from the population ❌~~
    - ~~Number of categories / units must be large ❌~~
    - ~~Variation must be Gaussian ❌~~
      - Gaussian priors can learn non-Gaussian distributions

    ## Practical Difficulties
    - Using multiple clusters simultaneously (e.g. participants AND stories)
    - Sampling efficiency -- recoding (e.g. centered/non-centered priors)
    - Partial pooling on other parameters (e.g. slopes) or unobserved confounds?

    These difficulties will be addressed in upcoming lectures on Multi-level models. 😅
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Fixed Effects, Multilevel Models, & Mundlak Machines
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Random Confounds
    """)
    return


@app.cell(hide_code=True)
def random_confound_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "X"),
            ("G", "Y"),
            ("X", "Y"),
            ("Z", "Y")
        ], 
        node_props={
            "G": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Estimand**: Influence of $X$ on $Y$

    - Outcome $Y$ (e.g. tadpole survival)
    - Individual-level traits $X$
    - Group-level traits $Z$
    - Unobserved tank effects $G$
      - affects both $X$ and $Y$
      - e.g. tank temperature
      - creates a backdoor path from $X$ to $Y$ (via fork)
      - We can't directly measure $G$.
      - However, if we have repeat observations, there's some tricks up our sleeve that we can use.

    **Estimator?**

    1. Fixed Effects model
    2. Multi-level model
    3. Mundlak Machine

    ### Simulate some data
    """)
    return


@app.cell
def random_confound_simulation():
    # Generate data
    np.random.seed(12)

    N_GROUPS = 30
    N_IDS = 200
    ALPHA = -2  # p(Y) < .5 on average
    BETA_ZY = -0.5
    BETA_XY = 1

    # Group-level data
    GROUP = np.random.choice(np.arange(N_GROUPS).astype(int), size=N_IDS, replace=True)
    Z = stats.norm.rvs(size=N_GROUPS)  # Observed group traits
    U = stats.norm(0, 1.5).rvs(size=N_GROUPS)  # Unobserved group confound

    # Individual-level data
    X = stats.norm.rvs(U[GROUP])  # Observed individual traits
    _p = utils.invlogit(ALPHA + BETA_XY * X + U[GROUP] + BETA_ZY * Z[GROUP])
    Y = stats.bernoulli.rvs(p=_p)
    return BETA_XY, BETA_ZY, GROUP, N_GROUPS, X, Y, Z


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Group sizes are imbalanced: some groups have many individuals, others only one or two
    """)
    return


@app.cell(hide_code=True)
def group_size_histogram(GROUP, N_GROUPS):
    plt.hist(GROUP, bins=N_GROUPS)
    plt.xlabel("group")
    plt.ylabel("individuals")
    plt.title("Distribution of Groups")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Naive Model

    $$
    \begin{align*}
    Y_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha + \beta_{X}X_i + \beta_{Z} Z_{G[i]} \\
    \alpha &\sim \text{Normal}(0, 10) \\
    \beta_* &\sim \text{Normal}(0, 1)
    \end{align*}
    $$
    """)
    return


@app.cell
def naive_fit(GROUP, X, Y, Z):
    with pm.Model() as naive_model:

        # Priors
        _fixed_sigma = 10  # no pooling
        _alpha = pm.Normal("alpha", 0, _fixed_sigma)  # no group effects
        _beta_XY = pm.Normal("beta_XY", 0, 1)
        _beta_ZY = pm.Normal("beta_ZY", 0, 1)

        # Likelihood
        _p = pm.math.invlogit(_alpha + _beta_XY * X + _beta_ZY * Z[GROUP])
        pm.Bernoulli("Y_obs", p=_p, observed=Y)
        naive_inference = cached_sample()
    return (naive_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fixed effect Model

    $$
    \begin{align*}
    Y_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha_{G[i]} + \beta_{X}X_i + \beta_{Z} Z_{G[i]} \\
    \alpha_j &\sim \text{Normal}(0, 10) \\
    \beta_* &\sim \text{Normal}(0, 1)
    \end{align*}
    $$

    - Estimate a different average rate for each group **without pooling**
    - $\beta_Z$ and $\beta_X$ are global parameters
    - Accounts for group-level confounding effect of $G$ (via offsets for each group)
    - Problems:
      - Cannot identify any group-level effects $Z$
        - $Z$ is unidentifiable because it's added globally
          - there are an infinite number of equivalent combinations of $\alpha_{G[i]} + \beta_Z Z_{G[i]}$
          - can't separate contribution of $\beta_Z$ from $\alpha$
        - Can't include group-level predictors with a fixed-effects model
      - Inefficient
    """)
    return


@app.cell
def fixed_effects_fit(GROUP, N_GROUPS, X, Y, Z):
    with pm.Model() as fixed_effects_model:

        # Priors
        _fixed_sigma = 10  # no pooling
        _alpha = pm.Normal("alpha", 0, _fixed_sigma, shape=N_GROUPS)
        _beta_XY = pm.Normal("beta_XY", 0, 1)
        _beta_ZY = pm.Normal("beta_ZY", 0, 1)

        # Likelihood
        _p = pm.math.invlogit(_alpha[GROUP] + _beta_XY * X + _beta_ZY * Z[GROUP])
        pm.Bernoulli("Y_obs", p=_p, observed=Y)
        fixed_effect_inference = cached_sample()
    return (fixed_effect_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multilevel Model

    $$
    \begin{align*}
    Y_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha_{G[i]} + \beta_{X}X_i + \beta_{Z} Z_{G[i]} \\
    \beta_* &\sim \text{Normal}(0, 1) \\
    \alpha_j &\sim \text{Normal}(\bar a, \tau) \\
    \bar a &\sim \text{Normal}(0, 1) \\
    \tau &\sim \text{Exponential}(1)
    \end{align*}
    $$

    - Estimate a different average rate for each group **with partial pooling**
    - better estimates for $G$ effects
    - at the expense of estimating $X$ effects
      - compromises estimating the confound so that it can get better estimates for the group
    - CAN identify $Z$ effects
      - CAN incorporate group-level predictors
    """)
    return


@app.cell
def group_multilevel_fit(GROUP, N_GROUPS, X, Y, Z):
    with pm.Model() as group_multilevel_model:

        # Priors
        _alpha_bar = pm.Normal("alpha_bar", 0, 1)
        _tau = pm.Exponential('tau', 1)

        # NOTE: non-centered prior reparameterization for numerical stability
        _z = pm.Normal("z", 0, 1, shape=N_GROUPS)
        _alpha = _alpha_bar + _z * _tau

        _beta_XY = pm.Normal("beta_XY", 0, 1)
        _beta_ZY = pm.Normal("beta_ZY", 0, 1)

        # Likelihood
        _p = pm.math.invlogit(_alpha[GROUP] + _beta_XY * X + _beta_ZY * Z[GROUP])
        pm.Bernoulli("Y_obs", p=_p, observed=Y)
        group_multilevel_inference = cached_sample(target_accept=.95)
    return (group_multilevel_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare Models
    """)
    return


@app.cell
def compare_posteriors_helper(BETA_XY, BETA_ZY):
    def compare_model_posteriors(inferences, labels, variable='beta_XY'):
        """Helper function to plot model posteriors
        """
        groundtruth = BETA_XY if variable == 'beta_XY' else BETA_ZY
        for ii, inference in enumerate(inferences):
            utils.plot_dist(
            inference.posterior[variable].values,
            color=f'C{ii}',
            label=labels[ii]
        )
        plt.axvline(groundtruth, label="actual", color='k', linestyle='--')
        plt.title(f"posterior {variable}")
        plt.legend()

    return (compare_model_posteriors,)


@app.cell(hide_code=True)
def beta_xy_three_models_plot(
    compare_model_posteriors,
    fixed_effect_inference,
    group_multilevel_inference,
    naive_inference,
):
    compare_model_posteriors(
        inferences=[naive_inference, fixed_effect_inference, group_multilevel_inference],
        labels=['naive', 'fixed', 'multi-level']
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Fixed effect models are better (though not great in this simulation) than the multi-level (and naive) models at estimating the individual-level effect $\beta_{XY}$ in the presence of the group confound, but they can't identify any group-level effects.
    """)
    return


@app.cell(hide_code=True)
def beta_zy_three_models_plot(
    compare_model_posteriors,
    fixed_effect_inference,
    group_multilevel_inference,
    naive_inference,
):
    compare_model_posteriors(
        inferences=[naive_inference, fixed_effect_inference, group_multilevel_inference],
        labels=['naive', 'fixed', 'multi-level'],
        variable='beta_ZY'
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Multi-level models can identify the group effects, though it can't identify the confound. Fixed effect models can't say much about the group level effects.

    - If interested completely in _prediction_, multi-level models are better
    - If interested in _inference_ (i.e. via do-calculus or counterfactuals), you might want to use fixed effects (though group-level prediction accuracy may be poor)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Mundlak Machines
    """)
    return


@app.cell(hide_code=True)
def mundlak_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "X"),
            ("G", "Y"),
            ("G", "X_bar"),
            ("X", "Y"),
            ("Z", "Y")
        ], 
        node_props={
            "G": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Statistical Model

    $$
    \begin{align*}
    Y_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha_{G[i]} + \beta_{X}X_i + \beta_{Z} Z_{G[i]} + \beta_{\bar X} \bar X_{G[i]}\\
    \beta_* &\sim \text{Normal}(0, 1) \\
    \alpha_j &\sim \text{Normal}(\bar a, \tau) \\
    \bar a &\sim \text{Normal}(0, 1) \\
    \tau &\sim \text{Exponential}(1)
    \end{align*}
    $$

    - Estimate a different average rate for each group **with partial pooling**
    - Takes advantage of the idea that conditioning on a descendant in a graph can (at least partially) deconfound the parent/ancestor variable.
    - Uses the group-level mean as child of the confounding variables to reduce its confounding effect.
    - Problems:
      - somewhat inefficient
        - because groups have different size, we also need to consider the uncertainty in the estimation of the group-level means, which the Mundlak machine ignores
    """)
    return


@app.cell
def mundlak_fit(GROUP, N_GROUPS, X, Y, Z):
    # Group-level average
    Xbar = np.array([X[GROUP==g].mean() for g in range(N_GROUPS)])

    with pm.Model() as mundlak_model:
        # Priors
        _alpha_bar = pm.Normal("alpha_bar", 0, 1)
        _tau = pm.Exponential("tau", 1)
        _z = pm.Normal("z", 0, 1, shape=N_GROUPS)

        # Note: uncentered reparameterization
        _alpha = _alpha_bar + _z * _tau

        _beta_XY = pm.Normal("beta_XY", 0, 1)
        _beta_ZY = pm.Normal("beta_ZY", 0, 1)
        _beta_Xbar_Y = pm.Normal("beta_XbarY", 0, 1)

        # Likelihood
        _p = pm.math.invlogit(_alpha[GROUP] + _beta_XY * X + _beta_ZY * Z[GROUP] + _beta_Xbar_Y * Xbar[GROUP])
        pm.Bernoulli("Y_obs", p=_p, observed=Y)
        mundlak_inference = cached_sample(target_accept=.95)
    return (mundlak_inference,)


@app.cell(hide_code=True)
def beta_xy_four_models_plot(
    compare_model_posteriors,
    fixed_effect_inference,
    group_multilevel_inference,
    mundlak_inference,
    naive_inference,
):
    compare_model_posteriors(
        inferences=[naive_inference, fixed_effect_inference, group_multilevel_inference, mundlak_inference],
        labels=['naive', 'fixed', 'multi-level', 'Mundlak']
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def beta_zy_four_models_plot(
    compare_model_posteriors,
    fixed_effect_inference,
    group_multilevel_inference,
    mundlak_inference,
    naive_inference,
):
    compare_model_posteriors(
        inferences=[naive_inference, fixed_effect_inference, group_multilevel_inference, mundlak_inference],
        labels=['naive', 'fixed', 'multi-level', 'Mundlak'],
        variable='beta_ZY'
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Mundlak machines are able to capture both Treatment and Group-level effects
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Latent Mundlak Machine (aka "Full Luxury Bayes")

    - We model not only the observed outcome $Y$ (as a function of $X, G, Z$), but also the treatment $X$ (as a function of the confound G).
    - Unlike the Mundlak machine, which collapses $\bar X$ into a point estimate, thus ignoring the uncertainty in that mean estimate, the Latent Mundlak estimates this uncertainty by including the submodel on $X$.
    - akin to a measurement error model

    So the original graph:
    """)
    return


@app.cell(hide_code=True)
def latent_mundlak_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "X"),
            ("G", "Y"),
            ("X", "Y"),
            ("Z", "Y")
        ],
        node_props={
            "G": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ...is modeled (simultaneously) as two sub-graphs each with their own regression sub-model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1. $X$ sub-model
    """)
    return


@app.cell(hide_code=True)
def x_submodel_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "X"),
        ], 
        node_props={
            "G": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    X_i &\sim \text{Normal}(\mu_{X_i}, \sigma_X) \\
    \mu_{X_i} &= \alpha_X + \beta_{GX} U_{G[i]} \\
    U_G &\sim \text{Normal}(0, 1) \\
    \alpha_X &\sim \text{Normal}(0, 1) \\
    \beta_{GX} &\sim \text{Exponential}(1) \\
    \sigma_X &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 2. $Y$ sub-model
    """)
    return


@app.cell(hide_code=True)
def y_submodel_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "Y"),
            ("X", "Y"),
            ("Z", "Y")
        ], 
        node_props={
            "G": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    Y_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha_{Y,G[i]} + \beta_{XY}X_i + \beta_{ZY} Z_{G[i]} + \beta_{GY} U_{G[i]}\\
    U_G &\sim \text{Normal}(0, 1) \\
    \beta_* &\sim \text{Normal}(0, 1) \\
    \alpha_{Y,j} &\sim \text{Normal}(\bar a, \tau) \\
    \bar a &\sim \text{Normal}(0, 1) \\
    \tau &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell
def latent_mundlak_fit(GROUP, N_GROUPS, X, Y, Z):
    with pm.Model() as latent_mundlak_model:
        # Unobserved variable (could use various priors here)
        _G = pm.Normal("u_X", 0, 1, shape=N_GROUPS)

        # ----------
        # X sub-model
        # X Priors
        _alpha_X = pm.Normal("alpha_X", 0, 1)
        _beta_GX = pm.Exponential("beta_GX", 1)
        _sigma_X = pm.Exponential("sigma_X", 1)

        # X Likelihood
        _mu_X = _alpha_X + _beta_GX * _G[GROUP]
        _X = pm.Normal("X", _mu_X, _sigma_X, observed=X)

        # ----------
        # Y sub-model
        # Y priors
        _tau = pm.Exponential("tau", 1)

        # Note: uncentered reparameterization
        _z = pm.Normal("z", 0, 1, size=N_GROUPS)
        _alpha_bar = pm.Normal("alpha_bar", 0, 1)
        _alpha = _alpha_bar + _tau * _z

        _beta_XY = pm.Normal("beta_XY", 0, 1)
        _beta_ZY = pm.Normal("beta_ZY", 0, 1)
        _beta_GY = pm.Normal("beta_GY", 0, 1)

        # Y likelihood
        _p = pm.math.invlogit(_alpha[GROUP] + _beta_XY * _X + _beta_ZY * Z[GROUP] + _beta_GY * _G[GROUP])
        pm.Bernoulli("Y", p=_p, observed=Y)

        latent_mundlak_inference = cached_sample(target_accept=.95)
    return (latent_mundlak_inference,)


@app.cell(hide_code=True)
def beta_xy_all_models_plot(
    compare_model_posteriors,
    fixed_effect_inference,
    group_multilevel_inference,
    latent_mundlak_inference,
    mundlak_inference,
    naive_inference,
):
    compare_model_posteriors(
        inferences=[naive_inference, fixed_effect_inference, group_multilevel_inference, mundlak_inference, latent_mundlak_inference],
        labels=['naive', 'fixed', 'multi-level', 'Mundlak', 'Luxury']
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def beta_zy_all_models_plot(
    compare_model_posteriors,
    fixed_effect_inference,
    group_multilevel_inference,
    latent_mundlak_inference,
    mundlak_inference,
    naive_inference,
):
    compare_model_posteriors(
        inferences=[naive_inference, fixed_effect_inference, group_multilevel_inference, mundlak_inference, latent_mundlak_inference],
        labels=['naive', 'fixed', 'multi-level', 'Mundlak', 'Luxury'],
        variable='beta_ZY'
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Random Confounds: Summary
    - Should you use fixed effects? sometimes, but generally Full-luxury is the way to go
    - Should you use Mundlak Machine / Average X?
      - Sometimes: it does simplify numerical computation, but at the expense of losing uncertainty estimation.
      - Usually have the compute power anyways, so why not just use FLB?
    - Use full generative model
      - no single solution, so just be explicit about model
    """)
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
