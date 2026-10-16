import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import inspect

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    import xarray as xr
    import arviz as az
    from scipy import stats
    from matplotlib import pyplot as plt
    from seaborn import pairplot

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 03 - Geocentric Models](https://youtu.be/tNOu-SEacNU?si=rp_4kWbSVvQChAiv)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Linear Regression
    - **Geocentric**
      - unreasonably good a approximations, despite it always being incorrect
      - can be used as a cog in a causal analysis system, despite being an innaccurate mechanistic model of real phenonena
    - **Gaussian**
      - General error model
      - Abstracts away details, allowing us to make macro inferences, without having to incorporate micro phenonena



    # Why Normal?
    ## Two arguments

    1. Generative: summed fluctuations (see below) tend toward Normal distribution
    2. Inferential: For estimating mean and variance, the Normal is the least informative (fewest assumptions), in the maximum entropy sense

    > Variables do not need to be Normally-distributed in order estimate the correct mean and variance using a Gaussian error model.

    ## Generating Normal distirbution from summation of decisions
    - Simulate a group of people randomly walking left-right, starting from a central location
    - Resulting positions are the summation of many left-and right -- the result is Normally-distributed
    - Normal distribution falls out processes where deviations are summed (also products)
    """)
    return


@app.cell(hide_code=True)
def _():
    _n_people = 10000
    _n_steps = 1000
    _step_size = 0.1
    _steps = _step_size * (2 * stats.bernoulli(p=0.5).rvs(size=(_n_people, _n_steps)) - 1)
    _positions = np.round(np.sum(_steps, axis=1))
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(_axs[0])
    plt.axvline(0, color='k', linestyle='--')
    for _ii, _pos in enumerate(_positions[::15]):
        _color = 'C1' if _pos > 0 else 'C0'
        plt.scatter(x=_pos, y=_ii, color=_color, alpha=0.7, s=10)
    plt.xlim([-10, 10])
    plt.yticks([])
    plt.title('positions')

    # Plot histogram
    _position_unique, _position_counts = np.unique(_positions, return_counts=True)
    _positive_idx = _position_unique > 0
    _negative_idx = _position_unique <= 0
    plt.sca(_axs[1])
    plt.bar(_position_unique[_positive_idx], _position_counts[_positive_idx], width=0.5, color='C1')
    plt.bar(_position_unique[_negative_idx], _position_counts[_negative_idx], width=0.5, color='C0')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlim([-10, 10])
    plt.yticks([])
    plt.title('distribution')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Drawing the Owl
    1. State clear **question**
    2. Sketch causal **assumptions** -- DAG
    3. Define a **generative model** -- generate data
    4. Use generative model to buile (AND TEST) an **estimator**
    5. Profit (through analyzing real data, possibly gaining info to iterate on the model)

    # Linear Regression

    ## Howell Dataset

    ### (1) Question Estimand
    - Describe the association between **weight** and **height**
    - We'll focus on **adult weight** -- Adult height is approximately linear
    """)
    return


@app.cell
def _():
    HOWELL = utils.load_data('Howell1')
    ADULT_HOWELL = HOWELL[HOWELL.age >= 18]
    HOWELL.head()
    return ADULT_HOWELL, HOWELL


@app.cell(hide_code=True)
def _(ADULT_HOWELL, HOWELL):
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(_axs[0])
    utils.plot_scatter(HOWELL.height, HOWELL.weight, color='C1')
    plt.title('Howell Dataset')
    utils.plot_scatter(ADULT_HOWELL.height, ADULT_HOWELL.weight)
    plt.sca(_axs[1])
    utils.plot_scatter(ADULT_HOWELL.height, ADULT_HOWELL.weight)
    plt.title('Adults')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### (2) Scientific Model
    How does **height** influence **weight**?

    $$
    \begin{align}
    H \rightarrow W \\
    W = f(H)
    \end{align}
    $$

    i.e. *"Weight is some function of height"*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### (3) Generative Models

    Options

    1. Dynamic - relationship changes over time
    2. Static - constant trend over time

    $$W = f(H, U)$$

    *"Weight is a function of height, H and unobserved stuff, U"*
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("U", "W")
        ],
        node_props={
            "U": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Linear regression model

    We need a function that maps adult weight as a proportion of height plus some unobserved/unaccounted-for causes. Enter **Linear Regression**:

    $$W = \beta H + U$$

    ### Generative model description:
    $$
    \begin{align}
    W_i &\sim \beta H_i + U_i \\
    U_i &\sim \text{Normal}(0, \sigma) \\
    H_i &\sim \text{Uniform}(130, 170)
    \end{align}
    $$

    ### Describing models
    - Variables on the left
    - Definition on right
    - $\sim$ indicates sampling from a distribution
      - e.g. $H_i \sim \text{Uniform}(130, 170)$ is definition that _height is distributed uniformly between 120 and 170_
    - $=$ indicates expectation or (deterministic) equality
      - e.g. $W_i \sim \beta H_i + U_i$ is definition of equation for _expected weight_
    - subscripts $i$ indicates index of a observation/individual
    - generally code will be written in opposite direction, because you need variables defined in order to be referenced/composed
    """)
    return


@app.function
def simulate_weight(H, beta, sigma):
    """
    Generative model describe above, simulate weight given height, H,
    proportional coefficient beta, and the standard deviation of
    unobserved (Normally-distributed) noise, sigma.
    """
    n_heights = len(H)

    # unobserved noise
    U = stats.norm.rvs(0, sigma, size=n_heights)
    return beta * H + U


@app.cell
def _():
    # Fixed seed, so the simulated data (and the fits below) are the same on
    # every run, and the cached fits can be reused
    np.random.seed(3)

    n_heights = 200
    MIN_HEIGHT = 130
    MAX_HEIGHT = 170
    H = stats.uniform.rvs(MIN_HEIGHT, MAX_HEIGHT - MIN_HEIGHT, size=n_heights)
    W = simulate_weight(H, beta=0.5, sigma=5)
    return H, MAX_HEIGHT, MIN_HEIGHT, W


@app.cell(hide_code=True)
def _(H, W):
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(_axs[0])
    plt.hist(H, bins=25)
    plt.xlabel('height')
    plt.ylabel('frequency')
    plt.sca(_axs[1])
    utils.plot_scatter(H, W)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('W ~ H')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Linear Regression

    Estimate how the average weight changes with a change in height:

    $$E[W_i | H_i] = \alpha + \beta H_i$$

    - $E[W_i | H_i]$: **average weight** conditioned on height
    - $\alpha$: **intercept** of line
    - $\beta$: **slope** of line

    ### Posterior Distribution

    $$
    p(\alpha, \beta, \sigma) = \frac{p(W_i | \alpha, \beta, \sigma) p(\alpha, \beta, \sigma)}{Z}
    $$

    - The only estimator in Bayesian data analysis

    - $p(\alpha, \beta, \sigma)$ -- **Posterior**: Probability of a specific line (model)
    - $p(W_i | \alpha, \beta, \sigma)$ -- **Likelihood**: The number of ways the generative proces (line) could have produced the data
        - aka the "Garden of Forking Data"
    - $p(\alpha, \beta, \sigma)$ -- **Prior**: the previous Posterior (sometimes with no data)
    - $Z$ -- **normalizing constant**

    Common parameterization

    $$
    \begin{align}
    W_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta H_i
    \end{align}
    $$

    _$W$ is distributed normally with mean $\mu$ that is a linear function of $H$_
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Grid Approximate Posterior
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For the following grid approximation simulation, we'll use a utility function `utils.simulate_2_parameter_bayesian_learning` for simulating general Bayeisan posterior update simulation. For the API, see `utils.py`
    """)
    return


@app.cell
def _():
    mo.plain_text(inspect.getdoc(utils.simulate_2_parameter_bayesian_learning))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Functions for `simulate_2_parameter_bayesian_learning`
    """)
    return


# Model function required for simulation
@app.function
def linear_model(x, intercept, slope):
    return intercept + slope * x


# Posterior function required for simulation
@app.function
def linear_regression_posterior(x_obs, y_obs, intercept_grid, slope_grid, likelihood_prior_std = 1.0):

    # Convert params to 1-d arrays
    if np.ndim(intercept_grid) > 1:
        intercept_grid = intercept_grid.ravel()

    if np.ndim(slope_grid):
        slope_grid = slope_grid.ravel()

    log_prior_intercept = stats.norm(0, 1).logpdf(intercept_grid)
    log_prior_slope = stats.norm(0, 1).logpdf(slope_grid)

    log_likelihood = np.array(
        [
            stats.norm(
                intercept + slope * x_obs,
                likelihood_prior_std
            ).logpdf(y_obs) for intercept, slope in zip(intercept_grid, slope_grid)
        ]
    ).sum(axis=1)

    # Posterior is equal to the product of likelihood and priors (here a sum in log scale)
    log_posterior = log_likelihood + log_prior_intercept + log_prior_slope

    # Convert back to natural scale
    return np.exp(log_posterior - log_posterior.max())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulating Posterior Updates
    """)
    return


@app.cell
def _():
    # Generate standardized regression data for demo
    np.random.seed(123)
    _RESOLUTION = 100
    _N_DATA_POINTS = 64

    # Ground truth parameters
    _SLOPE = 0.5
    _INTERCEPT = -1

    _x = stats.norm().rvs(size=_N_DATA_POINTS)
    _y = _INTERCEPT + _SLOPE * _x + stats.norm.rvs(size=_N_DATA_POINTS) * 0.25

    _slope_grid = np.linspace(-2, 2, _RESOLUTION)
    _intercept_grid = np.linspace(-2, 2, _RESOLUTION)

    # Vary the sample size to show how the posterior adapts to more and more data
    _figs = []
    for _n_samples in [0, 2, 4, 8, 16, 32, 64]:
        # Run the simulation
        utils.simulate_2_parameter_bayesian_learning(
            x_obs=_x[:_n_samples],
            y_obs=_y[:_n_samples],
            param_a_grid=_intercept_grid,
            param_b_grid=_slope_grid,
            true_param_a=_INTERCEPT,
            true_param_b=_SLOPE,
            model_func=linear_model,
            posterior_func=linear_regression_posterior,
            param_labels=['intercept', 'slope'],
            data_range_x=(-3, 3), data_range_y=(-3, 3)
        )
        _figs.append(plt.gcf())
    mo.vstack(_figs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Enough Grid Approximation
    McElreath uses Quadratic Approximation--`quap`--for the first half of the lectures, which _can_ speed up model fitting for continuous models that have posteriors that can be approximated with a multi-dimensional Normal distribution. However, we'll just use PyMC MCMC implementations for all examples without loss of generality. For the earlier examples in the lecture series where `quap` is being used, MCMC samples perfectly fast anyways.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## (4) Validate the model
    ### Validate Assumptions with **Prior Predictive Distribution**
    - Priors should express scientific knowledge, _but softly_
    - For example, when Height is 0, Weight shoudl be 0, right?
    - Weight should increase (on average) with height -- i.e. $\beta > 0$
    - Weight (kg) should be less than Height (cm)
    - variances should be positive

    $$
    \begin{align}
    \alpha &\sim \text{Normal}(0, 10) \\
    \beta &\sim \text{Normal}(0, 1) \\
    \sigma &\sim \text{Uniform}(0, 10) \\
    \end{align}
    $$

    ### More on Priors
    - We can understand the implications of priors by running simulations
    - **There are no correct priors**, only those that are scientifically justifiable
    - Priors are less important with simple models
    - **Priors are very important in complex models**
    """)
    return


@app.cell
def _():
    _n_simulations = 100
    _alphas = np.random.normal(0, 10, _n_simulations)
    _betas = np.random.uniform(0, 1, _n_simulations)  # beta should all be positive
    _heights = np.linspace(130, 170, 3)
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _a, _b in zip(_alphas, _betas):
        _weights = _a + _b * _heights
        plt.plot(_heights, _weights, color='C0')
    plt.xlabel('height, H (cm)')
    plt.ylabel('E[W | H] (kg)')
    plt.ylim((30, 70))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulation-based Validation & Calibration
    - Simulate data with varying parameters
    - Vary data-generating parameters (e.g. slope) that are analogous to the model; make sure the estimator tracks
    - Make sure that at large sample sizes, data-generating parameters can be recovered
    - Same for confounds/unkowns
    """)
    return


@app.cell
def _(H, W):
    linear_regression_inferences = []
    linear_regression_models = []
    sample_sizes = [1, 2, 10, 20, 50, len(H)]
    for _sample_size in sample_sizes:
        print(f'Sample size: {_sample_size}')
        with pm.Model() as _model:
            # Mutable data for posterior predictive visualization
            _H = pm.Data('H', H[:_sample_size], dims='obs_id')

            # Priors
            _sigma = pm.Uniform('sigma', 0, 10)  # Noise variance
            _alpha = pm.Normal('alpha', 0, 10)  # Intercept
            _beta = pm.Uniform('beta', 0, 1)  # slope

            # Likelihood
            _mu = _alpha + _beta * _H
            pm.Normal('W_obs', _mu, _sigma, observed=W[:_sample_size], dims='obs_id')

            # Sample posterior
            _inference = cached_sample(target_accept=0.99)
            linear_regression_inferences.append(_inference)
            linear_regression_models.append(_model)
    return linear_regression_inferences, linear_regression_models, sample_sizes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Test Model Validity with Posterior Predictive Distribution
    Below we show:
    - how the posterior becomes more specific with different numbers of observations
    - how the posterior is "made of lines" -- there are an infinite number of possible lines that can be drawn from the posterior
    - confidence intervals can be established to communicate the uncertainty of the posterior's fit to the data
    """)
    return


@app.cell
def _(MAX_HEIGHT, MIN_HEIGHT):
    MIN_WEIGHT = 55
    MAX_WEIGHT = 90

    def plot_linear_regression_posterior_predictive(model, inference, min_height=MIN_HEIGHT, max_height=MAX_HEIGHT, min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT, N_SHOW=20):
        # Sample Posterior Predictive with full range of heights
        xs = np.linspace(min_height, max_height, 30)
        H_ = xr.DataArray(xs)
        with model:
            pm.set_data({'H': H_})
            ppd = pm.sample_posterior_predictive(inference, var_names=['W_obs'], predictions=True, return_inferencedata=True, progressbar=False)

        # Plot Posterior Predictive HDI
        utils.plot_hdi(H_, ppd.predictions['W_obs'], color='k', fill_kwargs=dict(alpha=0.1))

        # Plot Posterior
        posterior = inference.posterior
        lines = posterior['alpha'] + posterior['beta'] * H_
        for l in lines[0, :N_SHOW, :]:
            plt.plot(xs, l, color='k', alpha=0.5, zorder=20)

        # Formatting
        plt.xticks(np.arange(min_height, max_height + 1, 10))
        plt.xlim([min_height, max_height])
        plt.xlabel('hieght, H (cm)')
        plt.ylim([min_weight, max_weight])
        plt.yticks(np.arange(min_weight, max_weight, 10))
        plt.ylabel('weight, W (kg)')

    return (plot_linear_regression_posterior_predictive,)


@app.cell(hide_code=True)
def _(
    H,
    W,
    linear_regression_inferences,
    linear_regression_models,
    plot_linear_regression_posterior_predictive,
    sample_sizes,
):
    _fig, _axs = plt.subplots(2, 3, figsize=(10, 8))
    for _ii, (_sample_size, _model, _inference) in enumerate(zip(sample_sizes, linear_regression_models, linear_regression_inferences)):
        plt.sca(_axs[_ii // 3][_ii % 3])

        # Plot training data
        plt.scatter(H[:_sample_size], W[:_sample_size], s=80, zorder=20, alpha=0.5)
        plot_linear_regression_posterior_predictive(_model, _inference)
        plt.title(f'N={_sample_size}')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## (5) Analyse the data
    """)
    return


@app.cell
def _(ADULT_HOWELL):
    with pm.Model() as howell_model:
        # Mutable data for posterior predictive / visualization
        _H = pm.Data('H', ADULT_HOWELL.height.values, dims='obs_ids')

        # priors
        _sigma = pm.Uniform('sigma', 0, 10)  # Noise variance
        _alpha = pm.Normal('alpha', 0, 10)  # Intercept
        _beta = pm.Uniform('beta', 0, 1)  # Slope

        # likelihood
        _mu = _alpha + _beta * _H
        pm.Normal('W_obs', _mu, _sigma, observed=ADULT_HOWELL.weight.values, dims='obs_ids')

        # Sample posterior
        howell_inference = cached_sample(target_accept=0.99)
    return howell_inference, howell_model


@app.cell
def _(howell_inference):
    az.summary(howell_inference)
    return


@app.cell(hide_code=True)
def _(howell_inference):
    _grid = pairplot(
        pd.DataFrame(
            {
                "alpha": howell_inference.posterior.sel(chain=0)['alpha'],
                "beta": howell_inference.posterior.sel(chain=0)['beta'],
                "sigma": howell_inference.posterior.sel(chain=0)['sigma'],
            }
        ),
        diag_kind="kde",
        plot_kws={"alpha": .25},
        height=2
    )
    _grid.figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Obey The Law:
    - **parameters are not independent**
    - **parameters cannot be interpreted in isolation**

    Instead...**Push out posterior predictions**

    Below, we again show:
    - how the posterior is "made of lines" -- there are an infinite number of possible lines that can be drawn from the posterior
    - confidence intervals can be established to communicate the uncertainty of the posterior's fit to the data
    """)
    return


@app.cell(hide_code=True)
def _(
    ADULT_HOWELL,
    howell_inference,
    howell_model,
    plot_linear_regression_posterior_predictive,
):
    _fig, _ax = plt.subplots(figsize=(6, 6))
    plt.scatter(ADULT_HOWELL.height.values, ADULT_HOWELL.weight.values, s=80, zorder=20, alpha=.5)
    plot_linear_regression_posterior_predictive(
        howell_model,
        howell_inference,
        min_height=135,
        max_height=175,
        min_weight=30,
        max_weight=65
    )
    plt.title(f"N={len(ADULT_HOWELL)}")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
