import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import inspect
    import time

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
      - unreasonably good approximations, despite it always being incorrect
      - can be used as a cog in a causal analysis system, despite being an inaccurate mechanistic model of real phenomena
    - **Gaussian**
      - General error model
      - Abstracts away details, allowing us to make macro inferences, without having to incorporate micro phenomena



    # Why Normal?
    ## Two arguments

    1. Generative: summed fluctuations (see below) tend toward Normal distribution
    2. Inferential: For estimating mean and variance, the Normal is the least informative (fewest assumptions), in the maximum entropy sense

    > Variables do not need to be Normally-distributed in order to estimate the correct mean and variance using a Gaussian error model.

    ## Generating Normal distribution from summation of decisions
    - Simulate a group of people randomly walking left-right, starting from a central location
    - Resulting positions are the summation of many left and right steps -- the result is Normally-distributed
    - Normal distribution falls out of processes where deviations are summed (also products)
    """)
    return


@app.cell(hide_code=True)
def random_walk_normal():
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
    ### Try it: summed steps become Normal
    Change how many steps each of 10,000 people takes, and how long each step is. With only a few steps the positions are lumpy; as steps add up, their distribution approaches the Normal curve with standard deviation $\text{step size} \times \sqrt{\text{steps}}$.
    """)
    return


@app.cell(hide_code=True)
def walk_controls():
    walk_n_steps = mo.ui.slider(steps=[1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000], value=5, label="Steps per person", show_value=True)
    walk_step_size = mo.ui.slider(0.1, 1, step=0.1, value=0.5, label="Step size", show_value=True)
    return walk_n_steps, walk_step_size


@app.cell(hide_code=True)
def walk_explorer(walk_n_steps, walk_step_size):
    _rng = np.random.default_rng(1)
    _n_people = 10_000
    _n_steps, _step_size = walk_n_steps.value, walk_step_size.value
    _steps = _step_size * (2 * _rng.integers(0, 2, size=(_n_people, _n_steps)) - 1)
    _positions = _steps.sum(axis=1)

    _position_unique, _position_counts = np.unique(np.round(_positions, 6), return_counts=True)
    _spacing = 2 * _step_size  # positions after n steps are 2 * step size apart
    _sd = _step_size * np.sqrt(_n_steps)
    _xs = np.linspace(-4 * _sd, 4 * _sd, 200)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    plt.bar(_position_unique, _position_counts / (_n_people * _spacing), width=0.8 * _spacing, color='C0', label='positions')
    plt.plot(_xs, stats.norm(0, _sd).pdf(_xs), color='k', linewidth=2, label=f'Normal(0, {_sd:.2f})')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlim([-4 * _sd - _spacing, 4 * _sd + _spacing])
    plt.yticks([])
    plt.xlabel('position')
    plt.legend()
    plt.title(f'{_n_people:,} people after {_n_steps} steps')

    mo.vstack([mo.hstack([walk_n_steps, walk_step_size], justify="start", gap=2), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Drawing the Owl
    1. State clear **question**
    2. Sketch causal **assumptions** -- DAG
    3. Define a **generative model** -- generate data
    4. Use generative model to build (AND TEST) an **estimator**
    5. Profit (through analyzing real data, possibly gaining info to iterate on the model)

    # Linear Regression

    ## Howell Dataset

    ### (1) Question & Estimand
    - Describe the association between **weight** and **height**
    - We'll focus on **adult weight** -- the relationship between adult height and weight is approximately linear
    """)
    return


@app.cell
def load_howell():
    HOWELL = utils.load_data('Howell1')
    ADULT_HOWELL = HOWELL[HOWELL.age >= 18]
    HOWELL.head()
    return ADULT_HOWELL, HOWELL


@app.cell(hide_code=True)
def howell_scatter(ADULT_HOWELL, HOWELL):
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
def weight_dag():
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
    W_i &= \beta H_i + U_i \\
    U_i &\sim \text{Normal}(0, \sigma) \\
    H_i &\sim \text{Uniform}(130, 170)
    \end{align}
    $$

    ### Describing models
    - Variables on the left
    - Definition on right
    - $\sim$ indicates sampling from a distribution
      - e.g. $H_i \sim \text{Uniform}(130, 170)$ is definition that _height is distributed uniformly between 130 and 170_
    - $=$ indicates expectation or (deterministic) equality
      - e.g. $W_i = \beta H_i + U_i$ is the deterministic equation for _weight_
    - subscript $i$ indicates the index of an observation/individual
    - generally code will be written in opposite direction, because you need variables defined in order to be referenced/composed
    """)
    return


@app.function
def simulate_weight(H, beta, sigma):
    """
    Generative model described above, simulate weight given height, H,
    proportional coefficient beta, and the standard deviation of
    unobserved (Normally-distributed) noise, sigma.
    """
    n_heights = len(H)

    # unobserved noise
    U = stats.norm.rvs(0, sigma, size=n_heights)
    return beta * H + U


@app.cell
def simulate_data():
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
def simulated_data_plot(H, W):
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
    p(\alpha, \beta, \sigma | W_i) = \frac{p(W_i | \alpha, \beta, \sigma) p(\alpha, \beta, \sigma)}{Z}
    $$

    - The only estimator in Bayesian data analysis

    - $p(\alpha, \beta, \sigma | W_i)$ -- **Posterior**: Probability of a specific line (model)
    - $p(W_i | \alpha, \beta, \sigma)$ -- **Likelihood**: The number of ways the generative process (line) could have produced the data
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
    For the following grid approximation simulation, we'll use a utility function `utils.simulate_2_parameter_bayesian_learning` for simulating general Bayesian posterior updates. For the API, see `utils.py`
    """)
    return


@app.cell
def grid_learning_docs():
    mo.plain_text(inspect.getdoc(utils.simulate_2_parameter_bayesian_learning))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Functions for `simulate_2_parameter_bayesian_learning`
    """)
    return


@app.function
def linear_model(x, intercept, slope):
    return intercept + slope * x


@app.function
def linear_regression_posterior(x_obs, y_obs, intercept_grid, slope_grid, likelihood_prior_std = 1.0):

    # Convert params to 1-d arrays
    if np.ndim(intercept_grid) > 1:
        intercept_grid = intercept_grid.ravel()

    if np.ndim(slope_grid) > 1:
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
def grid_data():
    # Generate standardized regression data for demo
    np.random.seed(123)
    GRID_RESOLUTION = 100
    GRID_N_DATA_POINTS = 64

    # Ground truth parameters
    GRID_SLOPE = 0.5
    GRID_INTERCEPT = -1

    grid_x = stats.norm().rvs(size=GRID_N_DATA_POINTS)
    grid_y = GRID_INTERCEPT + GRID_SLOPE * grid_x + stats.norm.rvs(size=GRID_N_DATA_POINTS) * 0.25

    slope_grid = np.linspace(-2, 2, GRID_RESOLUTION)
    intercept_grid = np.linspace(-2, 2, GRID_RESOLUTION)
    return (
        GRID_INTERCEPT,
        GRID_N_DATA_POINTS,
        GRID_SLOPE,
        grid_x,
        grid_y,
        intercept_grid,
        slope_grid,
    )


@app.cell(hide_code=True)
def grid_controls(GRID_N_DATA_POINTS):
    get_grid_n_samples, set_grid_n_samples = mo.state(0)


    def _add_grid_points(n):
        set_grid_n_samples(lambda current: min(current + n, GRID_N_DATA_POINTS))


    grid_add_one_button = mo.ui.button(label="+1 point", on_click=lambda _: _add_grid_points(1))
    grid_add_eight_button = mo.ui.button(label="+8 points", on_click=lambda _: _add_grid_points(8))
    grid_reset_button = mo.ui.button(label="Reset", on_click=lambda _: set_grid_n_samples(0))
    return (
        get_grid_n_samples,
        grid_add_eight_button,
        grid_add_one_button,
        grid_reset_button,
    )


@app.cell
def grid_learning(
    GRID_INTERCEPT,
    GRID_N_DATA_POINTS,
    GRID_SLOPE,
    get_grid_n_samples,
    grid_add_eight_button,
    grid_add_one_button,
    grid_reset_button,
    grid_x,
    grid_y,
    intercept_grid,
    slope_grid,
):
    # Add data points to show how the posterior adapts to more and more data
    grid_n_samples = get_grid_n_samples()
    utils.simulate_2_parameter_bayesian_learning(
        x_obs=grid_x[:grid_n_samples],
        y_obs=grid_y[:grid_n_samples],
        param_a_grid=intercept_grid,
        param_b_grid=slope_grid,
        true_param_a=GRID_INTERCEPT,
        true_param_b=GRID_SLOPE,
        model_func=linear_model,
        posterior_func=linear_regression_posterior,
        param_labels=['intercept', 'slope'],
        data_range_x=(-3, 3), data_range_y=(-3, 3)
    )
    mo.vstack([
        mo.hstack([grid_add_one_button, grid_add_eight_button, grid_reset_button], justify="start"),
        mo.md(f"**Data points used:** {grid_n_samples} / {GRID_N_DATA_POINTS}"),
        plt.gcf(),
    ])
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
    - For example, when Height is 0, Weight should be 0, right?
    - Weight should increase (on average) with height -- i.e. $\beta > 0$
    - Weight (kg) should be less than Height (cm)
    - variances should be positive

    $$
    \begin{align}
    \alpha &\sim \text{Normal}(0, 10) \\
    \beta &\sim \text{Uniform}(0, 1) \\
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
def prior_predictive():
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
    ### Try it: explore the priors
    Change the priors and see which lines they allow before seeing any data. The dashed lines mark two things we know: weight can't be negative, and an adult's weight in kg is less than their height in cm.
    """)
    return


@app.cell(hide_code=True)
def prior_controls():
    prior_alpha_sd = mo.ui.slider(1, 50, value=10, label=r"$\alpha \sim \text{Normal}(0, s)$, $s$", show_value=True)
    prior_beta_family = mo.ui.dropdown(["Uniform(0, b)", "Normal(0, b)"], value="Uniform(0, b)", label=r"$\beta$ prior")
    prior_beta_scale = mo.ui.slider(0.1, 5, step=0.1, value=1, label="$b$", show_value=True)
    prior_n_lines = mo.ui.slider(10, 300, step=10, value=100, label="Lines", show_value=True)
    return prior_alpha_sd, prior_beta_family, prior_beta_scale, prior_n_lines


@app.cell(hide_code=True)
def prior_explorer(
    prior_alpha_sd,
    prior_beta_family,
    prior_beta_scale,
    prior_n_lines,
):
    _rng = np.random.default_rng(1)
    _n_lines = prior_n_lines.value
    _alphas = _rng.normal(0, prior_alpha_sd.value, _n_lines)
    if prior_beta_family.value.startswith("Uniform"):
        _betas = _rng.uniform(0, prior_beta_scale.value, _n_lines)
    else:
        _betas = _rng.normal(0, prior_beta_scale.value, _n_lines)

    _heights = np.linspace(130, 170, 3)
    _weights = _alphas[:, None] + _betas[:, None] * _heights
    _implausible = ((_weights < 0) | (_weights > _heights)).any(axis=1).mean()

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    for _w in _weights:
        plt.plot(_heights, _w, color='C0', alpha=0.5)
    plt.axhline(0, color='k', linestyle='--')
    plt.plot(_heights, _heights, color='k', linestyle='--')
    plt.xlabel('height, H (cm)')
    plt.ylabel('E[W | H] (kg)')
    plt.ylim((-100, 250))
    plt.title(f"{_implausible:.0%} of lines leave the plausible range")

    mo.vstack([
        mo.hstack([prior_alpha_sd, prior_n_lines], justify="start", gap=2),
        mo.hstack([prior_beta_family, prior_beta_scale], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulation-based Validation & Calibration
    - Simulate data with varying parameters
    - Vary data-generating parameters (e.g. slope) that are analogous to the model; make sure the estimator tracks
    - Make sure that at large sample sizes, data-generating parameters can be recovered
    - Same for confounds/unknowns
    """)
    return


@app.cell
def fit_by_sample_size(H, W):
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
def posterior_predictive_helper(MAX_HEIGHT, MIN_HEIGHT):
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
        plt.xlabel('height, H (cm)')
        plt.ylim([min_weight, max_weight])
        plt.yticks(np.arange(min_weight, max_weight, 10))
        plt.ylabel('weight, W (kg)')

    return (plot_linear_regression_posterior_predictive,)


@app.cell(hide_code=True)
def sample_size_posteriors(
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
    ### Try it: can the estimator recover the truth?
    Pick the true slope $\beta$, the noise $\sigma$ and the sample size; the cells below simulate data from the generative model and fit the same PyMC model as above. The fit runs when you release a slider, and takes a few seconds the first time for each setting.
    """)
    return


@app.cell(hide_code=True)
def recovery_controls():
    recovery_beta = mo.ui.slider(0, 1, step=0.05, value=0.5, label=r"True $\beta$", show_value=True, debounce=True)
    recovery_sigma = mo.ui.slider(1, 9, step=0.5, value=5, label=r"True $\sigma$", show_value=True, debounce=True)
    recovery_n = mo.ui.slider(steps=[2, 5, 10, 20, 50, 100, 200, 500], value=20, label="Sample size", show_value=True, debounce=True)
    mo.hstack([recovery_beta, recovery_sigma, recovery_n], justify="start", gap=2)
    return recovery_beta, recovery_n, recovery_sigma


@app.cell
def recovery_fit(
    MAX_HEIGHT,
    MIN_HEIGHT,
    recovery_beta,
    recovery_n,
    recovery_sigma,
):
    np.random.seed(4)
    recovery_H = stats.uniform.rvs(MIN_HEIGHT, MAX_HEIGHT - MIN_HEIGHT, size=recovery_n.value)
    recovery_W = simulate_weight(recovery_H, beta=recovery_beta.value, sigma=recovery_sigma.value)

    with pm.Model() as recovery_model:
        # Same priors as the model above
        _sigma = pm.Uniform('sigma', 0, 10)
        _alpha = pm.Normal('alpha', 0, 10)
        _beta = pm.Uniform('beta', 0, 1)

        _mu = _alpha + _beta * recovery_H
        pm.Normal('W_obs', _mu, _sigma, observed=recovery_W)
        recovery_inference = cached_sample(target_accept=0.99, progressbar=False)
    return recovery_H, recovery_W, recovery_inference


@app.cell(hide_code=True)
def recovery_plot(
    MAX_HEIGHT,
    MIN_HEIGHT,
    recovery_H,
    recovery_W,
    recovery_beta,
    recovery_inference,
    recovery_n,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4))

    plt.sca(_axs[0])
    _xs = np.linspace(MIN_HEIGHT, MAX_HEIGHT, 30)
    _posterior = recovery_inference.posterior.sel(chain=0)
    for _a, _b in zip(_posterior['alpha'].values[:20], _posterior['beta'].values[:20]):
        plt.plot(_xs, _a + _b * _xs, color='k', alpha=0.3)
    plt.scatter(recovery_H, recovery_W, s=60, alpha=0.5, zorder=20)
    plt.xlabel('height, H (cm)')
    plt.ylabel('weight, W (kg)')
    plt.title(f'N={recovery_n.value}')

    plt.sca(_axs[1])
    utils.plot_dist(recovery_inference.posterior['beta'], label='posterior')
    plt.axvline(recovery_beta.value, color='k', linestyle='--', label=r'true $\beta$')
    plt.xlim([0, 1])
    plt.xlabel(r'$\beta$')
    plt.legend()
    plt.title(r'Posterior of $\beta$')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## (5) Analyse the data
    """)
    return


@app.cell
def howell_fit(ADULT_HOWELL):
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
def howell_summary(howell_inference):
    az.summary(howell_inference)
    return


@app.cell(hide_code=True)
def howell_pairplot(howell_inference):
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
def howell_posterior_predictive(
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
    mo.md(r"""
    ## Bonus: vectorizing the grid posterior

    `linear_regression_posterior` above loops over every point of the parameter grid. With a $100 	imes 100$ grid that is 10,000 iterations, and each one builds a new `stats.norm(...)` object, so a single posterior takes several seconds even when there is no data.

    NumPy *broadcasting* evaluates the whole grid in one call. Give the grid parameters a column axis (`[:, None]`) and the data a row axis (`[None, :]`); the likelihood is then a matrix with **one row per grid point and one column per observation**. Summing each row gives the log-likelihood of each $(lpha, eta)$ pair, exactly what the loop computes one pair at a time.

    After the timing comparison, the grid-learning demo runs again with this version. It is fast enough to follow a slider as you drag it.
    """)
    return


@app.function
def linear_regression_posterior_vectorized(x_obs, y_obs, intercept_grid, slope_grid, likelihood_prior_std=1.0):
    intercept_grid = np.ravel(intercept_grid)
    slope_grid = np.ravel(slope_grid)

    log_prior_intercept = stats.norm(0, 1).logpdf(intercept_grid)
    log_prior_slope = stats.norm(0, 1).logpdf(slope_grid)

    # Shape (n_grid_points, n_observations): row i holds the predicted mean of every
    # observation under the i-th (intercept, slope) pair
    predicted_mean = intercept_grid[:, None] + slope_grid[:, None] * x_obs[None, :]
    log_likelihood = stats.norm(predicted_mean, likelihood_prior_std).logpdf(y_obs[None, :]).sum(axis=1)

    log_posterior = log_likelihood + log_prior_intercept + log_prior_slope
    return np.exp(log_posterior - log_posterior.max())


@app.cell
def bonus_vectorized_comparison(grid_x, grid_y, intercept_grid, slope_grid):
    # Same posterior, computed both ways on all 64 data points of the grid demo
    _intercepts, _slopes = np.meshgrid(intercept_grid, slope_grid)

    _start = time.perf_counter()
    _loop_posterior = linear_regression_posterior(grid_x, grid_y, _intercepts, _slopes)
    _loop_seconds = time.perf_counter() - _start

    _start = time.perf_counter()
    _vectorized_posterior = linear_regression_posterior_vectorized(grid_x, grid_y, _intercepts, _slopes)
    _vectorized_seconds = time.perf_counter() - _start

    mo.md(f"""
    | Version | Time |
    |---|---|
    | loop | {_loop_seconds:.2f} s |
    | vectorized | {_vectorized_seconds * 1000:.1f} ms |

    Largest difference between the two posteriors: {np.abs(_loop_posterior - _vectorized_posterior).max():.1e}
    """)
    return


@app.cell(hide_code=True)
def bonus_grid_controls(GRID_N_DATA_POINTS):
    bonus_grid_n_samples = mo.ui.slider(
        0, GRID_N_DATA_POINTS, value=0, label="Number of data points", show_value=True, full_width=True
    )
    return (bonus_grid_n_samples,)


@app.cell
def bonus_grid_learning(
    GRID_INTERCEPT,
    GRID_SLOPE,
    bonus_grid_n_samples,
    grid_x,
    grid_y,
    intercept_grid,
    slope_grid,
):
    # The grid-learning demo again, now with the vectorized posterior: drag the slider
    utils.simulate_2_parameter_bayesian_learning(
        x_obs=grid_x[:bonus_grid_n_samples.value],
        y_obs=grid_y[:bonus_grid_n_samples.value],
        param_a_grid=intercept_grid,
        param_b_grid=slope_grid,
        true_param_a=GRID_INTERCEPT,
        true_param_b=GRID_SLOPE,
        model_func=linear_model,
        posterior_func=linear_regression_posterior_vectorized,
        param_labels=['intercept', 'slope'],
        data_range_x=(-3, 3), data_range_y=(-3, 3)
    )
    mo.vstack([bonus_grid_n_samples, plt.gcf()])
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
