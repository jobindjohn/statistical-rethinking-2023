import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from functools import partial

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from matplotlib import pyplot as plt
    from seaborn import pairplot

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 16 - Gaussian Processes](https://youtu.be/Y2ZLt4iOrXU?si=NN2oor2yEz3YSoLY)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Returning to Oceanic Technology Dataset

    We'll now deal with some of the confounds we glossed over, now that we have some more advanced statistical tools
    """)
    return


@app.cell
def _():
    KLINE = utils.load_data("Kline2")
    KLINE
    return (KLINE,)


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("P", "T"),
            ("C", "T"),
            ("U", "T"),
            ("U", "P")
        ],
        node_props={
            "U": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Total tools in society $T$ (outcome)
    - Populuation $P$ (treatment)
    - Contact level $C$
    - Unobserved confounds $U$, e.g.
      - materials
      - weather

    There should be **spatial covariation**:
    - islands closer to one another should have similar unobserved confounds
    - islands should share historical context
    - we don't have to know the exact dymanics of the contexts, we can handle statistically at a macro scale

    ## Review of Techonological Innovation/Loss model

    - Derived from steady state of difference equation $\Delta T = \alpha P^ \beta - \gamma T$
      - $\alpha$ is rate of innovation
      - $P$ is population
      - $\beta$ is elasticity rate (diminishing returns)
      - $\gamma$ is loss (e.g. forgetting) rate over time
    - Provides the expected number of tools over the long-run:
      - $\hat T = \frac{\alpha P ^ \beta}{\gamma}$
      - use $\lambda= \hat T$ as the expected rate parameter for a $\text{Poisson}(\lambda)$ distribution
    - **How can we encode spatial covariance in this model?**

    ## Let's start of with an easier model that ignores population

    $$
    \begin{align*}
        T_i &\sim \text{Poisson}(\lambda_i) \\
        \lambda_i &= \bar \alpha + \alpha_{S[i]} \\
        \begin{pmatrix}
            \alpha_1 \\
            \alpha_2 \\
            \vdots \\
            \alpha_{10}
        \end{pmatrix} &= \text{MVNormal}
        \left(
            \begin{bmatrix}
                0 \\
                0 \\
                \vdots \\
                0
            \end{bmatrix}, \textbf{K}
        \right)
    \end{align*}
    $$

    - $\alpha_{S[i]}$ are the variable intercepts for each society
      - These will have covariation based on their spatial distance from one another
      - closer societies have similar offsets
    - $\mathbf{K}$ the covariance amongst societies
      - models covariance as a function distance
      - though $\mathbf{K}$ has many parameters (45 covariances), we should be able to leverage spatial regularity to estimate far fewer effective parameters. This is good b.c. we only have 10 data points 😅
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Gaussian Processes (in the abstract)

    - Uses a **kernel function** to generate covariance matrices
      - Requires far fewer parameters than the covariance matrix entries
      - Can generate covariance matrices of any dimenions (i.e. "infinite dimensional generalization of the `MNNormal`")
      - Can generate a prediction for any point

    - Kernel function calculates the covariance of two points based on metric of comparison, e.g.
      - spatial distance
      - difference in time
      - difference in age
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Different Kernel Functions
    """)
    return


@app.function
def plot_kernel_function(
    kernel_function,
    max_distance=1,
    resolution=100,
    label=None,
    ax=None,
    **line_kwargs
):
    """Helper to plot a kernel function"""
    X = np.linspace(0, max_distance, resolution)[:, None]
    covariance = kernel_function(X, X)
    distances = np.linspace(0, max_distance, resolution)
    if ax is not None:
        plt.sca(ax)
    utils.plot_line(distances, covariance[0, :], label=label, **line_kwargs)
    plt.xlim([0, max_distance])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("|X1-X2|")
    plt.ylabel("covariance")
    if label is not None:
        plt.legend()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The Quadratic (L2) Kernel Function
    - aka RGB
    - aka Gaussian Kernel

    $$
    K(x_i, x_j) = \eta^2 \exp \left(- \frac{(x_i - x_j)^2}{\sigma^2} \right)
    $$
    """)
    return


@app.cell
def _():
    def quadratic_distance_kernel(X0, X1, eta=1, sigma=.5):
        # Use linear algebra identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        X0_norm = np.sum(X0 ** 2, axis=-1)
        X1_norm = np.sum(X1 ** 2, axis=-1)
        squared_distances = X0_norm[:, None] + X1_norm[None, :] - 2 * X0 @ X1.T
        rho = 1 / sigma ** 2
        return eta ** 2 * np.exp(-rho * squared_distances)

    plot_kernel_function(quadratic_distance_kernel)
    plt.gcf()
    return (quadratic_distance_kernel,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Ornstein-Uhlenbeck Kernel

    $$
    \begin{align*}
    K(x_i, x_j) &= \eta^2 \exp \left(- \frac{|x_i - x_j|}{\sigma} \right) \\
    \sigma &= \frac{1}{\rho}
    \end{align*}
    $$
    """)
    return


@app.cell
def _():
    def ornstein_uhlenbeck_kernel(X0, X1, eta_squared=1, rho=4):
        distances = np.abs(X1[None, :] - X0[:, None])
        return eta_squared * np.exp(-rho * distances)

    plot_kernel_function(ornstein_uhlenbeck_kernel)
    plt.gcf()
    return (ornstein_uhlenbeck_kernel,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Periodic Kernel

    $$
    K(x_i, x_j) = \eta^2 \exp \left(- \frac{\sin^2((x_i - x_j) / p)}{\sigma^2} \right)
    $$

    with additional periodicity parameter, $p$
    """)
    return


@app.cell
def _():
    def periodic_kernel(X0, X1, eta=1, sigma=1, periodicity=.5):
        distances = np.sin((X1[None, :] - X0[:, None]) * periodicity) ** 2
        rho = 2 / sigma ** 2
        return eta ** 2 * np.exp(-rho * distances)

    plot_kernel_function(periodic_kernel, max_distance=6)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Gaussian Process Prior

    Using the Gaussian/L2 Kernel Function to generate our covariance $\mathbf{K}$, we can adjust a few parameters

    - Varying $\sigma$ controls the bandwidth of the kernel function
      - smaller $\sigma$ makes covariance fall off quickly with space
      - larger $\sigma$ allow covariance to extend larger distances

    - Varying $\eta$ controls the maximum degree of covariance

    #### 1-D Examples
    Below we draw functions from a 1-D Guassian process, varying either $\sigma$ or $\eta$ to demonstrate the effect of the parameters on the spatial correlation of the samples.
    """)
    return


@app.function
def plot_gaussian_process(
    X,
    samples=None,
    mean=None,
    cov=None,
    X_obs=None,
    Y_obs=None,
    uncertainty_prob=.89
):
    X = X.ravel()

    # Plot GP samples
    for ii, sample in enumerate(samples):
        label = "GP samples" if not ii else None
        utils.plot_line(X, sample, color=f"C{ii}", linewidth=1, label=label)

    # Add GP mean, if provided
    if mean is not None:
        mean = mean.ravel()
        utils.plot_line(X, mean, color='k', label='GP mean')

        # Add uncertainty around mean; requires covariance matrix
        if cov is not None:
            z = stats.norm.ppf(1 - (1 - uncertainty_prob) / 2)
            uncertainty = z * np.sqrt(np.diag(cov))
            plt.fill_between(
                X,
                mean + uncertainty,
                mean - uncertainty,
                alpha=0.1,
                color='gray',
                zorder=1,
                label='GP uncertainty'
            )

    # Add any training data points, if provided
    if X_obs is not None:
        utils.plot_scatter(X_obs, Y_obs, color='k', label='observations', zorder=100, alpha=1)

    plt.xlim([X.min(), X.max()])
    plt.ylim([-5, 5])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


@app.function
def plot_gaussian_process_prior(kernel_function, n_samples=3, figsize=(10, 5), resolution=100):
    X = np.linspace(-5, 5, resolution)[:, None]

    prior = gaussian_process_prior(X, kernel_function)
    samples = prior.rvs(n_samples)

    _, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])
    plot_gaussian_process(X, samples=samples)

    plt.sca(axs[1])
    plot_kernel_function(kernel_function, color='k')
    plt.title(f"kernel function")
    return axs


@app.function
def gaussian_process_prior(X_pred, kernel_function):
    """Initializes a Gaussian Process prior distribution for provided Kernel function"""
    mean = np.zeros(X_pred.shape).ravel()
    cov = kernel_function(X_pred, X_pred)
    return stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Varying $\sigma$
    """)
    return


@app.cell(hide_code=True)
def _(quadratic_distance_kernel):
    _eta = 1
    _figs = []
    for _sigma in [.25, .5, 1]:
        _kernel_function = partial(quadratic_distance_kernel, eta=_eta, sigma=_sigma)
        _axs = plot_gaussian_process_prior(_kernel_function, n_samples=5)
        _axs[0].set_title(f"prior: $\\eta$={_eta}; $\\sigma=${_sigma}")
        _figs.append(_axs[0].figure)
    mo.vstack(_figs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Varying $\eta$
    """)
    return


@app.cell
def _(quadratic_distance_kernel):
    sigma = .5
    _figs = []
    for eta in [.25, .75, 1]:
        _kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
        _axs = plot_gaussian_process_prior(_kernel_function, n_samples=5)
        _axs[0].set_title(f"prior: $\\eta$={eta}; $\\sigma=${sigma}")
        _figs.append(_axs[0].figure)
    mo.vstack(_figs)
    return eta, sigma


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Gaussian Process Posterior

    As the model observes data, it adjusts the posterior to account for that data while also incorporating the smoothness constraints of the prior.
    """)
    return


@app.function
def gaussian_process_posterior(
    X_obs,
    Y_obs,
    X_pred,
    kernel_function,
    sigma_y=1e-6,
    smoothing_factor=1e-6
):
    """Initializes a Gaussian Process Posterior distribution"""

    # Observation covariance
    K_obs = kernel_function(X_obs, X_obs)
    K_obs_noise = sigma_y ** 2 * np.eye(len(X_obs))
    K_obs += K_obs_noise
    K_obs_inv = np.linalg.inv(K_obs)

    # Prediction grid covariance
    K_pred = kernel_function(X_pred, X_pred)
    K_pred_smooth = smoothing_factor * np.eye(len(X_pred))
    K_pred += K_pred_smooth

    # Covariance between observations and prediction grid
    K_obs_pred = kernel_function(X_obs, X_pred)

    # Posterior
    posterior_mean = K_obs_pred.T.dot(K_obs_inv).dot(Y_obs)
    posterior_cov = K_pred - K_obs_pred.T.dot(K_obs_inv).dot(K_obs_pred)

    return stats.multivariate_normal(
        mean=posterior_mean.ravel(),
        cov=posterior_cov,
        allow_singular=True
    )


@app.function
def plot_gaussian_process_posterior(
    X_obs,
    Y_obs,
    X_pred,
    kernel_function,
    sigma_y=1e-6,
    n_samples=3,
    figsize=(10, 5),
    resolution=100
):
    X = np.linspace(-5, 5, resolution)[:, None]

    posterior = gaussian_process_posterior(X_obs, Y_obs, X, kernel_function, sigma_y=sigma_y)
    samples = posterior.rvs(n_samples)

    _, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])

    plot_gaussian_process(
        X_pred,
        samples=posterior.rvs(size=n_samples),
        mean=posterior.mean,
        cov=posterior.cov,
        X_obs=X_obs,
        Y_obs=Y_obs
    )

    plt.sca(axs[1])
    plot_kernel_function(kernel_function, color='k')
    plt.title(f"kernel function")
    return axs


@app.cell
def _(eta, quadratic_distance_kernel, sigma):
    # Generate some training data
    _X_pred = np.linspace(-5, 5, 100)[:, None]
    _X_obs = np.linspace(-4, 4, 4)[:, None]
    _Y_obs = np.sin(_X_obs) ** 2 + _X_obs

    # Initialize the kernel function
    _sigma_y = .25
    _sigma_kernel = .75
    _eta_kernel = 1
    _kernel_function = partial(quadratic_distance_kernel, eta=_eta_kernel, sigma=_sigma_kernel)

    # Plot posterior
    _axs = plot_gaussian_process_posterior(_X_obs, _Y_obs, _X_pred, _kernel_function, sigma_y=_sigma_y, n_samples=3)
    _axs[0].set_title(f"posterior")
    _axs[1].set_title(f"kernel function:\n$\\eta$={eta}; $\\sigma=${sigma}")
    _axs[0].figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Distance-based model

    $$
    \begin{align}
    T_i &\sim \text{Poisson}(\lambda_i) \\
    \log(\lambda_i) &= \bar \alpha + \alpha_{S[i]} \\
    \begin{pmatrix}
    \alpha_1 \\
    \alpha_2 \\
    \vdots \\
    \alpha_{10}
    \end{pmatrix} &= \text{MVNormal}\left(
    \begin{bmatrix}
    0 \\
    0 \\
    \vdots \\
    0
    \end{bmatrix}, \textbf{K}
    \right) \\
    k_{i,j} &= \eta^2\exp(-\rho^2D_{i,j}^2)) \\
    \eta^2 &\sim \text{Exponential}(2) \\
    \rho^2 &\sim \text{Exponential}(0.5)
    \end{align}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Islands Distance Matrix
    """)
    return


@app.cell
def _():
    # Distance matrix D_{i, j}
    ISLANDS_DISTANCE_MATRIX = utils.load_data('islands_distance_matrix')
    ISLANDS_DISTANCE_MATRIX.set_index(ISLANDS_DISTANCE_MATRIX.columns, inplace=True)
    ISLANDS_DISTANCE_MATRIX.style.format(precision=1).background_gradient(cmap='viridis').set_caption("Distance in Thousands of km")
    return (ISLANDS_DISTANCE_MATRIX,)


@app.cell
def _(ISLANDS_DISTANCE_MATRIX, KLINE):
    # Data / coords
    CULTURE_ID, CULTURE = pd.factorize(KLINE.culture.values)
    ISLAND_DISTANCES = ISLANDS_DISTANCE_MATRIX.values.astype(float)
    TOOLS = KLINE.total_tools.values.astype(int)
    coords = {"culture": CULTURE}

    with pm.Model(coords=coords) as distance_model:

        # Priors
        _alpha_bar = pm.Normal("alpha_bar", 3, 0.5)
        _eta_squared = pm.Exponential("eta_squared", 2)
        _rho_squared = pm.Exponential("rho_squared", 0.5)

        # Gaussian Process
        _kernel_function = _eta_squared * pm.gp.cov.ExpQuad(input_dim=1, ls=_rho_squared)
        _GP = pm.gp.Latent(cov_func=_kernel_function)
        _alpha = _GP.prior("alpha", X=ISLAND_DISTANCES, dims="culture")

        # Likelihood
        _lambda_T = pm.math.exp(_alpha_bar + _alpha[CULTURE_ID])
        pm.Poisson("T", _lambda_T, dims='culture', observed=TOOLS)
    return CULTURE_ID, ISLAND_DISTANCES, TOOLS, coords, distance_model


@app.cell
def _(distance_model):
    pm.model_to_graphviz(distance_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Check model with prior-predictive simulation
    """)
    return


@app.cell
def _(quadratic_distance_kernel):
    def plot_predictive_covariance(predictive, n_samples=30, color='C0', label=None):

        eta_samples = predictive['eta_squared'].values[0, :n_samples] ** .5
        sigma_samples = 1 / predictive['rho_squared'].values[0, :n_samples] ** .5

        for ii, (eta, sigma) in enumerate(zip(eta_samples, sigma_samples)):
            label = label if ii == 0 else None

            kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
            plot_kernel_function(kernel_function, color=color, label=label, alpha=.5, linewidth=5, max_distance=7)

    return (plot_predictive_covariance,)


@app.cell
def _(distance_model, plot_predictive_covariance):
    with distance_model:
        distance_prior_predictive = pm.sample_prior_predictive(random_seed=12).prior

    plot_predictive_covariance(distance_prior_predictive, label='prior')
    plt.ylim([0, 2])
    plt.title("Prior Covariance Functions")
    plt.gcf()
    return (distance_prior_predictive,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model the data
    """)
    return


@app.cell
def _(distance_model):
    with distance_model:
        distance_inference = cached_sample(target_accept=.99)
    return (distance_inference,)


@app.cell
def _(distance_inference):
    az.summary(distance_inference, var_names=["~alpha_rotated_"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posterior Predictions

    #### Compare the Prior to the Posterior

    Ensure that we've learned something
    """)
    return


@app.cell(hide_code=True)
def _(
    distance_inference,
    distance_prior_predictive,
    plot_predictive_covariance,
):
    plot_predictive_covariance(distance_prior_predictive, color='k', label='prior')
    plot_predictive_covariance(distance_inference.posterior, label='posterior')
    plt.ylim([0, 2])
    plt.gcf()
    return


@app.cell
def _(ISLANDS_DISTANCE_MATRIX, ISLAND_DISTANCES, KLINE):
    def distance_to_covariance(distance, eta_squared, rho_squared):
        return eta_squared * np.exp(-rho_squared * distance ** 2)


    def calculate_posterior_mean_covariance_matrix(inference):
        posterior_mean = inference.posterior.mean(dim=('chain', 'draw'))

        posterior_eta_squared = posterior_mean['eta_squared'].values
        posterior_rho_squared = posterior_mean['rho_squared'].values

        print("Posterior Mean Kernel parameters:")
        print("eta_squared:", posterior_eta_squared)
        print("rho_squared:", posterior_rho_squared)

        model_covariance = np.zeros_like(ISLANDS_DISTANCE_MATRIX).astype(float)
        for ii in range(10):
            for jj in range(10):
                model_covariance[ii, jj] = distance_to_covariance(
                    ISLAND_DISTANCES[ii, jj],
                    eta_squared=posterior_eta_squared,
                    rho_squared=posterior_rho_squared
                )

        return model_covariance

    def plot_posterior_mean_covariance_matrix(covariance_matrix, clim=(0, .4)):
        plt.matshow(covariance_matrix)
        plt.xticks(np.arange(10), labels=KLINE.culture, rotation=90)
        plt.yticks(np.arange(10), labels=KLINE.culture)
        plt.clim(clim)
        plt.grid(False)
        plt.colorbar()
        plt.title("posterior mean covariance matrix");

    return (
        calculate_posterior_mean_covariance_matrix,
        plot_posterior_mean_covariance_matrix,
    )


@app.cell
def _(
    calculate_posterior_mean_covariance_matrix,
    distance_inference,
    plot_posterior_mean_covariance_matrix,
):
    distance_model_covariance = calculate_posterior_mean_covariance_matrix(distance_inference)
    plot_posterior_mean_covariance_matrix(distance_model_covariance)
    plt.gcf()
    return (distance_model_covariance,)


@app.cell
def _(KLINE, distance_model_covariance):
    def plot_kline_model_covarance(covariance_matric, min_alpha=.01, alpha_gain=1, max_cov=None):
        plt.subplots(figsize=(10, 5))
        # Plot covariance
        max_cov = covariance_matric.max() if max_cov is None else max_cov
        for ii in range(10):
            for jj in range(10):
                if ii != jj:
                    lat_a = KLINE.loc[ii, 'lat']
                    lon_a = KLINE.loc[ii, 'lon2']

                    lat_b = KLINE.loc[jj, 'lat']
                    lon_b = KLINE.loc[jj, 'lon2']
                    cov = covariance_matric[ii, jj]
                    alpha = (min_alpha + (1 - min_alpha) * (cov / max_cov)) ** (1 / alpha_gain)
                    plt.plot((lon_a, lon_b), (lat_a, lat_b), linewidth=3, color='k', alpha=alpha)

        plt.scatter(KLINE.lon2, KLINE.lat, s=KLINE.population * .01, zorder=10, alpha=.9)

        # Annotate
        for ii in range(10):
            plt.annotate(
                KLINE.loc[ii, 'culture'],
                (KLINE.loc[ii, 'lon2'] + 1.5, KLINE.loc[ii, 'lat']),
                zorder=11,
                color='C1',
                fontsize=12,
                fontweight='bold'
            )
        plt.xlabel("longitude")
        plt.ylabel("latitude");
        plt.axis('tight')

    plot_kline_model_covarance(distance_model_covariance, max_cov=.4)
    plt.title("Pure Spatial, nothing else")
    plt.gcf()
    return (plot_kline_model_covarance,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stratify by population size

    $$
    \begin{align}
    T_i &\sim \text{Poisson}(\lambda_i) \\
    \lambda_i &= \frac{\bar \alpha P^\beta}{\gamma} \exp(\alpha_{S[i]}) \\
        \begin{pmatrix}
            \alpha_1 \\
            \alpha_2 \\
            \vdots \\
            \alpha_{10}
        \end{pmatrix} &= \text{MVNormal}
        \left(
            \begin{bmatrix}
                0 \\
                0 \\
                \vdots \\
                0
            \end{bmatrix}, \textbf{K}
        \right) \\
        k_{i,j} &= \eta^2\exp(-\rho^2D_{i,j}^2)) \\
        \bar \alpha, \beta, \gamma &\sim \text{Exponential}(1) \\
        \eta^2 &\sim \text{Exponential}(2) \\
        \rho^2 &\sim \text{Exponential}(0.5)
    \end{align}
    $$

    - We now fold our steady state equation into the intercept/baseline tools equation
    - Include an additional $\exp$ multiplier to handle society-level offsets. Idea here being that
      - if $\alpha_{S[i]}=0$, then the society is average, and we fall back to the expected value of the difference equation i.e. multiplier is one
      - if $\alpha_{S[i]}>0$ then there is a boost in tool production. This boost is shared amongst spatially-local societies
    """)
    return


@app.cell
def _(CULTURE_ID, ISLAND_DISTANCES, KLINE, TOOLS, coords):
    with pm.Model(coords=coords) as distance_population_model:

        _population = pm.Data('log_population', KLINE.logpop.values.astype(float))
        _CULTURE_ID = pm.Data("CULTURE_ID", CULTURE_ID)

        # Priors
        _alpha_bar = pm.Exponential("alpha_bar", 1)
        _gamma = pm.Exponential("gamma", 1)
        _beta = pm.Exponential("beta", 1)
        _eta_squared = pm.Exponential("eta_squared", 2)
        _rho_squared = pm.Exponential("rho_squared", 2)

        # Gaussian Process
        _kernel_function = _eta_squared * pm.gp.cov.ExpQuad(input_dim=1, ls=_rho_squared)
        _GP = pm.gp.Latent(cov_func=_kernel_function)
        _alpha = _GP.prior("alpha", X=ISLAND_DISTANCES, dims="culture")

        # Likelihood
        _lambda_T = (_alpha_bar / _gamma * _population[_CULTURE_ID] ** _beta) * pm.math.exp(_alpha[_CULTURE_ID])
        pm.Poisson("T", _lambda_T, observed=TOOLS, dims='culture')
    return (distance_population_model,)


@app.cell
def _(distance_population_model):
    pm.model_to_graphviz(distance_population_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior Predictive
    """)
    return


@app.cell
def _(distance_population_model, plot_predictive_covariance):
    with distance_population_model:
        distance_population_prior_predictive = pm.sample_prior_predictive(random_seed=12).prior

    plot_predictive_covariance(distance_population_prior_predictive, label='prior')
    plt.ylim([0, 1])
    plt.title("Prior Covariance Functions")
    plt.gcf()
    return


@app.cell
def _(distance_population_model):
    with distance_population_model:
        distance_population_inference = cached_sample(target_accept=.99)
    return (distance_population_inference,)


@app.cell(hide_code=True)
def _(
    distance_inference,
    distance_population_inference,
    plot_predictive_covariance,
):
    plot_predictive_covariance(distance_inference.posterior, color='k', label='empty model')
    plot_predictive_covariance(distance_population_inference.posterior, label='population')
    plt.ylim([0, 1])
    plt.gcf()
    return


@app.cell
def _(
    calculate_posterior_mean_covariance_matrix,
    distance_population_inference,
    plot_posterior_mean_covariance_matrix,
):
    distance_population_model_covariance = calculate_posterior_mean_covariance_matrix(distance_population_inference)
    plot_posterior_mean_covariance_matrix(distance_population_model_covariance)
    plt.gcf()
    return (distance_population_model_covariance,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can see that by including population, the degree of covariance required to explain tool use is diminished, compared to the distance-only model. i.e. population has explained away a lot more of the variation.
    """)
    return


@app.cell(hide_code=True)
def _(distance_population_model_covariance, plot_kline_model_covarance):
    plot_kline_model_covarance(distance_population_model_covariance, max_cov=.4)
    plt.title("Stratifying by Population")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Fit population-only model for comparison
    I _think_ this is what McElreath is doing to construct the plot that combines population size, tool use and island covariance (next plot below). Specifically, we estimate a population-only model using the analytical solution, but includes no covariance amongst islands. We then compare add the covariance plot on top of that model to demonstrate that there is still some information left on the table regarding covariances, even when including population in the model.
    """)
    return


@app.cell
def _(KLINE, TOOLS, eta):
    ETA = 5.1  # Exponential hyperparmeter taken from Lecture 10 notes
    with pm.Model() as population_model:

        # Note: raw population here, not log/standardized
        _population = pm.Data("population", KLINE.logpop.values)

        # Priors
        # innovation rate
        _alpha = pm.Exponential("alpha", eta)

        # Contact-level elasticity
        _beta = pm.Exponential("beta", eta)

        # G lobal technology loss rate
        _gamma = pm.Exponential("gamma", eta)

        # Likelihood using difference equation equilibrium as mean Poisson rate
        _lamb = (_alpha * (_population ** _beta)) / _gamma
        pm.Poisson("tools", _lamb, observed=TOOLS)

        population_inference = cached_sample(tune=2000, target_accept=.98)
        population_inference = pm.sample_posterior_predictive(population_inference, extend_inferencedata=True)
    return (population_inference,)


@app.cell
def _(KLINE, distance_population_model_covariance, population_inference):
    def plot_kline_model_population_covariance(covariance_matric, min_alpha=.01, alpha_gain=1, max_cov=None):

        # Plot covariancef
        max_cov = covariance_matric.max() if max_cov is None else max_cov
        for ii in range(10):
            for jj in range(10):
                if ii != jj:
                    logpop_a = KLINE.loc[ii, 'logpop']
                    tools_a = KLINE.loc[ii, 'total_tools']

                    logpop_b = KLINE.loc[jj, 'logpop']
                    tools_b = KLINE.loc[jj, 'total_tools']
                    cov = covariance_matric[ii, jj]
                    alpha = (min_alpha + (1 - min_alpha) * (cov / max_cov)) ** (1 / alpha_gain)
                    plt.plot((logpop_a, logpop_b), (tools_a, tools_b), linewidth=3, color='k', alpha=alpha)

        plt.scatter(KLINE.logpop, KLINE.total_tools, s=KLINE.population * .01, zorder=10)

        # Annotate
        for ii in range(10):
            plt.annotate(
                KLINE.loc[ii, 'culture'],
                (KLINE.loc[ii, 'logpop'] + .1, KLINE.loc[ii, 'total_tools']),
                zorder=11,
                color='C1',
                fontsize=12,
                fontweight='bold'
            )
        plt.xlabel("log population")
        plt.ylabel("total tools")
        plt.axis('tight');


    def plot_kline_population_only_model_posterior_predictive(inference):
        ppd = inference.posterior_predictive['tools']
        utils.plot_hdi(
            x=KLINE.logpop,
            y=ppd, color="C1",
            hdi_prob=.89,
            fill_kwargs={"alpha": 0.1},
        )

        plt.plot(
            KLINE.logpop,
            ppd.mean(dim=('chain', 'draw')),
            color='C1', linewidth=4, alpha=.5,
            label='Population-only Model'
        )
        plt.legend()

    plot_kline_model_population_covariance(distance_population_model_covariance, max_cov=0.4)
    plot_kline_population_only_model_posterior_predictive(population_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Though Population alone accounts for a lot of the variance in tool use (blue curve), there's still quite a bit of residual island-island similarity that explains tool use (i.e. some of the black lines still have some weight).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Phylogenetic Regression
    ## Dataset: Primates301
    """)
    return


@app.cell
def _():
    PRIMATES301 = utils.load_data("Primates301")
    PRIMATES301.head()
    return (PRIMATES301,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - 301 Primate species
    - Life history traits
    - Body Mass $M$, Brain Size $B$, Social Group Size $G$
    - Measurement error, unobserved confounding
    - Missing data, **we'll fucus on the complete case analysis** in this lecture

    #### Filter out 151 **Complete Cases**
    Drop osbservations missing either Body Mass, Brain Size, or Group size
    """)
    return


@app.cell
def _(PRIMATES301):
    PRIMATES = PRIMATES301.query(
        "brain.notnull() and body.notnull() and group_size.notnull()",
        engine='python'
    )
    PRIMATES
    return (PRIMATES,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### There's lot's of Covariation amongst variables
    """)
    return


@app.cell(hide_code=True)
def _(PRIMATES):
    _grid = pairplot(
        PRIMATES.assign(
            log_brain_volume=lambda x: np.log(x['brain']),
            log_body_mass=lambda x: np.log(x['body']),
            log_group_size=lambda x: np.log(x['group_size']),
        )[['log_brain_volume', 'log_body_mass', 'log_group_size']]
    )
    _grid.figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    There's no way from the sample alone to know what causes what
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Causal Salad
    - Throwing all predictors into a model, then interpreting their coefficients causally
    - Using prediction methods as a stand-in for causality
    - Throwing all factors into "salad" of regression terms
        - performing model selection base on prediction (we showed earlier in [Lecture 07 - Fitting Over & Under](<Lecture 07 - Fitting Over & Under.ipynb>) why this is bad to do)
        - and interpreting the resulting coefficients as causal effects (we've shown why this is bad many times)
    - Controlling for phylogeny is important to do, but it's often done mindlessly (with the "salad" approach)
    - Regression with phylogeny still requires a causal model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Different Hypotheses required different causal models (DAGs)

    ### Social Brain Hypothesis
    - Group Size $G$ Effects Brain Size $B$

    #### Body Mass $M$ is a shared cause  for both $M$ and $B$
    This is the DAG we'll focus on in this Lecture
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("M", "G"),
            ("M", "B")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Body Mass $M$ is a mediator for both $M$ and $B$
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("G", "M"),
            ("M", "B")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Brain Size $B$ actually causes Group Size $G$; $M$ still a shared cause of both
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("B", "G"),
            ("M", "G"),
            ("M", "B")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Unobserved confounds make naive regression on traits w/out controlling for history is dangerous
    - History induces a number of unobserved shared confounds that we need to try to control for.
    - phylogenetic history can work as a proxy for these shared confounds.
    - We only have current measurements of outcome of history
      - actual phylogenies do not exist
      - different parts of a genome can have different histories

    BUT, say that we _do_ have an inferred phylogeny (as we do in this lecture), we use this phylogeny, in conjunction with Gaussian Process Regression to model the similarity amongst species as a proxy for shared history
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("M", "G"),
            ("M", "B"),
            ("u", "M"),
            ("u", "G"),
            ("u", "B"),
            ("h", "u")
        ],
        node_props={
            "h": {"label": "history\n(phylogeny)"},
            "u": {"style": "dashed", "label": "confounds, u"},
            "unobserved": {"style": "dashed"},
    
        },

        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Phylogenetic Regression
    - There's a long history of genetic evolution
    - We only get to measure the current end state of that process
    - In principle, common genetic histories should induce covariation amongst species
    - We should hopefully be able to model this covariation to average over all the possible histories (micro state) that could have led to genetic similarity (macro state)

    ### Two conjoint problems

    1. What is the history (phylogeny)?
    2. How do we use it to model causes and control for confounds?

    #### 1. What is the history (phylogeny)?

    How do we estimate or identify a phylogeny?

    ##### Difficulties
    - high degree of uncertainty
    - processes are non-stationary
      - different parts of the genome have different histories
      - crossing-over effects
    - available inferential tools
      - exploring tree space is difficult
      - repurposing software from other domains for studying biology
    - phylogenies (just like social networks) do not exist.
      - They are abstractions we use to capture regularities in data

    #### 2. How do we use it to model causes?

    ...say we've obtained a phylogeny, now what?

    ##### Approaches
    - no default approach
    - Gaussian Processes are default approach
      - use phylogeny as proxy for "genetic distance"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Two equivalent formulations of Linear Regression

    You can always re-express a linear regression with a draw from a multi-variate Normal distribution

    ### Classical Linear Regression
    The dataset $\textbf{B}$ is modeled as $D$ independent samples $B_i$ from a random a univariate normal.
    $$
    \begin{align}
    B_i &\sim \text{Normal}(\mu_i, \sigma^2) \\
    \mu_i &= \alpha + \beta_G G_i + \beta_M M_i \\
    \alpha &\sim \text{Normal}(0, 1) \\
    \beta_{G, M} &\sim \text{Normal}(0, 1) \\
    \sigma &\sim \text{Exponential}(1)
    \end{align}
    $$

    ### $\text{MVNormal}$ Formulation
    The dataset is modeled as a single $D$-vector $\textbf{B}$ sampled from a $\text{MVNormal}$. Has more of a Gaussian-process flavor to it; allows us to formulate the Linear regressoin in order to include covariation amongst the outputs.

    $$
    \begin{align}
    \textbf{B} &\sim \text{MVNormal}(\mu, \textbf{K}) \\
    \mathbf{K} &= \mathbf{I}\sigma^2 \\
    \mu_i &= \alpha + \beta_G G_i + \beta_M M_i \\
    \alpha &\sim \text{Normal}(0, 1) \\
    \beta_{G, M} &\sim \text{Normal}(0, 1) \\
    \sigma &\sim \text{Exponential}(1)
    \end{align}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Classic Linear Regression
    """)
    return


@app.cell
def _(PRIMATES):
    # Preprocessing used in the Lecture
    G = utils.standardize(np.log(PRIMATES.group_size.values))
    M = utils.standardize(np.log(PRIMATES.body.values))
    B = utils.standardize(np.log(PRIMATES.brain.values))
    return B, G, M


@app.cell
def _(B, G, M):
    with pm.Model() as vanilla_lr_model:
        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _beta_G = pm.Normal("beta_G", 0, .5)
        _beta_M = pm.Normal("beta_M", 0, .5)
        _sigma = pm.Exponential("sigma", 1)

        # Likelihood
        _mu = _alpha + _beta_G * G + _beta_M * M
        pm.Normal('B', mu=_mu, sigma=_sigma, observed=B)

        vanilla_lr_inference = cached_sample()
    return (vanilla_lr_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### $\text{MVNormal}$ Linear Regression
    """)
    return


@app.cell
def _(B, G, M):
    with pm.Model() as vector_lr_model:

        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _beta_G = pm.Normal("beta_G", 0, 0.5)
        _beta_M = pm.Normal("beta_M", 0, 0.5)
        _sigma = pm.Exponential("sigma", 1)

        # Likelihood
        _K = np.eye(len(B)) * _sigma ** 2
        _mu = _alpha + _beta_G * G + _beta_M * M
        pm.MvNormal('B', mu=_mu, cov=_K, observed=B)

        vector_lr_inference = cached_sample()
    return (vector_lr_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Verify the two formulations provide the same parameter estimates
    We're able to recover the same estimates as in the lecture
    """)
    return


@app.cell
def _(vanilla_lr_inference):
    az.summary(vanilla_lr_inference, round_to=2)
    return


@app.cell
def _(vector_lr_inference):
    az.summary(vector_lr_inference, round_to=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From Model to Kernel
    We'd like to **incorporate some residual $u_i$** into our linear regression that **adjusts the expecation in a way that encodes the shared history species**

    $$
    \begin{align}
    \textbf{B} &\sim \text{MVNormal}(\mu, \textbf{K}) \\
    \mathbf{K} &= \mathbf{I}\sigma^2 \\
    \mu_i &= \alpha + \beta_G G_i + \beta_M M_i + u_i\\
    \alpha &\sim \text{Normal}(0, 1) \\
    \beta_{G, M} &\sim \text{Normal}(0, 1) \\
    \sigma &\sim \text{Exponential}(1)
    \end{align}
    $$

    ### Phylogenetic distance as a proxy for species covariance
    - Covariance falls off with **phylogenetic distance**
      - path length between leaf nodes in tree (e.g. see dendrogram below)
    - Can use a Gaussian process in much the same way we did for island distances in the Oceanic Tools analysis
    """)
    return


@app.cell
def _(PRIMATES):
    PRIMATES_DISTANCE_MATRIX301 = utils.load_data("primates301_distance_matrix").values
    # Filter out incomplete cases
    PRIMATES_DISTANCE_MATRIX = PRIMATES_DISTANCE_MATRIX301[PRIMATES.index][:, PRIMATES.index]

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.matshow(PRIMATES_DISTANCE_MATRIX)
    _ax.grid(None)
    _fig.colorbar(_im, orientation='vertical')
    plt.xlabel('species ID')
    plt.ylabel('species ID')
    _ax.set_title("Primate Phylogenetic Distance Matrix")
    _fig
    return (PRIMATES_DISTANCE_MATRIX,)


@app.cell(hide_code=True)
def _(PRIMATES, PRIMATES_DISTANCE_MATRIX):
    # Plot the distance matrix
    _dists = squareform(PRIMATES_DISTANCE_MATRIX)
    _linkage_matrix = linkage(_dists, "single")
    _fig, _ = plt.subplots(figsize=(8, 20))
    dendrogram(
        _linkage_matrix,
        orientation='right',
        distance_sort=True,
        labels=PRIMATES.name.tolist()
    )
    plt.title("primate phylogeny")
    plt.grid(False)
    plt.xticks([])
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Kernel Function as proxy for evolutionary dynamics
    """)
    return


@app.cell
def _(ornstein_uhlenbeck_kernel):
    def brownian_motion_kernel(X0, X1, a = 1, c = 1):
        k =  1 / a * ((X0 - c) @ (X1 - c).T)[::-1, ::-1]
        return k

    _fig, _ = plt.subplots(figsize=(5, 5))
    plot_kernel_function(partial(brownian_motion_kernel, a=22000), max_distance=150, label='Linear\n"Brownian Motion"')
    plot_kernel_function(partial(ornstein_uhlenbeck_kernel, rho=1/20), max_distance=150, color='C1', label='Ornstein-Uhlenbeck\n"Damped Brownian Motion"')
    plt.xlabel("phylogenetic distance")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Evolutionary model + tree structure = pattern of covariation observed at the tips

    Common simple models of evolutionary dynamics:

    - Brownian motion - implies in linear covariance kernel function
    - Damped Brownian motion - implies an L1 / **Ornstein-Uhlenbeck** covariance kernel function
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Phylogenetic "regression" model

    $$
    \begin{align}
    \textbf{B} &\sim \text{MVNormal}(\mu, \textbf{K}) \\
    \mathbf{K} &= \eta^2 \exp(-\rho D_{ij}) &\text{Ornstein-Uhlenbeck kernel}\\
    \mu_i &= \alpha + \beta_G G_i + \beta_M M_i \\
    \alpha &\sim \text{Normal}(0, 1) \\
    \beta_{G, M} &\sim \text{Normal}(0, 1) \\
    \eta^2 &\sim \text{HalfNormal}(1, 0.25) &\text{Maximum covariance prior} \\
    \rho &\sim \text{HalfNormal}(3, 0.25) &\text{Covariance decline rate prior}
    \end{align}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Prior Samples Ornstein-Uhlenbeck Kernel Model
    """)
    return


@app.cell
def _(ornstein_uhlenbeck_kernel):
    _eta_squared_samples =  pm.draw(pm.TruncatedNormal.dist(mu=1, sigma=.25, lower=.01), 100) ** .5
    _rho_samples =  pm.draw(pm.TruncatedNormal.dist(mu=3, sigma=.25, lower=.01), 100)

    _fig, _ = plt.subplots(figsize=(5, 5))
    for _ii, (_eta_squared, _rho) in enumerate(zip(_eta_squared_samples, _rho_samples)):
            _label = 'prior' if _ii == 0 else None
            _kernel_function = partial(ornstein_uhlenbeck_kernel, eta_squared=_eta_squared, rho=_rho)
            plot_kernel_function(_kernel_function, color='C0', label=_label, alpha=.5, linewidth=2, max_distance=1)

    plt.ylim([-.05, 1.5])
    plt.xlabel("phylogenetic distance")
    plt.ylabel("covariance")
    plt.title("Covariance Function Prior")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Distance-only model

    - Get the Gaussian-process part to work, then add details
    - Set $\beta_M = \beta_G = 0$
    """)
    return


@app.cell
def _(PRIMATES_DISTANCE_MATRIX):
    # Rescale distances from 0 to 1
    D = (PRIMATES_DISTANCE_MATRIX / PRIMATES_DISTANCE_MATRIX.max())
    assert D.max() == 1
    return (D,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### PyMC5 Implementation

    Below is an implementation that uses PyMC's Gaussian process module.

    - In the Oceanic Tools models above, we didn't need the Gaussian process to have a mean function because all variables were standardized, thus we can use the default mean function of $0$
    - For phylogenetic regression with we no longer want a zero mean function, but we want the mean function that depends on $M$ and $G$, but not a function of the phylogenetic distance matrix. We can implement a custom mean function in pymc to handle this
    - Also, because our likelihood is a MVNormal, we can take advantage of conjugacy between the GP prior and the Normal likelihood, which returns a GP posterior. This means that we should use `pm.gp.Marginal` instead of `pm.gp.Latent`, which uses closed form solutions ot the posterior to speed up learning.
    """)
    return


@app.cell
def _(G, M):
    class MeanBodyMassSocialGroupSize(pm.gp.mean.Linear):
        """Custom mean function that separates covariates from phylogeny"""
        def __init__(self, alpha, beta_G, beta_M):
            self.alpha = alpha
            self.beta_G = beta_G
            self.beta_M = beta_M

        def __call__(self, X):
            return self.alpha + self.beta_G * G + self.beta_M * M

    return (MeanBodyMassSocialGroupSize,)


@app.cell
def _(B, D, MeanBodyMassSocialGroupSize, PRIMATES):
    PRIMATE_ID, PRIMATE = pd.factorize(PRIMATES['name'], sort=False)
    with pm.Model(coords={"primate": PRIMATE}) as intercept_only_phylogenetic_mvn_model:
        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _sigma = pm.Exponential("sigma", 1)

        # Intercept-only model
        _beta_M = 0
        _beta_G = 0

        # Define the mean function
        _mean_func = MeanBodyMassSocialGroupSize(_alpha, _beta_G, _beta_M)

        # Phylogenetic distance covariance
        _eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.01)
        _rho = pm.TruncatedNormal("rho", 3, .25, lower=.01)

        # For Ornstein-Uhlenbeck kernel we can use Matern 1/2 or Exponential covariance Function
        _cov_func = _eta_squared * pm.gp.cov.Matern12(1, ls=_rho)
        # cov_func = eta_squared * pm.gp.cov.Exponential(1, ls=rho)

        # Init the GP
        _gp = pm.gp.Marginal(mean_func=_mean_func, cov_func=_cov_func)

        # Likelihood
        _gp.marginal_likelihood("B", X=D, y=B, sigma=_sigma)
        intercept_only_phylogenetic_mvn_inference = cached_sample(target_accept=0.9)
    return (
        intercept_only_phylogenetic_mvn_inference,
        intercept_only_phylogenetic_mvn_model,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean `alpha` and covariance defined by the kernel. I find that these direct `MVNormal` implementations track better with the lecture
    """)
    return


@app.cell
def _():
    def generate_L1_kernel_matrix(D, eta_squared, rho, smoothing=0.1):
        K = eta_squared * pm.math.exp(-rho * D)

        # Smooth the diagonal of the covariance matrix
        N = D.shape[0]
        K += np.eye(N) * smoothing
        return K

    # with pm.Model() as intercept_only_phylogenetic_mvn_model:

    #     # Priors
    #     alpha = pm.Normal("alpha", 0, 1)

    #     # Phylogenetic distance covariance
    #     eta_squared = pm.TruncatedNormal("eta_squared", 1, 0.25, lower=.001)
    #     rho = pm.TruncatedNormal("rho", 3, 0.25, lower=.001)

    #     # Ornstein-Uhlenbeck kernel
    #     K = pm.Deterministic('K', generate_L1_kernel_matrix(D, eta_squared, rho))
    #     # K = pm.Deterministic('K', eta_squared * pm.math.exp(-rho * D))

    #     # Likelihood
    #     mu = pm.math.ones_like(B) * alpha
    #     pm.MvNormal("B", mu=mu, cov=K, observed=B)
    #     intercept_only_phylogenetic_mvn_inference = pm.sample(target_accept=.98)
    return


@app.cell
def _(intercept_only_phylogenetic_mvn_inference):
    az.summary(intercept_only_phylogenetic_mvn_inference, var_names=['alpha', 'eta_squared', 'rho'])
    return


@app.cell
def _(intercept_only_phylogenetic_mvn_model):
    # Sample the prior for comparison
    with intercept_only_phylogenetic_mvn_model:
        intercept_only_phylogenetic_mvn_prior = pm.sample_prior_predictive().prior

    az.summary(intercept_only_phylogenetic_mvn_prior.to_dataset(), var_names=["alpha", "eta_squared", "rho"])
    return (intercept_only_phylogenetic_mvn_prior,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare the Prior and Posterior for distance-only model
    """)
    return


@app.cell
def _(D):
    def calculate_mean_covariance_matrix(inference):
        def distance_to_covariance(distance, eta_squared, rho):
            return eta_squared * np.exp(-rho * np.abs(distance))

        inference_mean = inference.mean(dim=('chain', 'draw'))

        eta_squared = inference_mean['eta_squared'].values
        rho = inference_mean['rho'].values

        print("Mean Kernel parameters:")
        print("eta_squared:", eta_squared)
        print("rho:", rho)

        n_species = D.shape[0]
        model_covariance = np.zeros_like(D).astype(float)
        for ii in range(n_species):
            for jj in range(n_species):
                model_covariance[ii, jj] = distance_to_covariance(
                    D[ii, jj],
                    eta_squared=eta_squared,
                    rho=rho
                )
        return model_covariance

    def plot_mean_covariance_matrix(inference, title=None, max_cov=1.0):
        mean_cov = calculate_mean_covariance_matrix(inference)

        plt.imshow(mean_cov)
        # plt.imshow(inference['K'].mean(dim=('chain', 'draw')).values)
        plt.grid(None)
        plt.clim((0, max_cov))
        plt.colorbar();
        plt.xlabel('species ID')
        plt.ylabel('species ID')
        plt.title(title)

    return (plot_mean_covariance_matrix,)


@app.cell(hide_code=True)
def _(intercept_only_phylogenetic_mvn_prior, plot_mean_covariance_matrix):
    # Prior
    plot_mean_covariance_matrix(
        intercept_only_phylogenetic_mvn_prior,
        title='Prior Covariance\nIntercept-only phylogenetic Model'
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(intercept_only_phylogenetic_mvn_inference, plot_mean_covariance_matrix):
    # Posterior
    plot_mean_covariance_matrix(
        intercept_only_phylogenetic_mvn_inference.posterior,
        title='Prior Covariance\nIntercept-only phylogenetic Model'
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare the Prior and Posterior kernel functions
    """)
    return


@app.cell
def _(ornstein_uhlenbeck_kernel):
    def plot_predictive_kernel_function(predictive, n_samples=200, color='C0', label=None):

        eta_squared_samples = predictive['eta_squared'].values[0, :n_samples]
        rho_samples = predictive['rho'].values[0, :n_samples]

        for ii, (eta_squared_, rho_) in enumerate(zip(eta_squared_samples, rho_samples)):
            label = label if ii == 0 else None

            kernel_function = partial(ornstein_uhlenbeck_kernel, eta_squared=eta_squared_, rho=rho_)
            plot_kernel_function(kernel_function, color=color, label=label, alpha=.1, linewidth=2, max_distance=1)

        plt.ylim([-.05, 2.5]);
        plt.xlabel("phylogenetic distance")

    return (plot_predictive_kernel_function,)


@app.cell(hide_code=True)
def _(
    intercept_only_phylogenetic_mvn_inference,
    intercept_only_phylogenetic_mvn_prior,
    plot_predictive_kernel_function,
):
    plot_predictive_kernel_function(intercept_only_phylogenetic_mvn_prior, label='prior', color='k')
    plot_predictive_kernel_function(intercept_only_phylogenetic_mvn_inference.posterior, label='posterior B', color='C1')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can see comparing the prior and posterior distirbution over kernel functions, that the model has learned from the data and attenuated our expectation on maximum covariance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the full Phylogentic model.

    This model stratifies by Group size $G$ and Body Mass $M$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an implementation that uses PyMC's Gaussian process module.
    """)
    return


@app.cell
def _(B, D, MeanBodyMassSocialGroupSize, coords):
    with pm.Model(coords=coords) as full_phylogenetic_mvn_model:

        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _sigma = pm.Exponential("sigma", 1)

        # Intercept-only model
        _beta_M = pm.Normal("beta_M", 0, 1)
        _beta_G = pm.Normal("beta_G", 0, 1)

        # Define the mean function
        _mean_func = MeanBodyMassSocialGroupSize(_alpha, _beta_G, _beta_M)

        # Phylogenetic distance covariance
        _eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.01)
        _rho = pm.TruncatedNormal("rho", 3, .25, lower=.01)

        # For Ornstein-Uhlenbeck kernel we can use Matern 1/2 or Exponential covariance Function
        _cov_func = _eta_squared * pm.gp.cov.Matern12(1, ls=_rho)
        # cov_func = eta_squared * pm.gp.cov.Exponential(1, ls=rho)

        # Init the GP
        _gp = pm.gp.Marginal(mean_func=_mean_func, cov_func=_cov_func)

        # Likelihood
        _gp.marginal_likelihood("B", X=D, y=B, sigma=_sigma)
        full_phylogenetic_mvn_inference = cached_sample(target_accept=0.9)

    # with pm.Model() as full_phylogenetic_mvn_model:

    #     # Priors
    #     alpha = pm.Normal("alpha", 0, 1)
    #     beta_G = pm.Normal("beta_G", 0, 0.5)
    #     beta_M = pm.Normal("beta_M", 0, 0.5)

    #     # Phylogenetic distance covariance
    #     eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.001)
    #     rho = pm.TruncatedNormal("rho", 3, .25, lower=.001)
    #     K = pm.Deterministic('K', generate_L1_kernel_matrix(D, eta_squared, rho))
    #     # K = pm.Deterministic('K', eta_squared * pm.math.exp(-rho * D))

    #     mu = alpha + beta_G * G + beta_M * M

    #     # Likelihood
    #     pm.MvNormal("B", mu=mu, cov=K, observed=B)
    #     full_phylogenetic_mvn_inference = pm.sample(target_accept=.98)
    return (full_phylogenetic_mvn_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the lecture.
    """)
    return


@app.cell
def _(full_phylogenetic_mvn_inference):
    az.summary(full_phylogenetic_mvn_inference, var_names=["alpha", "eta_squared", "rho", "beta_M", "beta_G"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare covariance terms in distance-only and full phylogenetic regression models
    """)
    return


@app.cell(hide_code=True)
def _(full_phylogenetic_mvn_inference, plot_mean_covariance_matrix):
    # Posterior
    plot_mean_covariance_matrix(
        full_phylogenetic_mvn_inference.posterior,
        title='Prior Covariance\nIntercept-only phylogenetic Model'
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(
    full_phylogenetic_mvn_inference,
    intercept_only_phylogenetic_mvn_inference,
    intercept_only_phylogenetic_mvn_prior,
    plot_predictive_kernel_function,
):
    plot_predictive_kernel_function(intercept_only_phylogenetic_mvn_prior, label='prior', color='k')
    plot_predictive_kernel_function(intercept_only_phylogenetic_mvn_inference.posterior, label='posterior B', color='C1')
    plot_predictive_kernel_function(full_phylogenetic_mvn_inference.posterior, label='posterior BMG')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can see that when including regressors for body mass and group size, the amount of phylogenetic covaration used to explain brain size is greatly diminished, compared to the phylogenetic-distance-only model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The results for full-phylogenetic model reported here make sense to me too, given that we can explain away a lot of the variability in brain size by primarily body mass, after we've controlled for phylogenetic history; this in turn makes the phylogenetic history receive less weight in the model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Influence of Group Size on Brain Size
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("M", "G"),
            ("M", "B"),
            ("u", "M"),
            ("u", "G"),
            ("u", "B"),
            ("h", "u")
        ],
        node_props={
            "h": {"label": "history\n(phylogeny)", "color": "lightgray", "fontcolor": "lightgray"},
            "u": {"style": "dashed", "label": "confounds, u", "color": "lightgray", "fontcolor": "lightgray"},
            "unobserved": {"style": "dashed", "color": "lightgray", "fontcolor": "lightgray"},  
        },
        edge_props={
            ("G", "B"): {"color": "blue"},
            ("M", "G"): {"color": "red"},
            ("M", "B"): {"color": "red"},
            ("u", "B"): {"color": "lightgray"},
            ("u", "M"): {"color": "lightgray"},
            ("u", "G"): {"color": "lightgray"},
            ("h", "u"): {"color": "lightgray",}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - We ignore the confounds due to genetic history (light gray)
      - thus we don't try to control for potential confounds due to genetic history.
    - To get <span style="color:blue">the direct effect of $G$ on $B$</span> we'll also stratify by $M$ to block <span style="color:red">the backdoor path through the fork passing through $M$ </span>.
    """)
    return


@app.cell
def _(B, G, M):
    with pm.Model() as ordinary_model:

        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _beta_G = pm.Normal("beta_G", 0, 0.5)
        _beta_M = pm.Normal("beta_M", 0, 0.5)

        # Independent species (equal variance)
        _sigma = pm.Exponential("sigma", 1)
        _K = np.eye(len(B)) * _sigma

        _mu = _alpha + _beta_G * G + _beta_M * M

        pm.MvNormal("B", mu=_mu, cov=_K, observed=B)
        ordinary_inference = cached_sample()
    return (ordinary_inference,)


@app.cell
def _(ordinary_inference):
    az.summary(ordinary_inference, var_names=["alpha", "beta_M", "beta_G"])
    return


@app.cell
def _(full_phylogenetic_mvn_inference, ordinary_inference):
    def plot_primate_posterior_comparison(variable, ax, title=None):
        utils.plot_dist(ordinary_inference.posterior[variable], color='k', label='ordinary', ax=ax)
        utils.plot_dist(full_phylogenetic_mvn_inference.posterior[variable], color='C0', label='Ornstein-Uhlenbeck', ax=ax)
        ax.axvline(0, color='k', linestyle='--');
        ax.set_xlabel(f"$\\{variable}$")
        ax.set_ylabel("density");
        ax.set_title(title)

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_primate_posterior_comparison('beta_G', _axs[0], "Effect of Group Size\non Brain Size")
    plot_primate_posterior_comparison('beta_M', _axs[1], "Effect of Body Mass\non Brain size")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    By trying to control for potential confounds due to genetic history, we have nearly halved the estimated effect of Group size on Brain size $\beta_G$, though it's still mostly positive.

    This estimator can also give us the direct effect of Body Mass $M$ on Brain Size $B$ (by blocking the Pipe through $G$). When we look at the $\beta_M$, we see that there is a slight increase in the effect of body mass on brain size when controlling for genetic history confounds, though the effect is large for both estimators indicating that body mass is a large driver of brain size.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary: Phylogenetic Regression

    #### Potential Issues

    - What about uncertainty in phylogeny?
      - better to perform phylogenic inference simultaneously -- i.e. estimate a posterior on phylogenies
    - What about reciprocal causation?
      - feedback between organism and environment
      - there's no unidirectional cause
      - thus regression is likely not the best option
      - new were methods include differential equations to pose the problem as a multi-objective optimization problem
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary: Gaussian Processes
    - Provides means for (local) partial pooling for continuous groups/categories
    - General approximation engine -- good for prediction
    - Robust to overfitting
    - Sensitive to priors: prior predictive simiulation is very important
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
