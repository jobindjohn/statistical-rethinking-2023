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
    from scipy.linalg import sqrtm
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 14 Correlated Features](https://www.youtube.com/watch?v=Es44-Bp1aKo&t)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Revisiting Bangladesh Fertility
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "C"),
            ("K", "C"),
            ("U", "C"),
            ("D", "C"),
            ("D", "U"),
            ("U", "K"),
            ("A", "K"),
            ("D", "K")
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - **Estimand 1**: Contraceptive use $C$ in each District $D$
    - **Estimand 2**: Effect of "urbanity" $U$
    - **Estimand 3**: Effect of # of kids $K$ and age $A$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Districts are _CLUSTERS_

    #### Rural / Urban breakdown
    Below we show the contraceptive use for rural (only including a district-level intercept $\alpha_{D[i]}$) vs urban (including an additional intercept $\beta_{D[i]}$ for urban districts) groups. These plow was generated in [Lecture 13 - Multilevel Adventures](<Lecture 13 - Multilevel Adventures.ipynb>).
    """)
    return


@app.cell
def _():
    utils.display_image("fertility_posterior_means_rural_urban.png", width=1000)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The plot shows:

    - Rural populations are, on average are less likely to use contraception than urban areas
      - the dashed line in the top plot is lower than that in the lower plot
    - There are fewer urban samples, so there is a larger range of uncertaint for urban areas
      - the error blue bars associated with urban popluations are larger than for rural areas, where there were more polls sampled.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Urbanism is a _Correlated Feature_

    Below we plot the probability of contraceptive use for urban $p(C | U=1)$ vs rural $p(C|U=0)$ areas. These plow was generated in [Lecture 13 - Multilevel Adventures](<Lecture 13 - Multilevel Adventures.ipynb>).
    """)
    return


@app.cell
def _():
    utils.display_image("fertility_p_C_rural_urban.png")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The plot shows that contraceptive use between urban/rural observations is correlated (cc>0.7)

    ### 🤔
    - If we know about the contraceptive use in rural areas, you can make a better guess about urban contraceptive use because of this correlation
    - **We are leaving information on the table by not letting the Golem--i.e. our statistical model--leverage this feature correlation**

    This lecture focuses on building statistical models that can capture correlation information amongst features.
    - makes inference more efficient by using partial pooling across features

    > #### Estimating Correlation and Partial Pooling Demo

    > McElreath goes on to show a demo of Bayesian updating in a model with a correlated features. Ιt's great, and I recommend going over the demo a few times, but I'm too lazy to implement it (nor clever enough to do so without an animation, which I'm trying to avoid). It'll go on the TODO list.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Adding Correlated Features
    - **One prior distribution for each _cluster/subgroup_**  $\rightarrow a_j \sim \text{Normal}(\bar a , \sigma)$; allows partial pooling
    - If **one feature**, then **one-dimensional prior**
      - $a_j \sim \text{Normal}(\bar a, \sigma)$
    - **N-dimensional distribution for N features**
      - rather than independent distributions (as we've been doing in previous lectures), we use joint distributions for features
      - $a_{j, 1..N} \sim \text{MVNormal}(A, \Sigma)$
      - e.g. $[\alpha_j, \beta_j] \sim \text{MVNormal}([\bar \alpha, \bar \beta], \Sigma_{\alpha, \beta})$
    - **Hard part**: learning associations/correlations amongst features
      - "Estimating means is easy"
      - "Estimating standard deviations is hard"
      - "Estimating correlations is very hard"
      - That said, usually good idea to try to lear correlations:
        - if you don't have enough samples / signal, you'll fall back on the prior
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Previous model -- Using _uncorrelated_ urban features
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "C"),
            ("K", "C"),
            ("U", "C"),
            ("D", "C"),
            ("D", "U"),
            ("U", "K"),
            ("A", "K"),
            ("D", "K")
        ],
        node_props={
            "U": {"color": "red"},
            "D": {"color": "red"},
            "C": {"color": "red"}
        },
        edge_props={
            ("U", "K"): {"color": "red"},
            ("U", "C"): {"color": "red"},
            ("K", "C"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > This is the uncentered implementation, more-or-less copy-pasted from the previous Lecture
    """)
    return


@app.cell
def _():
    FERTILITY = utils.load_data("bangladesh")

    USES_CONTRACEPTION = FERTILITY['use.contraception'].values.astype(int)
    DISTRICT_ID, _ = pd.factorize(FERTILITY.district)
    DISTRICT = np.arange(1, 62).astype(int)  # note: district 54 has no data so we create it's dim by hand
    URBAN_ID, URBAN = pd.factorize(FERTILITY.urban, sort=True)

    with pm.Model(coords={"district": DISTRICT}) as uncorrelated_model:

        # Mutable data
        _urban = pm.Data("urban", URBAN_ID)

        # Priors -- priors for $\alpha$ and $\beta$ are separate, independent Normal distributions
        # District offset
        _alpha_bar = pm.Normal("alpha_bar", 0, 1)  # the average district
        _sigma = pm.Exponential('sigma', 1)  # variation amongst districts

        # Uncentered parameterization
        _z_alpha = pm.Normal("z_alpha", 0, 1, dims='district')
        _alpha = pm.Deterministic("alpha", _alpha_bar + _z_alpha * _sigma, dims='district')

        # District / urban interaction
        _beta_bar = pm.Normal("beta_bar", 0, 1)  # the average urban effect
        _tau = pm.Exponential('tau', 1)  # variation amongst urban

        # Uncentered parameterization
        _z_beta = pm.Normal("z_beta", 0, 1, dims='district')
        _beta = pm.Deterministic("beta", _beta_bar + _z_beta * _tau, dims='district')

        # Recored p(contraceptive)
        _p_C = pm.Deterministic("p_C", pm.math.invlogit(_alpha + _beta))
        _p_C_urban = pm.Deterministic("p_C_urban", pm.math.invlogit(_alpha + _beta))
        _p_C_rural = pm.Deterministic("p_C_rural", pm.math.invlogit(_alpha))

        # Likelihood
        _p = pm.math.invlogit(_alpha[DISTRICT_ID] + _beta[DISTRICT_ID] * _urban)
        _C = pm.Bernoulli("C", p=_p, observed=USES_CONTRACEPTION)

        uncorrelated_inference = cached_sample(target_accept=.95)
    return (
        DISTRICT,
        DISTRICT_ID,
        URBAN_ID,
        USES_CONTRACEPTION,
        uncorrelated_inference,
        uncorrelated_model,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Demonstrate that feature priors are independent in uncorrelated model
    Below we sample from the prior, and show that the $\alpha$ and $\beta$ parameters for all districts are aligned with the cardinal axes, indicating no correlation
    """)
    return


@app.function
def plot_2d_guassian_ci(R, mean=[0, 0], std=[1, 1], ci_prob=0.89, **plot_kwargs):
    """
    Plot the `ci_prob`% confidence interval for a 2D Gaussian defined
    by correlation matrix `R`, center `mean`, and standard deviations
    `std` along the marginals.
    """
    # Create a circle
    angles = np.linspace(0, 2 * np.pi, 100)
    radius = stats.norm.ppf(ci_prob)
    circle = radius * np.vstack([np.cos(angles), np.sin(angles)])

    # Warp circle using the covariance matrix
    D = np.diag(std)
    cov = D @ R @ D
    ellipse = sqrtm(np.matrix(cov)) @ circle
    plt.plot(
        ellipse[0] + mean[0], ellipse[1] + mean[1], linewidth=3, **plot_kwargs
    )


@app.cell
def _(uncorrelated_model):
    with uncorrelated_model:
        _prior_pred = pm.sample_prior_predictive()

    _districts = [1, 10, 30, 61]
    _n_districts = len(_districts)
    _fig, _axs = plt.subplots(_n_districts, 1, figsize=(6, _n_districts * 4))
    for _ii, _district in enumerate(_districts):
        _district_prior = _prior_pred.prior.sel(district=_district)
        plt.sca(_axs[_ii])
        utils.plot_dist(_district_prior['alpha'], _district_prior['beta'], ax=_axs[_ii], contourf_kwargs={"cmap": "Reds"})
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xlabel("$\\alpha$")
        plt.ylabel("$\\beta$")
        plt.title(f"District {_district} Prior")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Above we can see the priors for the uncorrelated model provide. We can tell they are uncorrelated because
    - the primary axes of the distribution lie orthogonally along the axes of parameter values.
    - i.e. there's no "tilt" of th PDF countour
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multivariate Normal Prior

    $$
    \begin{align*}
    [\alpha_j, \beta_j] &= \text{MVNormal}([\bar \alpha, \bar \beta], R, [\sigma, \tau]) \\
    [\alpha_j, \beta_j]&:= \text{features for district } j \\
    R &:= \text{correlation matrix amongst features} \\
    [\sigma, \tau] &:= \text{(independent) standard deviations for each parameter } j \\
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Samples from multivariate prior
    """)
    return


@app.cell
def _():
    np.random.seed(123)
    _sigma = 1.0
    _tau = 2.0
    _R = np.array([[1, .8], [.8, 1]])  # correlation matrix


    _stds = np.array([_sigma, _tau])
    _D = np.diag(_stds)
    _Sigma = _D @ _R @ _D  # convert correlation matrix to covariance matrix
    _mvnorm = stats.multivariate_normal(cov=_Sigma)

    def _mvn_norm_pdf(xs, ys, ):
        return _mvnorm.pdf(np.vstack([xs, ys]).T)

    _RESOLUTION = 100
    _xs = _ys = np.linspace(-4, 4, _RESOLUTION)
    utils.plot_2d_function(_xs, _ys, _mvn_norm_pdf, cmap="Reds")

    _n_samples = 50
    _samples = _mvnorm.rvs(_n_samples)
    utils.plot_scatter(_samples[:, 0], _samples[:, 1], color='darkred', alpha=1, label='samples')
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.legend()

    print(f"Standard deviations\n{_stds}")
    print(f"Correlation Matrix\n{_R}")
    print(f"Covariance Matrix\n{_Sigma}")
    print(f"Cholesky Matrix\n{np.linalg.cholesky(_Sigma)}")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model with correlated urban features

    $$
    \begin{align*}
    C_i &\sim \text{Bernouill}(p_i) \\
    p_i &= \alpha_{D[i]} + \beta_{D[i]} U_i \\
    [\alpha_j, \beta_j] &\sim \text{MVNormal}([\bar \alpha, \bar \beta], R, [\sigma, \tau]) \\
    \sigma &\sim \text{Exponential}(1) \\
    \tau &\sim \text{Exponential}(1) \\
    R &\sim \text{LKJCorr}(4)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### On the `LKJCorr` prior

    - Named after Lewandowski-Kurowicka-Joe, the authors of the original paper on the distribution
    - **Prior for correlation matrices**
    - Takes a shape parameter $\eta>0$
      - $\eta=1$ results in unifrom distribution over correlation matrices
      - $\eta \rightarrow \infty$, R approaches an identity matrix
    - changes the distribution of correlation matrices
    """)
    return


@app.cell(hide_code=True)
def _():
    _etas = [1, 4, 10, 20, 100]
    for _ii, _eta in enumerate(_etas):
        _rv = pm.LKJCorr.dist(n=2, eta=_eta)
        _chol = pm.draw(_rv, 1000)  # PyMC 6 draws the Cholesky factor of the correlation matrix
        _samples = (_chol @ _chol.mT)[:, 0, 1]
        utils.plot_dist(_samples, color=f"C{_ii}", label=f"LKJCorr$(\\eta={_eta})$")
    plt.legend()
    plt.xlabel("correlation")
    plt.ylabel("density")
    plt.title("LKJCorr distribution of correlations")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ####  Correlation Matrices Sampled from LKJCorr
    """)
    return


@app.function
def sample_correlation_matrix_LKJ(dim=2, eta=4, n_samples=1):

    # Samples; PyMC 6's LKJCorr returns the Cholesky factor L of each
    # correlation matrix, so the correlation matrix is L @ L.T
    rv = pm.LKJCorr.dist(n=dim, eta=eta)
    chol = pm.draw(rv, n_samples).reshape(n_samples, dim, dim)
    return chol @ chol.mT


@app.function
def sample_LKJCorr_prior(eta=1, n_samples=20, ax=None):
    plt.sca(ax)
    for _ in range(n_samples):

        R = sample_correlation_matrix_LKJ(eta=eta)

        # Sample a random standard deviation along each dimension
        std = stats.expon(.1).rvs(2)  # exponential
        # std = np.random.rand(2) * 10  # uniform
        # std = np.abs(np.random.randn(2))  # half-normal

        plot_2d_guassian_ci(R, std=std, color='C0', alpha=.25);

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.title(f"Random Correlations Sampled\nfrom LKJCorr({eta}) prior");


@app.cell(hide_code=True)
def _():
    _etas = [1, 4, 10, 1000]
    _n_etas = len(_etas)
    _fig, _axs = plt.subplots(_n_etas, 1, figsize=(6, 6 * _n_etas))
    for _ii, _eta in enumerate(_etas):
        sample_LKJCorr_prior(eta=_eta, ax=_axs[_ii])
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    we can see that as $\eta$ becomes larger, the axes of the correlation matrices become more alighed with the axes of the parameter space, providing diagonal covariance samples
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model with correlated features
    #### A couple of notes ⚠️
    - We jump straight to the implementation with uncentered priors here, skipping the centered version. For details on the uncentered implementation, see the [PyMC docs](https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/case_studies/LKJ.html)
    - We use `LKJCholeskyCov` instead `LKJCorr` for improved numerical stability and performance. For details, see https://discourse.pymc.io/t/uses-of-lkjcholeskycov-and-lkjcorr/3629/3
    - We don't need to implement the `Chol_to_Corr` function in the lecture, as we can just pull the correlations directly out of the `LKJCholeskyCov` distribution
    """)
    return


@app.cell
def _(DISTRICT, DISTRICT_ID, URBAN_ID, USES_CONTRACEPTION):
    N_CORRELATED_FEATURES = 2
    N_DISTRICTS = 61
    ETA = 4

    with pm.Model(coords={"district": DISTRICT}) as correlated_model:

        # Mutable data for posterior predictions
        _urban = pm.Data("urban", URBAN_ID)

        # Priors
        # Feature correlation
        _chol, _corr, _stds = pm.LKJCholeskyCov(
            "Rho",
            eta=ETA,
            n=N_CORRELATED_FEATURES,
            sd_dist=pm.Exponential.dist(1, size=N_CORRELATED_FEATURES)  # variation for params
        )

        # Uncentered parameterization for each variable
        _z = pm.Normal("z", 0, 1, shape=(N_DISTRICTS, N_CORRELATED_FEATURES))
        _v = pm.Deterministic("v", _chol.dot(_z.T).T)

        _alpha_bar = pm.Normal("alpha_bar", 0, 1)  # the average district
        _alpha = pm.Deterministic("alpha", _alpha_bar + _v[:, 0])

        _beta_bar = pm.Normal("beta_bar", 0, 1)  # the average urban effect
        _beta = pm.Deterministic("beta", _beta_bar + _v[:, 1])

        # Record values for reporting
        pm.Deterministic("feature_cov", _chol.dot(_chol.T))
        pm.Deterministic("feature_corr", _corr)
        pm.Deterministic("feature_std", _stds)
        pm.Deterministic("p_C", pm.math.invlogit(_alpha + _beta), dims='district')
        pm.Deterministic("p_C_urban", pm.math.invlogit(_alpha + _beta), dims='district')
        pm.Deterministic("p_C_rural", pm.math.invlogit(_alpha), dims='district')

        # Likelihood
        _p = pm.math.invlogit(_alpha[DISTRICT_ID] + _beta[DISTRICT_ID] * _urban)
        _C = pm.Bernoulli("C", p=_p, observed=USES_CONTRACEPTION)

        correlated_inference = cached_sample(target_accept=.95)
    return correlated_inference, correlated_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyzing feature correlation
    """)
    return


@app.cell
def _(correlated_inference):
    correlated_posterior_mean = correlated_inference.posterior.mean(dim=('chain', 'draw'))
    print(f'Feature Covariance\n', correlated_posterior_mean['feature_cov'].values)
    print(f'Feature Correlation\n', correlated_posterior_mean['feature_corr'].values)
    print(f'Feature Std\n', correlated_posterior_mean['feature_std'].values)
    return (correlated_posterior_mean,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare to prior
    """)
    return


@app.cell
def _(correlated_model):
    with correlated_model:
        # Add prior samples
        correlated_prior_predictive = pm.sample_prior_predictive()
    return (correlated_prior_predictive,)


@app.function
def plot_posterior_mean_alpha_beta(posterior_mean, title=None):
    alphas = posterior_mean['alpha'].values
    betas =  posterior_mean['beta'].values
    alpha_mean = posterior_mean['alpha'].mean().values
    beta_mean = posterior_mean['beta'].mean().values

    # Correlated feature params
    if "feature_corr" in posterior_mean:
        posterior_R = posterior_mean['feature_corr'].values
        posterior_std = posterior_mean['feature_std'].values

    # Independent feature params
    else:
        cc = np.corrcoef(alphas, betas)[0][1]
        posterior_R = np.array([[1, cc], [cc, 1]])
        posterior_std = np.array([posterior_mean['sigma'], posterior_mean['tau']])

    plot_2d_guassian_ci(
        R=posterior_R,
        mean=(alpha_mean, beta_mean),
        std=posterior_std,
        color='C0'
    )

    utils.plot_scatter(
        xs=posterior_mean['alpha'],
        ys=posterior_mean['beta'],
        color='k', label='district param means'
    )

    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlim([-2, 1])
    plt.ylim([-2, 2])
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.legend();
    plt.title(title)


@app.cell(hide_code=True)
def _(
    correlated_inference,
    correlated_posterior_mean,
    correlated_prior_predictive,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot correlation distribution for posterior and prior
    plt.sca(_axs[0])
    _plot_kwargs = {"linewidth": 3}
    utils.plot_dist(
        correlated_inference.posterior['feature_corr'][:, :, 0, 1],
        label='posterior', plot_kwargs=_plot_kwargs,
        ax=_axs[0]
    )
    utils.plot_dist(
        correlated_prior_predictive.prior['feature_corr'][:, :, 0, 1],
        color='k',
        label='prior', plot_kwargs=_plot_kwargs,
        ax=_axs[0]
    )

    plt.axvline(0, color='gray', linestyle='--')
    plt.xlim([-1, 1])
    plt.xlabel("correlation between $\\alpha$ and $\\beta$")
    plt.ylabel("density")
    plt.legend()

    plt.sca(_axs[1])

    plot_posterior_mean_alpha_beta(correlated_posterior_mean)
    plt.suptitle("Correlated Features", fontsize=18)
    _fig
    return


@app.cell
def _(correlated_posterior_mean, uncorrelated_inference):
    uncorrelated_posterior_mean = uncorrelated_inference.posterior.mean(dim=('chain', 'draw'))

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))

    plt.sca(_axs[0])
    plot_posterior_mean_alpha_beta(uncorrelated_posterior_mean, title='uncorrelated features')

    plt.sca(_axs[1])
    plot_posterior_mean_alpha_beta(correlated_posterior_mean, title='correlated features')
    _fig
    return (uncorrelated_posterior_mean,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Comparing to the model that does not model feature correlations: the model with uncorrelated features exhibits a much weaker negative correlation (i.e. the red ellipse on the left is less "tilted" downward.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Comparing models on the outcome scale.
    """)
    return


@app.cell
def _(DISTRICT):
    def plot_p_C_district_rural_urban(posterior_mean, title=None):
        for ii, (label, p_C) in enumerate(zip(['rural', 'urban'], ['p_C_rural', 'p_C_urban'])):
            color = f"C{ii}"
            plot_data = posterior_mean[p_C]
            utils.plot_scatter(DISTRICT, plot_data, label=label, color=color, alpha=.8)

            # Add quantile indicators
            ql, median, qu = np.quantile(plot_data, [.25, .5, .975])
            error = np.array((median-ql, qu-median))[:, None]
            plt.errorbar(
                x=0,
                y=median,
                yerr=error,
                color=color,
                capsize=5,
                linewidth=2,
                label=f'95%IPR:({ql:0.2f}, {qu:0.2f}))'
            )
            plt.axhline(median, color=color)


        plt.axhline(.5, color='gray', linestyle='--', label='p=0.5')
        plt.legend(loc='upper right')
        plt.ylim([0, 1])
        plt.xlabel("district")
        plt.ylabel("prob. contraceptive use")
        plt.title(title)

    return (plot_p_C_district_rural_urban,)


@app.cell(hide_code=True)
def _(
    correlated_posterior_mean,
    plot_p_C_district_rural_urban,
    uncorrelated_posterior_mean,
):
    _fig, _axs = plt.subplots(2, 1, figsize=(10, 8))

    plt.sca(_axs[0])
    plot_p_C_district_rural_urban(correlated_posterior_mean, title='correlated features')

    plt.sca(_axs[1])
    plot_p_C_district_rural_urban(uncorrelated_posterior_mean, title='uncorrelated features')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Modeling feature correlation allows urban areas to capture urban variance differently. Though the effect is subtle, the correlated feature model (top) exhibits smaller Inner-percentile range (IPR) (0.45-0.65) than the uncorrelated feature model (0.41-0.65).
    - ⚠️ This is actually the opposite of what was reported in the lecture. My intuition would be that there would be less posterior variability because we're sharing information across features, and thus reducing our uncertainty. But I could easily be doing or interpreting something incorrectly here.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Looking at outcome space
    """)
    return


@app.function
def plot_pC_rural_urban_xy(posterior_mean, label, color):
    utils.plot_scatter(
        xs=posterior_mean['p_C_rural'],
        ys=posterior_mean['p_C_urban'],
        label=label,
        color=color, alpha=.8
    )

    plt.axhline(.5, color='gray', linestyle='--')
    plt.axvline(.5, color='gray', linestyle='--')
    plt.xlabel("$p(C)$ (rural)")
    plt.ylabel("$p(C)$ (urban)")
    plt.legend()
    plt.axis('square')


@app.function
def plot_pC_xy_movement(posterior_mean_a, posterior_mean_b):
    xs_a = posterior_mean_a['p_C_rural']
    xs_b = posterior_mean_b['p_C_rural']
    ys_a = posterior_mean_a['p_C_urban']
    ys_b = posterior_mean_b['p_C_urban']

    for ii, (xa, xb, ya, yb) in enumerate(zip(xs_a, xs_b, ys_a, ys_b)):
        label = '$p(C) $ change' if not ii else None
        plt.plot((xa, xb), (ya, yb), linewidth=.5, color='C0', label=label)
    plt.legend(loc="upper left")


@app.cell(hide_code=True)
def _(correlated_posterior_mean, uncorrelated_posterior_mean):
    plot_pC_rural_urban_xy(uncorrelated_posterior_mean, label='uncorrelated features', color='k')
    plot_pC_rural_urban_xy(correlated_posterior_mean, label='correlated features', color='C0')
    plot_pC_xy_movement(uncorrelated_posterior_mean, correlated_posterior_mean)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Including feature correlations

    - Points move because the transfer of information across features provides better estimates of $p(C)$
    - There's a negative correlation between rural areas and the **difference** between urban contraceptive use in urban areas
      - When a district's rural areas have high contraceptive use, the difference will be smaller (i.e. urban areas will also have high contraceptive use)
      - When rural contraceptive use is low, urban areas will be more different (i.e. urban areas will still tend to have higher contraceptive use)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Review: Correlated Varying Effects
    Priors that **learn correlation** amongst features. Provides the following benefits
    1. partial pooling across features -- memory, shared information improves efficiency
    2. learns correlations -- this is the _focus_ of research questions

    💡 Varying effects can be correlated _and can still be learned_ without priors that incorporate correlation structure. However, incorporating correlations explicitly allows partial pooling, and is thus more efficient and provides more explicit information about those correlations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Non-centered (aka Transformed) Priors
    ## Inconvenient Posteriors
    - Inefficient MCMC caused by steep curvature in the neg-log-likelihood
    - Hamiltonian Monte Carlo has trouble exploring steep surface
      - leading to "divergent" transistions
      - "punching through wall of skate park"
      - detected when the Hamiltonian changes value dramatically between start/end of proposal
    - Transforming the prior to be a "smoother skate park" helps
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Devil's Funnel prior

    $$
    \begin{align*}
    v &\sim \text{Normal}(0, \sigma_v) \\
    x &\sim \text{Normal}(0, \exp({v}))
    \end{align*}
    $$

    As $\sigma_v$ increases, a nasty trough forms in the prior probability surface that is difficult for Hamiltonian dynamics to sample--i.e. the skatepark is too steep and narrow.

    I'm too lazy to code up the fancy HMC animation that McElreath uses to visualize each divergent path. However, we can still verifty that the number of divergences increases when sampling the devil's funnel as we increase the v prior's $\sigma$ in the devil's funnel prior.

    ### Centered-prior models
    """)
    return


@app.cell
def _():
    _xs = _vs = np.linspace(-4, 4, 100)

    def _prior_devils_funnel(xs, vs, sigma_v=.5):

        log_prior_v = stats.norm(0, sigma_v).logpdf(vs)
        log_prior_x = stats.norm(0, np.exp(vs)).logpdf(xs)

        log_prior = log_prior_v + log_prior_x
        return np.exp(log_prior - log_prior.max())


    # Loop over sigma_v, and show that divergences increase
    # as the depth and narrowness of the trough increases
    # for low values of v
    _sigma_vs = [.5, 1, 3]
    _n_sigma_vs = len(_sigma_vs)
    _fig, _axs = plt.subplots(3, 1, figsize=(4, 4 * _n_sigma_vs))

    for _ii, _sigma_v in enumerate(_sigma_vs):
        with pm.Model() as _centered_devils_funnel:
            _v = pm.Normal('v', 0, _sigma_v)
            _x = pm.Normal('x', 0, pm.math.exp(_v))
            _centered_devils_funnel_inference = cached_sample()
            _n_divergences = _centered_devils_funnel_inference.sample_stats.diverging.sum().values

        _prior = partial(_prior_devils_funnel, sigma_v=_sigma_v)
        plt.sca(_axs[_ii])
        utils.plot_2d_function(_xs, _vs, _prior, colors="gray", ax=_axs[_ii])
        plt.xlabel("x")
        plt.ylabel("v")
        plt.title(f"$v \\sim Normal(0, {_sigma_v})$\n# Divergences: {_n_divergences}")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What to do?
    - Smaller step size: handles steepness better, but takes longer to explore the posterior
    - Re-parameterize (tranform) to make surface smoother

    ## Non-centered prior
    We add an auxilary variable $z$ that has a smooth probability surface. We then sample that auxilary variable, and transform it to obtain the target variable distribution. For the case of the devi's funnel prior:

    $$
    \begin{align*}
    v &\sim \text{Normal}(0, \sigma_v) \\
    z &\sim \text{Normal}(0, 1) \\
    x &= z \exp(v)
    \end{align*}
    $$
    """)
    return


@app.cell
def _():
    _xs = _vs = np.linspace(-4, 4, 100)
    _sigma_vs = [.5, 1, 3]
    _n_sigma_vs = len(_sigma_vs)
    _fig, _axs = plt.subplots(3, 1, figsize=(4, 4 * _n_sigma_vs))

    def _prior_noncentered_devils_funnel(xs, vs, sigma_v=.5):
        """Reparamterize into a smoother auxilary variable space"""
        log_prior_v = stats.norm(0, sigma_v).logpdf(vs)
        log_prior_z = stats.norm(0, 1).logpdf(xs)
        log_prior = log_prior_v + log_prior_z
        return np.exp(log_prior - log_prior.max())

    for _ii, _sigma_v in enumerate(_sigma_vs):
        with pm.Model() as _noncentered_devils_funnel:
            _v = pm.Normal('v', 0, _sigma_v)
            _z = pm.Normal('z', 0, 1)
            # Record x for reporting
            _x = pm.Deterministic('x', _z * pm.math.exp(_v))
            noncentered_devils_funnel_inference = cached_sample()
            _n_divergences = noncentered_devils_funnel_inference.sample_stats.diverging.sum().values

        _prior = partial(_prior_noncentered_devils_funnel, sigma_v=_sigma_v)
        plt.sca(_axs[_ii])
        utils.plot_2d_function(_xs, _vs, _prior, colors="gray", ax=_axs[_ii])
        plt.xlabel("z")
        plt.ylabel("v")
        plt.title(f"$v \\sim$ Normal(0, {_sigma_v})\n# Divergences: {_n_divergences}")
    _fig
    return (noncentered_devils_funnel_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - By reparameterizing, we get to sample multi-dimensional Normal distributions, which are smoother parabolas in the log space.

    - We can see that the number of divergence is consistently reduced for all values of prior variance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Check HMC Diagnostics
    """)
    return


@app.cell
def _(noncentered_devils_funnel_inference):
    utils.check_sampling(noncentered_devils_funnel_inference)
    az.summary(noncentered_devils_funnel_inference)['r_hat']
    return


@app.cell
def _(noncentered_devils_funnel_inference):
    az.plot_trace_dist(noncentered_devils_funnel_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Diagnostics look good 👍
    - `Rhat`s = 1
    - $v$ and $z$ chains exhibit "fuzzy caterpillar" behavior
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cholesky Factors & Correlation Matrices
    - Cholesky factor, $\bf L$ provides an efficient means for encoding a correlation matrix $\Omega$ (requires fewer floating point numbers than the full correlation matrix)
    - $\Omega = \bf LL^T$
    - $\bf L$ is a lower-triangular matrix
    - we can sample data with correlation $\Omega$, and standard deviations $\sigma_i$, using the Cholesky factorization $\bf L$ of $\Omega$ as follows:

    $$
    X_\Omega \sim \text{diag}(\sigma_i)\bf{LZ},
    $$
    where $\bf Z$ is a matrix of z-scores sampled from a standard normal distribution
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Demonstrate reconstructing correlation matrix from Cholesky factor
    """)
    return


@app.cell
def _():
    orig_corr = np.array([[1, .6], [.6, 1]])
    print("Ground-truth Correlation Matrix\n", orig_corr)

    L = np.linalg.cholesky(orig_corr)
    print("\nCholesky Matrix of correlation matrix, L\n", L)
    print("\nReconstructed correlation matrix from L\n", L @ L.T)
    return L, orig_corr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Demonstrate sampling random corrleation matrix from z-scores and the Cholesky factor
    """)
    return


@app.cell
def _(L, orig_corr):
    np.random.seed(12345)
    _N = 100_000
    _orig_sigmas = [2, .5]

    # Matrix of z-score samples
    _Z = stats.norm().rvs(size=(2, _N))
    print("Raw z-score samples are indpendent\n", np.corrcoef(_Z[0], _Z[1]))

    # Transform Z-scores using the Cholesky factorization of the correlation matrix
    _sLZ = np.diag(_orig_sigmas) @ L @ _Z

    _corr_sLZ = np.corrcoef(_sLZ[0], _sLZ[1])
    print("\nCholesky-transformed z-scores have original correlation encoded\n", _corr_sLZ)
    assert np.isclose(orig_corr[0, 1], _corr_sLZ[0, 1], atol=.01)

    _std_sLZ = _sLZ.std(axis=1)
    print("\nOriginal std devs also encoded in transformed z-scores:\n", _std_sLZ)
    assert np.isclose(_orig_sigmas[0], _std_sLZ[0], atol=.01)
    assert np.isclose(_orig_sigmas[1], _std_sLZ[1], atol=.01)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## When to use Transformed Priors
    - depends on context
    - Centered:
      - Lot's of data in each cluster/subgroup
      - likelihood-dominant scenario
    - Non-centered:
      - Many clusters/subgroups, with sparse data in some of the clusters
      - prior-dominant scenario
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
