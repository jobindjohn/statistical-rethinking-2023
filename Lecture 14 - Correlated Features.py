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
def fertility_dag():
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
    Below we show the contraceptive use for rural (only including a district-level intercept $\alpha_{D[i]}$) vs urban (including an additional intercept $\beta_{D[i]}$ for the urban part of each district) groups. This plot was generated in [Lecture 13 - Multilevel Adventures](<Lecture 13 - Multilevel Adventures.py>).
    """)
    return


@app.cell
def rural_urban_posterior_image():
    utils.display_image("fertility_posterior_means_rural_urban.png", width=1000)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The plot shows:

    - Rural populations are, on average, less likely to use contraception than urban areas
      - the dashed line in the top plot is lower than that in the lower plot
    - There are fewer urban samples, so there is a larger range of uncertainty for urban areas
      - the blue error bars associated with urban populations are larger than for rural areas, where more women were surveyed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Urbanism is a _Correlated Feature_

    Below we plot the probability of contraceptive use for urban $p(C | U=1)$ vs rural $p(C|U=0)$ areas. This plot was generated in [Lecture 13 - Multilevel Adventures](<Lecture 13 - Multilevel Adventures.py>).
    """)
    return


@app.cell
def rural_urban_correlation_image():
    utils.display_image("fertility_p_C_rural_urban.png")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The plot shows that rural and urban contraceptive use are correlated across districts (cc ≈ 0.75)

    ### 🤔
    - If we know about the contraceptive use in rural areas, you can make a better guess about urban contraceptive use because of this correlation
    - **We are leaving information on the table by not letting the Golem--i.e. our statistical model--leverage this feature correlation**

    This lecture focuses on building statistical models that can capture correlation information amongst features.
    - makes inference more efficient by using partial pooling across features

    > #### Estimating Correlation and Partial Pooling Demo

    > McElreath goes on to show a demo of Bayesian updating in a model with correlated features. It's great, and I recommend going over the demo a few times. The correlated pooling demo further down, after the correlated model is fit, shows the same idea one district at a time.
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
      - That said, usually a good idea to try to learn correlations:
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
def uncorrelated_dag():
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
            ("D", "C"): {"color": "red"},
            ("U", "C"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > This is the uncentered implementation, more-or-less copy-pasted from the previous lecture
    """)
    return


@app.cell
def uncorrelated_fit():
    FERTILITY = utils.load_data("bangladesh")

    USES_CONTRACEPTION = FERTILITY['use.contraception'].values.astype(int)
    DISTRICT = np.arange(1, 62).astype(int)  # note: district 54 has no data, but still gets a parameter
    DISTRICT_ID = FERTILITY.district.values - 1  # index into DISTRICT, so district d is entry d - 1
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

        # Record p(contraceptive)
        _p_C_urban = pm.Deterministic("p_C_urban", pm.math.invlogit(_alpha + _beta), dims='district')
        _p_C_rural = pm.Deterministic("p_C_rural", pm.math.invlogit(_alpha), dims='district')

        # Likelihood
        _p = pm.math.invlogit(_alpha[DISTRICT_ID] + _beta[DISTRICT_ID] * _urban)
        _C = pm.Bernoulli("C", p=_p, observed=USES_CONTRACEPTION)

        uncorrelated_inference = cached_sample(target_accept=.95)
    return (
        DISTRICT,
        DISTRICT_ID,
        FERTILITY,
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
def plot_2d_gaussian_ci(R, mean=[0, 0], std=[1, 1], ci_prob=0.89, **plot_kwargs):
    """
    Plot the ellipse that holds `ci_prob` of the probability of a 2D
    Gaussian defined by correlation matrix `R`, center `mean`, and
    standard deviations `std` along the marginals.
    """
    # Create a circle
    angles = np.linspace(0, 2 * np.pi, 100)
    radius = np.sqrt(stats.chi2(df=2).ppf(ci_prob))  # squared radius of a 2D standard normal is chi2(2)
    circle = radius * np.vstack([np.cos(angles), np.sin(angles)])

    # Warp circle using the covariance matrix
    D = np.diag(std)
    cov = D @ R @ D
    ellipse = sqrtm(cov) @ circle
    plt.plot(
        ellipse[0] + mean[0], ellipse[1] + mean[1], linewidth=3, **plot_kwargs
    )


@app.cell
def uncorrelated_prior_plot(uncorrelated_model):
    with uncorrelated_model:
        _prior_pred = pm.sample_prior_predictive(random_seed=1)

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
    Above we can see the priors that the uncorrelated model provides. We can tell they are uncorrelated because
    - the primary axes of the distribution lie orthogonally along the axes of parameter values.
    - i.e. there's no "tilt" of the PDF contour
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multivariate Normal Prior

    $$
    \begin{align*}
    [\alpha_j, \beta_j] &\sim \text{MVNormal}([\bar \alpha, \bar \beta], R, [\sigma, \tau]) \\
    [\alpha_j, \beta_j]&:= \text{features for district } j \\
    R &:= \text{correlation matrix amongst features} \\
    [\sigma, \tau] &:= \text{(independent) standard deviations for each feature} \\
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
def mvn_prior_samples():
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
def mvn_controls():
    mvn_sigma = mo.ui.slider(0.25, 2.5, step=0.05, value=1.0, label="$\\sigma$ (sd of $\\alpha$)", show_value=True)
    mvn_tau = mo.ui.slider(0.25, 2.5, step=0.05, value=2.0, label="$\\tau$ (sd of $\\beta$)", show_value=True)
    mvn_rho = mo.ui.slider(-0.95, 0.95, step=0.05, value=0.8, label="$\\rho$ (correlation)", show_value=True)
    mo.vstack([
        mo.md(
            "**Explore the multivariate Normal.** The left panel shows 100 fixed z-scores. "
            "The right panel shows the same points after the transform $\\text{diag}(\\sigma, \\tau)\\,\\mathbf{L}\\,z$, "
            "where $\\mathbf{L}$ is the Cholesky factor of the correlation matrix (see the Cholesky section in the bonus). "
            "The colors follow each point through the transform; the rings are the 89% regions."
        ),
        mo.hstack([mvn_sigma, mvn_tau, mvn_rho], justify="start"),
    ])
    return mvn_rho, mvn_sigma, mvn_tau


@app.cell(hide_code=True)
def mvn_explorer(mvn_rho, mvn_sigma, mvn_tau):
    _stds = np.array([mvn_sigma.value, mvn_tau.value])
    _R = np.array([[1, mvn_rho.value], [mvn_rho.value, 1]])
    _L = np.linalg.cholesky(_R)
    _Sigma = np.diag(_stds) @ _R @ np.diag(_stds)
    _mvnorm = stats.multivariate_normal(cov=_Sigma)

    _Z = np.random.default_rng(123).standard_normal((2, 100))
    _X = np.diag(_stds) @ _L @ _Z
    _colors = np.arctan2(_Z[1], _Z[0])

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(_axs[0])
    plot_2d_gaussian_ci(np.eye(2), color="gray")
    _axs[0].scatter(_Z[0], _Z[1], c=_colors, cmap="twilight", s=30)
    _axs[0].set(xlim=[-6, 6], ylim=[-6, 6], xlabel="$z_1$", ylabel="$z_2$", title="z-scores", aspect="equal")

    _grid = np.linspace(-6, 6, 100)
    utils.plot_2d_function(_grid, _grid, lambda xs, ys: _mvnorm.pdf(np.vstack([xs, ys]).T), ax=_axs[1], cmap="Reds")
    plot_2d_gaussian_ci(_R, std=_stds, color="gray")
    _axs[1].scatter(_X[0], _X[1], c=_colors, cmap="twilight", s=30)
    _axs[1].set(xlim=[-6, 6], ylim=[-6, 6], xlabel="$\\alpha$", ylabel="$\\beta$", title="$\\text{diag}(\\sigma, \\tau)\\,\\mathbf{L}\\,z$", aspect="equal")
    _fig.tight_layout()


    def _bmatrix(M):
        _rows = " \\\\ ".join(" & ".join(f"{_v:.2f}" for _v in _row) for _row in M)
        return f"\\begin{{bmatrix}} {_rows} \\end{{bmatrix}}"


    mo.vstack([
        _fig,
        mo.md(
            f"$$\\Sigma = {_bmatrix(_Sigma)} \\qquad \\mathbf{{L}} = {_bmatrix(_L)} "
            f"\\qquad \\text{{diag}}(\\sigma, \\tau)\\,\\mathbf{{L}} = {_bmatrix(np.diag(_stds) @ _L)}$$"
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model with correlated urban features

    $$
    \begin{align*}
    C_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha_{D[i]} + \beta_{D[i]} U_i \\
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
      - $\eta=1$ results in a uniform distribution over correlation matrices
      - $\eta \rightarrow \infty$, R approaches an identity matrix
    - changes the distribution of correlation matrices
    """)
    return


@app.cell(hide_code=True)
def lkj_correlation_density_plot():
    _etas = [1, 4, 10, 20, 100]
    for _ii, _eta in enumerate(_etas):
        _rv = pm.LKJCorr.dist(n=2, eta=_eta)
        _chol = pm.draw(_rv, 1000, random_seed=_ii)  # PyMC 6 draws the Cholesky factor of the correlation matrix
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
def sample_correlation_matrix_LKJ(dim=2, eta=4, n_samples=1, random_seed=None):

    # Samples; PyMC 6's LKJCorr returns the Cholesky factor L of each
    # correlation matrix, so the correlation matrix is L @ L.T
    rv = pm.LKJCorr.dist(n=dim, eta=eta)
    chol = pm.draw(rv, n_samples, random_seed=random_seed).reshape(n_samples, dim, dim)
    return chol @ chol.mT


@app.function
def sample_LKJCorr_prior(eta=1, n_samples=20, ax=None, seed=None):
    plt.sca(ax)
    rng = np.random.default_rng(seed)
    # Slow: each sample_correlation_matrix_LKJ call compiles a new pm.draw function, so
    # 20 ellipses x 4 etas take about 30 s. Drawing all matrices in one call takes about 1 s:
    #     Rs = sample_correlation_matrix_LKJ(eta=eta, n_samples=n_samples, random_seed=rng)
    #     for R in Rs:
    #         ...
    for _ in range(n_samples):

        R = sample_correlation_matrix_LKJ(eta=eta, random_seed=rng)[0]

        # Sample a random standard deviation along each dimension
        std = stats.expon(scale=1).rvs(2, random_state=rng)  # exponential, as in the model's prior
        # std = np.random.rand(2) * 10  # uniform
        # std = np.abs(np.random.randn(2))  # half-normal

        plot_2d_gaussian_ci(R, std=std, color='C0', alpha=.25);

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.title(f"Random Correlations Sampled\nfrom LKJCorr({eta}) prior");


@app.cell(hide_code=True)
def lkj_ellipses_plot():
    _etas = [1, 4, 10, 1000]
    _n_etas = len(_etas)
    _fig, _axs = plt.subplots(_n_etas, 1, figsize=(6, 6 * _n_etas))
    for _ii, _eta in enumerate(_etas):
        sample_LKJCorr_prior(eta=_eta, ax=_axs[_ii], seed=_ii)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can see that as $\eta$ becomes larger, the axes of the ellipses become more aligned with the axes of the parameter space, providing diagonal covariance samples
    """)
    return


@app.cell(hide_code=True)
def lkj_controls():
    lkj_eta = mo.ui.slider(steps=[0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 10, 20, 50, 100, 1000], value=4, label="$\\eta$", show_value=True)
    mo.vstack([
        mo.md(
            "**Explore the LKJCorr prior.** For two features, the correlation $r$ has density proportional to "
            "$(1 - r^2)^{\\eta - 1}$: $\\eta < 1$ favors strong correlations, $\\eta = 1$ is flat, and $\\eta > 1$ "
            "pulls the correlation toward 0. The ellipses use unit standard deviations, so only the correlation changes their shape."
        ),
        lkj_eta,
    ])
    return (lkj_eta,)


@app.cell(hide_code=True)
def lkj_explorer(lkj_eta):
    _rng = np.random.default_rng(4)
    _R = sample_correlation_matrix_LKJ(eta=lkj_eta.value, n_samples=4000, random_seed=_rng)
    _r = _R[:, 0, 1]
    _rs = np.linspace(-0.99, 0.99, 400)
    _density = stats.beta(lkj_eta.value, lkj_eta.value).pdf((_rs + 1) / 2) / 2  # r = 2u - 1 with u ~ Beta(eta, eta)

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 5))
    _axs[0].hist(_r, bins=np.linspace(-1, 1, 61), density=True, color="C0", alpha=0.4, label="4000 draws")
    _axs[0].plot(_rs, _density, color="k", label="exact density")
    _axs[0].set(xlim=[-1, 1], xlabel="correlation", ylabel="density", title=f"LKJCorr($\\eta$={lkj_eta.value})")
    _axs[0].legend()

    plt.sca(_axs[1])
    for _Ri in _R[:20]:
        plot_2d_gaussian_ci(_Ri, color="C0", alpha=0.3)
    _axs[1].set(xlim=[-3, 3], ylim=[-3, 3], xlabel="$\\alpha$", ylabel="$\\beta$", title="20 sampled 89% ellipses", aspect="equal")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model with correlated features
    #### A couple of notes ⚠️
    - We jump straight to the implementation with uncentered priors here, skipping the centered version. For details on the uncentered implementation, see the [PyMC docs](https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/case_studies/LKJ.html)
    - We use `LKJCholeskyCov` instead of `LKJCorr` for improved numerical stability and performance. For details, see https://discourse.pymc.io/t/uses-of-lkjcholeskycov-and-lkjcorr/3629/3
    - We don't need to implement the `Chol_to_Corr` function in the lecture, as we can just pull the correlations directly out of the `LKJCholeskyCov` distribution
    """)
    return


@app.cell
def correlated_fit(DISTRICT, DISTRICT_ID, URBAN_ID, USES_CONTRACEPTION):
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
def correlated_posterior_summary(correlated_inference):
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
def correlated_prior_predictive(correlated_model):
    with correlated_model:
        # Add prior samples
        correlated_prior_predictive = pm.sample_prior_predictive(random_seed=2)
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

    plot_2d_gaussian_ci(
        R=posterior_R,
        mean=(alpha_mean, beta_mean),
        std=posterior_std,
        color='C0',
        label='89% ellipse',
    )

    utils.plot_scatter(
        xs=posterior_mean['alpha'],
        ys=posterior_mean['beta'],
        color='k', label='district param means'
    )

    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlim([-2.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.legend();
    plt.title(title)


@app.cell(hide_code=True)
def correlation_posterior_plot(
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
def correlated_vs_uncorrelated_plot(
    correlated_posterior_mean,
    uncorrelated_inference,
):
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
    Comparing to the model that does not model feature correlations: the model with uncorrelated features exhibits a much weaker negative correlation (i.e. the red ellipse on the left is less "tilted" downward). The uncorrelated model has no correlation parameter, so its ellipse uses the correlation of the district means.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Comparing models on the outcome scale.
    """)
    return


@app.cell
def outcome_scale_plot_helper(DISTRICT):
    def plot_p_C_district_rural_urban(posterior_mean, title=None):
        for ii, (label, p_C) in enumerate(zip(['rural', 'urban'], ['p_C_rural', 'p_C_urban'])):
            color = f"C{ii}"
            plot_data = posterior_mean[p_C]
            utils.plot_scatter(DISTRICT, plot_data, label=label, color=color, alpha=.8)

            # Add quantile indicators
            ql, median, qu = np.quantile(plot_data, [.025, .5, .975])
            error = np.array((median-ql, qu-median))[:, None]
            plt.errorbar(
                x=0,
                y=median,
                yerr=error,
                color=color,
                capsize=5,
                linewidth=2,
                label=f'95% IPR: ({ql:0.2f}, {qu:0.2f})'
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
def outcome_scale_plot(
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
    - The bars show how far the district posterior means spread (the 2.5% to 97.5% inter-percentile range, IPR), not how uncertain any one district is.
    - Modeling the feature correlation pulls the urban means closer together: their standard deviation across districts is 0.071 in the correlated model (top) against 0.090 in the uncorrelated model (bottom), and the urban IPR is (0.37, 0.65) against (0.34, 0.65). The rural means spread slightly more (0.085 against 0.072).
    - The uncertainty of each district barely changes: the average posterior standard deviation of urban $p(C)$ is 0.121 against 0.123.
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
def outcome_space_plot(correlated_posterior_mean, uncorrelated_posterior_mean):
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
    - There's a negative correlation between rural contraceptive use ($\alpha_j$) and the urban **difference** ($\beta_j$)
      - When a district's rural areas have high contraceptive use, the difference will be smaller (i.e. urban areas will also have high contraceptive use)
      - When rural contraceptive use is low, urban areas will be more different (i.e. urban areas will still tend to have higher contraceptive use)
    """)
    return


@app.cell
def correlated_pooling_helper(DISTRICT, FERTILITY):
    def grid_correlated_district(k_rural, n_rural, k_urban, n_urban, rho, alpha_bar, beta_bar, sigma, tau, grid=np.linspace(-5, 5, 401)):
        """Exact grid posterior of one district's (alpha, beta), with the population values held fixed."""
        alpha, beta = np.meshgrid(grid, grid, indexing="ij")
        cov = np.array([[sigma**2, rho * sigma * tau], [rho * sigma * tau, tau**2]])
        log_prior = stats.multivariate_normal([alpha_bar, beta_bar], cov).logpdf(np.dstack([alpha, beta]))

        def log_likelihood(k, n, log_odds):
            # k of n women use contraception, each with probability invlogit(log_odds)
            return -k * np.logaddexp(0, -log_odds) - (n - k) * np.logaddexp(0, log_odds)

        log_posterior = log_prior + log_likelihood(k_rural, n_rural, alpha) + log_likelihood(k_urban, n_urban, alpha + beta)
        posterior = np.exp(log_posterior - log_posterior.max())
        return alpha, beta, posterior / posterior.sum()


    # Number of women (n) and contraceptive users (k) per district, rural and urban
    _counts = (
        FERTILITY.groupby(["district", "urban"])["use.contraception"]
        .agg(["sum", "count"])
        .unstack("urban", fill_value=0)
    )
    DISTRICT_COUNTS = pd.DataFrame({
        "k_rural": _counts[("sum", 0)],
        "n_rural": _counts[("count", 0)],
        "k_urban": _counts[("sum", 1)],
        "n_urban": _counts[("count", 1)],
    }).reindex(DISTRICT, fill_value=0).astype(int)
    return DISTRICT_COUNTS, grid_correlated_district


@app.cell(hide_code=True)
def correlated_pooling_controls(DISTRICT_COUNTS, correlated_posterior_mean):
    _labels = {}
    for _d, _row in DISTRICT_COUNTS.iterrows():
        if _row.n_rural + _row.n_urban == 0:
            _note = " (no data)"
        elif _row.n_urban == 0:
            _note = " (no urban women)"
        elif _row.n_rural == 0:
            _note = " (no rural women)"
        else:
            _note = ""
        _labels[f"District {_d}{_note}"] = _d

    _fitted_rho = float(correlated_posterior_mean["feature_corr"][0, 1])
    pooling_district = mo.ui.dropdown(_labels, value="District 24 (no urban women)", label="District")
    pooling_rho = mo.ui.slider(-0.95, 0.95, step=0.05, value=round(_fitted_rho * 20) / 20, label="$\\rho$", show_value=True)
    pooling_hide_urban = mo.ui.checkbox(label="hide this district's urban data")
    mo.vstack([
        mo.md(
            "**Correlated pooling demo.** Pick a district and a correlation $\\rho$ between $\\alpha_j$ and $\\beta_j$. "
            "The other population values are fixed at the correlated model's posterior means, so the district's posterior "
            "can be computed exactly on a grid. Dashed lines show the same posterior with $\\rho = 0$. "
            "With $\\rho \\neq 0$, the rural data also moves the urban estimate. With the fitted negative $\\rho$, "
            "low rural use predicts a larger urban difference $\\beta_j$, so the urban estimate moves toward the average. "
            "District 24 has low rural use and no urban women; district 34 with its urban data hidden shows the effect most."
        ),
        mo.hstack([pooling_district, pooling_rho, pooling_hide_urban], justify="start"),
    ])
    return pooling_district, pooling_hide_urban, pooling_rho


@app.cell(hide_code=True)
def correlated_pooling_explorer(
    DISTRICT_COUNTS,
    correlated_posterior_mean,
    grid_correlated_district,
    pooling_district,
    pooling_hide_urban,
    pooling_rho,
):
    _alpha_bar = float(correlated_posterior_mean["alpha_bar"])
    _beta_bar = float(correlated_posterior_mean["beta_bar"])
    _sigma, _tau = correlated_posterior_mean["feature_std"].values
    _c = DISTRICT_COUNTS.loc[pooling_district.value]
    _k_urban, _n_urban = (0, 0) if pooling_hide_urban.value else (_c.k_urban, _c.n_urban)


    def _invlogit(x):
        return 1 / (1 + np.exp(-x))


    def _p_density(log_odds, posterior):
        # Bin on the log-odds grid (alpha + beta also lands on it), then change variables to p
        _step = np.diff(np.unique(log_odds))[0]
        _edges = np.arange(log_odds.min() - _step / 2, log_odds.max() + _step, _step)
        _density = np.histogram(log_odds.ravel(), bins=_edges, weights=posterior.ravel())[0] / _step
        _p = _invlogit((_edges[:-1] + _edges[1:]) / 2)
        return _p, _density / (_p * (1 - _p))


    def _region_level(posterior, prob=0.89):
        # density level that encloses `prob` of the grid posterior
        _sorted = np.sort(posterior.ravel())[::-1]
        return _sorted[np.searchsorted(np.cumsum(_sorted), prob)]


    _fig, _axs = plt.subplots(1, 2, figsize=(11, 5))
    _means = {}
    for _rho, _style in [(pooling_rho.value, "-"), (0.0, "--")]:
        _alpha, _beta, _post = grid_correlated_district(
            _c.k_rural, _c.n_rural, _k_urban, _n_urban, _rho, _alpha_bar, _beta_bar, _sigma, _tau
        )
        _axs[0].contour(_alpha, _beta, _post, levels=[_region_level(_post)], colors="k", linestyles=_style)
        _axs[0].plot([], [], color="k", linestyle=_style, label=f"posterior, $\\rho$ = {_rho:.2f}")
        for _ii, (_area, _log_odds) in enumerate([("rural", _alpha), ("urban", _alpha + _beta)]):
            _means[(_area, _rho)] = float((_invlogit(_log_odds) * _post).sum())
            _p, _density = _p_density(_log_odds, _post)
            _axs[1].plot(_p, _density, color=f"C{_ii}", linestyle=_style, label=f"{_area}, $\\rho$ = {_rho:.2f}")

    plt.sca(_axs[0])
    plot_2d_gaussian_ci(
        np.array([[1, pooling_rho.value], [pooling_rho.value, 1]]),
        mean=(_alpha_bar, _beta_bar), std=(_sigma, _tau), color="gray", alpha=0.5,
        label="population prior",
    )
    _axs[0].set(xlim=[-3, 2], ylim=[-2.5, 3.5], xlabel="$\\alpha_j$", ylabel="$\\beta_j$", title="89% regions")
    _axs[0].legend(loc="upper right")

    for _ii, (_area, _k, _n) in enumerate([("rural", _c.k_rural, _c.n_rural), ("urban", _c.k_urban, _c.n_urban)]):
        if _n > 0:
            _hidden = _area == "urban" and pooling_hide_urban.value
            _axs[1].axvline(_k / _n, color="gray" if _hidden else f"C{_ii}", linestyle=":", label=f"{_area} observed" + (" (hidden)" if _hidden else ""))
    _axs[1].set(xlim=[-0.02, 1.02], xlabel="prob. contraceptive use", ylabel="density", title=f"District {pooling_district.value}")
    _axs[1].legend(loc="best", fontsize=9)
    _fig.tight_layout()


    def _observed(k, n):
        return f"{k}/{n} = {k / n:.2f}" if n else "no women"


    _table = pd.DataFrame({
        "observed": [_observed(_c.k_rural, _c.n_rural), _observed(_c.k_urban, _c.n_urban) + (" (hidden)" if pooling_hide_urban.value and _c.n_urban else "")],
        f"posterior mean, ρ = {pooling_rho.value:.2f}": [round(_means[("rural", pooling_rho.value)], 3), round(_means[("urban", pooling_rho.value)], 3)],
        "posterior mean, ρ = 0": [round(_means[("rural", 0.0)], 3), round(_means[("urban", 0.0)], 3)],
    }, index=["rural", "urban"])
    mo.vstack([_table, _fig])
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
      - leading to "divergent" transitions
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

    Instead of the HMC animation McElreath uses to visualize each divergent path, we verify that the number of divergences increases when sampling the devil's funnel as we increase the v prior's $\sigma$ in the devil's funnel prior. The funnel explorer further down also plots the draws and marks the divergent transitions.

    ### Centered-prior models
    """)
    return


@app.cell
def centered_funnel():
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
    - Re-parameterize (transform) to make surface smoother

    ## Non-centered prior
    We add an auxiliary variable $z$ that has a smooth probability surface. We then sample that auxiliary variable, and transform it to obtain the target variable distribution. For the case of the devil's funnel prior:

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
def noncentered_funnel():
    _xs = _vs = np.linspace(-4, 4, 100)
    _sigma_vs = [.5, 1, 3]
    _n_sigma_vs = len(_sigma_vs)
    _fig, _axs = plt.subplots(3, 1, figsize=(4, 4 * _n_sigma_vs))

    def _prior_noncentered_devils_funnel(xs, vs, sigma_v=.5):
        """Reparameterize into a smoother auxiliary variable space"""
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

    - The non-centered version has no divergences at any of the three values of $\sigma_v$, against 4, 108 and 405 for the centered version.
    """)
    return


@app.cell(hide_code=True)
def funnel_controls():
    funnel_sigma_v = mo.ui.slider(steps=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0], value=1.0, label="$\\sigma_v$", show_value=True)
    funnel_sample_button = mo.ui.run_button(label="Sample both models")
    mo.vstack([
        mo.md(
            "**Funnel explorer.** Pick $\\sigma_v$ and sample the centered and the non-centered funnel. "
            "Divergent transitions are marked in red. A new $\\sigma_v$ takes a few seconds to sample; it is cached after that."
        ),
        mo.hstack([funnel_sigma_v, funnel_sample_button], justify="start"),
    ])
    return funnel_sample_button, funnel_sigma_v


@app.cell
def funnel_fits(funnel_sample_button, funnel_sigma_v):
    mo.stop(not funnel_sample_button.value, mo.md("Pick $\\sigma_v$ and click **Sample both models**."))

    with pm.Model() as funnel_centered_model:
        _v = pm.Normal('v', 0, funnel_sigma_v.value)
        _x = pm.Normal('x', 0, pm.math.exp(_v))
        funnel_centered_inference = cached_sample()

    with pm.Model() as funnel_noncentered_model:
        _v = pm.Normal('v', 0, funnel_sigma_v.value)
        _z = pm.Normal('z', 0, 1)
        _x = pm.Deterministic('x', _z * pm.math.exp(_v))
        funnel_noncentered_inference = cached_sample()
    return funnel_centered_inference, funnel_noncentered_inference


@app.cell(hide_code=True)
def funnel_explorer(
    funnel_centered_inference,
    funnel_noncentered_inference,
    funnel_sigma_v,
):
    _v_floor = stats.norm(0, funnel_sigma_v.value).ppf(0.01)
    _rows = []
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for _ax, (_label, _idata) in zip(_axs, [("centered", funnel_centered_inference), ("non-centered", funnel_noncentered_inference)]):
        _x = _idata.posterior["x"].values.ravel()
        _v = _idata.posterior["v"].values.ravel()
        _divergent = _idata.sample_stats["diverging"].values.ravel()
        _summary = az.summary(_idata, var_names=["v"])
        _rows.append({
            "model": _label,
            "divergences": int(_divergent.sum()),
            "ESS bulk, v": int(_summary.loc["v", "ess_bulk"]),
            "r̂, v": _summary.loc["v", "r_hat"],
            "lowest v": round(float(_v.min()), 2),
        })
        _ax.scatter(_x[~_divergent], _v[~_divergent], s=4, color="gray", alpha=0.3, label="draws")
        _ax.scatter(_x[_divergent], _v[_divergent], s=12, color="C0", label="divergent")
        _ax.axhline(_v_floor, color="k", linestyle="--", label="1% quantile of v")
        _ax.set(title=f"{_label}: {int(_divergent.sum())} divergences", xlabel="x")
    _axs[0].set(xlim=[-6, 6], ylim=[-3.5 * funnel_sigma_v.value, 3.5 * funnel_sigma_v.value], ylabel="v")
    _axs[0].legend(loc="upper right")
    _fig.tight_layout()

    mo.vstack([
        pd.DataFrame(_rows).set_index("model"),
        _fig,
        mo.md(
            "The neck of the funnel, at low $v$, is where $x$ is squeezed close to 0. A step size that works in the wide mouth "
            "is far too large there, so the centered sampler diverges at the edge of the neck and rarely gets inside; "
            "1% of the draws should fall below the dashed line. The non-centered sampler draws $v$ and $z$ independently, "
            "so the neck is no harder to reach than the mouth."
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Check HMC Diagnostics
    """)
    return


@app.cell
def funnel_diagnostics(noncentered_devils_funnel_inference):
    utils.check_sampling(noncentered_devils_funnel_inference)
    az.summary(noncentered_devils_funnel_inference)['r_hat']
    return


@app.cell
def funnel_trace_plot(noncentered_devils_funnel_inference):
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
    X_\Omega = \text{diag}(\sigma_i)\bf{LZ},
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
def cholesky_example():
    orig_corr = np.array([[1, .6], [.6, 1]])
    print("Ground-truth Correlation Matrix\n", orig_corr)

    L = np.linalg.cholesky(orig_corr)
    print("\nCholesky Matrix of correlation matrix, L\n", L)
    print("\nReconstructed correlation matrix from L\n", L @ L.T)
    return L, orig_corr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Demonstrate sampling correlated data from z-scores and the Cholesky factor
    """)
    return


@app.cell
def cholesky_sampling_demo(L, orig_corr):
    np.random.seed(12345)
    _N = 100_000
    _orig_sigmas = [2, .5]

    # Matrix of z-score samples
    _Z = stats.norm().rvs(size=(2, _N))
    print("Raw z-score samples are independent\n", np.corrcoef(_Z[0], _Z[1]))

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
      - Lots of data in each cluster/subgroup
      - likelihood-dominant scenario
    - Non-centered:
      - Many clusters/subgroups, with sparse data in some of the clusters
      - prior-dominant scenario
    """)
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
