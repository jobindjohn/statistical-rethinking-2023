import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import inspect
    from functools import partial

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    from scipy import stats
    from matplotlib import pyplot as plt

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 09 - Modeling Events](https://www.youtube.com/watch?v=Zi6N3GLUJmw)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # UC Berkeley Admissions

    ## Dataset
    - 4562 Graduate student applications
    - Stratified by
      - Department
      - Gender

    Goal is to identify gender discrimination by admissions officers
    """)
    return


@app.cell
def _():
    ADMISSIONS = utils.load_data("UCBadmit")
    ADMISSIONS
    return (ADMISSIONS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modeling Events
    - **Events**: discrete, unordered outcomes
    - Observations are counts/aggregates
    - Unknowns are probabilities $p$ of event ocurring, or odds of those probabilities $\frac{p}{1-p}$
    - All things we stratify by interact -- generally never independent in real life
    - Often deal with the Log Odds of $p = \log \left (\frac{p}{1-p} \right)$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Admissions: Drawing the owl 🦉

    1. Estimands(s)
    2. Scientific Models(s)
    3. Statistical Models(s)
    4. Analysis
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Estimand(s)

    ## Which path defines "discrimination"

    ### Direct Discrimination (Causal Effect)
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A")
        ],
        node_props={
            "G": {"label": "Gender, G", "color": "red"},
            "D": {"label": "Department, D"},
            "A": {"label": "Admission rate, A", "color": "red"},
        },
        edge_props={
            ("G", "A"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - aka "Institutional discrimination"
    - Referees are biased for or against a particular group
    - Usually the type we're interested in identifying if it exists
    - Requires strong causal assumptions

    Here, deparment, D is a mediator -- this is a common structure in social sciences, where categorical status (e.g. gender) effects some mediating context (e.g. occupation), both of which affect a target outcome (wage). Examples
    - wage discrimination
    - hiring
    - scientific awards
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Indirect Discrimination
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A")
        ],
        node_props={
            "G": {"label": "Gender, G", "color": "red"},
            "D": {"label": "Department, D"},
            "A": {"label": "Admission rate, A", "color": "red"},
        },
        edge_props={
            ("G", "D"): {"color": "red"},
            ("D", "A"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - aka "Structural discrimination"
    - e.g. Gender affects a person's interests, and therefore the department they will apply to
    - Affects overall admission rates
    - Requires strong causal assumptions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Total Discrimination
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A")
        ],
        node_props={
            "G": {"label": "Gender, G", "color": "red"},
            "D": {"label": "Department, D"},
            "A": {"label": "Admission rate, A", "color": "red"},
        },
        edge_props={
            ("G", "D"): {"color": "red"},
            ("G", "A"): {"color": "red"},
            ("D", "A"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - aka "Experienced discrimination"
    - through both direct and indirect routes
    - Requires mild assumptions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Estimands & Estimators
    - Each of the **different estimands require a different estimators**
    - Often the thing we **can estimate** often isn't what we **want to estimate**
    - e.g. Total discrimination may be easier to estimate, but is less actionable
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Unobserved Confounds
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A"),
            ("U", "D"),
            ("U", "A")
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
    It's always possible there are also confounds between the mediator and some unobserved confounds. We will ignore these for now.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Scientific Model(s):

    ## Simulate the process
    Below is a generative model of the review/admission process
    """)
    return


@app.cell
def _():
    np.random.seed(123)
    _n_samples = 1000

    GENDER = stats.bernoulli.rvs(p=.5, size=_n_samples)

    # Groundtruth parameters
    # Gender 1 tends to apply to department 1
    P_DEPARTMENT = np.where(GENDER==0, .3, .8)

    # Acceptance rates matrices -- Department x Gender
    UNBIASED_ACCEPTANCE_RATES = np.array([[.1, .1], [.3, .3]])  # No *direct* gender bias

    # Biased acceptance:
    # - dept 0 accepts gender 0 at 50% of unbiased rate
    # - dept 1 accepts gender 0 at 67% of unbiased rate
    BIASED_ACCEPTANCE_RATES = np.array([[.05, .1], [.2, .3]])   # *direct* gender bias present

    DEPARTMENT = stats.bernoulli.rvs(p=P_DEPARTMENT, size=_n_samples).astype(int)


    def simulate_admissions_data(bias_type):
        """Simulate admissions data using the global params above"""
        acceptance_rate = BIASED_ACCEPTANCE_RATES if bias_type == 'biased' else UNBIASED_ACCEPTANCE_RATES
        acceptance = stats.bernoulli.rvs(p=acceptance_rate[DEPARTMENT, GENDER])

        return pd.DataFrame(
            np.vstack([GENDER, DEPARTMENT, acceptance]).T,
            columns=['gender', 'department', 'accepted']
        ), acceptance_rate

    for _bias_type in ['unbiased', 'biased']:

        _fake_admissions, _acceptance_rate = simulate_admissions_data(_bias_type)

        _gender_acceptance_counts = _fake_admissions.groupby(["gender", "accepted"]).count()['department']
        _gender_acceptance_counts.name = None

        _gender_department_counts = _fake_admissions.groupby(["gender", "department"]).count()['accepted']
        _gender_department_counts.name = None

        _observed_acceptance_rates = _fake_admissions.groupby("gender").mean()['accepted']
        _observed_acceptance_rates.name = None

        print()
        print('-' * 30)
        print(_bias_type.upper())
        print('-' * 30)
        print(f'Department Acceptance rate:\n{_acceptance_rate}')
        print(f'\nGender-Department Frequency:\n{_gender_department_counts}')
        print(f'\nGender-Acceptance Frequency:\n{_gender_acceptance_counts}')
        print(f'\nOverall Acceptance Rates:\n{_observed_acceptance_rates}')
    return (
        BIASED_ACCEPTANCE_RATES,
        UNBIASED_ACCEPTANCE_RATES,
        simulate_admissions_data,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulated data properties

    #### Both scenarios

    - Gender 0 tends to apply to department 0
    - Gender 1 tends to apply to department 1

    #### Unbiased scenario:
    - due to lower acceptance rates in dept 0 and tendency of gender 0 to apply to dept 0, gender 0 has a lower overall rejection rate compared to gender 1
    - due to higher acceptance rates in dept 1 and tendency of gender 1 to apply to dept 1, gender 1 has a higher overall acceptance rate compared to gender 0
    - even in the case of no (direct) gender discrimination, **there is still indirect discrimination based on tendency of genders to apply to different departments, and the unqual likelihood that each department accepts students.**

    #### Biased scenario
    - overall acceptance rates are lower (due to baseline reduction in gender 0 acceptance rates across both departments)
    - in the scenario where there is actual department bias, **we see a similar overall _pattern_ of discrimination as the unbiased case due to the indirect effect.**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Statistical Models
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modeling Events
    - **We observe** counts of events
    - **We estimate** probabilities -- or, rather, the log-odds of events ocurring

    ### Linear Models
    Expected value is linear (additive) combination of parameters

    $$
    \begin{align*}
    Y_i &\sim \mathcal{N}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta_X X_i + ...
    \end{align*}
    $$

    b.c. Normal distribution is **unbounded**, so too is the expected value of the linear model.

    ### Event Models
    Discrete events either occur, taking the value 1, or they do not, taking the value 0. This **puts bounds** on the expected value of an event. Namely the bounds are on the interval $(0, 1)$

    ## Generalized Linear Models & Link Functions

    - Expected value is **some function $f(\mathbb{E}[\theta])$ of an additive combination of the parameters**.

    $$
    f(\mathbb{E}[\theta]) = \alpha + \beta_X X_i + ...
    $$

    - Generally, this function $f()$, called the **link function**, will have a specific form **that is associated with the likelihood distribution**.
    - the link function will have an **inverse link function** $ f^{-1}(X_i, \theta)$ such that
    - to reiterate **link functions are matched to distributions**

    $$
    \mathbb{E}[\theta] = f^{-1}(\alpha + \beta_X X_i + ...)
    $$

    - In the linear regression case the likelihood model is Gaussian, and the associated link function (and inverse link function) is just the identity, $I()$. (left plot below)
    """)
    return


@app.cell(hide_code=True)
def _():
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))
    _xs = np.linspace(-3, 3, 100)
    plt.sca(_axs[0])
    _alpha = 0
    _beta = .75

    plt.plot(_xs, _alpha + _beta * _xs)
    plt.axhline([0], linestyle='--', color='gray')
    plt.axhline([1], linestyle='--', color='gray')
    plt.xlim([-3, 3])
    plt.ylim([-.5, 1.5])
    plt.xlabel('x')
    plt.ylabel('$I(\\alpha + \\beta X$)')
    plt.title("Linear Model\n(continous outcomes)")

    plt.sca(_axs[1])
    plt.plot(_xs, utils.invlogit(_alpha + _beta * _xs))
    plt.axhline([0], linestyle='--', color='gray')
    plt.axhline([1], linestyle='--', color='gray')
    plt.xlim([-3, 3])
    plt.ylim([-.5, 1.5])
    plt.xlabel('x')
    plt.ylabel('$\\frac{1}{e^{-(\\alpha + \\beta X)}}$')
    plt.title("Logistic Model\n(binary outcomes)")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Logit Link Function and Logisitic Inverse function
    ### Binary outcomes

    - **Logistic regression** used to model the probability of an event ocurring
    - Likelihood function is the Bernouilli
    - Associated link function is the **log-odds, or logit**: $f(p) = \frac{p}{1-p}$

    $$
    \begin{align*}
    Y_i &\sim \text{Bernoulli}(p_i) \\
    f(p_i) &= \alpha + \beta_X X_i + ...
    \end{align*}
    $$

    - the inverse link function is the **inverse logit** aka **logistic** function (right plot above): $f^{-1}(X_i, \theta) = \frac{1}{1 + e^{-(\alpha + \beta_X X_i + ...)}}$
      - defining $\alpha + \beta_X X + ... = q$ (ignoring the data index $i$ for now), the derivation of the inverse logit is as follows
    $$
    \begin{align}
    \log \left(\frac{p}{1-p}\right) &= \alpha + \beta_X X + ... = q \\
    \frac{p}{1-p} &= e^{q} \\
    p &= (1-p) e^{q} \\
    p + p e^{q} &= e^{q} \\
    p(1 + e^{q}) &= e^{q} \\
    p &= \frac{ e^{q}}{(1 + e^{q})} \\
    p &= \frac{ e^{q}}{(1 + e^{q})} \frac{e^{-q}}{e^{-q}}\\
    p &= \frac{1}{(1 + e^{-q})} \\
    \end{align}
    $$

    ### logit is a harsh transform
    Interpreting the log odds can be difficult at first, but in time becomes easier
    - log-odds scale
    - $\text{logit}(p)=0, p=0.5$
    - $\text{logit}(p)=-\infty, p=0$
      - rule of thumb: $\text{logit}(p)<-4$ means event is unlikely (hardly ever)
    - $\text{logit}(p)=\infty, p=1$
      - rule of thumb: $\text{logit}(p)>4$ means event is very likely (nearly always)
    """)
    return


@app.cell(hide_code=True)
def _():
    _fig, _ax = plt.subplots()
    _log_odds = np.linspace(-4, 4, 100)
    utils.plot_line(_log_odds, utils.invlogit(_log_odds), label=None)
    plt.axhline(0.5, linestyle="--", color='k', label="p=0.5")
    plt.xlabel("$logit(p)$")
    plt.ylabel("$p$")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Bayesian Updating for Logistic Regression
    For the following simulation, we'll use a custom utility function `utils.simulate_2_parameter_bayesian_learning` for simulating general Bayeisan posterior update simulation. Here's the API for that function (for more details see `utils.py`)
    """)
    return


@app.cell
def _():
    mo.plain_text(inspect.getdoc(utils.simulate_2_parameter_bayesian_learning))
    return


@app.function
def logistic_model(x, alpha, beta):
    return utils.invlogit(alpha + beta * x)


@app.function
def logistic_regression_posterior(x_obs, y_obs, alpha_grid, beta_grid, likelihood_prior_std = .01):

    # Convert params to 1-d arrays
    if np.ndim(alpha_grid) > 1:
        alpha_grid = alpha_grid.ravel()

    if np.ndim(beta_grid):
        beta_grid = beta_grid.ravel()

    log_prior_intercept = stats.norm(0, 1).logpdf(alpha_grid)
    log_prior_slope = stats.norm(0, 1).logpdf(beta_grid)

    log_likelihood = np.array(
        [
            stats.bernoulli(
                p=utils.invlogit(a + b * x_obs)
            ).logpmf(y_obs) for a, b in zip(alpha_grid, beta_grid)
        ]
    ).sum(axis=1)

    # Posterior is equal to the product of likelihood and priors (here a sum in log scale)
    log_posterior = log_likelihood + log_prior_intercept + log_prior_slope

    # Convert back to natural scale
    return np.exp(log_posterior - log_posterior.max())


@app.cell
def _():
    # Generate data for demo
    np.random.seed(123)
    _RESOLUTION = 100
    _N_DATA_POINTS = 128

    # Ground truth parameters
    _ALPHA = -1
    _BETA = 2

    _x = stats.norm(0, 1).rvs(size=_N_DATA_POINTS)
    _p_y = utils.invlogit(_ALPHA + _BETA * _x)
    _y = stats.bernoulli.rvs(p=_p_y)

    _alpha_grid = np.linspace(-3, 3, _RESOLUTION)
    _beta_grid = np.linspace(-3, 3, _RESOLUTION)

    # Vary the sample size to show how the posterior adapts to more and more data
    _figs = []
    for _n_samples in [0, 2, 4, 8, 16, 32, 64, 128]:
        # Run the simulation
        utils.simulate_2_parameter_bayesian_learning(
            x_obs=_x[:_n_samples],
            y_obs=_y[:_n_samples],
            param_a_grid=_alpha_grid,
            param_b_grid=_beta_grid,
            true_param_a=_ALPHA,
            true_param_b=_BETA,
            model_func=logistic_model,
            posterior_func=logistic_regression_posterior,
            param_labels=['$\\alpha$', '$\\beta$'],
            data_range_x=(-2, 2), data_range_y=(-.05, 1.05)
        )
        _figs.append(plt.gcf())
    mo.vstack(_figs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Priors for logistic regression

    **logit link function is a harsh transform**
    - Logit compresses parameter distributions
    - $x > +4 \rightarrow $ event basically always occurs
    - $x < -4 \rightarrow$ event basically never occurs
    """)
    return


@app.cell
def _():
    _n_samples = 10000
    _fig, _axs = plt.subplots(3, 2, figsize=(8, 8))
    for _ii, _std in enumerate([10, 1.5, 1]):

        # Alpha prior distribution
        _alphas = stats.norm(0, _std).rvs(_n_samples)
        plt.sca(_axs[_ii][0])
        utils.plot_dist(_alphas, color='C1')
        plt.xlim([-30, 30])
        plt.ylabel('density')
        if _ii == 2:
            plt.xlabel("alpha")

        # Resulting event probability distribution
        _ps = utils.invlogit(_alphas)
        plt.sca(_axs[_ii][1])
        utils.plot_dist(_ps, color="C0")
        plt.xlim([0, 1])
        plt.ylabel('density')
        plt.title(f"$\\alpha \\sim \\mathcal{{N}}(0, {_std})$")
        if _ii == 2:
            plt.xlabel("p(event)")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior Predictive Simulations
    """)
    return


@app.function
def gaussian_2d_pdf(xs, ys, mean=(0, 0), covariance=np.eye(2)):
    return np.array(stats.multivariate_normal(mean, covariance).pdf(np.vstack([xs, ys]).T))


@app.cell
def _():
    _pdf = partial(gaussian_2d_pdf)

    _xs =  _ys = np.linspace(-3, 3, 100)
    # utils.plot_2d_function(xs, ys, pdf, cmap='gray_r')
    return


@app.function
def plot_logistic_prior_predictive(n_samples = 20, prior_std=.5):

    _, axs = plt.subplots(1, 2, figsize=(10, 4))
    plt.sca(axs[0])
    min_x, max_x = -prior_std, prior_std
    xs = np.linspace(min_x, max_x, 100)
    ys = xs

    pdf = partial(gaussian_2d_pdf, covariance=np.array([[prior_std ** 2, 0],[0, prior_std ** 2]]))
    utils.plot_2d_function(xs, ys, pdf, cmap='gray_r')

    plt.axis("square")
    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    # Sample some parameters from prior
    βs = stats.norm.rvs(0, prior_std, size=n_samples)
    αs = stats.norm.rvs(0, prior_std, size=n_samples)

    for b, a in zip(βs, αs):
        plt.scatter(a, b)

    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.title(f"Samples from prior with std={prior_std}")

    # Resulting sigmoid functions
    plt.sca(axs[1])
    min_x, max_x = -3, 3
    xs = np.linspace(min_x, max_x, 100)
    for a, b in zip(αs, βs):
        logit_p = a + xs * b
        p = utils.invlogit(logit_p)
        plt.plot(xs, p)

    plt.xlabel('x')
    plt.ylabel('p = invlogit(x)')
    plt.xlim([-3, 3])
    plt.ylim([-.05, 1.05])
    plt.axhline(0.5, linestyle='--', color='k')
    plt.title(f"Resulting Logistic Models")


@app.cell
def _():
    plot_logistic_prior_predictive(prior_std=0.5)
    plt.gcf()
    return


@app.cell
def _():
    plot_logistic_prior_predictive(prior_std=1.)
    plt.gcf()
    return


@app.cell
def _():
    plot_logistic_prior_predictive(prior_std=10)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Statistical models for admissions
    Again, the estimator will depend on the estimand
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Total Causal Effect
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A")
        ],
        node_props={
            "G": {"color": "red"},
            "A": {"color": "red"},
        },
        edge_props={
            ("G", "A"): {"color": "red"},
            ("D", "A"): {"color": "red"},
            ("G", "D"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Stratify by only Gender**. Don't stratify by Department b.c. it's a Pipe (mediator) that we do not want to block

    $$
    \begin{align*}
    A_i &\sim \text{Bernoulli}(p=p_i) \\
    \text{logit}(p_i) &= \alpha[G_i] \\
    \alpha &= [\alpha_0, \alpha_1] \\
    \alpha_j &\sim \text{Normal}(0, 1)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Direct Causal Effect
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A")
        ],
        node_props={
            "G": {"color": "red"},
            "A": {"color": "red"},
        },
        edge_props={
            ("G", "A"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Stratify by Gender and Department** to block the Pipe

    $$
    \begin{align*}
    A_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha[G_i, D_i] \\
    \alpha &= \left[{\begin{array}{cc} \alpha_{0,0}, \alpha_{0,1} \\ \alpha_{1,0}, \alpha_{1,1} \end{array}}\right] \\
    \alpha_{j,k} &\sim \text{Normal}(0, 1)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fitting Total Causal Effect Model
    """)
    return


@app.function
def fit_total_effect_model_admissions(data):
    """Fit total effect model, stratifying by gender.

    Note
    ----
    We use the Binomial regression, so simulated observation-level
    data from above must be pre-aggregated. (see the function
    `aggregate_admissions_data` below). We could have
    used a Bernoulli likelihood (Logistic regression) for the simulated
    data, but given that we only have aggregates for the real
    UC Berkeley data we choose this more general formulation.
    """

    # Set up observations / coords
    n_admitted = data['admit'].values
    n_applications = data['applications'].values

    gender_coded, gender = pd.factorize(data['applicant.gender'].values)

    with pm.Model(coords={"gender": gender}) as total_effect_model:

        # Mutable data for running any gender-based counterfactuals
        gender_coded = pm.Data("gender_id", gender_coded, dims="obs_ids")

        alpha = pm.Normal('alpha', 0, 1, dims="gender")

        # Record the inverse linked probabilities for reporting
        pm.Deterministic("p_accept", pm.math.invlogit(alpha), dims='gender')

        p = pm.math.invlogit(alpha[gender_coded])

        likelihood = pm.Binomial("accepted", n=n_applications, p=p, observed=n_admitted, dims="obs_ids")
        total_effect_inference = cached_sample()

    return total_effect_model, total_effect_inference


@app.function
def aggregate_admissions_data(raw_admissions_data):
    """Aggregate simulated data from `simulate_admissions_data`, which
    is in long format to short format, by so that it can be modeled
    using Binomial likelihood. We also recode column names to have the
    same fields as the UC Berkeley dataset so that we can use the
    same model-fitting functions.
    """

    # Aggregate applications, admissions, and rejections. Rename
    # columns to have the  same fields as UC Berkeley dataset
    applications = raw_admissions_data.groupby(["gender", "department"]).count().reset_index()
    applications.rename(columns={"accepted": "applications"}, inplace=True)

    admits = raw_admissions_data.groupby(["gender", "department"]).sum().reset_index()
    admits.rename(columns={"accepted": "admit"}, inplace=True)

    data = applications.merge(admits)
    data.loc[:, "reject"] = data.applications - data.admit

    # Code gender & department. For easier comparison to lecture,
    # we use 1-indexed gender/department indicators like McElreath
    data.loc[:, "applicant.gender"] = data.gender.apply(lambda x: 2 if x else 1)
    data.loc[:, "dept"] = data.department.apply(lambda x: 2 if x else 1)

    return data[["applicant.gender", "dept", "applications", "admit", "reject"]]


@app.function
def plot_admissions_model_posterior(inference, figsize=(6, 2)):
    fig_alpha, ax = plt.subplots(figsize=figsize)

    # Plot alphas
    utils.plot_forest(inference, var_names=['alpha'], hdi_prob=.89, ax=ax)

    # Plot inver-linked acceptance probabilities
    fig_p_accept, ax = plt.subplots(figsize=figsize)
    utils.plot_forest(inference, var_names=['p_accept'], hdi_prob=.89, ax=ax)

    return mo.vstack([az.summary(inference), fig_alpha, fig_p_accept])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Total Effect model on BIASED simulated admissions
    """)
    return


@app.cell
def _(simulate_admissions_data):
    # Fixed seed, so the simulated data (and the fits below) are the same on
    # every run, and the cached fits can be reused
    np.random.seed(1)

    SIMULATED_BIASED_ADMISSIONS, _ = simulate_admissions_data("biased")
    SIMULATED_BIASED_ADMISSIONS = aggregate_admissions_data(SIMULATED_BIASED_ADMISSIONS)
    SIMULATED_BIASED_ADMISSIONS
    return (SIMULATED_BIASED_ADMISSIONS,)


@app.cell
def _(SIMULATED_BIASED_ADMISSIONS):
    total_effect_model_simulated_biased, total_effect_inference_simulated_biased = fit_total_effect_model_admissions(SIMULATED_BIASED_ADMISSIONS)
    return (total_effect_inference_simulated_biased,)


@app.cell
def _(total_effect_inference_simulated_biased):
    plot_admissions_model_posterior(total_effect_inference_simulated_biased, (6, 1.5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Total Effect model on UNBIASED simulated admissions
    """)
    return


@app.cell
def _(simulate_admissions_data):
    # Fixed seed, so the simulated data (and the fits below) are the same on
    # every run, and the cached fits can be reused
    np.random.seed(2)

    SIMULATED_UNBIASED_ADMISSIONS, _ = simulate_admissions_data("unbiased")
    SIMULATED_UNBIASED_ADMISSIONS = aggregate_admissions_data(SIMULATED_UNBIASED_ADMISSIONS)
    SIMULATED_UNBIASED_ADMISSIONS
    return (SIMULATED_UNBIASED_ADMISSIONS,)


@app.cell
def _(SIMULATED_UNBIASED_ADMISSIONS):
    total_effect_model_simulated_unbiased, total_effect_inference_simulated_unbiased = fit_total_effect_model_admissions(SIMULATED_UNBIASED_ADMISSIONS)
    return (total_effect_inference_simulated_unbiased,)


@app.cell
def _(total_effect_inference_simulated_unbiased):
    plot_admissions_model_posterior(total_effect_inference_simulated_unbiased, (6, 1.5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitting Direct Causal Effect Model
    """)
    return


@app.function
def fit_direct_effect_model_admissions(data):
    """Fit total effect model, stratifying by gender.

    Note
    ----
    We use the Binomial likelihood, so simulated observation-level data from
    above must be pre-aggregated. (see the function `aggregate_admissions_data`
    below). We could have used a Bernoulli likelihood for the simulated data,
    but given that we only have aggregates for the real UC Berkeley data we
    choose this more general formulation.
    """

    # Set up observations / coords
    n_admitted = data['admit'].values
    n_applications = data['applications'].values

    department_coded, department = pd.factorize(data['dept'].values)
    gender_coded, gender = pd.factorize(data['applicant.gender'].values)

    with pm.Model(coords={"gender": gender, "department": department}) as direct_effect_model:

        # Mutable data for running any gender-based or department-based counterfactuals
        gender_coded = pm.Data("gender_id", gender_coded, dims="obs_ids")
        department_coded = pm.Data("department_id", department_coded, dims="obs_ids")
        n_applications = pm.Data("n_applications", n_applications, dims="obs_ids")

        alpha = pm.Normal('alpha', 0, 1, dims=["department", "gender"])

        # Record inverse linked probabilities for reporting
        pm.Deterministic("p_accept", pm.math.invlogit(alpha), dims=['department', 'gender'])

        p = pm.math.invlogit(alpha[department_coded, gender_coded])

        likelihood = pm.Binomial("accepted", n=n_applications, p=p, observed=n_admitted, dims="obs_ids")
        direct_effect_inference = cached_sample()

    return direct_effect_model, direct_effect_inference


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Direct Effect model to BIASED simulated admissions data
    """)
    return


@app.cell
def _(SIMULATED_BIASED_ADMISSIONS):
    direct_effect_model_simulated_biased, direct_effect_inference_simulated_biased = fit_direct_effect_model_admissions(SIMULATED_BIASED_ADMISSIONS)
    return (direct_effect_inference_simulated_biased,)


@app.cell
def _(direct_effect_inference_simulated_biased):
    plot_admissions_model_posterior(direct_effect_inference_simulated_biased)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For comparison, here's the ground truth biased admission rates, which we're able to mostly recover:
    """)
    return


@app.cell
def _(BIASED_ACCEPTANCE_RATES):
    # Department x Gender
    BIASED_ACCEPTANCE_RATES
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Direct Effect model to UNBIASED simulated admissions data
    """)
    return


@app.cell
def _(SIMULATED_UNBIASED_ADMISSIONS):
    direct_effect_model_simulated_unbiased, direct_effect_inference_simulated_unbiased = fit_direct_effect_model_admissions(SIMULATED_UNBIASED_ADMISSIONS)
    return (direct_effect_inference_simulated_unbiased,)


@app.cell
def _(direct_effect_inference_simulated_unbiased):
    plot_admissions_model_posterior(direct_effect_inference_simulated_unbiased)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For comparison, here's the ground truth unbiased admission rates, which were able to recover:
    """)
    return


@app.cell
def _(UNBIASED_ACCEPTANCE_RATES):
    # Department x Gender
    UNBIASED_ACCEPTANCE_RATES
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Analyze the UC Berkeley Admissions data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Review of the counts dataset
    - we'll use these counts data to model log odds of acceptance rate for gender/department
    - note that we'll be using Binomial Regression, which is equivalent to Bernoulli regression, but operates on aggregate counts, as oppose to individual binary trials
      - The examples above were actually implemented as Binomial Regression so that we can re-use code and demonstrate general patterns of analysis
      - Either way, you'll get the same inference using both approaches
    """)
    return


@app.cell
def _(ADMISSIONS):
    ADMISSIONS
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Total Effect model to UC Berkeley admissions data

    > Don't forget to look at diagnostics, which we'll skip here
    """)
    return


@app.cell
def _(ADMISSIONS):
    total_effect_model, total_effect_inference = fit_total_effect_model_admissions(ADMISSIONS)
    return (total_effect_inference,)


@app.cell
def _(total_effect_inference):
    plot_admissions_model_posterior(total_effect_inference, figsize=(6, 1.5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Total Causal Effect
    """)
    return


@app.cell
def _(total_effect_inference):
    # Total Causal Effect
    _female_p_accept = total_effect_inference.posterior["p_accept"].sel(gender="female")
    _male_p_accept = total_effect_inference.posterior["p_accept"].sel(gender="male")
    _contrast = _male_p_accept - _female_p_accept

    _fig, _ax = plt.subplots(figsize=(8, 3))
    utils.plot_dist(_contrast)
    plt.axvline(0, linestyle="--", color='black', label='No difference')
    plt.xlabel("gender contrast (acceptance probability)")
    plt.ylabel("density")
    plt.title("Total Causal Effect (Experienced Discrimination):\npositive values indicate male advantage")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Direct Effect model to UC Berkeley admissions data
    """)
    return


@app.cell
def _(ADMISSIONS):
    direct_effect_model, direct_effect_inference = fit_direct_effect_model_admissions(ADMISSIONS)
    return (direct_effect_inference,)


@app.cell
def _(direct_effect_inference):
    plot_admissions_model_posterior(direct_effect_inference, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(ADMISSIONS, direct_effect_inference):
    # Fancier plot of department/gender acceptance probability distributions
    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _ii, _dept in enumerate(ADMISSIONS.dept.unique()):
        _color = f"C{_ii}"
        for _gend in ADMISSIONS["applicant.gender"].unique():
            _label = f"{_dept}:{_gend}"
            _linestyle = "-." if _gend == 'female' else "-"
            _post = direct_effect_inference.posterior['p_accept'].sel(department=_dept, gender=_gend)
            utils.plot_dist(_post, label=_label, color=_color, plot_kwargs={"linestyle": _linestyle})
    plt.xlabel("acceptance probability")
    plt.legend(title='Deptartment:Gender')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Direct Causal Effect
    """)
    return


@app.cell
def _(ADMISSIONS, direct_effect_inference):
    # Direct Causal Effect
    _fig, _ax = plt.subplots(figsize=(8, 4))

    for _ii, _dept in enumerate(ADMISSIONS.dept.unique()):
        _color = f"C{_ii}"
        _label = f"{_dept}"

        _female_p_accept = direct_effect_inference.posterior["p_accept"].sel(gender="female", department=_dept)
        _male_p_accept = direct_effect_inference.posterior["p_accept"].sel(gender="male", department=_dept)

        _contrast = _male_p_accept - _female_p_accept
        utils.plot_dist(_contrast, color=_color, label=_label)

    plt.xlabel("gender contrast (acceptance probability)")
    plt.ylabel("density")
    plt.title("Direct Causal Effect (Institutional Discrimination):\npositive values indicate male advantage")
    plt.axvline(0, linestyle="--", color='black', label='No difference')
    plt.legend(title='Department')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Average Direct Effect
    - The causal effect of gender **averaged across departments** (marginalize)
    - Depends on the distribution of applications to each deparment
    - This is easy to do as a simulation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Post-stratification
    - **Re-weight estimates** for the target population
    - allows us to apply model fit from one university to estiamte causal impact at another university with a different distribution of departments
    - Here, we use the empirical distribution for re-weighting estimates
    """)
    return


@app.cell
def _(ADMISSIONS, direct_effect_inference):
    # Use the empirical distribution of departments -- we'd update this for a different university
    _applications_per_dept = ADMISSIONS.groupby('dept').sum()['applications'].values
    _total_applications = _applications_per_dept.sum()
    _department_weight = _applications_per_dept / _total_applications

    _female_alpha = direct_effect_inference.posterior.sel(gender='female')['alpha']
    _male_alpha = direct_effect_inference.posterior.sel(gender='male')['alpha']

    _weighted_female_alpha = _female_alpha * _department_weight
    _weighed_male_alpha = _male_alpha * _department_weight

    _contrast = _weighed_male_alpha - _weighted_female_alpha

    _fig, _ax = plt.subplots(figsize=(8, 4))
    utils.plot_dist(_contrast)

    plt.axvline(0, linestyle='--', color='k', label='No Difference')
    plt.xlabel("causal effect of perceived gender")
    plt.ylabel("density")
    plt.title("Average Direct Causal of Gender (perception),\npositive values indicate male advantage")
    plt.xlim([-.3, .3])
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To verify the averaging process, we can look at the contrast of the `p_accept` samples from the posterior, which provides similar results. However, looking ath the posterior obviously wouldn't work for making predictions for an out-of-sample university however.
    """)
    return


@app.cell
def _(direct_effect_inference):
    _direct_posterior = direct_effect_inference.posterior

    # Select by gender; note: p_accept already has link function applied
    _female_posterior = _direct_posterior.sel(gender='female')['p_accept']  # shape: (chain x draw x department)
    _male_posterior = _direct_posterior.sel(gender='male')['p_accept'] # shape: (chain x draw x department)
    _contrast = _male_posterior - _female_posterior # shape: (chain x draw x department)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    # marginalize / collapse the contrast across all departments
    utils.plot_dist(_contrast)
    plt.axvline(0, linestyle='--', color='k', label='No Difference')
    plt.xlim([-.3, .3])
    plt.xlabel("causal effect of perceived gender")
    plt.ylabel("density")
    plt.title("Posterior contrast of $p_{accept}$,\npositive values indicate male advantage")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discrimination?
    Hard to say
    - Big structural effects
    - Distribution of applicants could be due to discrimination
    - **Confounds are likely**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Survival Analysis
    - Counts often modeled as time-to-event (i.e. Exponential or Gamma distribution)  -- Goal is to estimate event rate
    - Tricky b.c. ignoring **censored data** can lead to inferential errors
      - **Left censored**: we don't know when the timer started
      - **Right censored** the observation period finished before the event had time to happen
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Austin Cat Adoption

    - 20k cats
    - Events: adopted (1) or not (0)
    - Censoring mechanisms
      - death before adoption
      - escape
      - not adopted yet

    **Goal**: determine if Black are adopted at a lower rate than non-Black cats.
    """)
    return


@app.cell
def _():
    CATS = utils.load_data('AustinCats')
    CATS.head()
    return (CATS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Modeling outcome variable: `days_to_event`
    Two go-to distributions for modeling time-to-event

    **Exponential Distribution**:
    - simpler of the two (single parameter)
    - assumes constant rate
    - maximum entropy distribution amongst the set of non-negative continuous distributions that have the same average rate.
    - assumes a single thing to occur (e.g. part failure) before an event occurs (machine breakdown)

    **Gamma Distribution**
    - more complex of the two (two parameters)
    - maximum entropy distribution amongst the set of distributions with the same mean and average log.
    - assumes multiple things to happen (e.g. part failures) before an event occurs (machine breakdown)
    """)
    return


@app.cell(hide_code=True)
def _():
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4))

    _days = np.linspace(0, 100, 100)

    plt.sca(_axs[0])
    for _scale in [20, 50]:
        _expon = stats.expon(loc=.01, scale=_scale)
        utils.plot_line(_days, _expon.pdf(_days), label=f"$\\lambda$=1/{_scale}")
    plt.xlabel("days")
    plt.ylabel("density")
    plt.legend()
    plt.title("Exponential Distribution")


    plt.sca(_axs[1])
    _N = 8
    _alpha=10
    for _scale in [2, 5]:
        _gamma = stats.gamma(a=10, loc=_alpha, scale=_scale)
        utils.plot_line(_days, _gamma.pdf(_days), label=f"$\\alpha$={_alpha}, $\\beta$={_scale}")
    plt.xlabel("days")
    plt.ylabel("density")
    plt.legend()
    plt.title("Gamma Distribution")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Censored and un-censored observations
    - Observed data use the Cumulative distribution; i.e. the **probability that the event occurred by time x**
    - Unobserved (censored) data instead require the Complementary of the CDF; i.e. the **probability that the event hasn't happened yet**.
    """)
    return


@app.cell(hide_code=True)
def _():
    _xs = np.linspace(0, 5, 100)
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(_axs[0])
    _cdfs = {}
    for _lambda in [0.5, 1]:
        _cdfs[_lambda] = _ys = 1 - np.exp(-_lambda * _xs)
        utils.plot_line(_xs, _ys, label=f"$\\lambda={_lambda}$")

    plt.xlabel("x")
    plt.ylabel("$1 - \\exp(-x)$")
    plt.legend()
    plt.title("CDF\np(event happened before or at time x| $\\lambda$)", fontsize=12)

    plt.sca(_axs[1])
    for _lambda in [0.5, 1]:
        _ys = 1 - _cdfs[_lambda]
        utils.plot_line(_xs, _ys, label=f"$\\lambda={_lambda}$")

    plt.xlabel("x")
    plt.ylabel("$\\exp(-x)$")
    plt.legend()
    plt.title("CCDF\np(event hasn't happened before or at time x| $\\lambda$)", fontsize=12)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Statistical Model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    D_i | A_i &= 1 \sim \text{Exponential}(\lambda_i) \\
    D_i | A_i &= 0 \sim \text{ExponentialCCDF}(\lambda_i) \\
    \lambda_i &= \frac{1}{\mu_i} \\
    \log \mu_i &= \alpha_{\text{cat color}[i]} \\
    \alpha_{Black, Other} &\sim \text{Exponential}(\gamma)
    \end{align*}
    $$

    - $D_i | A_i = 1$ - observed adoptions
    - $D_i | A_i = 1$ - not-yet-observed adoptions
    - $\alpha_{\text{cat color}[i]}$ log average time-to-adoption for each cat color
    - $\log \mu_i$ -- link function ensures $\alpha$s are positive
    - $\lambda_i = \frac{1}{\mu_i}$ simplifies estimating average **time-to-adoption**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Finding reasonable hyperparameter for $\alpha$
    We'll need to determine a reasonable data for the Exponential prior mean parameter $\gamma$. To do so, we'll look at the empirical distribution of time to adoption:
    """)
    return


@app.cell
def _(CATS):
    # determine reasonable prior for alpha
    _fig, _ax = plt.subplots(figsize=(6, 3))
    CATS[CATS.out_event == 'Adoption'].days_to_event.hist(bins=100)
    plt.xlim([0, 500])
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using the above empirical historgram, we see that a majority of the probablity mass is between zero and 200, so let's use 50 as the expected wait time.
    """)
    return


@app.cell
def _(CATS):
    CAT_COLOR_ID, CAT_COLOR = pd.factorize(CATS.color.apply(lambda x: "Other" if x != 'Black' else x))
    ADOPTED_ID, ADOPTED = pd.factorize(CATS.out_event.apply(lambda x: "Other" if x != "Adoption" else 'Adopted'))
    DAYS_TO_ADOPTION = CATS.days_to_event.values.astype(float)
    LAMBDA = 50

    with pm.Model(coords={"cat_color": CAT_COLOR}) as adoption_model:

        # Censoring
        _right_censoring = DAYS_TO_ADOPTION.copy()
        _right_censoring[ADOPTED_ID == 0] = DAYS_TO_ADOPTION.max()

        # Priors
        _gamma = 1 / LAMBDA
        _alpha = pm.Exponential("alpha", _gamma, dims="cat_color")

        # Likelihood
        _log_adoption_rate = 1 / _alpha[CAT_COLOR_ID]
        pm.Censored(
            "adopted",
            pm.Exponential.dist(lam=_log_adoption_rate),
            lower=None,
            upper=_right_censoring,
            observed=DAYS_TO_ADOPTION
        )
        adoption_inference = cached_sample()
    return (adoption_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Poor black kittes 🐈‍⬛
    It appears that black cats DO take longer to get adopted.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior Summary
    """)
    return


@app.cell(hide_code=True)
def _(adoption_inference):
    _fig, _ax = plt.subplots(figsize=(4, 2))
    utils.plot_forest(adoption_inference, var_names=['alpha'], hdi_prob=.89, ax=_ax)
    plt.xlabel("days to adoption")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior distributions
    """)
    return


@app.cell(hide_code=True)
def _(adoption_inference):
    _fig, _ax = plt.subplots()
    for _ii, _cat_color in enumerate(["Black", "Other"]):
        _color = 'black' if _cat_color == 'Black' else "C0"
        _posterior_alpha = adoption_inference.posterior.sel(cat_color=_cat_color)['alpha']
        utils.plot_dist(_posterior_alpha, color=_color, label=_cat_color)
    plt.xlabel("waiting time")
    plt.ylabel("density")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(adoption_inference):
    _days_until_adoption = np.linspace(1, 100)
    _n_posterior_samples = 25

    _fig, _ax = plt.subplots()
    for _cat_color in ['Black', 'Other']:
        _color = "black" if _cat_color == 'Black' else "C0"
        _posterior = adoption_inference.posterior.sel(cat_color=_cat_color)
        _alpha_samples = _posterior.alpha.sel(chain=0)[:_n_posterior_samples].values
        for _ii, _alpha in enumerate(_alpha_samples):
            _label = _cat_color if _ii == 0 else None
            plt.plot(_days_until_adoption, np.exp(-_days_until_adoption/_alpha), color=_color, alpha=.25, label=_label)

    plt.xlabel("days until adoption")
    plt.ylabel("fraction")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
