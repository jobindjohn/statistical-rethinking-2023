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
    - 4526 Graduate student applications
    - Stratified by
      - Department
      - Gender

    Goal is to identify gender discrimination by admissions officers
    """)
    return


@app.cell
def admissions_data():
    ADMISSIONS = utils.load_data("UCBadmit")
    ADMISSIONS
    return (ADMISSIONS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modeling Events
    - **Events**: discrete, unordered outcomes
    - Observations are counts/aggregates
    - Unknowns are probabilities $p$ of an event occurring, or the odds of those probabilities $\frac{p}{1-p}$
    - All things we stratify by interact -- generally never independent in real life
    - We often work with the log-odds of $p$: $\log \left(\frac{p}{1-p} \right)$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Admissions: Drawing the owl 🦉

    1. Estimand(s)
    2. Scientific Model(s)
    3. Statistical Model(s)
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
def direct_discrimination_dag():
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

    Here, department, D is a mediator -- this is a common structure in social sciences, where categorical status (e.g. gender) affects some mediating context (e.g. occupation), both of which affect a target outcome (wage). Examples
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
def indirect_discrimination_dag():
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
def total_discrimination_dag():
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
def unobserved_confound_dag():
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
def admissions_simulation():
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
    - due to lower acceptance rates in dept 0 and tendency of gender 0 to apply to dept 0, gender 0 has a lower overall acceptance rate compared to gender 1
    - due to higher acceptance rates in dept 1 and tendency of gender 1 to apply to dept 1, gender 1 has a higher overall acceptance rate compared to gender 0
    - even in the case of no (direct) gender discrimination, **there is still indirect discrimination based on tendency of genders to apply to different departments, and the unequal likelihood that each department accepts students.**

    #### Biased scenario
    - overall acceptance rates are lower (due to baseline reduction in gender 0 acceptance rates across both departments)
    - in the scenario where there is actual department bias, **we see a similar overall _pattern_ of discrimination as the unbiased case due to the indirect effect.**
    """)
    return


@app.cell(hide_code=True)
def admissions_sim_controls():
    admissions_sim = mo.ui.dictionary({
        "p_dept1_g0": mo.ui.slider(0, 1, step=0.05, value=0.3, label="P(dept 1 | gender 0)", show_value=True),
        "p_dept1_g1": mo.ui.slider(0, 1, step=0.05, value=0.8, label="P(dept 1 | gender 1)", show_value=True),
        "rate_d0_g0": mo.ui.slider(0, 1, step=0.05, value=0.1, label="dept 0, gender 0", show_value=True),
        "rate_d0_g1": mo.ui.slider(0, 1, step=0.05, value=0.1, label="dept 0, gender 1", show_value=True),
        "rate_d1_g0": mo.ui.slider(0, 1, step=0.05, value=0.3, label="dept 1, gender 0", show_value=True),
        "rate_d1_g1": mo.ui.slider(0, 1, step=0.05, value=0.3, label="dept 1, gender 1", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
    #### Explore: direct and indirect discrimination
    The overall acceptance rate of a gender is each department's acceptance rate, weighted by how often that gender applies there. So the gap between the genders splits into a **direct** part (different acceptance rates within the same department) and an **indirect** part (applying to different departments). This is computed exactly, without simulating applicants.

    The sliders start at the unbiased scenario. For the biased scenario, set gender 0's acceptance rates to 0.05 (dept 0) and 0.2 (dept 1).
    """),
        mo.hstack([
            mo.vstack([mo.md("**Where each gender applies**"), admissions_sim["p_dept1_g0"], admissions_sim["p_dept1_g1"]]),
            mo.vstack([
                mo.md("**Acceptance rates**"),
                admissions_sim["rate_d0_g0"], admissions_sim["rate_d0_g1"],
                admissions_sim["rate_d1_g0"], admissions_sim["rate_d1_g1"],
            ]),
        ], justify="start", gap=4),
    ])
    return (admissions_sim,)


@app.cell(hide_code=True)
def admissions_sim_explorer(admissions_sim):
    _v = admissions_sim.value
    # Department x Gender, like the acceptance rate matrices above
    _p_dept = np.array([[1 - _v["p_dept1_g0"], 1 - _v["p_dept1_g1"]], [_v["p_dept1_g0"], _v["p_dept1_g1"]]])
    _rates = np.array([[_v["rate_d0_g0"], _v["rate_d0_g1"]], [_v["rate_d1_g0"], _v["rate_d1_g1"]]])

    # Overall acceptance rate of each gender: department rates weighted by where that gender applies
    _contributions = _p_dept * _rates
    _overall = _contributions.sum(axis=0)

    # Split the gap (gender 1 - gender 0) into a direct and an indirect part
    _total = _overall[1] - _overall[0]
    _direct = (_p_dept[:, 0] * (_rates[:, 1] - _rates[:, 0])).sum()  # same departments, different rates
    _indirect = ((_p_dept[:, 1] - _p_dept[:, 0]) * _rates[:, 1]).sum()  # different departments

    _fig, (_ax_rates, _ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    _bottom = np.zeros(2)
    for _dept in range(2):
        _ax_rates.bar([0, 1], _contributions[_dept], bottom=_bottom, width=.5, color=f"C{_dept}", label=f"from dept {_dept}")
        _bottom += _contributions[_dept]
    for _g in range(2):
        _ax_rates.text(_g, _overall[_g], f"{_overall[_g]:.3f}", ha="center", va="bottom")
    _ax_rates.set_xticks([0, 1], ["gender 0", "gender 1"])
    _ax_rates.set_xlim([-.6, 1.6])
    _ax_rates.set_ylim([0, min(1.05, max(.1, 1.25 * _overall.max()))])
    _ax_rates.set_ylabel("overall acceptance rate")
    _ax_rates.set_title("Overall acceptance rate by gender")
    _ax_rates.legend(loc="upper left")

    _gaps = [_total, _direct, _indirect]
    _ax_gap.barh([2, 1, 0], _gaps, color=["gray", "C0", "C1"], height=.5)
    for _y, _gap in zip([2, 1, 0], _gaps):
        _ax_gap.text(_gap, _y, f" {_gap:+.3f} ", va="center", ha="left" if _gap >= 0 else "right")
    _ax_gap.axvline(0, color="k", linewidth=1)
    _lim = max(.05, 1.4 * np.abs(_gaps).max())
    _ax_gap.set_xlim([-_lim, _lim])
    _ax_gap.set_yticks([2, 1, 0], ["total", "direct\n(same departments)", "indirect\n(different departments)"])
    _ax_gap.set_xlabel("acceptance rate gap, gender 1 − gender 0")
    _ax_gap.set_title("total = direct + indirect")
    _fig.tight_layout()
    _fig
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
    - **We estimate** probabilities -- or, rather, the log-odds of events occurring

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

    - Generally, this function $f()$, called the **link function**, will have a specific form **that is associated with the likelihood distribution**. To reiterate, **link functions are matched to distributions**
    - the link function has an **inverse link function** $f^{-1}(X_i, \theta)$ such that

    $$
    \mathbb{E}[\theta] = f^{-1}(\alpha + \beta_X X_i + ...)
    $$

    - In the linear regression case the likelihood model is Gaussian, and the associated link function (and inverse link function) is just the identity, $I()$. (left plot below)
    """)
    return


@app.cell(hide_code=True)
def linear_vs_logistic_plot():
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
    plt.ylabel('$I(\\alpha + \\beta X)$')
    plt.title("Linear Model\n(continuous outcomes)")

    plt.sca(_axs[1])
    plt.plot(_xs, utils.invlogit(_alpha + _beta * _xs))
    plt.axhline([0], linestyle='--', color='gray')
    plt.axhline([1], linestyle='--', color='gray')
    plt.xlim([-3, 3])
    plt.ylim([-.5, 1.5])
    plt.xlabel('x')
    plt.ylabel('$\\frac{1}{1 + e^{-(\\alpha + \\beta X)}}$')
    plt.title("Logistic Model\n(binary outcomes)")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Logit Link Function and Logistic Inverse function
    ### Binary outcomes

    - **Logistic regression** used to model the probability of an event occurring
    - Likelihood function is the Bernoulli
    - Associated link function is the **log-odds, or logit**: $f(p) = \log \left(\frac{p}{1-p}\right)$

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
def logistic_curve_plot():
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
    For the following simulation, we'll use a custom utility function `utils.simulate_2_parameter_bayesian_learning` for simulating general Bayesian posterior update simulation. Here's the API for that function (for more details see `utils.py`)
    """)
    return


@app.cell
def bayesian_learning_docs():
    mo.plain_text(inspect.getdoc(utils.simulate_2_parameter_bayesian_learning))
    return


@app.function
def logistic_model(x, alpha, beta):
    return utils.invlogit(alpha + beta * x)


@app.function
def logistic_regression_posterior(x_obs, y_obs, alpha_grid, beta_grid):

    # Convert params to 1-d arrays
    if np.ndim(alpha_grid) > 1:
        alpha_grid = alpha_grid.ravel()

    if np.ndim(beta_grid) > 1:
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
def logistic_updating_demo():
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
def logistic_updating_controls():
    logistic_updating_n = mo.ui.slider(0, 128, step=1, value=8, label="number of observations", show_value=True, full_width=True)
    mo.vstack([
        mo.md(r"""
    #### Explore: updating one observation at a time
    Same model and priors as above, on a new simulated dataset with the same true parameters ($\alpha = -1$, $\beta = 2$). Here the grid posterior is vectorized (the log-likelihood of all observations at all grid points in one array operation), so it updates instantly. The demo above loops over the grid points in Python.
    """),
        logistic_updating_n,
    ])
    return (logistic_updating_n,)


@app.cell(hide_code=True)
def logistic_updating_explorer(logistic_updating_n):
    _ALPHA, _BETA = -1, 2
    _rng = np.random.default_rng(123)
    _x_all = _rng.normal(size=128)
    _y_all = _rng.binomial(1, utils.invlogit(_ALPHA + _BETA * _x_all))
    _n = logistic_updating_n.value
    _x, _y = _x_all[:_n], _y_all[:_n]

    # Vectorized grid posterior: (beta x alpha x observation) log-odds in one array
    _grid = np.linspace(-3, 3, 150)
    _alphas, _betas = np.meshgrid(_grid, _grid)
    _logit_p = _alphas[..., None] + _betas[..., None] * _x
    _log_likelihood = (_y * _logit_p - np.logaddexp(0, _logit_p)).sum(axis=-1)  # Bernoulli log-pmf on the logit scale
    _log_posterior = _log_likelihood + stats.norm(0, 1).logpdf(_alphas) + stats.norm(0, 1).logpdf(_betas)
    _posterior = np.exp(_log_posterior - _log_posterior.max())

    # Sample parameters from the grid posterior
    _idx = np.random.default_rng(0).choice(_posterior.size, size=20, p=(_posterior / _posterior.sum()).ravel())
    _alpha_samples, _beta_samples = _alphas.ravel()[_idx], _betas.ravel()[_idx]

    _fig, (_ax_post, _ax_data) = plt.subplots(1, 2, figsize=(11, 4.5))
    _ax_post.contour(_alphas, _betas, _posterior, cmap="gray_r")
    _ax_post.scatter(_alpha_samples, _beta_samples, color="C0", s=12, alpha=.6, label="posterior samples")
    _ax_post.scatter([_ALPHA], [_BETA], color="C1", marker="x", s=80, label="true parameters")
    _ax_post.set_aspect("equal")
    _ax_post.set_xlabel("$\\alpha$")
    _ax_post.set_ylabel("$\\beta$")
    _ax_post.set_title(f"posterior after {_n} observations")
    _ax_post.legend(loc="lower left", fontsize=8)

    _xs = np.linspace(-2, 2, 100)
    for _a, _b in zip(_alpha_samples, _beta_samples):
        _ax_data.plot(_xs, utils.invlogit(_a + _b * _xs), color="C0", alpha=.25)
    _ax_data.plot(_xs, utils.invlogit(_ALPHA + _BETA * _xs), color="C1", linestyle="--", label="true model")
    _ax_data.scatter(_x, _y, color="k", s=20, zorder=3, label="observations")
    _ax_data.set_xlim([-2, 2])
    _ax_data.set_ylim([-.05, 1.05])
    _ax_data.set_xlabel("x")
    _ax_data.set_ylabel("p(y = 1)")
    _ax_data.set_title("20 logistic curves sampled from the posterior")
    _ax_data.legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    _fig
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
def logit_prior_plots():
    _n_samples = 10000
    _rng = np.random.default_rng(12)
    _fig, _axs = plt.subplots(3, 2, figsize=(8, 8))
    for _ii, _std in enumerate([10, 1.5, 1]):

        # Alpha prior distribution
        _alphas = stats.norm(0, _std).rvs(_n_samples, random_state=_rng)
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


@app.function
def plot_logistic_prior_predictive(n_samples = 20, prior_std=.5, random_seed=None):

    _, axs = plt.subplots(1, 2, figsize=(10, 4))
    plt.sca(axs[0])
    min_x, max_x = -3 * prior_std, 3 * prior_std
    xs = np.linspace(min_x, max_x, 100)
    ys = xs

    pdf = partial(gaussian_2d_pdf, covariance=np.array([[prior_std ** 2, 0],[0, prior_std ** 2]]))
    utils.plot_2d_function(xs, ys, pdf, cmap='gray_r')

    plt.axis("square")
    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    # Sample some parameters from prior
    rng = np.random.default_rng(random_seed)
    βs = stats.norm.rvs(0, prior_std, size=n_samples, random_state=rng)
    αs = stats.norm.rvs(0, prior_std, size=n_samples, random_state=rng)

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
    plt.ylabel('p = invlogit($\\alpha + \\beta x$)')
    plt.xlim([-3, 3])
    plt.ylim([-.05, 1.05])
    plt.axhline(0.5, linestyle='--', color='k')
    plt.title("Resulting Logistic Models")


@app.cell
def prior_predictive_std_05():
    plot_logistic_prior_predictive(prior_std=0.5, random_seed=1)
    plt.gcf()
    return


@app.cell
def prior_predictive_std_1():
    plot_logistic_prior_predictive(prior_std=1., random_seed=1)
    plt.gcf()
    return


@app.cell
def prior_predictive_std_10():
    plot_logistic_prior_predictive(prior_std=10, random_seed=1)
    plt.gcf()
    return


@app.cell(hide_code=True)
def logit_prior_controls():
    logit_prior_sigma = mo.ui.slider(
        steps=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10],
        value=1.5,
        label="prior std, σ",
        show_value=True,
        full_width=True,
    )
    mo.vstack([
        mo.md(r"""
    #### Explore: how wide should a log-odds prior be?
    Set the prior $\alpha, \beta \sim \mathcal{N}(0, \sigma)$ and see what it implies on the probability scale. Beyond $|\text{logit}(p)| > 4.6$, i.e. $p < 0.01$ or $p > 0.99$, the logit squashes everything against 0 and 1, so a wide prior on the log-odds scale is a strong prior for extreme probabilities.
    """),
        logit_prior_sigma,
    ])
    return (logit_prior_sigma,)


@app.cell(hide_code=True)
def logit_prior_explorer(logit_prior_sigma):
    _sigma = logit_prior_sigma.value
    _rng = np.random.default_rng(12)
    _ps = utils.invlogit(_rng.normal(0, _sigma, size=20000))
    _logit_99 = np.log(99)
    _extreme = 2 * stats.norm(0, _sigma).sf(_logit_99)

    _fig, (_ax_alpha, _ax_p, _ax_curves) = plt.subplots(1, 3, figsize=(14, 4))
    _xs = np.linspace(-25, 25, 1001)
    _ax_alpha.plot(_xs, stats.norm(0, _sigma).pdf(_xs), color="C1")
    _ax_alpha.axvspan(-_logit_99, _logit_99, color="gray", alpha=.15, label="0.01 < p < 0.99")
    _ax_alpha.set_xlabel("$\\alpha$ (log-odds)")
    _ax_alpha.set_ylabel("density")
    _ax_alpha.set_title(f"$\\alpha \\sim \\mathcal{{N}}(0, {_sigma:g})$")
    _ax_alpha.legend(loc="upper left", fontsize=8)

    _ax_p.hist(_ps, bins=np.linspace(0, 1, 51), density=True, color="C0", alpha=.7)
    _ax_p.set_xlim([0, 1])
    _ax_p.set_xlabel("p(event)")
    _ax_p.set_title(f"implied prior on p\n{_extreme:.0%} of draws have p < 0.01 or p > 0.99")

    _xs = np.linspace(-3, 3, 100)
    for _a, _b in zip(_rng.normal(0, _sigma, size=30), _rng.normal(0, _sigma, size=30)):
        _ax_curves.plot(_xs, utils.invlogit(_a + _b * _xs), color="C0", alpha=.4)
    _ax_curves.axhline(0.5, linestyle="--", color="k")
    _ax_curves.set_ylim([-.05, 1.05])
    _ax_curves.set_xlabel("x")
    _ax_curves.set_ylabel("p = invlogit($\\alpha + \\beta x$)")
    _ax_curves.set_title("30 logistic curves from the prior")
    _fig.tight_layout()
    _fig
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
def total_effect_dag():
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
def direct_effect_dag():
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
    \alpha &= \left[{\begin{array}{cc} \alpha_{0,0} & \alpha_{0,1} \\ \alpha_{1,0} & \alpha_{1,1} \end{array}}\right] \\
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

    # Plot inverse-linked acceptance probabilities
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
def simulated_biased_data(simulate_admissions_data):
    # Fixed seed, so the simulated data (and the fits below) are the same on
    # every run, and the cached fits can be reused
    np.random.seed(1)

    SIMULATED_BIASED_ADMISSIONS, _ = simulate_admissions_data("biased")
    SIMULATED_BIASED_ADMISSIONS = aggregate_admissions_data(SIMULATED_BIASED_ADMISSIONS)
    SIMULATED_BIASED_ADMISSIONS
    return (SIMULATED_BIASED_ADMISSIONS,)


@app.cell
def fit_total_simulated_biased(SIMULATED_BIASED_ADMISSIONS):
    total_effect_model_simulated_biased, total_effect_inference_simulated_biased = fit_total_effect_model_admissions(SIMULATED_BIASED_ADMISSIONS)
    return (total_effect_inference_simulated_biased,)


@app.cell
def plot_total_simulated_biased(total_effect_inference_simulated_biased):
    plot_admissions_model_posterior(total_effect_inference_simulated_biased, (6, 1.5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Total Effect model on UNBIASED simulated admissions
    """)
    return


@app.cell
def simulated_unbiased_data(simulate_admissions_data):
    # Fixed seed, so the simulated data (and the fits below) are the same on
    # every run, and the cached fits can be reused
    np.random.seed(2)

    SIMULATED_UNBIASED_ADMISSIONS, _ = simulate_admissions_data("unbiased")
    SIMULATED_UNBIASED_ADMISSIONS = aggregate_admissions_data(SIMULATED_UNBIASED_ADMISSIONS)
    SIMULATED_UNBIASED_ADMISSIONS
    return (SIMULATED_UNBIASED_ADMISSIONS,)


@app.cell
def fit_total_simulated_unbiased(SIMULATED_UNBIASED_ADMISSIONS):
    total_effect_model_simulated_unbiased, total_effect_inference_simulated_unbiased = fit_total_effect_model_admissions(SIMULATED_UNBIASED_ADMISSIONS)
    return (total_effect_inference_simulated_unbiased,)


@app.cell
def plot_total_simulated_unbiased(total_effect_inference_simulated_unbiased):
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
    """Fit direct effect model, stratifying by gender and department.

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
def fit_direct_simulated_biased(SIMULATED_BIASED_ADMISSIONS):
    direct_effect_model_simulated_biased, direct_effect_inference_simulated_biased = fit_direct_effect_model_admissions(SIMULATED_BIASED_ADMISSIONS)
    return (direct_effect_inference_simulated_biased,)


@app.cell
def plot_direct_simulated_biased(direct_effect_inference_simulated_biased):
    plot_admissions_model_posterior(direct_effect_inference_simulated_biased)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For comparison, here's the ground truth biased admission rates, which we're able to mostly recover:
    """)
    return


@app.cell
def biased_rates_truth(BIASED_ACCEPTANCE_RATES):
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
def fit_direct_simulated_unbiased(SIMULATED_UNBIASED_ADMISSIONS):
    direct_effect_model_simulated_unbiased, direct_effect_inference_simulated_unbiased = fit_direct_effect_model_admissions(SIMULATED_UNBIASED_ADMISSIONS)
    return (direct_effect_inference_simulated_unbiased,)


@app.cell
def plot_direct_simulated_unbiased(direct_effect_inference_simulated_unbiased):
    plot_admissions_model_posterior(direct_effect_inference_simulated_unbiased)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For comparison, here's the ground truth unbiased admission rates, which we're able to recover:
    """)
    return


@app.cell
def unbiased_rates_truth(UNBIASED_ACCEPTANCE_RATES):
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
    - note that we'll be using Binomial Regression, which is equivalent to Bernoulli regression, but operates on aggregate counts, as opposed to individual binary trials
      - The examples above were actually implemented as Binomial Regression so that we can re-use code and demonstrate general patterns of analysis
      - Either way, you'll get the same inference using both approaches
    """)
    return


@app.cell
def admissions_data_review(ADMISSIONS):
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
def fit_total_ucb(ADMISSIONS):
    total_effect_model, total_effect_inference = fit_total_effect_model_admissions(ADMISSIONS)
    return (total_effect_inference,)


@app.cell
def plot_total_ucb(total_effect_inference):
    plot_admissions_model_posterior(total_effect_inference, figsize=(6, 1.5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Total Causal Effect
    """)
    return


@app.cell
def total_effect_contrast(total_effect_inference):
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
def fit_direct_ucb(ADMISSIONS):
    direct_effect_model, direct_effect_inference = fit_direct_effect_model_admissions(ADMISSIONS)
    return (direct_effect_inference,)


@app.cell
def plot_direct_ucb(direct_effect_inference):
    plot_admissions_model_posterior(direct_effect_inference, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def department_gender_posteriors(ADMISSIONS, direct_effect_inference):
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
    plt.legend(title='Department:Gender')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Direct Causal Effect
    """)
    return


@app.cell
def direct_effect_contrast(ADMISSIONS, direct_effect_inference):
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
    - Depends on the distribution of applications to each department
    - This is easy to do as a simulation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Post-stratification
    - **Re-weight estimates** for the target population
    - allows us to apply model fit from one university to estimate causal impact at another university with a different distribution of departments
    - Here, we use the empirical distribution of applications across departments: each posterior draw of the contrast comes from a department picked in proportion to its applications. This simulates the contrast an applicant at Berkeley experiences
    """)
    return


@app.cell
def post_stratification(ADMISSIONS, direct_effect_inference):
    # Use the empirical distribution of departments -- we'd update this for a different university
    _applications_per_dept = ADMISSIONS.groupby('dept')['applications'].sum()
    _department_weight = (_applications_per_dept / _applications_per_dept.sum()).reindex(
        direct_effect_inference.posterior.department.values
    ).values

    # Contrast on the probability scale, per department: (chain x draw x department)
    _p_accept = direct_effect_inference.posterior['p_accept']
    department_contrast = _p_accept.sel(gender='male') - _p_accept.sel(gender='female')
    _contrast = department_contrast.transpose('chain', 'draw', 'department').values.reshape(-1, department_contrast.sizes['department'])

    # Simulate applications: each posterior draw gets a department, picked in
    # proportion to the department's share of applications
    _rng = np.random.default_rng(1)
    _department_draws = _rng.choice(len(_department_weight), size=len(_contrast), p=_department_weight)
    _post_stratified_contrast = _contrast[np.arange(len(_contrast)), _department_draws]

    # Average direct effect: the weighted average of the department contrasts
    _average_effect = _contrast @ _department_weight

    _fig, _ax = plt.subplots(figsize=(8, 4))
    utils.plot_dist(_post_stratified_contrast, label="post-stratified contrast")
    plt.axvline(
        _average_effect.mean(), color='C1',
        label=f"weighted average: {_average_effect.mean():.3f} ± {_average_effect.std():.3f}",
    )
    plt.axvline(0, linestyle='--', color='k', label='No Difference')
    plt.xlabel("gender contrast (acceptance probability)")
    plt.ylabel("density")
    plt.title("Average Direct Causal Effect of Gender (perception),\npositive values indicate male advantage")
    plt.xlim([-.3, .3])
    plt.legend()
    _fig
    return (department_contrast,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For comparison, here are the same department contrasts pooled with **equal weights**, as if every department received the same number of applications. The average moves from $-0.036$ to $-0.029$: department A, where women have the largest advantage, now counts for 1/6 of the applications instead of 21%. The average direct effect depends on the population we average over, which is why post-stratification needs a target population. Try other populations in the explorer below.
    """)
    return


@app.cell
def equal_weight_contrast(direct_effect_inference):
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
    plt.title("Contrast of $p_{accept}$, departments pooled with equal weights,\npositive values indicate male advantage")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def post_strat_controls(ADMISSIONS):
    _applications = ADMISSIONS.pivot_table(index="dept", columns="applicant.gender", values="applications", aggfunc="sum")
    post_strat_population = mo.ui.dropdown(
        options={
            "All Berkeley applications": _applications.sum(axis=1),
            "Equal weights": _applications.sum(axis=1) * 0 + 1,
            "Women's applications": _applications["female"],
            "Men's applications": _applications["male"],
        },
        value="All Berkeley applications",
        label="population to average over",
    )
    mo.vstack([
        mo.md(r"""
    #### Explore: the average direct effect depends on the population
    The average direct effect is a weighted average of the department contrasts above, with weights from the target population. Women applied mostly to departments C to F, where the contrasts are close to zero. Men applied mostly to A and B, and A is where women have the largest advantage.
    """),
        post_strat_population,
    ])
    return (post_strat_population,)


@app.cell(hide_code=True)
def post_strat_explorer(department_contrast, post_strat_population):
    _weights = post_strat_population.value.reindex(department_contrast.department.values).values.astype(float)
    _weights = _weights / _weights.sum()
    _contrast = department_contrast.transpose("chain", "draw", "department").values.reshape(-1, len(_weights))

    # Post-stratified contrast: each posterior draw comes from a department picked by the weights
    _rng = np.random.default_rng(1)
    _draws = _rng.choice(len(_weights), size=len(_contrast), p=_weights)
    _mixture = _contrast[np.arange(len(_contrast)), _draws]
    _average = _contrast @ _weights

    _fig, (_ax_weights, _ax_effect) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]})
    _departments = department_contrast.department.values
    _ax_weights.bar(_departments, _weights, color="C1", alpha=.7)
    for _dept, _w, _c in zip(_departments, _weights, _contrast.mean(axis=0)):
        _ax_weights.text(_dept, _w, f"{_c:+.2f}", ha="center", va="bottom", fontsize=9)
    _ax_weights.set_ylim([0, max(.4, 1.2 * _weights.max())])
    _ax_weights.set_xlabel("department")
    _ax_weights.set_ylabel("weight")
    _ax_weights.set_title("department weights\n(labels: mean contrast)")

    plt.sca(_ax_effect)
    utils.plot_dist(_mixture, label="post-stratified contrast")
    plt.axvline(_average.mean(), color="C1", label=f"weighted average: {_average.mean():.3f} ± {_average.std():.3f}")
    plt.axvline(0, linestyle="--", color="k", label="No Difference")
    plt.xlim([-.3, .3])
    plt.xlabel("gender contrast (acceptance probability)")
    plt.ylabel("density")
    plt.title(f"Average direct effect for: {post_strat_population.selected_key}\npositive values indicate male advantage")
    plt.legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
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

    - 22,356 cats
    - Events: adopted (1) or not (0)
    - Censoring mechanisms
      - death before adoption
      - escape
      - not adopted yet

    **Goal**: determine if Black cats are adopted at a lower rate than non-Black cats.
    """)
    return


@app.cell
def cats_data():
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
def exponential_gamma_plot():
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4))

    _days = np.linspace(0, 100, 100)

    plt.sca(_axs[0])
    for _scale in [20, 50]:
        _expon = stats.expon(scale=_scale)
        utils.plot_line(_days, _expon.pdf(_days), label=f"$\\lambda$=1/{_scale}")
    plt.xlabel("days")
    plt.ylabel("density")
    plt.legend()
    plt.title("Exponential Distribution")


    plt.sca(_axs[1])
    _shape = 10
    for _scale in [2, 5]:
        _gamma = stats.gamma(a=_shape, scale=_scale)
        utils.plot_line(_days, _gamma.pdf(_days), label=f"shape={_shape}, scale={_scale}")
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
def cdf_ccdf_plot():
    _xs = np.linspace(0, 5, 100)
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(_axs[0])
    _cdfs = {}
    for _lambda in [0.5, 1]:
        _cdfs[_lambda] = _ys = 1 - np.exp(-_lambda * _xs)
        utils.plot_line(_xs, _ys, label=f"$\\lambda={_lambda}$")

    plt.xlabel("x")
    plt.ylabel("$1 - \\exp(-\\lambda x)$")
    plt.legend()
    plt.title("CDF\np(event happened before or at time x| $\\lambda$)", fontsize=12)

    plt.sca(_axs[1])
    for _lambda in [0.5, 1]:
        _ys = 1 - _cdfs[_lambda]
        utils.plot_line(_xs, _ys, label=f"$\\lambda={_lambda}$")

    plt.xlabel("x")
    plt.ylabel("$\\exp(-\\lambda x)$")
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
    D_i | A_i = 1 &\sim \text{Exponential}(\lambda_i) \\
    D_i | A_i = 0 &\sim \text{ExponentialCCDF}(\lambda_i) \\
    \lambda_i &= \frac{1}{\mu_i} \\
    \mu_i &= \alpha_{\text{cat color}[i]} \\
    \alpha_{Black, Other} &\sim \text{Exponential}(\gamma)
    \end{align*}
    $$

    - $D_i | A_i = 1$ - observed adoptions
    - $D_i | A_i = 0$ - not-yet-observed adoptions: the cat wasn't adopted within $D_i$ days (censored)
    - $\alpha_{\text{cat color}[i]}$ average time-to-adoption for each cat color
    - the Exponential prior keeps the $\alpha$s positive. (In the lecture McElreath uses a log link, $\log \mu_i = \alpha_{\text{cat color}[i]}$, with a Normal prior on $\alpha$; the code below puts the prior on $\mu$ directly)
    - $\lambda_i = \frac{1}{\mu_i}$ simplifies estimating average **time-to-adoption**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Finding reasonable hyperparameter for $\alpha$
    We'll need to determine a reasonable value for the rate $\gamma$ of the Exponential prior (its mean is $1 / \gamma$). To do so, we'll look at the empirical distribution of time to adoption:
    """)
    return


@app.cell
def adoption_days_histogram(CATS):
    # determine reasonable prior for alpha
    _fig, _ax = plt.subplots(figsize=(6, 3))
    CATS[CATS.out_event == 'Adoption'].days_to_event.hist(bins=100)
    plt.xlim([0, 500])
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using the above empirical histogram, we see that a majority of the probability mass is between zero and 200, so let's use 50 days as the prior expected wait time ($\gamma = 1/50$).
    """)
    return


@app.cell
def adoption_model_fit(CATS):
    CAT_COLOR_ID, CAT_COLOR = pd.factorize(CATS.color.apply(lambda x: "Other" if x != 'Black' else x))
    ADOPTED = (CATS.out_event == "Adoption").values
    DAYS_TO_ADOPTION = CATS.days_to_event.values.astype(float)
    LAMBDA = 50

    with pm.Model(coords={"cat_color": CAT_COLOR}) as adoption_model:

        # Censoring: a cat that wasn't adopted is right-censored at its own
        # days_to_event (it was still waiting then). Adopted cats get an infinite
        # upper bound, so their days are used as observed events.
        _right_censoring = np.where(ADOPTED, np.inf, DAYS_TO_ADOPTION)

        # Priors
        _gamma = 1 / LAMBDA
        _alpha = pm.Exponential("alpha", _gamma, dims="cat_color")

        # Likelihood
        _adoption_rate = 1 / _alpha[CAT_COLOR_ID]
        pm.Censored(
            "adopted",
            pm.Exponential.dist(lam=_adoption_rate),
            lower=None,
            upper=_right_censoring,
            observed=DAYS_TO_ADOPTION
        )
        adoption_inference = cached_sample()
    return (
        ADOPTED,
        CAT_COLOR,
        CAT_COLOR_ID,
        DAYS_TO_ADOPTION,
        LAMBDA,
        adoption_inference,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Poor black kitties 🐈‍⬛
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
def adoption_forest(adoption_inference):
    _fig, _ax = plt.subplots(figsize=(4, 2))
    utils.plot_forest(adoption_inference, var_names=['alpha'], hdi_prob=.89, ax=_ax)
    plt.xlabel("average days to adoption")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior distributions
    """)
    return


@app.cell(hide_code=True)
def adoption_posteriors(adoption_inference):
    _fig, _ax = plt.subplots()
    for _ii, _cat_color in enumerate(["Black", "Other"]):
        _color = 'black' if _cat_color == 'Black' else "C0"
        _posterior_alpha = adoption_inference.posterior.sel(cat_color=_cat_color)['alpha']
        utils.plot_dist(_posterior_alpha, color=_color, label=_cat_color)
    plt.xlabel("average days to adoption")
    plt.ylabel("density")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def adoption_survival_curves(adoption_inference):
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
    plt.ylabel("fraction not yet adopted")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def censoring_controls():
    censoring_treatment = mo.ui.radio(
        options=["Censored, as in the model above", "Dropped", "Adopted on their last day"],
        value="Censored, as in the model above",
        label="Cats that weren't adopted are:",
    )
    mo.vstack([
        mo.md(r"""
    #### Explore: what if we ignore censoring?
    About half of the cats weren't adopted (transferred, died, still waiting, ...). Compare three ways of treating them. The posterior of the average waiting time $\mu$ is computed exactly on a grid, with the same Exponential prior as above, so it updates instantly.

    - **Censored**: each non-adopted cat says "not adopted within $D_i$ days" (the CCDF). This reproduces the MCMC fit above.
    - **Dropped**: only adopted cats count, so the waiting times look shorter.
    - **Adopted on their last day**: shorter still.

    The dashed lines are the Kaplan-Meier estimate of the fraction not yet adopted, computed directly from the data while accounting for censoring.
    """),
        censoring_treatment,
    ])
    return (censoring_treatment,)


@app.cell(hide_code=True)
def censoring_explorer(
    ADOPTED,
    CAT_COLOR,
    CAT_COLOR_ID,
    DAYS_TO_ADOPTION,
    LAMBDA,
    censoring_treatment,
):
    def _kaplan_meier(days, adopted):
        """Fraction not yet adopted after each adoption time, accounting for censoring"""
        times = np.unique(days[adopted])
        at_risk = len(days) - np.searchsorted(np.sort(days), times, side="left")
        n_adopted = np.bincount(np.searchsorted(times, days[adopted]), minlength=len(times))
        return times, np.cumprod(1 - n_adopted / at_risk)


    _is_black = CAT_COLOR[CAT_COLOR_ID] == "Black"
    _mu_grid = np.linspace(15, 75, 1201)
    _t = np.linspace(0, 150, 151)

    _fig, (_ax_post, _ax_surv) = plt.subplots(1, 2, figsize=(12, 4))
    for _cat_color, _mask in [("Black", _is_black), ("Other", ~_is_black)]:
        _color = "black" if _cat_color == "Black" else "C0"
        _days, _adopted = DAYS_TO_ADOPTION[_mask], ADOPTED[_mask]
        if censoring_treatment.value == "Dropped":
            _n_adoptions, _total_days = _adopted.sum(), _days[_adopted].sum()
        elif censoring_treatment.value == "Adopted on their last day":
            _n_adoptions, _total_days = len(_days), _days.sum()
        else:
            _n_adoptions, _total_days = _adopted.sum(), _days.sum()

        # Exponential log-likelihood with mean mu: each adoption adds -log(mu), each cat adds
        # -days/mu (for a censored cat that is the log CCDF)
        _log_posterior = -_n_adoptions * np.log(_mu_grid) - _total_days / _mu_grid + stats.expon(scale=LAMBDA).logpdf(_mu_grid)
        _posterior = np.exp(_log_posterior - _log_posterior.max())
        _posterior /= _posterior.sum() * (_mu_grid[1] - _mu_grid[0])
        _mu_mean = (_mu_grid * _posterior).sum() * (_mu_grid[1] - _mu_grid[0])

        _ax_post.plot(_mu_grid, _posterior, color=_color, label=f"{_cat_color}: {_mu_mean:.1f} days")
        _ax_surv.plot(_t, np.exp(-_t / _mu_mean), color=_color, label=f"{_cat_color}, model")
        _times, _not_adopted = _kaplan_meier(_days, _adopted)
        _ax_surv.step(_times, _not_adopted, where="post", color=_color, linestyle="--", alpha=.7, label=f"{_cat_color}, data (Kaplan-Meier)")

    _ax_post.set_xlabel("average days to adoption, $\\mu$")
    _ax_post.set_ylabel("density")
    _ax_post.set_title("posterior of $\\mu$")
    _ax_post.legend()
    _ax_surv.set_xlim([0, 150])
    _ax_surv.set_ylim([0, 1.02])
    _ax_surv.set_xlabel("days")
    _ax_surv.set_ylabel("fraction not yet adopted")
    _ax_surv.set_title("survival curves at the posterior mean")
    _ax_surv.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
