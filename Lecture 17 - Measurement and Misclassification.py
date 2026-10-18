import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from itertools import product as iproduct

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
    # [Lecture 17 - Measurement & Misclassification](https://www.youtube.com/watch?v=mt9WKbQJrI4)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Probability, Intuition, and Pancakes 🥞

    McElreath poses a re-dressing of the [classic Monty Hall Problem](https://en.wikipedia.org/wiki/Monty_Hall_problem). The original problem uses an old gameshow called "Let's Make a Deal" (hosted by Monty Hall) as the backdrop for a scenario where the correct, optimal strategy for winning a game is given by following the rules of probability theory. What's interesting about the Monty Hall problem is that the optimal strategy doesn't align with our intuitions.

    In lecture, instead of opening doors to find donkeys or prizes, as was in the case of the Monty Hall problem, we have pancakes that are either burnt or perfectly cooked on either side. The thought experiment goes like this:

    - You have 3 pancakes:
      - One (likely the first one you cooked) is burnt on both sides
      - One (likely the next one cooked, after you're improving your cooking skills) is burnt on only one side
      - One is cooked perfectly on both sides
    - Say you are given a pancake at random and it is burned on one side
    - What is the probability that the other side is burned?

    Most folks would say 1/2, which intuitively _feels_ correct. However, the correct answer is given by Bayes rule:

    Let $U'$ and $D'$ be the events that the up side and the down side of the pancake are burnt:

    $$
    \begin{align*}
    p(D' | U') &= \frac{p(U', D')}{p(U')} &\text{Bayes Rule} \\
    p(U', D') &= 1/3 &\text{only one out of three pancakes has both sides burnt} \\
    p(U') &= (1/3 \times 2/2)  + (1/3 \times 1/2) + (1/3 \times 0) = 3/6 = 1/2 &\text{probability of initially observing upside burnt} \\
    p(D' | U') &= \frac{1/3}{1/2} = 2/3
    \end{align*}
    $$

    ## Avoid being clever

    - Being clever is unreliable
    - Being clever is opaque
    - You'll rarely beat the axioms of probability theory
    - Probability allows us to solve complex problems, if we stick to the rules
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Measurement Error

    - Many variables are proxies for what we want to measure (e.g. Descendant elemental confound)
    - It's common practice to ignore measurement error
      - assumes error is harmless, or will "average out"
    - Better to draw out the measurement error, think causally, and fall back on probability theory (don't be clever)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Myth: Measurement error can only decrease an effect, not increase

    - This is incorrect, measurement error has no predefined direction of causal effect
    - It can, in some cases, _increase_ an effect, as we'll demonstrate below
    """)
    return


@app.cell(hide_code=True)
def recall_bias_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("P", "C"),
            ("P", "P*"),
            ("e", "P*"),
            ("C", "e")
        ],
        node_props={
            "P": {"style": "dashed"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("C", "e"): {"label": "recall bias", "color": "blue", "fontcolor": "blue"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Child Income $C$
    - Actual Parental Income $P$, unobserved
    - Measured parental income $P^*$
    - Error $e$ in the reported parental income (e.g. due to recall bias)
    """)
    return


@app.function
def simulate_child_parent_income(beta_P=0, n_samples=500, title=None, random_seed=123):
    np.random.seed(random_seed)
    P = stats.norm().rvs(size=n_samples)
    C = stats.norm.rvs(beta_P * P)
    mu_P_star = 0.8 * P + 0.2 * C
    P_star = stats.norm.rvs(mu_P_star)

    with pm.Model() as model:
        sigma = pm.Exponential("sigma", 1)
        beta = pm.Normal("beta", 0, 1)
        alpha = pm.Normal("alpha", 0, 1)
        mu = alpha + beta * P_star
        pm.Normal("C", mu, sigma, observed=C)
        inference = cached_sample()

    utils.plot_dist(inference.posterior['beta'])
    plt.axvline(beta_P, color='k', linestyle='--', label="actual")
    plt.title(title)
    plt.legend()

    return mo.vstack([az.summary(inference, var_names=["alpha", "beta", "sigma"]), plt.gcf()])


@app.cell
def overestimate_example():
    simulate_child_parent_income(title="Measurement error makes\nus over-estimate effect")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the scenario above, measurement error can cause us to over-estimate the actual effect: there is no effect ($\beta_P = 0$), but recall bias makes $P^*$ depend on $C$
    """)
    return


@app.cell
def underestimate_example():
    simulate_child_parent_income(beta_P=0.75, title="Measurement error makes\nus under-estimate effect")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the scenario above, measurement error can cause us to under-estimate the actual effect

    - What happens depends on the details of the context
    - **There is no general rule to tell you what measurement error will do to your estimate**
    """)
    return


@app.cell(hide_code=True)
def measurement_error_controls():
    measurement_error_params = mo.ui.dictionary({
        "beta_P": mo.ui.slider(-1.0, 1.5, step=0.05, value=0.0, label="true effect $\\beta_P$", show_value=True),
        "recall_bias": mo.ui.slider(-0.5, 0.5, step=0.05, value=0.2, label="recall bias $b$", show_value=True),
        "noise": mo.ui.slider(0.0, 2.0, step=0.1, value=1.0, label="noise $\\sigma_e$", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: which way does measurement error push?
        The simulation above, with its settings exposed: $P \sim \mathcal{N}(0, 1)$, $C \sim \mathcal{N}(\beta_P P, 1)$ and $P^* \sim \mathcal{N}(0.8 P + b\,C, \sigma_e)$. The sliders start at the first example ($\beta_P = 0$). Noise alone pulls the estimated slope toward zero. Recall bias adds a path from $C$ back into the measurement, which can push the slope up or down, or flip its sign.

        The histogram shows the least-squares slopes of $C$ on $P^*$ in 500 simulated datasets of 500 people each; the Bayesian fits above give nearly the same estimates. The solid line is the slope the regression converges to with unlimited data.
        """),
        mo.hstack([measurement_error_params["beta_P"], measurement_error_params["recall_bias"], measurement_error_params["noise"]], justify="start", gap=2),
    ])
    return (measurement_error_params,)


@app.cell(hide_code=True)
def measurement_error_explorer(measurement_error_params):
    _v = measurement_error_params.value
    _beta_P, _b, _s = _v["beta_P"], _v["recall_bias"], _v["noise"]
    _rng = np.random.default_rng(123)
    _P = _rng.normal(size=(500, 500))  # 500 datasets of 500 people
    _C = _rng.normal(_beta_P * _P, 1)
    _P_star = 0.8 * _P + _b * _C + _s * _rng.normal(size=_P.shape)

    # Least-squares slope of C on P* in each dataset
    _P_star_c = _P_star - _P_star.mean(axis=1, keepdims=True)
    _C_c = _C - _C.mean(axis=1, keepdims=True)
    _slopes = (_P_star_c * _C_c).sum(axis=1) / (_P_star_c ** 2).sum(axis=1)

    # Slope with unlimited data: cov(C, P*) / var(P*)
    _limit = (0.8 * _beta_P + _b * (_beta_P ** 2 + 1)) / ((0.8 + _b * _beta_P) ** 2 + _b ** 2 + _s ** 2)

    if abs(_limit - _beta_P) < 0.02:
        _verdict = "about right"
    elif abs(_beta_P) < 1e-9:
        _verdict = "a spurious effect"
    elif _limit * _beta_P < 0:
        _verdict = "the wrong sign"
    elif abs(_limit) > abs(_beta_P):
        _verdict = "an over-estimate"
    else:
        _verdict = "an under-estimate"

    _fig, (_ax_data, _ax_slopes) = plt.subplots(1, 2, figsize=(11, 4))
    _ax_data.scatter(_P_star[0], _C[0], s=8, color="C0", alpha=0.4)
    _xs = np.array([_P_star[0].min(), _P_star[0].max()])
    _ax_data.plot(_xs, _C[0].mean() + _slopes[0] * (_xs - _P_star[0].mean()), color="C0", linewidth=2, label=f"fitted slope: {_slopes[0]:.2f}")
    _ax_data.plot(_xs, _beta_P * _xs, color="k", linestyle="--", label=f"true effect: {_beta_P:.2f}")
    _ax_data.set_xlabel("measured parental income $P^*$")
    _ax_data.set_ylabel("child income $C$")
    _ax_data.set_title("one simulated dataset")
    _ax_data.legend(fontsize=8, loc="upper left")

    _ax_slopes.hist(_slopes, bins=30, color="C0", alpha=0.6)
    _ax_slopes.axvline(_beta_P, color="k", linestyle="--", label="true effect $\\beta_P$")
    _ax_slopes.axvline(_limit, color="C0", linewidth=2, label="slope with unlimited data")
    _ax_slopes.set_xlabel("estimated slope of $C$ on $P^*$")
    _ax_slopes.set_ylabel("datasets")
    _ax_slopes.set_title(f"estimate {_limit:.2f} vs true {_beta_P:.2f}: {_verdict}")
    _ax_slopes.legend(fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Modeling Measurement

    ## Revisiting the Waffle House Divorce Dataset
    """)
    return


@app.cell
def load_waffle_divorce():
    WAFFLE_DIVORCE = utils.load_data("WaffleDivorce")
    WAFFLE_DIVORCE.head()
    return (WAFFLE_DIVORCE,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problems
    - Imbalance in evidence
    - Potential confounding
      - e.g. population size
    """)
    return


@app.cell(hide_code=True)
def marriage_divorce_errors_plot(WAFFLE_DIVORCE):
    utils.plot_scatter(
        WAFFLE_DIVORCE.Marriage,
        WAFFLE_DIVORCE.Divorce,
        label=None
    )

    utils.plot_errorbar(
        xs=WAFFLE_DIVORCE.Marriage,
        ys=WAFFLE_DIVORCE.Divorce,
        error_lower=WAFFLE_DIVORCE['Divorce SE'],
        error_upper=WAFFLE_DIVORCE['Divorce SE'],
        error_width=7
    )

    utils.plot_x_errorbar(
        xs=WAFFLE_DIVORCE.Marriage,
        ys=WAFFLE_DIVORCE.Divorce,
        error_lower=WAFFLE_DIVORCE['Marriage SE'],
        error_upper=WAFFLE_DIVORCE['Marriage SE'],
        error_width=7
    )

    plt.xlabel("marriage rate")
    plt.ylabel("divorce rate")
    plt.title("Errors not consistent across states")
    plt.gcf()
    return


@app.cell(hide_code=True)
def population_errors_plot(WAFFLE_DIVORCE):
    utils.plot_scatter(
        np.log(WAFFLE_DIVORCE.Population),
        WAFFLE_DIVORCE.Divorce,
        label=None
    )

    utils.plot_errorbar(
        xs=np.log(WAFFLE_DIVORCE.Population),
        ys=WAFFLE_DIVORCE.Divorce,
        error_lower=WAFFLE_DIVORCE['Divorce SE'],
        error_upper=WAFFLE_DIVORCE['Divorce SE'],
        error_width=7
    )
    plt.xlabel("log population")
    plt.ylabel("divorce rate")
    plt.title("Error decreases with state population")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Measured divorce model
    """)
    return


@app.cell(hide_code=True)
def full_measurement_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("M", "D"),
            ("A", "M"),
            ("A", "D"),
            ("M", "M*"),
            ("D", "D*"),
            ("eD", "D*"),
            ("eM", "M*"),
            ("A", "A*"),
            ("eA", "A*"),
            ("P", "eA"),
            ("P", "eM"),
            ("P", "eD"),
            ("P", "D")
        ],
        node_props={
            "unobserved": {"style": "dashed"},
            "A": {"style": "dashed"},
            "M": {"style": "dashed"},
            "D": {"style": "dashed"},
        },
        edge_props={
            ("P", "eM"): {"color": "blue"},
            ("P", "eD"): {"color": "blue", "label": "population confounds", "fontcolor": "blue"},
            ("P", "D"): {"style": "dashed", "label": " potential backdoor", "color": "red", "fontcolor": "red"},
            ("A", "A*"): {"color": "red"},
            ("P", "eA"): {"color": "purple"},
            ("eA", "A*"): {"color": "red"},
            ("A", "D"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Thinking like a graph

    - **Thinking like regression**: which _predictors_ do I include in the model?
      - stop doing this
      - To be fair, GLMs enable this approach
      - The scientific world is SO MUCH LARGER than GLMs (e.g. think about the difference equation in tool use example)
    - **Thinking like a graph**: how do I model the network of _causes_?
      - e.g. Full Luxury Bayes

    ## Let's start simpler
    """)
    return


@app.cell(hide_code=True)
def divorce_measurement_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("M", "D"),
            ("A", "M"),
            ("A", "D"),
            ("M", "M*"),
            ("D", "D*"),
            ("eD", "D*")
        ],
        node_props={
            "unobserved": {"style": "dashed"},
            "D": {"style": "dashed", "color": "red"},
            "M": {"color": "red"},
            "A": {"color": "red"},
            "eD": {"color": "blue"},
            "D*": {"color": "blue"},
        },
        edge_props={
            ("A", "M"): {"color": "red"},
            ("A", "D"): {"color": "red", "label": "Divorce Model", "fontcolor": "red"},
            ("M", "D"): {"color": "red"},
            ("eD", "D*"): {"color": "blue", "label": " Divorce Measurement\nError Model", "fontcolor": "blue"},
            ("D", "D*"): {"color": "blue"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2 submodels

    #### <span style="color:red">Divorce Model</span>

    $$
    \begin{align*}
    D_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta_A A_i + \beta_M M_i
    \end{align*}
    $$

    #### <span style="color:blue">Divorce Measurement Error Model</span>
    $$
    \begin{align*}
    D^*_i &= D_i + e_{D,i} \\
    e_{D,i} &\sim \text{Normal}(0, S_i) &S_i \text{ is the reported standard error for state } i
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the Divorce Measurement Error Model

    - two simultaneous regressions
      - One for $D$
        - Note: these are just parameters in the model, they are not observed 🤯
      - One for $D^*$
    """)
    return


@app.cell
def divorce_data(WAFFLE_DIVORCE):
    DIVORCE_STD = WAFFLE_DIVORCE['Divorce'].std(ddof=0)  # the same scale as utils.standardize
    DIVORCE = utils.standardize(WAFFLE_DIVORCE['Divorce']).values
    MARRIAGE = utils.standardize(WAFFLE_DIVORCE['Marriage']).values
    AGE = utils.standardize(WAFFLE_DIVORCE['MedianAgeMarriage']).values
    DIVORCE_SE = WAFFLE_DIVORCE['Divorce SE'].values / DIVORCE_STD
    STATE_ID, STATE = pd.factorize(WAFFLE_DIVORCE['Loc'])
    return AGE, DIVORCE, DIVORCE_SE, MARRIAGE, STATE, STATE_ID


@app.cell
def measured_divorce_fit(AGE, DIVORCE, DIVORCE_SE, MARRIAGE, STATE):
    coords = {'state': STATE}
    with pm.Model(coords=coords) as measured_divorce_model:
        _sigma = pm.Exponential("sigma", 1)
        _alpha = pm.Normal("alpha", 0, 0.2)
        _beta_A = pm.Normal("beta_A", 0, 0.5)
        _beta_M = pm.Normal("beta_M", 0, 0.5)

        _mu = _alpha + _beta_A * AGE + _beta_M * MARRIAGE
        _D = pm.Normal("D", _mu, _sigma, dims='state')
        pm.Normal("D*", _D, DIVORCE_SE, observed=DIVORCE)

        measured_divorce_inference = cached_sample()
    return coords, measured_divorce_inference


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior for True Divorce Rates
    """)
    return


@app.cell
def true_divorce_summary(measured_divorce_inference):
    az.summary(measured_divorce_inference, var_names=["D"])[:10]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### The effect of modeling the Divorce measurement error
    """)
    return


@app.function
def plot_change_xy(x_orig, y_orig, x_new, y_new, color='C0'):
    for ii, (xo, yo, xn, yn) in enumerate(zip(x_orig, y_orig, x_new, y_new)):
        label = 'change' if not ii else None
        plt.plot((xo, xn), (yo, yn), linewidth=2, alpha=.25, color=color, label=label)


@app.cell(hide_code=True)
def divorce_shrinkage_plot(AGE, DIVORCE, measured_divorce_inference):
    _D_true = measured_divorce_inference.posterior.mean(dim=('chain', 'draw'))['D'].values

    # Plot raw observations
    utils.plot_scatter(
        AGE,
        DIVORCE,
        color='k',
        label='observed (std)',
        zorder=100
    )

    # Plot divorce posterior
    utils.plot_scatter(
        AGE,
        _D_true,
        color='C0',
        label='posterior mean',
        zorder=100
    )

    plot_change_xy(AGE, DIVORCE, AGE, _D_true)

    # Add trendline
    # AGE and DIVORCE are standardized, so no need for offset
    _trend_slope = np.linalg.lstsq(AGE[:, None], _D_true)[0]
    _xs = np.linspace(-3, 3, 10)
    _ys = _xs * _trend_slope
    utils.plot_line(
        xs=_xs,
        ys=_ys,
        linestyle='--',
        color='C0',
        label='posterior trend',
        linewidth=1
    )

    plt.xlabel("marriage age (std)")
    plt.ylabel("divorce rate (std)")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - black points ignore measurement error
    - red points are posterior means for model that captures measurement error in divorce rates
    - thin pink lines demonstrate the movement of estimates by modeling measurement error
    - Modeling measurement error in divorce rates shrinks estimates for states with more uncertainty (larger standard errors) toward the main trend line (dashed red line)
      - partial pooling information across states
    - **You get this all for free by drawing the graph, and following the rules of probability**
    """)
    return


@app.cell(hide_code=True)
def shrinkage_controls(measured_divorce_inference):
    _sigma_mean = float(measured_divorce_inference.posterior["sigma"].mean())
    shrinkage_params = mo.ui.dictionary({
        "se_scale": mo.ui.slider(steps=[0.25, 0.5, 1, 2, 4], value=1, label="scale the standard errors by", show_value=True),
        "sigma": mo.ui.slider(0.05, 1.5, step=0.05, value=round(_sigma_mean * 20) / 20, label="spread around the regression $\\sigma$", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: what decides how far each state moves?
        Given the regression parameters, each state's true divorce rate $D_i$ has an exact Normal posterior: a precision-weighted average of its measurement $D^*_i$ (precision $1/S_i^2$) and its regression prediction $\mu_i = \alpha + \beta_A A_i + \beta_M M_i$ (precision $1/\sigma^2$). So each state moves the share $S_i^2 / (S_i^2 + \sigma^2)$ of the way from its measurement to its prediction.

        The explorer fixes $\alpha$, $\beta_A$ and $\beta_M$ at their posterior means and starts $\sigma$ at its posterior mean, so at the default settings the points are close to the posterior means in the plot above. Scale the standard errors up and every state slides toward its prediction; shrink them, or widen $\sigma$, and the measurements win.
        """),
        mo.hstack([shrinkage_params["se_scale"], shrinkage_params["sigma"]], justify="start", gap=2),
    ])
    return (shrinkage_params,)


@app.cell(hide_code=True)
def shrinkage_explorer(
    AGE,
    DIVORCE,
    DIVORCE_SE,
    MARRIAGE,
    STATE,
    measured_divorce_inference,
    shrinkage_params,
):
    _v = shrinkage_params.value
    _post = measured_divorce_inference.posterior
    _alpha, _beta_A, _beta_M = (float(_post[_name].mean()) for _name in ["alpha", "beta_A", "beta_M"])
    _mu = _alpha + _beta_A * AGE + _beta_M * MARRIAGE
    _se = _v["se_scale"] * DIVORCE_SE
    _weight = _se ** 2 / (_se ** 2 + _v["sigma"] ** 2)  # share of the way from D*_i to mu_i
    _D_shrunk = DIVORCE + _weight * (_mu - DIVORCE)

    _fig, (_ax_states, _ax_weight) = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.sca(_ax_states)
    utils.plot_scatter(AGE, DIVORCE, color='k', label='observed (std)', zorder=100)
    utils.plot_scatter(AGE, _D_shrunk, color='C0', label='posterior mean', zorder=100)
    plot_change_xy(AGE, DIVORCE, AGE, _D_shrunk)
    _ax_states.scatter(AGE, _mu, marker="x", s=20, color="C1", label="regression prediction $\\mu_i$")
    _ax_states.set_ylim(-2.5, 3.5)
    _ax_states.set_xlabel("marriage age (std)")
    _ax_states.set_ylabel("divorce rate (std)")
    _ax_states.legend(fontsize=8)

    _ses = np.linspace(0, _se.max() * 1.05, 200)
    _ax_weight.plot(_ses, _ses ** 2 / (_ses ** 2 + _v["sigma"] ** 2), color="C1")
    _ax_weight.scatter(_se, _weight, color="C0", zorder=10)
    for _i in np.argsort(_se)[-3:]:
        _ax_weight.annotate(STATE[_i], (_se[_i], _weight[_i]), textcoords="offset points", xytext=(-16, 5), fontsize=8)
    _ax_weight.set_ylim(0, 1.08)
    _ax_weight.set_xlabel("standard error $S_i$ (std units)")
    _ax_weight.set_ylabel("share of the way to the prediction")
    _ax_weight.set_title(f"$\\sigma$ = {_v['sigma']:.2f}: states move {_weight.mean():.0%} of the way on average")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Divorce and Marriage Measurement Error Model
    """)
    return


@app.cell(hide_code=True)
def divorce_marriage_measurement_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("M", "D"),
            ("A", "M"),
            ("A", "D"),
            ("M", "M*"),
            ("D", "D*"),
            ("eD", "D*")
        ],
        node_props={
            "unobserved": {"style": "dashed"},
            "D": {"style": "dashed", "color": "red"},
            "M": {"color": "darkcyan"},
            "A": {"color": "darkcyan"},
            "eD": {"color": "blue"},
            "D*": {"color": "blue"},
        },
        edge_props={
            ("A", "M"): {"color": "darkcyan", "label": "Marriage Model", "fontcolor": "darkcyan"},
            ("A", "D"): {"color": "red", "label": "Divorce Model", "fontcolor": "red"},
            ("M", "D"): {"color": "red"},
            ("M", "M*"): {"label": "Marriage Measurement\nError Model"},
            ("eD", "D*"): {"color": "blue", "label": " Divorce Measurement\nError Model", "fontcolor": "blue"},
            ("D", "D*"): {"color": "blue"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4 Submodels

    #### <span style="color:red">Divorce Model</span>

    $$
    \begin{align*}
    D_i &\sim \text{Normal}(\mu_{Di}, \sigma_D) \\
    \mu_{Di} &= \alpha_D + \beta_{AD} A_i + \beta_{MD} M_i
    \end{align*}
    $$

    #### <span style="color:blue">Divorce Measurement Error Model</span>
    $$
    \begin{align*}
    D^*_i &= D_i + e_{D,i} \\
    e_{D,i} &\sim \text{Normal}(0, S_{Di}) &S_{Di} \text{ is the reported standard error of the divorce rate}
    \end{align*}
    $$

    #### <span style="color:darkcyan">Marriage Rate Model</span>
    $$
    \begin{align*}
    M_i &\sim \text{Normal}(\mu_{Mi}, \sigma_M) \\
    \mu_{Mi} &= \alpha_M + \beta_{AM} A_i
    \end{align*}
    $$

    #### Marriage Rate Measurement Error Model
    $$
    \begin{align*}
    M^*_i &= M_i + e_{M,i} \\
    e_{M,i} &\sim \text{Normal}(0, S_{Mi}) &S_{Mi} \text{ is the reported standard error of the marriage rate}
    \end{align*}
    $$
    """)
    return


@app.cell
def measured_divorce_marriage_fit(
    AGE,
    DIVORCE,
    DIVORCE_SE,
    MARRIAGE,
    STATE,
    STATE_ID,
    WAFFLE_DIVORCE,
):
    MARRIAGE_STD = WAFFLE_DIVORCE['Marriage'].std(ddof=0)
    MARRIAGE_SE = WAFFLE_DIVORCE['Marriage SE'].values / MARRIAGE_STD

    _coords = {'state': STATE}

    with pm.Model(coords=_coords) as measured_divorce_marriage_model:

        # Marriage Rate Model
        _sigma_marriage = pm.Exponential("sigma_M", 1)
        _alpha_marriage = pm.Normal("alpha_M", 0, 0.2)
        _beta_AM = pm.Normal("beta_AM", 0, 0.5)

        # True marriage parameter
        _mu_marriage = _alpha_marriage + _beta_AM * AGE
        _M = pm.Normal("M", _mu_marriage, _sigma_marriage, dims='state')

        # Likelihood
        pm.Normal("M*", _M, MARRIAGE_SE, observed=MARRIAGE)

        # Divorce Model
        ## Priors
        _sigma_divorce = pm.Exponential("sigma_D", 1)
        _alpha_divorce = pm.Normal("alpha_D", 0, 0.2)
        _beta_AD = pm.Normal("beta_AD", 0, 0.5)
        _beta_MD = pm.Normal("beta_MD", 0, 0.5)

        # True divorce parameter
        _mu_divorce = _alpha_divorce + _beta_AD * AGE + _beta_MD * _M[STATE_ID]
        _D = pm.Normal("D", _mu_divorce, _sigma_divorce, dims='state')

        # Likelihood
        pm.Normal("D*", _D, DIVORCE_SE, observed=DIVORCE)

        measured_divorce_marriage_inference = cached_sample(target_accept=0.95)
    return (measured_divorce_marriage_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### The effect of modeling the Divorce and Marriage Rate measurement error
    """)
    return


@app.cell(hide_code=True)
def divorce_marriage_shrinkage_plot(
    DIVORCE,
    MARRIAGE,
    measured_divorce_marriage_inference,
):
    _D_true = measured_divorce_marriage_inference.posterior.mean(dim=('chain', 'draw'))['D'].values
    _M_true = measured_divorce_marriage_inference.posterior.mean(dim=('chain', 'draw'))['M'].values

    # Observations
    utils.plot_scatter(
        MARRIAGE,
        DIVORCE,
        color='k',
        label='observed (std)',
        zorder=100
    )

    # Posterior Means
    utils.plot_scatter(
        _M_true,
        _D_true,
        color='C0',
        label='posterior mean',
        zorder=100
    )

    # Change from modeling measurement error
    plot_change_xy(MARRIAGE, DIVORCE, _M_true, _D_true)

    # Add trendline
    _trend_slope = np.linalg.lstsq(_M_true[:, None], _D_true)[0]
    _xs = np.linspace(-3, 3, 10)
    _ys = _xs * _trend_slope
    utils.plot_line(
        xs=_xs,
        ys=_ys,
        color='C0',
        linestyle='--',
        linewidth=1,
        label='posterior trend'
    )

    plt.xlabel("marriage rate (std)")
    plt.ylabel("divorce rate (std)")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - When modeling measurement error for both Marriage and Divorce, we get estimates that move in multiple dimensions
    - Shrinkage (thin pink line) is toward the posterior trend (dashed red line)
      - the direction of movement is proportional to the uncertainty along the respective dimension
      - the degree of the movement is inversely proportional to the "quality" (in terms of std dev) of the data points
    - Again, we get this all for free by writing down the joint distribution and leaning on the axioms of probability
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compare causal effects for models that do and do not model measurement error
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit model that considers no measurement error
    """)
    return


@app.cell
def no_measurement_error_fit(AGE, DIVORCE, MARRIAGE, coords):
    with pm.Model(coords=coords) as no_measurement_error_model:

        # Divorce Model
        ## Priors
        _sigma = pm.Exponential("sigma", 1)
        _alpha = pm.Normal("alpha", 0, 0.2)
        _beta_A = pm.Normal("beta_A", 0, 0.5)
        _beta_M = pm.Normal("beta_M", 0, 0.5)

        # Likelihood
        _mu = _alpha + _beta_A * AGE + _beta_M * MARRIAGE
        pm.Normal("D", _mu, _sigma, observed=DIVORCE)

        no_measurement_error_inference = cached_sample(target_accept=0.95)
    return (no_measurement_error_inference,)


@app.cell(hide_code=True)
def measurement_error_effects_plot(
    measured_divorce_marriage_inference,
    no_measurement_error_inference,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(_axs[0])
    utils.plot_dist(no_measurement_error_inference.posterior['beta_A'], color='gray', label='ignoring error')
    utils.plot_dist(measured_divorce_marriage_inference.posterior['beta_AD'], color='C0', label='with error')
    plt.axvline(0, color='k', linestyle='--', label='no effect')
    plt.xlabel("$\\beta_A$")
    plt.ylabel("density")
    plt.legend()
    _axs[0].set_title("Effect of Age on Divorce Rate")

    plt.sca(_axs[1])
    utils.plot_dist(no_measurement_error_inference.posterior['beta_M'], color='gray', label='ignoring error')
    utils.plot_dist(measured_divorce_marriage_inference.posterior['beta_MD'], color='C0', label='with error')
    plt.axvline(0, color='k', linestyle='--', label='no effect')
    plt.xlabel("$\\beta_M$")
    plt.ylabel("density")
    plt.legend()
    _axs[1].set_title("Effect of Marriage on Divorce Rate")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Again, no clear rule for how measurement error will affect causal estimates

    - Estimated effect of age is attenuated when accounting for measurement error (left)
    - Estimated effect of marriage on divorce rate increases when modeling measurement error (right)
    - Effects of age and marriage found in previous examples could be due in part to measurement error and/or population confounds
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Unpredictable Errors

    - Modeling measurement error of Marriage increases the estimated effect of Marriage on Divorce
      - This is not intuitive
        - your intuitions are the devil 😈
      - Often results for nonlinear interactions are not intuitive -- rely on probability theory
      - Likely due to down-weighting the effect of unreliable, high-uncertainty datapoints, improving the estimate
    - Errors can "hurt" or "help", depending on goals
      - only honest option is to attempt to model them
      - do the best you can, analyze your data like you wish your colleagues would analyze their own.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Misclassification

    ## [Paternity in Himba pastoralist culture](https://www.science.org/doi/10.1126/sciadv.aay6195)

    - Unusual kinship systems (in a Western sense, anyway)
    - "Open" marriages
    - **Estimand**: proportion of children fathered by men in extra-marital relationships $p$
    - **Misclassification**: categorical version of measurement error
      - Paternity Test has some false positive rate $f$ (FPR, say 5%)
      - If the rate of extra-marital paternity is small, it may be on the same order as the FPR
        - thus often can't ignore the misclassification error
      - How do we include misclassification rate?
    """)
    return


@app.cell(hide_code=True)
def misclassification_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("F", "X"),
            ("F", "T"),
            ("T", "X"),
            ("M", "T"),
            ("M", "X"),
            ("X", "Xstar"),
            ("eX", "Xstar")
        ],
        node_props={
            "X": {"style": "dashed"},
            "unobserved": {"style": "dashed"}

        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - social father $F$
    - mother $M$
    - social ties (dyads) $T$
    - _actual_ extra-marital paternity $X$, unobserved
    - _measured_ extra-marital paternity $X^*$
    - misclassification errors $e_X$

    ### Generative Model
    $$
    \begin{align*}
    X_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha + \beta_M M_{\text{mother}[i]} + \beta_T T_i
    \end{align*}
    $$

    with mother traits $M_j \sim \text{Normal}(0, 1)$ and relationship strengths $T_i \sim \text{Normal}(0, 1)$. The simulation below uses $\alpha = 0$, $\beta_M = 1$ and $\beta_T = 3$.

    ### Measurement Model

    $$
    \begin{align*}
    P(X^*=1 | p_i) &= p_i + (1 - p_i)f \\
    P(X^*=0 | p_i) &= (1 - p_i)(1 - f)
    \end{align*}
    $$

    With
    - false-positive rate $f$
    - probability of extra-marital paternity $p$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Deriving the Misclassification Error Model graphically:

    Rather than being clever, write out all possible measurement outcomes
    """)
    return


@app.cell(hide_code=True)
def misclassification_tree():
    utils.draw_causal_graph(
        edge_list=[
            ("", "X=1"),
            ("", "X=0"),
            ("X=1", "X*=1"),
            ("X=0", "X*=1"),
            ("X=0", "X*=0"),

        ],
        node_props={
            "": {"shape": "point"}
        },
        edge_props={
            ("", "X=1"): {"label": " p"},
            ("", "X=0"): {"label": " 1-p", },
            ("X=1", "X*=1"): {"label": " 1"},
            ("X=0", "X*=1"): {"label": " f",},
            ("X=0", "X*=0"): {"label": " 1-f"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $p(X^*=1) = p + (1-p)f$

    The probability of observing $X^*=1$ given $p$ is the sum of the red paths below: an actual $X=1$ always tests positive, and an actual $X=0$ tests positive with probability $f$
    """)
    return


@app.cell(hide_code=True)
def misclassification_tree_x1():
    utils.draw_causal_graph(
        edge_list=[
            ("", "X=1"),
            ("", "X=0"),
            ("X=1", "X*=1"),
            ("X=0", "X*=1"),
            ("X=0", "X*=0"),

        ],
        node_props={
            "": {"shape": "point"},
            "X=1": {"color": "red"},
            "X=0": {"color": "red"},
            "X*=1": {"color": "red"},
        },
        edge_props={
            ("", "X=1"): {"label": " p", "color": "red"},
            ("", "X=0"): {"label": " 1-p", "color": "red"},
            ("X=1", "X*=1"): {"label": " 1", "color": "red"},
            ("X=0", "X*=1"): {"label": " f", "color": "red"},
            ("X=0", "X*=0"): {"label": " 1-f"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $p(X^*=0) = (1-p)(1-f)$

    The probability of observing $X^*=0$ given $p$ is the red path below
    """)
    return


@app.cell(hide_code=True)
def misclassification_tree_x0():
    utils.draw_causal_graph(
        edge_list=[
            ("", "X=1"),
            ("", "X=0"),
            ("X=1", "X*=1"),
            ("X=0", "X*=1"),
            ("X=0", "X*=0"),

        ],
        node_props={
            "": {"shape": "point"},
            "X=0": {"color": "red"},
            "X*=0": {"color": "red"},
        },
        edge_props={
            ("", "X=1"): {"label": " p"},
            ("", "X=0"): {"label": " 1-p", "color": "red"},
            ("X=0", "X*=0"): {"label": " 1-f",  "color": "red"},
            ("X=0", "X*=1"): {"label": " f"},
            ("X=1", "X*=1"): {"label": " 1"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Generating a Proxy Dataset

    AFAICT, the dataset in the Scelza et al. paper isn't publicly available, so let's simulate one from the process defined by the generative model.
    """)
    return


@app.cell
def simulate_paternity():
    np.random.seed(123)

    # Generate social network
    N_MOTHERS = 30
    N_FATHERS = 15

    MOTHER_IDS = np.arange(N_MOTHERS).astype(int)
    FATHER_IDS = np.arange(N_FATHERS).astype(int)

    MOTHER_TRAITS = stats.norm.rvs(size=N_MOTHERS)
    DYADS = np.array(list((iproduct(MOTHER_IDS, FATHER_IDS))))

    N_DYADS = len(DYADS)
    DYAD_ID = np.arange(N_DYADS)

    RELATIONSHIPS = stats.norm.rvs(size=N_DYADS)

    PATERNITY = pd.DataFrame(
        {
            "mother_id": DYADS[:, 0],
            "social_father_id": DYADS[:, 1],
            "dyad_id": DYAD_ID,
            "relationship": RELATIONSHIPS,
        }
    )

    # Generative model
    ALPHA = 0
    BETA_MOTHER_TRAIT = 1
    BETA_RELATIONSHIP = 3

    PATERNITY.loc[:, "mother_trait"] = PATERNITY.mother_id.apply(lambda x: MOTHER_TRAITS[x])
    _p_father = utils.invlogit(
        ALPHA + BETA_MOTHER_TRAIT * PATERNITY.mother_trait + BETA_RELATIONSHIP * PATERNITY.relationship
    )

    PATERNITY.loc[:, "p_father"] =  _p_father
    PATERNITY.loc[:, "is_father"] =  stats.bernoulli.rvs(_p_father)  # unobserved actual paternity

    # Measurement model: an actual extra-marital paternity always tests positive,
    # and any other dyad tests positive (a false positive) with probability FALSE_POSITIVE_RATE
    FALSE_POSITIVE_RATE = 0.05

    _false_positive = stats.bernoulli.rvs(FALSE_POSITIVE_RATE, size=N_DYADS)
    PATERNITY.loc[:, 'is_father*'] = np.where(PATERNITY.is_father == 1, 1, _false_positive)

    print('Actual average paternity rate: ', PATERNITY['is_father'].mean().round(3))
    print('Measured average paternity rate: ', PATERNITY['is_father*'].mean().round(3))
    print('False positives: ', ((PATERNITY.is_father == 0) & (PATERNITY['is_father*'] == 1)).sum())
    return DYAD_ID, FALSE_POSITIVE_RATE, N_DYADS, N_MOTHERS, PATERNITY


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We print out the actual and measured paternity rates above. The measured rate is higher because of the false positives. A model that accounts for misclassification should recover the actual rate, and one that ignores it should recover the measured rate.
    """)
    return


@app.cell
def paternity_table(PATERNITY):
    PATERNITY
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the model with misclassification error

    #### Notes
    - We never observe $X_i$, so we average over its two possible values: the probability of each test result $X^*_i$ sums over the paths in the diagrams above. That's why the model has no $X_i \sim \text{Bernoulli}(p_i)$ term
    - Each dyad then contributes the probability of the test result we actually observed:
      - if $X^*_i=1$ we use $p(X^*_i=1) = p_i + (1-p_i)f$
      - if $X^*_i=0$ we use $p(X^*_i=0) = (1-p_i)(1-f)$
    - To use these custom distributions, we leverage the `pm.Potential` function, which adds the log probability of each observation to the model
      - this is analogous to the `custom` function used in lecture
      - We use the vanilla implementation below, one that doesn't use the `logsumexp`, etc. functions, as it compiles just fine
    """)
    return


@app.cell
def paternity_measurement_fit(
    DYAD_ID,
    FALSE_POSITIVE_RATE,
    N_DYADS,
    N_MOTHERS,
    PATERNITY,
):
    MOM_ID = PATERNITY.mother_id.values
    IS_FATHER_STAR = PATERNITY['is_father*'].values  # observed test results

    with pm.Model() as paternity_measurement_model:

        _beta_M = pm.TruncatedNormal("beta_M", 0, 1, lower=0.01)
        _beta_T = pm.TruncatedNormal("beta_T", 0, 1, lower=0.01)

        _alpha = pm.Normal("alpha", 0, 1.5)
        _T = pm.Normal("T", 0, 1, shape=N_DYADS)    # Relationship strength
        _M = pm.Normal("M", 0, 1, shape=N_MOTHERS)  # Mother traits

        _p_father = pm.Deterministic('p_father', pm.math.invlogit(_alpha + _M[MOM_ID] * _beta_M + _T[DYAD_ID] * _beta_T))

        # Each dyad contributes the probability of its observed test result
        _p_x1 = _p_father[IS_FATHER_STAR == 1]
        _custom_log_p_x1 = pm.math.log(_p_x1 + (1 - _p_x1) * FALSE_POSITIVE_RATE)
        pm.Potential("X*=1", _custom_log_p_x1)

        _p_x0 = _p_father[IS_FATHER_STAR == 0]
        _custom_log_p_x0 = pm.math.log((1 - _p_x0) * (1 - FALSE_POSITIVE_RATE))
        pm.Potential("X*=0", _custom_log_p_x0)

        paternity_measurement_inference = cached_sample(target_accept=.95)
    return IS_FATHER_STAR, MOM_ID, paternity_measurement_inference


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit analogous model without accounting for misclassification
    """)
    return


@app.cell
def paternity_naive_fit(DYAD_ID, IS_FATHER_STAR, MOM_ID, N_DYADS, N_MOTHERS):
    with pm.Model() as paternity_model:

        _alpha = pm.Normal("alpha", 0, 1.5)
        _beta_M = pm.TruncatedNormal("beta_M", 0, 1, lower=0.01)
        _beta_T = pm.TruncatedNormal("beta_T", 0, 1, lower=0.01)

        _T = pm.Normal("T", 0, 1, shape=N_DYADS)    # Relationship strength
        _M = pm.Normal("M", 0, 1, shape=N_MOTHERS)  # Mother traits

        _p_father = pm.Deterministic('p_father', pm.math.invlogit(_alpha + _M[MOM_ID] * _beta_M + _T[DYAD_ID] * _beta_T))
        pm.Bernoulli('is_father', _p_father, observed=IS_FATHER_STAR)

        paternity_inference = cached_sample(target_accept=.95)
    return (paternity_inference,)


@app.cell(hide_code=True)
def paternity_comparison_plot(
    PATERNITY,
    paternity_inference,
    paternity_measurement_inference,
):
    _fig, _ = plt.subplots(figsize=(10, 4))
    utils.plot_dist(paternity_inference.posterior['p_father'], color='k')
    _p_father_mean = paternity_inference.posterior['p_father'].mean()
    plt.axvline(_p_father_mean, color='k', linestyle='--', label=f"Ignoring misclassification: {_p_father_mean:1.2f}")

    utils.plot_dist(paternity_measurement_inference.posterior['p_father'])
    _p_father_mean = paternity_measurement_inference.posterior['p_father'].mean()
    plt.axvline(_p_father_mean, linestyle='--', label=f"Modeling misclassification: {_p_father_mean:1.2f}")
    _actual_rate = PATERNITY['is_father'].mean()
    plt.axvline(_actual_rate, color='gray', linestyle=':', label=f"Actual rate: {_actual_rate:1.2f}")
    plt.xlim([0, 1])
    plt.xlabel("probability of extra-marital paternity, all dyads")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Including misclassification, the model recovers the _actual_ average paternity rate (about 0.48).
    - Without it, the model recovers the _measured_ rate (about 0.51), which the false positives inflate.
    - The posterior of the average rate is about as wide in both models (sd of about 0.02): accounting for misclassification moves the estimate, it doesn't make it more precise.
    - Each dyad has a single child, so the effect of relationship strength $\beta_T$ is only weakly identified: its posterior stays close to its prior.
    - With an actual rate near 0.5, 5% false positives shift the estimate only a little. The bias grows when the actual rate is on the same order as the false-positive rate; try it in the explorer below.
    """)
    return


@app.cell(hide_code=True)
def misclassification_controls():
    misclassification_params = mo.ui.dictionary({
        "p": mo.ui.slider(0.01, 0.6, step=0.01, value=0.05, label="actual rate $p$", show_value=True),
        "f": mo.ui.slider(0.0, 0.3, step=0.01, value=0.05, label="false-positive rate $f$", show_value=True),
        "fn": mo.ui.slider(0.0, 0.3, step=0.01, value=0.0, label="false-negative rate", show_value=True),
        "n": mo.ui.slider(steps=[50, 100, 200, 450, 1000, 2000, 5000], value=450, label="children tested", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: when does misclassification matter?
        A stripped-down version of the paternity model, with a single rate $p$ and no covariates. Each of $n$ children has $X \sim \text{Bernoulli}(p)$. The test reports $X^* = 1$ for an actual $X = 1$ (unless the optional false-negative rate strikes), and for an actual $X = 0$ with probability $f$. Both posteriors are computed exactly on a grid with a flat prior. The naive one treats $X^*$ as if it were $X$; the aware one uses $P(X^*=1) = p\,(1 - f_n) + (1-p)\,f$ with the known error rates.

        The simulated paternity data above has an actual rate near 0.5, where 5% false positives barely matter. The sliders start at $p = f = 0.05$, where the naive estimate is about double the actual rate.
        """),
        mo.hstack([
            mo.vstack([misclassification_params["p"], misclassification_params["n"]]),
            mo.vstack([misclassification_params["f"], misclassification_params["fn"]]),
        ], justify="start", gap=4),
    ])
    return (misclassification_params,)


@app.cell(hide_code=True)
def misclassification_explorer(misclassification_params):
    _v = misclassification_params.value
    _n = _v["n"]
    _rng = np.random.default_rng(17)
    _u = _rng.uniform(size=(3, 5000))  # the same random numbers for every setting, so the data change smoothly
    _X = _u[0, :_n] < _v["p"]
    _X_star = np.where(_X, _u[1, :_n] >= _v["fn"], _u[2, :_n] < _v["f"])
    _k = _X_star.sum()

    # Exact grid posteriors with a flat prior on p
    _ps = np.linspace(0.0005, 0.9995, 2000)
    _dp = _ps[1] - _ps[0]
    _p_positive = {
        "ignoring misclassification": _ps,
        "modeling misclassification": _ps * (1 - _v["fn"]) + (1 - _ps) * _v["f"],  # P(X*=1 | p)
    }

    _fig, (_ax_post, _ax_curve) = plt.subplots(1, 2, figsize=(11, 4))
    for (_label, _q), _color in zip(_p_positive.items(), ["k", "C0"]):
        _log_lik = stats.binom.logpmf(_k, _n, _q)
        _density = np.exp(_log_lik - _log_lik.max())
        _density /= _density.sum() * _dp
        _ax_post.plot(_ps, _density, color=_color, linewidth=2, label=f"{_label}: {(_ps * _density).sum() * _dp:.3f}")
    _ax_post.axvline(_X.mean(), color="gray", linestyle=":", label=f"actual rate in the sample: {_X.mean():.3f}")
    _ax_post.set_xlim(0, min(1, 2 * max(_v["p"], _k / _n) + 0.05))
    _ax_post.set_xlabel("p")
    _ax_post.set_ylabel("posterior density")
    _ax_post.set_title(f"{_k} of {_n} children test positive")
    _ax_post.legend(fontsize=8)

    _q_true = _v["p"] * (1 - _v["fn"]) + (1 - _v["p"]) * _v["f"]
    _ax_curve.plot(_ps, _p_positive["modeling misclassification"], color="C0", label="with misclassification")
    _ax_curve.plot(_ps, _ps, color="k", linestyle="--", label="without misclassification")
    _ax_curve.scatter([_v["p"]], [_q_true], color="C0", zorder=10)
    _ax_curve.set_xlabel("actual rate p")
    _ax_curve.set_ylabel("expected measured rate")
    _ax_curve.set_title(f"expected measured rate {_q_true:.3f}: {_q_true / _v['p']:.1f}× the actual rate")
    _ax_curve.legend(fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Measurement & Misclassification Horizons

    There are a number of problems and solutions related to modeling measurement error and misclassification

    - Item response theory (IRT)
      - e.g. NJ wine judging example earlier
    - Factor Analysis
    - Hurdle models
      - measurements need to cross a threshold before they are identifiable
    - Occupancy models
      - considering the existence of phenomena without detection
      - big in applied ecology: just because a species isn't detected doesn't mean it isn't there
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Floating Point Monsters

    - Working on the log scale avoids (er, minimizes) floating point problems: underflow (tiny probabilities rounding to zero) and loss of precision (e.g. $1 - p$ rounding to one)
    - "ancient weapons": special functions for working on the log scale:
      - `pm.math.logsumexp`: stably computes the log of the sum of exponentials of input elements.
      - `pm.math.log1mexp`: calculates $\log(1 - \exp(x))$
      - `log1p` and `log1m`: calculate $\log(1 + x)$ and $\log(1 - x)$ (`log1m` is defined below)

    ### Logs make sense

    $$
    \begin{align*}
    P(X^*=0) &= (1-p)(1-f) \\
    \log P(X^*=0) & = \log (1-p) + \log (1-f)  &\text{put on log scale (used by HMC)}
    \end{align*}
    $$

    ### Devil's in the detail

    Some terms can become numerically unstable

    - e.g. if $p \approx 0$, $1-p$ rounds to exactly 1, so $\log(1-p)$ returns $\log(1) = 0$ instead of $\approx -p$: the information in $p$ is lost
    - e.g. if $p \approx 1, \log(1-p) \approx \log(0) = -\infty$
    """)
    return


@app.cell
def log1m_naive_demo():
    for _p in [1e-2, 1e-10, 1e-90]:
        print('p:', _p, ', log(1-p):', np.log(1-_p))
    return


@app.cell
def log1m_stable_demo():
    def _log1m(x):
        return np.log1p(-x)

    for _p in [1e-2, 1e-10, 1e-90]:
        print('p:', _p, ', log1p(-p):', _log1m(_p))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `log1p` and Taylor series approximation for small $p$

    - when $p > 10^{-4}$, just calculate $\log(1 + p)$
    - when $p < 10^{-4}$, use Taylor expansion: $\log(1 + p) = p - \frac{p^2}{2} + \frac{p^3}{3} - ...$
    """)
    return


@app.cell
def my_log1p_demo():
    def my_log1p(x):
        if x > 1e-4:
            return np.log(1 + x)
        else:
            # second order Taylor expansion
            return x - x ** 2 / 2

    # Compare results to numpy log1p
    for _p in [1e-3, 1e-10, 1e-90]:
        print('p:', _p, ', numpy.log1p:', np.log1p(_p), ', my_log1p', my_log1p(_p))
    return


@app.cell(hide_code=True)
def floating_point_controls():
    floating_point_params = mo.ui.dictionary({
        "k": mo.ui.slider(1, 100, step=1, value=10, label="$p = 10^{-k}$ with k =", show_value=True),
        "L": mo.ui.slider(1, 1000, step=1, value=10, label="log-probabilities $a = -L$, $b = -L - 1$ with L =", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: floating point monsters
        **Left:** computing $1 - p$ first throws away the digits of $p$. The relative error of `np.log(1 - p)` grows as $p$ shrinks, and once $p$ is below about $10^{-16}$, $1 - p$ is exactly 1 and the result is 0. `np.log1p(-p)` stays accurate for any $p$.

        **Right:** $\log(e^a + e^b)$, like adding the two paths of $P(X^*=1)$ when both are tiny probabilities. Past $L \approx 745$, both $e^a$ and $e^b$ underflow to 0, so the naive log is $-\infty$. `logsumexp` subtracts the maximum first and stays exact.
        """),
        mo.hstack([floating_point_params["k"], floating_point_params["L"]], justify="start", gap=2),
    ])
    return (floating_point_params,)


@app.cell(hide_code=True)
def floating_point_explorer(floating_point_params):
    _v = floating_point_params.value
    _p = 10.0 ** -_v["k"]
    _a, _b = -float(_v["L"]), -float(_v["L"]) - 1
    with np.errstate(divide="ignore"):
        _naive_log1m = np.log(1 - _p)
        _naive_lse = np.log(np.exp(_a) + np.exp(_b))
    _stable_log1m = np.log1p(-_p)
    _stable_lse = np.logaddexp(_a, _b)  # the max-subtraction trick of pm.math.logsumexp


    def _rel_error(naive, stable):
        return abs(naive - stable) / abs(stable)


    _table = mo.md(f"""
    | | naive | stable | relative error |
    | --- | --- | --- | --- |
    | $\\log(1 - p)$ | `np.log(1 - p)` = {_naive_log1m:.10g} | `np.log1p(-p)` = {_stable_log1m:.10g} | {_rel_error(_naive_log1m, _stable_log1m):.2g} |
    | $\\log(e^a + e^b)$ | `np.log(np.exp(a) + np.exp(b))` = {_naive_lse:.10g} | logsumexp = {_stable_lse:.10g} | {_rel_error(_naive_lse, _stable_lse):.2g} |
    """)

    _fig, (_ax_log1m, _ax_lse) = plt.subplots(1, 2, figsize=(11, 4))
    _ps = np.logspace(-25, -1, 500)
    _errors = np.abs(np.log(1 - _ps) - np.log1p(-_ps)) / np.abs(np.log1p(-_ps))
    _ax_log1m.loglog(_ps, np.maximum(_errors, 1e-17), color="C0")
    if _v["k"] <= 25:
        _ax_log1m.scatter([_p], [max(_rel_error(_naive_log1m, _stable_log1m), 1e-17)], color="k", zorder=10, label=f"p = 1e-{_v['k']}")
        _ax_log1m.legend(fontsize=8)
    _ax_log1m.set_xlabel("p")
    _ax_log1m.set_ylabel("relative error of np.log(1 - p)")
    _ax_log1m.set_title("100% error once 1 - p rounds to 1")

    _Ls = np.arange(1, 1001)
    with np.errstate(divide="ignore"):
        _naive_curve = np.log(np.exp(-_Ls) + np.exp(-_Ls - 1.0))
    _underflow = _Ls[np.isneginf(_naive_curve)].min()
    _ax_lse.plot(_Ls, np.logaddexp(-_Ls, -_Ls - 1.0), color="C0", linewidth=5, alpha=0.4, label="logsumexp")
    _ax_lse.plot(_Ls, np.where(np.isneginf(_naive_curve), np.nan, _naive_curve), color="k", label="np.log(np.exp(a) + np.exp(b))")
    _ax_lse.axvline(_underflow, color="k", linestyle=":", label=f"naive result is -inf from L = {_underflow}")
    _ax_lse.scatter([_v["L"]], [_stable_lse], color="C0", zorder=10)
    _ax_lse.set_xlabel("L")
    _ax_lse.set_ylabel("log(exp(a) + exp(b))")
    _ax_lse.set_title(f"L = {_v['L']}: naive {_naive_lse:.4g}, logsumexp {_stable_lse:.4g}")
    _ax_lse.legend(fontsize=8)
    mo.vstack([_table, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `logsumexp`

    - Use when you need the log of a sum of terms
    - We want to stay on the log scale, because it is numerically stable
    - But the log of a sum doesn't split into a sum of logs (only the log of a product does), e.g.

    $$
    \begin{align*}
    P(X^*=1) &=  p + (1-p)f \\
    \log P(X^*=1) &=  \log [p + (1-p)f] \\
    \log P(X^*=1) &= \text{logsumexp}\big([\log(p),\ \log(1-p) + \log(f)]\big) \\
    \log P(X^*=1) &= \text{logsumexp}\big([\log(p),\ \text{log1m}(p) + \log(f)]\big)
    \end{align*}
    $$

    where $\text{logsumexp}([a, b]) = \log(e^a + e^b)$ is computed as $c + \log(e^{a-c} + e^{b-c})$ with $c = \max(a, b)$, so that nothing underflows.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Previous Paternity measurement model with log-scaling tricks
    """)
    return


@app.function
def log_sum_exp(a, b):
    """
    Simple, numerically-stable logsumexp implementation for two inputs.

    `pm.math.logsumexp` uses the same max-subtraction trick and is just
    as stable, but it takes a single tensor, so the two terms would first
    have to be stacked along a new axis.
    """
    c = pm.math.maximum(a, b)
    return c + pm.math.log(pm.math.exp(a - c) + pm.math.exp(b - c))


@app.function
def log1p(x):
    return pm.math.log1p(x)


@app.function
def log1m(x):
    return log1p(-x)


@app.cell
def paternity_measurement_lse_fit(
    DYAD_ID,
    FALSE_POSITIVE_RATE,
    IS_FATHER_STAR,
    MOM_ID,
    N_DYADS,
    N_MOTHERS,
):
    with pm.Model() as paternity_measurement_lse_model:

        _beta_M = pm.TruncatedNormal("beta_M", 0, 1, lower=0.01)
        _beta_T = pm.TruncatedNormal("beta_T", 0, 1, lower=0.01)

        _alpha = pm.Normal("alpha", 0, 1.5)
        _T = pm.Normal("T", 0, 1, shape=N_DYADS)    # Relationship strength
        _M = pm.Normal("M", 0, 1, shape=N_MOTHERS)  # Mother traits

        _p = pm.Deterministic('p', pm.math.invlogit(_alpha + _M[MOM_ID] * _beta_M + _T[DYAD_ID] * _beta_T))
        _f = FALSE_POSITIVE_RATE

        # # For reference: the original custom logp with vanilla parameterization
        # custom_log_p_x1 = pm.math.log(p + (1 - p) * f)
        # custom_log_p_x0 = pm.math.log((1 - p) * (1 - f))

        # Custom logp using log-space reparameterization, for each dyad's observed test result
        _p_x1 = _p[IS_FATHER_STAR == 1]
        _p_x0 = _p[IS_FATHER_STAR == 0]
        _custom_log_p_x1 = log_sum_exp(pm.math.log(_p_x1), log1m(_p_x1) + pm.math.log(_f))
        _custom_log_p_x0 = log1m(_p_x0) + log1m(_f)

        pm.Potential("X*=1", _custom_log_p_x1)
        pm.Potential("X*=0", _custom_log_p_x0)

        paternity_measurement_lse_inference = cached_sample(target_accept=.95)
    return (paternity_measurement_lse_inference,)


@app.cell(hide_code=True)
def parameterization_comparison_plot(
    paternity_measurement_inference,
    paternity_measurement_lse_inference,
):
    # Original vanilla parameterization
    utils.plot_dist(paternity_measurement_inference.posterior['p_father'])
    _p_father_mean = paternity_measurement_inference.posterior['p_father'].mean()
    plt.axvline(_p_father_mean, linestyle='--', label=f"Vanilla parameterization: {_p_father_mean:1.2f}")

    # Re-parameterization
    utils.plot_dist(paternity_measurement_lse_inference.posterior['p'], color='C1')
    _p_father_mean = paternity_measurement_lse_inference.posterior['p'].mean()
    plt.axvline(_p_father_mean, linestyle='--', color='C1', label=f"Re-parameterization: {_p_father_mean:1.2f}")
    plt.xlim([0, 1])

    plt.legend()
    plt.title("Different Parameterizations,\nEquivalent results")
    plt.gcf()
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
