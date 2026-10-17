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

    In lecure, instead of opening doors to find donkeys or prizes, as was in the case of the Monty Hall problem, we have pancakes that are either burnt or perfectly cooked on either side. The thought experiment goes like this:

    - You have 3 pancakes:
      - One (likely the first one you cooked) is burnt on both sides
      - One (likely the next one cooked, after you're improving your cooking skills) is burnt on only one side
      - One is cooked perfectly on both sides
    - Say you are given a pancake at random and it is burned on one side
    - What is the probability that the other side is burned?

    Most folks would say 1/2, which intuitively _feels_ correct. However, the correct answer is given by Bayes rule:

    If we define $U, D$ as observing upside $U$ or downsid $D$ hot being burnt and $U', D'$ as up or down sides being burnt

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
def _():
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
    - Error in Parental income reporting $P^*$ (e.g. due to recall bias)
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
def _():
    simulate_child_parent_income(title="Measurement error makes\nus over-estimate effect")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the scenario above, measurement error can cause us over-estimate the actual effect
    """)
    return


@app.cell
def _():
    simulate_child_parent_income(beta_P=0.75, title="Measurement error makes\nus under-estimate effect")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the scenario above, measurement error can cause us under-estimate the actual effect

    - What happens depends on the details of the context
    - **There is no general rule to tell you what measurement error will do to your estimate**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Modeling Measurment

    ## Revisiting the Wafflehouse Divorce Dataset
    """)
    return


@app.cell
def _():
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
def _(WAFFLE_DIVORCE):
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
def _(WAFFLE_DIVORCE):
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
def _():
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
            ("P", "eA"): {"color": "blue"},
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
    - **Thinking like a graph**:how do I model the network of _causes_?
      - e.g. Full Luxury Bayes

    ## Let's start simpler
    """)
    return


@app.cell(hide_code=True)
def _():
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
    D_i &\sim \text{Normal}(\mu_i) \\
    \mu_i &= \alpha + \beta_A A + \beta_M M
    \end{align*}
    $$

    #### <span style="color:blue">Divorce Measurement Error Model</span>
    $$
    \begin{align*}
    D^*_i &= D_i + e_D \\
    e_{D,i} &\sim \text{Normal}(0, S_i) &S_i \text{ is the standard deviation estimated from the sample}
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
      - One for $D*$
    """)
    return


@app.cell
def _(WAFFLE_DIVORCE):
    DIVORCE_STD = WAFFLE_DIVORCE['Divorce'].std()
    DIVORCE = utils.standardize(WAFFLE_DIVORCE['Divorce']).values
    MARRIAGE = utils.standardize(WAFFLE_DIVORCE['Marriage']).values
    AGE = utils.standardize(WAFFLE_DIVORCE['MedianAgeMarriage']).values
    DIVORCE_SE = WAFFLE_DIVORCE['Divorce SE'].values / DIVORCE_STD
    STATE_ID, STATE = pd.factorize(WAFFLE_DIVORCE['Loc'])
    N = len(WAFFLE_DIVORCE)
    return AGE, DIVORCE, DIVORCE_SE, MARRIAGE, STATE, STATE_ID


@app.cell
def _(AGE, DIVORCE, DIVORCE_SE, MARRIAGE, STATE):
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
def _(measured_divorce_inference):
    az.summary(measured_divorce_inference, var_names=["D"])[:10]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### The effect of modeling the Divorce measurment error
    """)
    return


@app.function
def plot_change_xy(x_orig, y_orig, x_new, y_new, color='C0'):
    for ii, (xo, yo, xn, yn) in enumerate(zip(x_orig, y_orig, x_new, y_new)):
        label = 'change' if not ii else None
        plt.plot((xo, xn), (yo, yn), linewidth=2, alpha=.25, color=color, label=label)


@app.cell(hide_code=True)
def _(AGE, DIVORCE, measured_divorce_inference):
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
    - red points are posterior means for model that captures measurement error in Divorcd rates
    - thin pink lines demonstrate the movement of estimates by modeling measurment error
    - Modeling measurement error in divorce rates shrinks estimates for states more uncertainty (larger standard errors) toward the main trend line (dashed red line)
      - partial pooling information across states
    - **You get this all for free by drawing the graph, and following the rules of probablity**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Divorce and Marriage Measurement Error Model
    """)
    return


@app.cell(hide_code=True)
def _():
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
    D^*_i &= D_i + e_D \\
    e_{D,i} &\sim \text{Normal}(0, S_{Di}) &S_{Di} \text{ is the divorce standard deviation estimated from the sample}
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
    M^*_i &= M_i + e_M \\
    e_{M,i} &\sim \text{Normal}(0, S_{Mi}) &S_{Mi} \text{ is the marriage standard deviation estimated from the sample}
    \end{align*}
    $$
    """)
    return


@app.cell
def _(AGE, DIVORCE, DIVORCE_SE, MARRIAGE, STATE, STATE_ID, WAFFLE_DIVORCE):
    MARRIAGE_STD = WAFFLE_DIVORCE['Marriage'].std()
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
    #### The effect of modeling the Divorce and Marriage Rate measurment error
    """)
    return


@app.cell(hide_code=True)
def _(DIVORCE, MARRIAGE, measured_divorce_marriage_inference):
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
      - the direction of movement is proporitonal to the uncertainty along the respective dimension
      - the degree of the movement is inversely proportional to the "quality" (in terms of std dev) of the data points
    - Again, we get this all for free by writing down joint distribution and leaning on the axioms of probability
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
def _(AGE, DIVORCE, MARRIAGE, coords):
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
def _(measured_divorce_marriage_inference, no_measurement_error_inference):
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
    #### Again, no clear rule for how measurment error will effect causal estimates

    - Estimated effect of age is attenuated when accounting for measurement error (left)
    - Estimated effect of marriage on divorce rate increases when modeling measurement error (right)
    - Effects of age and marriage found in previous examples could be due in part to measurement error and/or population confounds
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Unpredictable Errors

    - Modeling measurment error of Marriage increases the estiamted effect of Marriage on Divorce
      - This is not intuitive
        - your intuitions are the devil 😈
      - Often results for nonlinear interactions are not intuitive -- rely on probability theory
      - Likely due to down-weighting the effect of unreliable, high-uncertainty datapoints, improving the estimate
    - Errors can "hurt" or "help", depending on goals
      - only honest option is to attempt to model them
      - do the best you can, analyze your data like you wish your collegues would analyze their own.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Misclassification

    ## [Paternity in Himba pastoralist culture](https://www.science.org/doi/10.1126/sciadv.aay6195)

    - Unusual kinships systems (in a Western sense, anyway)
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
def _():
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
    \text{logit}(p_i) &= \alpha + \mu_{M[i]} + \delta_{T[i]}
    \end{align*}
    $$

    ### Measurement Model

    $$
    \begin{align*}
    P(X^*=1 | p_i) = p_i + (1 - p_i)f \\
    P(X^*=0 | p_i) = (1 - p_i)(1 - f)
    \end{align*}
    $$

    With
    - error rate $f$
    - probability of extra-marital paternity $p$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Deriving the Misclassification Error Model graphically:

    Not being clever, right out all possible measurement outcomes
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("", "X=1"),
            ("", "X=0"),
            ("X=0", "X*=1"),
            ("X=0", "X*=0"),
    
        ],
        node_props={
            "": {"shape": "point"}
        },
        edge_props={
            ("", "X=1"): {"label": " p"},
            ("", "X=0"): {"label": " 1-p", },
            ("X=0", "X*=1"): {"label": " f",},
            ("X=0", "X*=0"): {"label": " 1-f"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $p(X^*=1) = p + (1-p)f$

    The probability of observing $X=1$ given $p$ is the red path below
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("", "X=1"),
            ("", "X=0"),
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
            ("X=0", "X*=1"): {"label": " f", "color": "red"},
            ("X=0", "X*=0"): {"label": " 1-f"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $p(X^*=0) = (1-p)(1-f)$

    The probability of observing $X=0$ given $p$ is the red path below
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("", "X=1"),
            ("", "X=0"),
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
            ("X=0", "X*=1"): {"label": " f"}
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
def _():
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

    # Measurment model
    FALSE_POSITIVE_RATE = 0.05

    def _p_father_star(row):
        if row.is_father == 1:
            return row.p_father + (1 - row.p_father) * FALSE_POSITIVE_RATE
        return 1 - (1 - row.p_father) * (1 - FALSE_POSITIVE_RATE)

    PATERNITY.loc[:, 'p_father*'] = PATERNITY.apply(_p_father_star, axis=1)
    PATERNITY.loc[:, 'is_father*'] = stats.bernoulli.rvs(p=PATERNITY['p_father*'].values)

    print('Actual average paternity rate: ', PATERNITY['is_father'].mean().round(2))
    print('Measured average paternity rate: ', PATERNITY['is_father*'].mean().round(2))
    return DYAD_ID, FALSE_POSITIVE_RATE, N_DYADS, N_MOTHERS, PATERNITY


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We print out the actual, and observed paternity rates above. In principle, we should be able to recover these with our model
    """)
    return


@app.cell
def _(PATERNITY):
    PATERNITY
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the model with misclassification error

    #### Notes
    - we average over the unknown $X_i$, so we have no likelihood, and thus do not need the $X_i \sim \text{Bernoulli}(p_i)$ term in the model
    - But we still include an observation model, where we use two simultanous distributions, based on our observation:
      - if $X^*=1$ we use $p(X^*=1) = p + (1-p)f$
      - if $X^*=0$ we use $p(X^*=0) = (1-p)(1-f)$
      - To use these custom distributions, we leverage the `pm.Potential` function, which takes in the log probability of each observational distribution
        - this is analogous to the `custom` function used in lecture
      - We use the vanilla implementation below, one that doesn't use the `logsumexp`, etc. functions, as it compiles just fine
    """)
    return


@app.cell
def _(DYAD_ID, FALSE_POSITIVE_RATE, N_DYADS, N_MOTHERS, PATERNITY):
    MOM_ID = PATERNITY.mother_id
    MOM_TRAITS = PATERNITY.mother_trait.values

    with pm.Model() as paternity_measurement_model:

        _beta_M = pm.TruncatedNormal("beta_M", 0, 1, lower=0.01)
        _beta_T = pm.TruncatedNormal("beta_T", 0, 1, lower=0.01)

        _alpha = pm.Normal("alpha", 0, 1.5)
        _T = pm.Normal("T", 0, 1, shape=N_DYADS)    # Relationship strength
        _M = pm.Normal("M", 0, 1, shape=N_MOTHERS)  # Mother traits

        _p_father = pm.Deterministic('p_father', pm.math.invlogit(_alpha + _M[MOM_ID] * _beta_M + _T[DYAD_ID] * _beta_T))

        _custom_log_p_x1 = pm.math.log(_p_father + (1 - _p_father) * FALSE_POSITIVE_RATE)
        pm.Potential("X*=1", _custom_log_p_x1)

        _custom_log_p_x0 = pm.math.log((1 - _p_father) * (1 - FALSE_POSITIVE_RATE))
        pm.Potential("X*=0", _custom_log_p_x0)

        paternity_measurement_inference = cached_sample(target_accept=.95)
    return MOM_ID, paternity_measurement_inference


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit analogous model without accouting for misclassification
    """)
    return


@app.cell
def _(DYAD_ID, MOM_ID, N_DYADS, N_MOTHERS, PATERNITY):
    IS_FATHER_STAR = PATERNITY['is_father*'].values
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
def _(PATERNITY, paternity_inference, paternity_measurement_inference):
    _fig, _ = plt.subplots(figsize=(10, 4))
    utils.plot_dist(paternity_inference.posterior['p_father'], color='k')
    _p_father_mean = paternity_inference.posterior['p_father'].mean()
    plt.axvline(_p_father_mean, color='k', linestyle='--', label=f"Without Measurement Error: {_p_father_mean:1.2f}")

    utils.plot_dist(paternity_measurement_inference.posterior['p_father'])
    _p_father_mean = paternity_measurement_inference.posterior['p_father'].mean()
    plt.axvline(_p_father_mean, linestyle='--', label=f"With Measurement Error: {_p_father_mean:1.2f}")
    plt.xlim([0, 1])
    plt.legend();

    plt.legend();

    print('Actual average paternity rate: ', PATERNITY['is_father'].mean().round(2))
    print('Measured average paternity rate: ', PATERNITY['is_father*'].mean().round(2))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Including measurment error, we're able to accurately recover the _actual_ average paternity rate.
    - Without including the measurement error in the model, we recover the paternity rate observed in the presence of misclassification error.
    - At least for with this simulated dataset, including measurment error gives us a much tigher posterior on the estimated paternity rate, when compared to not modeling the misclassification error in the model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Measurement & Misclassification Horizons

    There are a number of problems and solutions related to modeling measurment error and misclassification

    - Item response theory (IRT)
      - e.g. NJ wine judging example earlier
    - Factor Analysis
    - Hurdle models
      - measurments need to cross a threshold before they are identifiable
    - Occupancy models
      - considering the existence of phenomena without detection
      - big in applied ecology: just because a species isn't detected doesn't mean it isn't there
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Floating Point Monsters

    - Working on log scale avoids (er, minimizes) issues associated with floating point arithmetic overflow (rounding to one) or underflow (rounding to zero)
    - "ancient weapons": special functions for working log scale:
      - `pm.logsumexp`: efficiently computes the log of the sum of exponentials of input elements.
      - `pm.math.log1mexp`: calculates $\log(1 - \exp(-x))$
      - `pm.log(1 - p)`

    ### Logs make sense

    $$
    \begin{align*}
    P(X^*=0) &= (1-p)(1-f) \\
    \log P(X^*=0) & = \log (1-p) + \log (1-f)  &\text{put on log scale (used by HMC)}
    \end{align*}
    $$

    ### Devil's in the detail

    Some terms can become numerically unstable

    - e.g. if $p \approx 0, \log(1-p) \approx \log(1) = 0$
    - e.g. if $p \approx 1, \log(1-p) \approx \log(0) = -\infty$
    """)
    return


@app.cell
def _():
    for _p in [1e-2, 1e-10, 1e-90]:
        print('p:', _p, ', log(1-p):', np.log(1-_p))
    return


@app.cell
def _():
    def _log1m(x):
        return np.log1p(-x)

    for _p in [1e-2, 1e-10, 1e-90]:
        print('p:', _p, ', log1p(1-p):', _log1m(_p))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `log1p` and Taylor series approximation for small $p$

    - when $p>e{-4}$, just calculate $\log(1 + p)$
    - when $p<e{-4}$, use Taylor expansion: $\log(1 + p) = p - \frac{p^2}{2} + \frac{p^3}{3} - ...$
    """)
    return


@app.cell
def _():
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
def _():
    mo.md(r"""
    ## `logsumexp`

    - Use when you need to take the sum of log of multiple terms.
    - Want the log because it will be numerically stable
    - But logs don't apply to sum, only products, e.g.

    $$
    \begin{align*}
    P(X^*=1) &=  p + (1-p)f \\
    \log P(X^*=1) &=  \log [p + (1-p)f] \\
    \log P(X^*=1) &= \text{pm.math.logsumexp}([\text{pm.math.log}(p), \text{pm.math.log}(1-p) + \text{pm.math.log}(f))]) \\
    \log P(X^*=1) &= \text{pm.math.logsumexp}([\text{pm.math.log}(p), \text{log1m}(p)) + \text{pm.math.log}(f))])
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Previous Paternity measurement model with log-scaling tricks
    """)
    return


@app.cell
def _(FALSE_POSITIVE_RATE):
    _p_father = 0.5
    # pm.math.logsumexp((pm.math.log(p_father), log1m(p_father) + )).eval()
    pm.math.log(FALSE_POSITIVE_RATE).astype('float32').eval()
    return


@app.function
def log_sum_exp(a, b):
    """
    Simple, numerically-stable logsumexp implementation for two inputs.

    Ironically, using `pm.math.logsumexp` for this implementation was
    providing numerically results.

    It looks like the PyMC5 implementation uses vanilla `exp`, `sum`,
    and `log` operators, which likely isn't numerically stable:

    https://www.pymc.io/projects/docs/en/stable/_modules/pytensor/tensor/math.html#logsumexp
    """
    c = pm.math.maximum(a, b)
    return c + pm.math.log(pm.math.exp(a - c) + pm.math.exp(b - c))


@app.function
def log1p(x):
    return pm.math.log(1 + x)


@app.function
def log1m(x):
    return log1p(-x)


@app.cell
def _(DYAD_ID, FALSE_POSITIVE_RATE, N_DYADS, N_MOTHERS, PATERNITY):
    _MOM_ID = PATERNITY.mother_id

    with pm.Model() as paternity_measurement_lse_model:

        _beta_M = pm.TruncatedNormal("beta_M", 0, 1, lower=0.01)
        _beta_T = pm.TruncatedNormal("beta_T", 0, 1, lower=0.01)

        _alpha = pm.Normal("alpha", 0, 1.5)
        _T = pm.Normal("T", 0, 1, shape=N_DYADS)    # Relationship strength
        _M = pm.Normal("M", 0, 1, shape=N_MOTHERS)  # Mother traits

        _p = pm.Deterministic('p', pm.math.invlogit(_alpha + _M[_MOM_ID] * _beta_M + _T[DYAD_ID] * _beta_T))
        _f = FALSE_POSITIVE_RATE

        # # For reference: the original custom logp with vanilla parameterization
        # custom_log_p_x1 = pm.math.log(p + (1 - p) * f)
        # custom_log_p_x0 = pm.math.log((1 - p) * (1 - f))

        # Custom logp using log-space reparameterization
        _custom_log_p_x1 = log_sum_exp(pm.math.log(_p), log1m(_p) + pm.math.log(_f))  # this line is problematic
        _custom_log_p_x0 = log1m(_p) + log1m(_f)

        pm.Potential("X*=1", _custom_log_p_x1)
        pm.Potential("X*=0", _custom_log_p_x0)

        paternity_measurement_lse_inference = cached_sample(target_accept=.95)
    return (paternity_measurement_lse_inference,)


@app.cell(hide_code=True)
def _(paternity_measurement_inference, paternity_measurement_lse_inference):
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
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
