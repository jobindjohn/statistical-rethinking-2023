import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    import xarray as xr
    import arviz as az
    import statsmodels.formula.api as smf
    from scipy import stats
    from matplotlib import pyplot as plt

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 05 - Elemental Confounds](https://youtu.be/mBEA7PKDmiYXsi=b4vr1l9rogryaiEB)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Causation & Association
    ## Does Wafflehouse cause divorceX
    """)
    return


@app.cell
def load_waffle_divorce():
    WAFFLEHOUSE_DIVORCE = utils.load_data("WaffleDivorce")
    WAFFLEHOUSE_DIVORCE.head()
    return (WAFFLEHOUSE_DIVORCE,)


@app.cell(hide_code=True)
def waffle_divorce_scatter(WAFFLEHOUSE_DIVORCE):
    _fig, _ax = plt.subplots()
    utils.plot_scatter(
        WAFFLEHOUSE_DIVORCE.WaffleHouses,
        WAFFLEHOUSE_DIVORCE.Divorce,
        zorder=60, alpha=1
    )
    for _ii, _row in WAFFLEHOUSE_DIVORCE.sort_values("Divorce", ascending=False).iloc[:25].iterrows():
        plt.annotate(_row.Loc, (_row.WaffleHouses+5, _row.Divorce), zorder=60)

    _ols_formula = "Divorce ~ WaffleHouses"
    _linear_divorce_ols = smf.ols(_ols_formula, data=WAFFLEHOUSE_DIVORCE).fit()
    utils.plot_line(
        WAFFLEHOUSE_DIVORCE.WaffleHouses,
        _linear_divorce_ols.predict(),
        color='C1', label=f"Linear Model\n{_ols_formula}"
    )

    plt.xlabel("Waffle Houses per million")
    plt.ylabel("divorce rate")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Waffle houses tend to be in the Southern US
    - There are also higher divorce rates in the Southern US
    - Southernness creates a strong statistical association between Waffle Houses and divorce rate.
    - It's not plausible that Waffle House causes divorce rate

    **Correlation is Common in Nature -- Causation is Rare**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Goal is to address some **Estimand** -- a scientific question
      - i.e. metaphorical Cake we'd like to create
    - We use an **Estimator**, which is a set of instructions to organize code and data to create an estimate of the estimand
      - metaphorical recipe
    - The resulting **Estimate** may or may not be the original Estimand you were aiming for. Many things can cause this
      - **bad recipe (e.g. estimator design or confounds -- the focus of this lecture)**
        - the right recipe will defend against confounds
      - bad execution of recipe (e.g. buggy code or interpretation)
      - bad data

    ### **Confounds**
    - features of the sample, and how we use it that can mislead us
    - confounds are diverse

    # The Periodic Table of Confounds
    - The Fork: $X \leftarrow Z \rightarrow Y$
    - The Pipe: $X \rightarrow Z \rightarrow Y$
    - The Collider: $X \rightarrow Z \leftarrow Y$
    - The Descendant: $X \rightarrow Z \rightarrow Y$, $Z \rightarrow A$
    """)
    return


@app.function
def summarize_discrete_confound_simulation(X, Y, Z, z_name="Z"):
    print(f'Correlation between X and Y: {np.corrcoef(X, Y)[0, 1]:1.2f}')
    print("Cross tabulation:")
    print(pd.crosstab(pd.Series(X, name="X"), pd.Series(Y, name="Y")))

    for z in [0, 1]:
        X_z = X[Z==z]
        Y_z = Y[Z==z]

        print(f'\nCorrelation between X and Y conditioned on {z_name}={z}: {np.corrcoef(X_z, Y_z)[0, 1]:1.2f}')
        print("Cross tabulation:")
        print(pd.crosstab(pd.Series(X_z, name="X"), pd.Series(Y_z, name="Y")))


@app.function
def fit_linear_models_to_simulated_data(data):
    models = {}
    models['unstratified'] = smf.ols("Y ~ X", data=data).fit()

    # Stratified Models
    for z in [0, 1]:
        models[f'Z={z}'] = smf.ols("Y ~ X", data=data[data.Z==z]).fit()
    return models


@app.function
def plot_sms_linear_model_fit(model, xs, label, color):
    """Helper function to plot linear models"""
    params = model.params
    ys = params.Intercept + params.X * xs
    return utils.plot_line(xs, ys, label=label, color=color)


@app.function
def plot_continuous_confound_simulation(data, title):
    """Helper function to plot simulations"""
    models = fit_linear_models_to_simulated_data(data)
    plt.subplots(figsize=(6, 6))

    xs = np.linspace(-4, 4, 20)
    for z in [0, 1]:
        color = f"C{np.abs(z - 1)}"  # Note: flip colormap
        utils.plot_scatter(data[data.Z==z].X, data[data.Z==z].Y, color=color)
        plot_sms_linear_model_fit(models[f'Z={z}'], xs, label=f'Z={z}', color=color)

    plot_sms_linear_model_fit(models['unstratified'], xs, label='total sample', color='black')

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.legend()
    plt.title(title)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Fork $X \leftarrow Z \rightarrow Y$
    - $Χ$ and $Y$ share "common cause" $Z$
    - common cause $Z$ induces an association between $Χ$ and $Y$
      - $X \not \perp Y$
    - once stratified by $Z$, no association between $Χ$ and $Y$ **for each level of $Z$**
      -  $X \perp Y | Z$
    """)
    return


@app.cell(hide_code=True)
def fork_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("Z", "X"), ("Z", "Y")
        ],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note. If $Z$ was the _only_ common cause, $X$ and $Y$ would be clones of one another. There are other non-modeled/unobserved influences of $X$ and $Y$--often referred to as error terms $e_{X,Y}$ that are not included in the graph above, but nonetheless are causing the difference between $X$ and $Y$, and should be modeled, often as noise in any generative model.
    """)
    return


@app.cell(hide_code=True)
def fork_dag_with_noise():
    utils.draw_causal_graph(
        edge_list=[
            ("Z", "X"), ("Z", "Y"), ("e_X", "X"), ("e_Y", "Y")
        ],
        node_props={
            "e_X": {"style": "dashed"},
            "e_Y": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fork Generative Process: Discrete Example
    Below we simulate a Fork generative process:

    $$
    \begin{align*}
    Z &\sim \text{Bernoulli}(0.5) \\
    X &\sim \text{Bernoulli}(p^*) \\
    Y &\sim \text{Bernoulli}(p^*) \\
    p^* &= 0.9 \times Z + 0.1 \times (1 - Z) \\
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def discrete_fork_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("Z", "X"),
            ("Z", "Y")
        ],
        edge_props={
            ("Z", "X"): {"label": "p*"},
            ("Z", "Y"): {"label": "p*"}
        },
        graph_direction="LR"
    )
    return


@app.cell
def simulate_discrete_fork():
    # Discrete Fork
    np.random.seed(321)
    _n_samples = 1000
    _Z = stats.bernoulli.rvs(p=.5, size=_n_samples)
    _p_star = _Z * .9 + (1 - _Z) * .1
    _X = stats.bernoulli.rvs(p=_p_star)
    _Y = stats.bernoulli.rvs(p=_p_star)
    discrete_fork = dict(X=_X, Y=_Y, Z=_Z)
    return (discrete_fork,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Show that:
    - $X \not\!\perp Y$
    - $X \perp Y | Z$
    """)
    return


@app.cell
def discrete_fork_summary(discrete_fork):
    summarize_discrete_confound_simulation(**discrete_fork)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fork Generative Process: Continuous Example

    $$
    \begin{align*}
    Z &\sim \text{Bernoulli}(0.5) \\
    X &\sim \text{Normal}(\mu^*, 1) \\
    Y &\sim \text{Normal}(\mu^*, 1) \\
    \mu^* &= 2Z -1 \\
    \end{align*}
    $$
    """)
    return


@app.cell
def simulate_continuous_fork():
    # Continous Fork
    np.random.seed(1234)
    _n_samples = 400

    _Z = stats.bernoulli.rvs(p=.5, size=_n_samples)

    _mu_star = 2 * _Z - 1
    _X = stats.norm.rvs(loc=_mu_star, size=_n_samples)
    _Y = stats.norm.rvs(loc=_mu_star, size=_n_samples)

    # Put simulated data into dataframe for statsmodels API
    fork_data = pd.DataFrame(np.vstack([_Z, _X, _Y]).T, columns=['Z', 'X', 'Y'])
    return (fork_data,)


@app.cell(hide_code=True)
def continuous_fork_plot(fork_data):
    plot_continuous_confound_simulation(fork_data, "The Fork\n$X \\leftarrow Z \\rightarrow Y$")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fork Example: Marriage Rates & Divorce Rates (& Waffle House!)
    🧇 As someone who grew up in the South, I love Wafflehouse, and by association, love this example.
    """)
    return


@app.cell
def wafflehouse_pairwise_plotter(WAFFLEHOUSE_DIVORCE):
    def plot_wafflehouse_pairwise(x_var, y_var, n_label=10, wafflehouse_divorce=WAFFLEHOUSE_DIVORCE):
        """Helper function for visualizing Wafflehouse data
        """
        # Scatter by South
        for south, label in enumerate([None, 'Southern\nStates']):
            xs = wafflehouse_divorce[wafflehouse_divorce['South'] == south][x_var].values
            ys = wafflehouse_divorce[wafflehouse_divorce['South'] == south][y_var].values

            utils.plot_scatter(
                xs,
                ys,
                label=label,
                color=f'C{np.abs(south-1)}'  # Note: flip binary colorscale
            )

        # Annotate extreme values
        loc = wafflehouse_divorce['Loc'].tolist()

        xs = wafflehouse_divorce[x_var].values
        ys = wafflehouse_divorce[y_var].values
        xs_mean = np.mean(xs)
        ys_mean = np.mean(ys)

        def distance_from_center_of_mass(x, y):
            return (x - xs_mean) ** 2 + (y - ys_mean) ** 2

        sorted_xyl = sorted(
            zip(xs, ys, loc),
            key=lambda z: distance_from_center_of_mass(z[0], z[1])
        )[::-1]

        for li in range(n_label):
            x, y, label = sorted_xyl[li]
            plt.annotate(label, (x + .1, y + .1))

        def camelcase_to_label(string):
            return re.sub(r'(X<!^)(X=[A-Z])', ' ', string).lower()

        plt.xlabel(camelcase_to_label(x_var))
        plt.ylabel(camelcase_to_label(y_var))

        plt.legend()

    return (plot_wafflehouse_pairwise,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Marrying the Owl
    1. Estimand
    2. Scientific Model
    3. Statistical Model
    4. Analysis

    ### (1) Estimand
    Causal effect of **Marriage** Rate, M on **Divorce** Rate, D
    """)
    return


@app.cell(hide_code=True)
def marriage_divorce_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("M", "D")
        ],
        node_props={
            "M": {"label": "Marriage Rate, M"},
            "D": {"label": "Divorce Rate, D"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def marriage_divorce_scatter(plot_wafflehouse_pairwise):
    _fig, _ax = plt.subplots()
    plot_wafflehouse_pairwise('Marriage', 'Divorce', n_label=15)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ...another potential cause is **Age at Marriage**
    """)
    return


@app.cell(hide_code=True)
def age_divorce_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "D")
        ],
        node_props={
            "A": {"label": "Median Marriage Age, A"},
            "D": {"label": "Divorce Rate, D"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def age_divorce_scatter(plot_wafflehouse_pairwise):
    _fig, _ax = plt.subplots()
    plot_wafflehouse_pairwise('MedianAgeMarriage', 'Divorce', n_label=15)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### (2) Scientific Model
    """)
    return


@app.cell(hide_code=True)
def age_marriage_divorce_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("M", "D"),
            ("A", "M"),
            ("A", "D")
        ],
        edge_props={
            ("A", "D"): {"label": " X"},
            ("M", "D"): {"label": " X"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def waffle_pairwise_scatters(plot_wafflehouse_pairwise):
    _fig, _axs = plt.subplots(1, 3, figsize=(12, 4))

    plt.sca(_axs[0])
    plot_wafflehouse_pairwise('Marriage', 'Divorce', n_label=15)

    plt.sca(_axs[1])
    plot_wafflehouse_pairwise('MedianAgeMarriage', 'Divorce', n_label=15)

    plt.sca(_axs[2])
    plot_wafflehouse_pairwise('MedianAgeMarriage', 'Marriage', n_label=15)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - **Is the effect of Marriage rate on Divorce rate solely the result of their common cause Age--i.e. the ForkX**
      - Can't tell from just looking at the scatter plots above.
    - We need to **break the Fork by stratifying by Age**

    #### Testing Scientific Model with Generative Process Model
    > Notes:
    > - McElreath mentions that we should always do this testing step in the standard analysis pipeline, but skips for sake of time, so we do go ahead and take a stab at it here.
    > - this simulation models the Marriage Process as a function of Age, and Divorce as a function of Age _and_ Marriage
    > - this simulation is in the space of standardized predictor variables
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Stratifying by a continuous variable

    $$
    \begin{align*}
    D &= f(A, M) \\
    D_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta_M M_i + \beta_A A_i \\
    \mu_i &= (\alpha + \beta_A A_i) + \beta_M M_i
    \end{align*}
    $$

    where $(\alpha + \beta_A A_i)$  can be thought of as an effective intercept for each continuous value of $A$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Generating data according to the statistical model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Statistical Fork

    $$
    \begin{align*}
    D &= f(A, M) \\
    D_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta_M M_i + \beta_A A_i \\
    \alpha &\sim \text{Normal}(\mu_X, \sigma_X) \\
    \beta_{A, M} &\sim \text{Normal}(\mu_X, \sigma_X) \\
    \sigma &\sim \text{Exponential}(\lambda_X)
    \end{align*}
    $$

    What parameters $\theta_X$  do we use for the priorsX **Enter Prior Predictive Simulation**

    #### 💡 When working with Linear Regression, it's always always a good idea to **standardize**
    - $X_{standardized} = \frac{(X - \bar X)}{\bar \sigma}$
    - $\bar X$ and $\bar \sigma$ are the sample mean and standard deviation
    - results in a mean of 0 and standard deviation of 1
    - simplifies prior definition
    - makes inference run smoother
    - can always reverse the standardization transformation (using $\bar X$ and $\bar \sigma$) to get back to the original space
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior Predictive Simulation
    Gives us better intuition of the types of lines the model can produce given a set of prior parameters
    """)
    return


@app.function
def simulate_linear_model_priors(std=10, n_simulations=200):

    xs = np.linspace(-3, 3, 10)
    for ii in range(n_simulations):
        alpha = stats.norm.rvs(0, std)
        beta = stats.norm.rvs(0, std)
        ys = alpha + beta * xs
        plt.plot(xs, ys, color='C0', alpha=.25)

    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel("median age at marriage (standardized)")
    plt.ylabel("Divorce rate (standardized)")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Too large of a prior variance results in unrealistic linear models
    """)
    return


@app.cell
def wide_prior_lines():
    _fig, _ax = plt.subplots()
    simulate_linear_model_priors(std=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### More reasonable priors for standardized variables
    """)
    return


@app.cell
def narrow_prior_lines():
    _fig, _ax = plt.subplots()
    simulate_linear_model_priors(std=.5)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore the prior
    Change the prior standard deviation of the intercept $\alpha$ and slope $\beta$. For standardized variables, values around $0.5$ keep most lines within the plausible range; much larger values produce absurdly steep lines.
    """)
    return


@app.cell(hide_code=True)
def prior_predictive_controls():
    prior_std = mo.ui.slider(steps=[0.1, 0.2, 0.5, 1, 2, 5, 10], value=0.5, label=r"Prior std of $\alpha$ and $\beta$", show_value=True)
    prior_n_lines = mo.ui.slider(10, 200, step=10, value=100, label="Lines", show_value=True)
    return prior_n_lines, prior_std


@app.cell(hide_code=True)
def prior_predictive_explorer(prior_n_lines, prior_std):
    np.random.seed(1)
    _fig, _ax = plt.subplots()
    simulate_linear_model_priors(std=prior_std.value, n_simulations=prior_n_lines.value)
    plt.title(f"$\\alpha, \\beta \\sim Normal(0, {prior_std.value})$")
    mo.vstack([
        mo.hstack([prior_std, prior_n_lines], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### (3) Statistical Model for Causal Effect of Marriage on Divorce

    Here's the full statistical model with more reasonable priors.

    $$
    \begin{align*}
    D &= f(A, M) \\
    D_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta_M M_i + \beta_A A_i \\
    \alpha &\sim \text{Normal}(0, 0.2) \\
    \beta_{A, M} &\sim \text{Normal}(0, 0.5) \\
    \sigma &\sim \text{Exponential}(1)
    \end{align*}
    $$
    where we've standardized all the input variables, so that we can use tight, zero-centered Normal priors for the intercept and slopes

    #### Run the statistical model on the simulated data
    Verify that we can recover the parameters from a generative process that matches the statistical model
    """)
    return


@app.cell
def divorce_simulator(WAFFLEHOUSE_DIVORCE):
    def simulate_divorce(
        beta_marriage_divorce=1,
        beta_age_divorce=-1,
        beta_age_marriage=0,
        alpha=0.0,
        sigma=0.5,
        random_seed=123
    ):
        N_STATES = 50
        np.random.seed(random_seed)

        # Age (standardized)
        age = stats.norm.rvs(size=N_STATES)

        # Marriage rate
        marriage_rate = age * beta_age_marriage + stats.norm(0, sigma).rvs(size=N_STATES)

        # Divorce Rate
        mu = alpha + marriage_rate * beta_marriage_divorce + age * beta_age_divorce
        divorce_rate = stats.norm.rvs(mu, sigma)

        simulated_divorce = pd.DataFrame(
            {
                "MedianAgeMarriage": age,
                "Marriage": marriage_rate,
                "Divorce": divorce_rate
            }
        )

        # Shuffle actual labels to retain same proportion of Southern/Other states
        # i.e. Southern state has no effect on parameters in this simulation.
        # `.to_numpy()` stops pandas from realigning the shuffled rows on the index.
        simulated_location = WAFFLEHOUSE_DIVORCE[['Loc', "South"]].sample(frac=1.0)
        simulated_divorce["Loc"] = simulated_location["Loc"].to_numpy()
        simulated_divorce["South"] = simulated_location["South"].to_numpy()
        return simulated_divorce

    return (simulate_divorce,)


@app.cell
def simulate_marriage_divorce(simulate_divorce):
    MARRIAGE_DIVORCE_SIMULATION_PARAMS = dict(
        beta_marriage_divorce=1,
        beta_age_divorce=-1,
        beta_age_marriage=1,
        alpha=0,
        sigma=0.5
    )
    SIMULATED_MARRIAGE_DIVORCE = simulate_divorce(**MARRIAGE_DIVORCE_SIMULATION_PARAMS, random_seed=1)
    return MARRIAGE_DIVORCE_SIMULATION_PARAMS, SIMULATED_MARRIAGE_DIVORCE


@app.cell(hide_code=True)
def simulated_marriage_divorce_scatters(
    SIMULATED_MARRIAGE_DIVORCE,
    plot_wafflehouse_pairwise,
):
    # Plot simulation
    _fig, _axs = plt.subplots(1, 3, figsize=(12, 4))

    plt.sca(_axs[0])
    plot_wafflehouse_pairwise('Marriage', 'Divorce', n_label=15, wafflehouse_divorce=SIMULATED_MARRIAGE_DIVORCE)

    plt.sca(_axs[1])
    plot_wafflehouse_pairwise('MedianAgeMarriage', 'Divorce', n_label=15, wafflehouse_divorce=SIMULATED_MARRIAGE_DIVORCE)

    plt.sca(_axs[2])
    plot_wafflehouse_pairwise('MedianAgeMarriage', 'Marriage', n_label=15, wafflehouse_divorce=SIMULATED_MARRIAGE_DIVORCE)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Breaking the fork
    Simulate 50 states with your own causal effects, then estimate the effect of $M$ on $D$ in two ways: from $M$ alone ($D \sim M$), and stratified by age ($D \sim M + A$). When $\beta_{AM}$ and $\beta_{AD}$ are both nonzero, $A$ is a fork, and only the stratified estimate recovers the true $\beta_{MD}$. Try $\beta_{MD} = 0$: on its own, $M$ still seems to affect $D$.

    These fits use least squares for speed; with 50 states, the Bayesian model gives nearly the same estimates.
    """)
    return


@app.cell(hide_code=True)
def breaking_fork_controls():
    fork_beta_am = mo.ui.slider(-2, 2, step=0.25, value=1, label=r"$\beta_{AM}$ (A ? M)", show_value=True)
    fork_beta_ad = mo.ui.slider(-2, 2, step=0.25, value=-1, label=r"$\beta_{AD}$ (A ? D)", show_value=True)
    fork_beta_md = mo.ui.slider(-2, 2, step=0.25, value=0, label=r"$\beta_{MD}$ (M ? D)", show_value=True)
    fork_resample = mo.ui.button(value=0, on_click=lambda count: count + 1, label="New states")
    return fork_beta_ad, fork_beta_am, fork_beta_md, fork_resample


@app.cell(hide_code=True)
def breaking_fork_explorer(
    fork_beta_ad,
    fork_beta_am,
    fork_beta_md,
    fork_resample,
    simulate_divorce,
):
    _sim = simulate_divorce(
        beta_marriage_divorce=fork_beta_md.value,
        beta_age_divorce=fork_beta_ad.value,
        beta_age_marriage=fork_beta_am.value,
        random_seed=1 + fork_resample.value,
    )
    _fits = {
        "D ~ M": smf.ols("Divorce ~ Marriage", data=_sim).fit(),
        "D ~ M + A": smf.ols("Divorce ~ Marriage + MedianAgeMarriage", data=_sim).fit(),
    }

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4))
    plt.sca(_axs[0])
    utils.plot_scatter(_sim.Marriage, _sim.Divorce)
    _xs = np.linspace(_sim.Marriage.min(), _sim.Marriage.max(), 10)
    _params = _fits["D ~ M"].params
    utils.plot_line(_xs, _params.Intercept + _params.Marriage * _xs, color="k", label="D ~ M")
    plt.xlabel("marriage rate (simulated)")
    plt.ylabel("divorce rate (simulated)")
    plt.legend()

    plt.sca(_axs[1])
    for _y, (_label, _fit) in enumerate(_fits.items()):
        _lower, _upper = _fit.conf_int(alpha=0.11).loc["Marriage"]
        plt.hlines(_y, _lower, _upper, color=f"C{_y}", linewidth=3)
        plt.scatter(_fit.params.Marriage, _y, color=f"C{_y}", s=80, zorder=3)
    plt.axvline(fork_beta_md.value, color="k", linestyle="--", label=r"true $\beta_{MD}$")
    plt.yticks([0, 1], list(_fits))
    plt.ylim(-0.5, 1.5)
    plt.xlabel("estimated effect of M on D (89% interval)")
    plt.legend()
    mo.vstack([
        mo.hstack([fork_beta_am, fork_beta_ad, fork_beta_md, fork_resample], justify="start", wrap=True),
        _fig,
    ])
    return


@app.cell
def marriage_divorce_model_params():
    MARRIAGE_DIVORCE_MODEL_PARAMS = ["alpha", "beta_marriage", "beta_age",  "sigma"]
    return (MARRIAGE_DIVORCE_MODEL_PARAMS,)


@app.function
def fit_marriage_divorce_model(data):
    with pm.Model() as divorce_model:

        # Observed Data
        age = pm.Data(
            "age", data["MedianAgeMarriage"].values, dims="obs_ids"
        )
        marriage_rate = pm.Data(
            "marriage_rate", data["Marriage"].values, dims="obs_ids"
        )
        divorce_rate = data["Divorce"].values

        sigma = pm.Exponential("sigma", 1)
        alpha = pm.Normal("alpha", 0, 0.2)
        beta_age = pm.Normal("beta_age", 0, 0.5)
        beta_marriage = pm.Normal("beta_marriage", 0, 0.5)

        ## Divorce process
        mu = alpha + beta_marriage * marriage_rate + beta_age * age
        pm.Normal(
            "divorce_rate",
            mu,
            sigma,
            observed=divorce_rate,
            dims="obs_ids"
        )
        divorce_inference = cached_sample(draws=10000, target_accept=.95)
    return divorce_model, divorce_inference


@app.function
def plot_divorce_model_posterior(
    inference,
    model_params,
    actual_params=None,
    ax=None
):
    """Forest plot of the posterior. `actual_params` maps each model
    parameter to the value used to simulate the data."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2))

    plt.sca(ax)
    utils.plot_forest(
        inference,
        var_names=model_params,
        ax=ax
    )
    title = "Posterior"
    if actual_params is not None:
        # utils.plot_forest draws the first parameter on the top row
        ys = np.arange(len(model_params))[::-1]
        for y, param in zip(ys, model_params):
            plt.scatter(actual_params[param], y, marker="x", s=100, color="k", zorder=4)
        title += " and\nActual Parameter Values (x)"
    plt.title(title)


@app.cell
def simulated_marriage_divorce_fit(SIMULATED_MARRIAGE_DIVORCE):
    simulated_marriage_divorce_model, simulated_marriage_divorce_inference = fit_marriage_divorce_model(SIMULATED_MARRIAGE_DIVORCE)
    return (simulated_marriage_divorce_inference,)


@app.cell
def simulated_marriage_divorce_summary(
    MARRIAGE_DIVORCE_MODEL_PARAMS,
    simulated_marriage_divorce_inference,
):
    az.summary(
        simulated_marriage_divorce_inference,
        var_names=MARRIAGE_DIVORCE_MODEL_PARAMS
    )
    return


@app.cell(hide_code=True)
def simulated_marriage_divorce_posterior(
    MARRIAGE_DIVORCE_MODEL_PARAMS,
    MARRIAGE_DIVORCE_SIMULATION_PARAMS,
    simulated_marriage_divorce_inference,
):
    plot_divorce_model_posterior(
        simulated_marriage_divorce_inference,
        model_params=MARRIAGE_DIVORCE_MODEL_PARAMS,
        actual_params={
            "alpha": MARRIAGE_DIVORCE_SIMULATION_PARAMS["alpha"],
            "beta_marriage": MARRIAGE_DIVORCE_SIMULATION_PARAMS["beta_marriage_divorce"],
            "beta_age": MARRIAGE_DIVORCE_SIMULATION_PARAMS["beta_age_divorce"],
            "sigma": MARRIAGE_DIVORCE_SIMULATION_PARAMS["sigma"],
        }
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Cool. We can recover our params with this model. Let's try it on real data.

    > Note: the true values sit near the edges of the 89% intervals. With $\beta_{AM} = 1$, $M$ and $A$ are strongly correlated, and the tight $\text{Normal}(0, 0.5)$ priors pull both slopes toward 0. Set `beta_age_marriage=0.5` in the simulation above for a closer recovery.
    """)
    return


@app.cell
def standardize_waffle_divorce(WAFFLEHOUSE_DIVORCE):
    # Generate a standardized version of the real data
    WAFFLEHOUSE_DIVORCE_STD = WAFFLEHOUSE_DIVORCE.copy()

    # standardize variables
    for _var in ["MedianAgeMarriage", "Marriage", "Divorce"]:
        WAFFLEHOUSE_DIVORCE_STD.loc[:, _var] = utils.standardize(WAFFLEHOUSE_DIVORCE_STD[_var])
    return (WAFFLEHOUSE_DIVORCE_STD,)


@app.cell
def marriage_divorce_fit(WAFFLEHOUSE_DIVORCE_STD):
    marriage_divorce_model, marriage_divorce_inference = fit_marriage_divorce_model(WAFFLEHOUSE_DIVORCE_STD)
    return marriage_divorce_inference, marriage_divorce_model


@app.cell(hide_code=True)
def marriage_divorce_posterior(
    MARRIAGE_DIVORCE_MODEL_PARAMS,
    marriage_divorce_inference,
):
    plot_divorce_model_posterior(marriage_divorce_inference, model_params=MARRIAGE_DIVORCE_MODEL_PARAMS)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulating Interventions

    #### Causal Effect of Marriage Rate on Divorce Rate
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### No intervention
    """)
    return


@app.cell(hide_code=True)
def no_intervention_dag():
    # Without Interventions
    utils.draw_causal_graph(
        edge_list=[
            ("M", "D"),
            ("A", "M"),
            ("A", "D")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $do(M)$
    - We "play God" by setting the value of $M$
    - Deletes the arrow entering $M$
      - thus breaks the association between $A$ and $M$
    """)
    return


@app.cell(hide_code=True)
def do_marriage_dag():
    # With Interventions do(M)
    utils.draw_causal_graph(
        edge_list=[
            ("M", "D"),
            ("A", "D")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we simulate two worlds where $M$ takes on very different values (e.g. 0 and +1 std) to identify the possible range of causal effects of Marriage on Divorce. In principle, we could do this for any number of values along the continuum of $M$. The slider below sets the size of the intervention.
    """)
    return


@app.cell(hide_code=True)
def intervention_controls():
    intervention_delta_m = mo.ui.slider(0.25, 3, step=0.25, value=1, label="Increase in $M$ (std)", show_value=True)
    return (intervention_delta_m,)


@app.cell
def intervention_explorer(intervention_delta_m, marriage_divorce_inference):
    np.random.seed(123)
    _n_simulations = int(1e3)
    _posterior = marriage_divorce_inference.posterior.sel(chain=0)

    # Sample Ages
    _age_samples = stats.norm.rvs(0, 1, size=_n_simulations)

    # Sample parameters from posterior
    _alpha_samples = np.random.choice(_posterior.alpha, size=_n_simulations, replace=True)
    _sigma_samples = np.random.choice(_posterior.sigma, size=_n_simulations, replace=True)
    _beta_age_samples = np.random.choice(_posterior.beta_age, size=_n_simulations, replace=True)
    _beta_marriage_samples = np.random.choice(_posterior.beta_marriage, size=_n_simulations, replace=True)

    _marriage_contrast = {}
    _n_std_increase = intervention_delta_m.value
    for _do_marriage in [0,  _n_std_increase]:  # Playing God, setting the value of Marriage Rate
        _mu = _alpha_samples + _beta_age_samples * _age_samples + _beta_marriage_samples * _do_marriage
        _marriage_contrast[_do_marriage] = stats.norm.rvs(_mu, _sigma_samples)

    _marriage_contrast = _marriage_contrast[_n_std_increase] - _marriage_contrast[0]
    _prob_lt_zero = (_marriage_contrast < 0).mean()
    _fig, _ax = plt.subplots()
    utils.plot_dist(_marriage_contrast)
    plt.axvline(0, color='k', linestyle='--')
    plt.title(
        "Causal effect of increasing Marriage Rate\n"
        f"by {_n_std_increase} standard deviations on Divorce Rate\n"
        f"$p(\\delta < 0) = {_prob_lt_zero:1.2}$"
    )
    mo.vstack([intervention_delta_m, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    But that's just one slice of the Marriage range. We can use pymc to look at the causal effect of shifting Marriage over a range of values.
    """)
    return


@app.cell
def marriage_counterfactual_ppd(
    marriage_divorce_inference,
    marriage_divorce_model,
):
    marriage_counterfactuals = xr.DataArray(np.linspace(-3, 3, 100))
    _age_counterfactuals = xr.DataArray(np.zeros_like(marriage_counterfactuals))  # set all to average age
    with marriage_divorce_model:
        pm.set_data({"marriage_rate": marriage_counterfactuals, "age": _age_counterfactuals})
        ppd_marriage_divorce = pm.sample_posterior_predictive(marriage_divorce_inference, extend_inferencedata=False)
    return marriage_counterfactuals, ppd_marriage_divorce


@app.cell(hide_code=True)
def marriage_counterfactual_plot(
    marriage_counterfactuals,
    ppd_marriage_divorce,
):
    _fig, _ax = plt.subplots()
    utils.plot_hdi(
        marriage_counterfactuals,
        ppd_marriage_divorce.posterior_predictive["divorce_rate"],
        color="C0", fill_kwargs={"alpha": .15}
    )
    plt.plot(
        marriage_counterfactuals,
        ppd_marriage_divorce.posterior_predictive["divorce_rate"].mean(dim=("chain", "draw")),
        color="C0"
    )
    plt.xlabel("marriage counterfactual (in standard deviations)")
    plt.ylabel("divorce rate (in std deviations)")
    plt.title("Causal effect of Marriage Rate on\nDivorce Rate in standard deviations")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Causal Effect of Age on Divorce RateX
    - how do we do implement $do(A)$X
    - There are no causal arrows that we need to remove for an intervention
    - **This requires a new model that ignores $M$**
      - we can then simulate any intervention on $A$
    - Why can we ignore $M$X
      - because $A \rightarrow M \rightarrow D$ is **a Pipe**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Pipe $X \rightarrow Z \rightarrow Y$

    - $Z$ is a "mediator" of $Χ$ and $Y$ ($Z$ does not affect $X$)
    - The influence of $Χ$ on $Y$ is transmitted through $Z$, thus $X$ and $Y$ are associated
      - $X \not \perp Y$
    - once stratified by $Z$, no association between $Χ$ and $Y$ **for each level of $Z$**
      -  $X \perp Y | Z$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - **Statistically very similar to the Fork**
    - **Causally very different from the Fork**
    """)
    return


@app.cell(hide_code=True)
def pipe_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "Y")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discrete Example

    Below we simulate a "Pipe" generative process:

    $$
    \begin{align*}
    X &\sim \text{Bernoulli}(0.5) \\
    Z &\sim \text{Bernoulli}(p*) \\
    Y &\sim \text{Bernoulli}(q*) \\
    p^* &= 0.9 X + 0.1 (1 - X) \\
    q^* &= 0.9 Z + 0.1 (1 - Z) \\
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def discrete_pipe_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "Y")
        ],
        edge_props={
            ("X", "Z"): {"label": "p*"},
            ("Z", "Y"): {"label": "q*"}
        },
        graph_direction="LR"
    )
    return


@app.cell
def simulate_discrete_pipe():
    np.random.seed(1234)
    _n_samples = 1000
    _X = stats.bernoulli.rvs(p=.5, size=_n_samples)

    _p_star = _X * .9 + (1 - _X) * .1
    _Z = stats.bernoulli.rvs(p=_p_star, size=_n_samples)

    _q_star = _Z * .9 + (1 - _Z) * .1
    _Y = stats.bernoulli.rvs(p=_q_star, size=_n_samples)
    discrete_pipe = dict(X=_X, Y=_Y, Z=_Z)
    return (discrete_pipe,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Demonstrate that:
    - $X \not \perp Y$
    - $X \perp Y | Z$
    """)
    return


@app.cell
def discrete_pipe_summary(discrete_pipe):
    summarize_discrete_confound_simulation(**discrete_pipe)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Why does this happenX
     - everything $X$ knows about $Y$, $Z$ knows
     - once you learn about $Z$, there is nothing else to learn about the association between $X$ and $Y$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Continuous Example

    $$
    \begin{align*}
    X &\sim \text{Normal}(0, 1) \\
    Z &\sim \text{Bernoulli}(p^*) \\
    Y &\sim \text{Normal}(\mu^*, 1) \\
    p^* &= \frac{1}{1 + e^{-X}} \\
    \mu^* &= 2Z - 1 \\
    \end{align*}
    $$
    """)
    return


@app.cell
def simulate_continuous_pipe():
    np.random.seed(1234)
    _n_samples = 1000

    _X = stats.norm.rvs(size=_n_samples)
    _p_star = utils.invlogit(_X)
    _Z = stats.bernoulli.rvs(p=_p_star, size=_n_samples)

    _mu_star = 2 * _Z - 1
    _Y = stats.norm.rvs(loc=_mu_star, size=_n_samples)

    pipe_data = pd.DataFrame(np.vstack([_Z, _X, _Y]).T, columns=['Z', 'X', 'Y'])
    return (pipe_data,)


@app.cell(hide_code=True)
def continuous_pipe_plot(pipe_data):
    plot_continuous_confound_simulation(pipe_data, "The Pipe\n$X \\rightarrow  Z \\rightarrow Y$")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pipe Example: Plant Growth
    """)
    return


@app.cell(hide_code=True)
def plant_growth_dag():
    utils.draw_causal_graph(
        [
            ("H0", "H1"),
            ("T", "H1"),
            ("T", "F"),
            ("F", "H1")
        ],
        node_props={
            "T": {"color": "red", "label": "T, treatment"},
            "F": {"label": "F, fungus"},
            "H0": {"label": "H0, height at t0"},
            "H1": {"label": "H1, height at t1"}
        },
        edge_props={
            ("T", "H1"): {"color": "red"},
            ("T", "F"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### To estimate the **Total Causal Effect** of Treatment, should we Stratify by FX

    **NO**
    - Stratifying by F would be a bad idea here if we wanted to estimate the total causal effect of T on H1, because of the $T \rightarrow F \rightarrow H1$ pipe in the graph.
    - Stratifying by F would block information about T's effect on H1 that flow through F.
    - NOTE: stratifying by $F$ would give the **Direct Causal Effect** of $T$ on $H1$

    This is an example of **post-treatment bias**.
    > Rule of Thumb: **consequence of the treatment should not be included in an estimator.**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Statistical Model for Causal Effect of Age on Divorce Rate
    McElreath mentions how we could/should build and test this model, but doesn't do this in the lecture, so we do it here!

    The Age-Divorce model is similar to the Marriage-Divorce model, but we no longer need to stratify by Marriage.

    $$
    \begin{align*}
    D &= f(A) \\
    D_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta_A A_i \\
    \alpha &\sim \text{Normal}(0, 0.2) \\
    \beta_A &\sim \text{Normal}(0, 0.5) \\
    \sigma &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell
def age_divorce_model_params():
    AGE_DIVORCE_MODEL_PARAMS = ["sigma", "alpha", "beta_age"]
    return (AGE_DIVORCE_MODEL_PARAMS,)


@app.function
def fit_age_divorce_model(data):
    with pm.Model() as divorce_model:

        # Observed Data
        age = pm.Data(
            "age", data["MedianAgeMarriage"].values, dims="obs_ids"
        )

        alpha = pm.Normal("alpha", 0, 0.2)
        beta_age = pm.Normal("beta_age", 0, 0.5)
        sigma = pm.Exponential("sigma", 1)

        ## Divorce process
        mu = alpha + beta_age * age
        pm.Normal(
            "divorce_rate",
            mu,
            sigma,
            observed=data["Divorce"].values,
            dims="obs_ids"
        )
        divorce_inference = cached_sample(draws=10000, target_accept=.95)
    return divorce_model, divorce_inference


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Test Age-Divorce model on simulated data
    """)
    return


@app.cell
def simulate_age_divorce(simulate_divorce):
    AGE_DIVORCE_SIMULATION_PARAMS = dict(
        beta_age_divorce=-1.,
        sigma=.5,
        alpha=0.,
        beta_marriage_divorce=0,
        beta_age_marriage=0
    )
    SIMULATED_AGE_DIVORCE = simulate_divorce(**AGE_DIVORCE_SIMULATION_PARAMS)
    return AGE_DIVORCE_SIMULATION_PARAMS, SIMULATED_AGE_DIVORCE


@app.cell(hide_code=True)
def simulated_age_divorce_scatter(
    SIMULATED_AGE_DIVORCE,
    plot_wafflehouse_pairwise,
):
    # Plot simulation
    _fig, _ax = plt.subplots(figsize=(4, 4))
    plot_wafflehouse_pairwise('MedianAgeMarriage', 'Divorce', n_label=15, wafflehouse_divorce=SIMULATED_AGE_DIVORCE)
    _fig
    return


@app.cell
def simulated_age_divorce_fit(SIMULATED_AGE_DIVORCE):
    simulated_age_divorce_model, simulated_age_divorce_inference = fit_age_divorce_model(SIMULATED_AGE_DIVORCE)
    return (simulated_age_divorce_inference,)


@app.cell(hide_code=True)
def simulated_age_divorce_posterior(
    AGE_DIVORCE_MODEL_PARAMS,
    AGE_DIVORCE_SIMULATION_PARAMS,
    simulated_age_divorce_inference,
):
    plot_divorce_model_posterior(
        simulated_age_divorce_inference,
        model_params=AGE_DIVORCE_MODEL_PARAMS,
        actual_params={
            "alpha": AGE_DIVORCE_SIMULATION_PARAMS["alpha"],
            "beta_age": AGE_DIVORCE_SIMULATION_PARAMS["beta_age_divorce"],
            "sigma": AGE_DIVORCE_SIMULATION_PARAMS["sigma"],
        }
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 🤘
    We can recover the simulation params.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Age-Divorce model on real data
    """)
    return


@app.cell
def age_divorce_fit(WAFFLEHOUSE_DIVORCE_STD):
    age_divorce_model, age_divorce_inference = fit_age_divorce_model(WAFFLEHOUSE_DIVORCE_STD)
    return age_divorce_inference, age_divorce_model


@app.cell(hide_code=True)
def age_divorce_posterior(AGE_DIVORCE_MODEL_PARAMS, age_divorce_inference):
    plot_divorce_model_posterior(
        age_divorce_inference,
        model_params=AGE_DIVORCE_MODEL_PARAMS
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Run Age Counterfactuals
    """)
    return


@app.cell
def age_counterfactual_ppd(age_divorce_inference, age_divorce_model):
    age_counterfactuals = xr.DataArray(np.linspace(-3, 3, 100))

    with age_divorce_model:
        pm.set_data({"age": age_counterfactuals})
        ppd_age = pm.sample_posterior_predictive(age_divorce_inference, extend_inferencedata=False)
    return age_counterfactuals, ppd_age


@app.cell(hide_code=True)
def age_counterfactual_plot(age_counterfactuals, ppd_age):
    _fig, _ax = plt.subplots()
    utils.plot_hdi(
        age_counterfactuals,
        ppd_age.posterior_predictive["divorce_rate"],
        color="C0", fill_kwargs={"alpha": .15}
    )
    plt.plot(
        age_counterfactuals,
        ppd_age.posterior_predictive["divorce_rate"].mean(dim=("chain", "draw")),
        color="C0"
    )
    plt.xlabel("age counterfactual (in standard deviations)")
    plt.ylabel("divorce rate (in std deviations)")
    plt.title("Causal effect of Age on Divorce Rate in standard deviations")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Collider $X \rightarrow Z \leftarrow Y$

    - $Z$ _caused jointly_ by $Χ$ and $Y$
    - $X$ and $Y$ have no shared causes, thus are not associated
      - $X \perp Y$
    - once stratified by $Z$, $Χ$ and $Y$ are now associated
      -  $X \not \perp Y | Z$
      -  stratifying by $Z$ provides information about $X$ and $Y$
    """)
    return


@app.cell(hide_code=True)
def collider_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Y", "Z")
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discrete Example

    Below we simulate a "Collider" generative process:

    $$
    \begin{align*}
    X &\sim \text{Bernoulli}(0.5) \\
    Y &\sim \text{Bernoulli}(0.5) \\
    Z &\sim \text{Bernoulli}(p^*) \\
    p^* &= \begin{cases}
        0.9 ,& \text{if } X + Y \gt 0\\
        0.05,              & \text{otherwise}
    \end{cases}
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def discrete_collider_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "p_star"),
            ("Y", "p_star"),
            ("p_star", "Z")
        ],
        node_props={"p_star": {"label": "p*", "color": "green", "fontcolor": "green"}},
        edge_props={("p_star", "Z"): {"color": "green"}},
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note: the structure above should not be confused with the "Descendant" discussed later. Here $p^*$ is a deterministic function of $X$ and $Y$ that defines the probability distribution over $Z$, whereas in the Descendant, the downstream variable (e.g. $A$) "leaks" information about the collider.
    """)
    return


@app.cell
def simulate_discrete_collider():
    np.random.seed(123)
    _n_samples = 1000

    _X = stats.bernoulli.rvs(p=.5, size=_n_samples)
    _Y = stats.bernoulli.rvs(p=.5, size=_n_samples)

    # p(z | x, y)
    _p_star = np.where(_X + _Y > 0, 0.9, 0.05)
    _Z = stats.bernoulli.rvs(p=_p_star, size=_n_samples)
    discrete_collider = dict(X=_X, Y=_Y, Z=_Z)
    return (discrete_collider,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Demonstrate that:
    - $X \perp Y$
    - $X \not  \perp Y | Z$
    """)
    return


@app.cell
def discrete_collider_summary(discrete_collider):
    summarize_discrete_confound_simulation(**discrete_collider)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Continuous Example

    $$
    \begin{align*}
    X &\sim \text{Normal}(0, 1) \\
    Y &\sim \text{Normal}(0, 1) \\
    Z &\sim \text{Bernoulli}(p^*) \\
    p^* &= \frac{1}{1 + e^{-(\alpha + \beta_{XZ}X + \beta_{YZ}Y)}} \\
    \end{align*}
    $$
    """)
    return


@app.cell
def simulate_continuous_collider():
    np.random.seed(1234)

    _n_samples = 500
    _X = stats.norm.rvs(size=_n_samples)
    _Y = stats.norm.rvs(size=_n_samples)

    _beta_XZ = 2
    _beta_YZ = 2
    _alpha = -2

    _p_star = utils.invlogit(_alpha + _beta_XZ * _X + _beta_YZ * _Y)
    _Z = stats.bernoulli.rvs(p=_p_star)

    collider_data = pd.DataFrame(np.vstack([_Z, _X, _Y]).T, columns=['Z', 'X', 'Y'])
    return (collider_data,)


@app.cell(hide_code=True)
def continuous_collider_plot(collider_data):
    plot_continuous_confound_simulation(collider_data, "The Collider\n$X \\rightarrow  Z \\leftarrow Y$")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Total sample shows no association
    - Stratifying by Z shows negative association

    ### Why does this happenX

    **Thresholding effect**
    - If either $X$ OR $Y$ are big enough, it will result in a conditional value (the green deterministic function $p*$ above) that is large enough to associate the sample with $Z=1$.
    - Similarly if either $X$ OR $Y$ are small enough, it will associate the sample with $Z=0$.
    - If one of those is high/low enough while the other is average or lower/larger, this will induce a negative correlation on average.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Confound playground
    Pick an elemental confound and change how strongly $Z$ is tied to $X$ and $Y$: $1$ matches the examples above, and $0$ removes the links. Compare the black line for the total sample with the colored lines stratified by $Z$. For the Fork and the Pipe, stratifying by $Z$ removes the association between $X$ and $Y$; for the Collider, stratifying creates one.
    """)
    return


@app.cell(hide_code=True)
def confound_playground_controls():
    confound_type = mo.ui.radio(["Fork", "Pipe", "Collider"], value="Fork", label="Confound", inline=True)
    confound_strength = mo.ui.slider(0, 3, step=0.25, value=1, label="Strength of the links to $Z$", show_value=True)
    confound_n = mo.ui.slider(steps=[100, 200, 500, 1000], value=500, label="Samples", show_value=True)
    return confound_n, confound_strength, confound_type


@app.cell(hide_code=True)
def confound_playground_explorer(confound_n, confound_strength, confound_type):
    np.random.seed(1234)
    _n = confound_n.value
    _s = confound_strength.value
    if confound_type.value == "Fork":
        _Z = stats.bernoulli.rvs(p=.5, size=_n)
        _X = stats.norm.rvs(loc=_s * (2 * _Z - 1), size=_n)
        _Y = stats.norm.rvs(loc=_s * (2 * _Z - 1), size=_n)
        _title = "The Fork\n$X \\leftarrow Z \\rightarrow Y$"
    elif confound_type.value == "Pipe":
        _X = stats.norm.rvs(size=_n)
        _Z = stats.bernoulli.rvs(p=utils.invlogit(_s * _X))
        _Y = stats.norm.rvs(loc=_s * (2 * _Z - 1), size=_n)
        _title = "The Pipe\n$X \\rightarrow Z \\rightarrow Y$"
    else:
        _X = stats.norm.rvs(size=_n)
        _Y = stats.norm.rvs(size=_n)
        _Z = stats.bernoulli.rvs(p=utils.invlogit(-2 + 2 * _s * (_X + _Y)))
        _title = "The Collider\n$X \\rightarrow Z \\leftarrow Y$"

    plot_continuous_confound_simulation(pd.DataFrame({"Z": _Z, "X": _X, "Y": _Y}), _title)
    _rows = [("total sample", np.corrcoef(_X, _Y)[0, 1])] + [
        (f"Z = {_z}", np.corrcoef(_X[_Z == _z], _Y[_Z == _z])[0, 1]) for _z in [0, 1]
    ]
    _table = mo.md("| | corr(X, Y) |\n|---|---|\n" + "\n".join(f"| {_k} | {_v:.2f} |" for _k, _v in _rows))
    mo.vstack([
        mo.hstack([confound_type, confound_strength, confound_n], justify="start", gap=2),
        mo.hstack([plt.gcf(), _table], justify="start", align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Collider Example: Grant Awards
    $N \rightarrow A \leftarrow T$
    - To be awarded $A$, Grants must be sufficiently either Newsworthy $N$ or Trustworthy $T$
    - Few grants are high on both
    - results in negative association for awarded grants
    - Post-selection bias
    """)
    return


@app.cell
def grant_awards_plot():
    np.random.seed(123)
    _n_samples = 200

    # N and T are independent
    _N = stats.norm.rvs(size=_n_samples)
    _T = stats.norm.rvs(size=_n_samples)

    # Award criterion; either are large enough to threshold
    _A = np.where(_N + _T > 2, 1, 0)

    _fig, _ax = plt.subplots()
    for _awarded in [0, 1]:
        _color = 'gray' if not _awarded else 'C0'
        _N_A = _N[_A==_awarded]
        _T_A = _T[_A==_awarded]
        utils.plot_scatter(_N_A, _T_A, color=_color)

    _awarded_model = smf.ols("T ~ N", data=pd.DataFrame({"N": _N_A, "T": _T_A})).fit()
    utils.plot_line(_N_A, _awarded_model.predict(), color='k', label='Post-selection\nTrend')
    plt.xlabel("Newsworthiness")
    plt.ylabel("Trustworthiness")
    plt.axis("square")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Selection threshold
    Change how strict the selection is. Newsworthiness and trustworthiness are independent among all proposals, but the stricter the threshold, the more negative their correlation among the awarded grants.
    """)
    return


@app.cell(hide_code=True)
def selection_threshold_controls():
    grant_threshold = mo.ui.slider(0, 3, step=0.25, value=2, label="Award if $N + T >$", show_value=True)
    grant_n = mo.ui.slider(steps=[200, 500, 1000, 2000], value=200, label="Proposals", show_value=True)
    return grant_n, grant_threshold


@app.cell(hide_code=True)
def selection_threshold_explorer(grant_n, grant_threshold):
    np.random.seed(123)
    _N = stats.norm.rvs(size=grant_n.value)
    _T = stats.norm.rvs(size=grant_n.value)
    _A = (_N + _T > grant_threshold.value).astype(int)

    _fig, _ax = plt.subplots()
    utils.plot_scatter(_N[_A == 0], _T[_A == 0], color="gray", label="rejected")
    utils.plot_scatter(_N[_A == 1], _T[_A == 1], color="C0", label="awarded")
    _title = f"All proposals: r = {np.corrcoef(_N, _T)[0, 1]:.2f}"
    if _A.sum() >= 3:
        _N_A, _T_A = _N[_A == 1], _T[_A == 1]
        _awarded_model = smf.ols("T ~ N", data=pd.DataFrame({"N": _N_A, "T": _T_A})).fit()
        utils.plot_line(_N_A, _awarded_model.predict(), color="k", label="Post-selection\nTrend")
        _title += f"\nAwarded ({_A.sum()}): r = {np.corrcoef(_N_A, _T_A)[0, 1]:.2f}"
    plt.xlabel("Newsworthiness")
    plt.ylabel("Trustworthiness")
    plt.axis("square")
    plt.legend()
    plt.title(_title)
    mo.vstack([
        mo.hstack([grant_threshold, grant_n], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Age, Happiness, and Marriage

    $A \rightarrow M \leftarrow H$

    - The older you are, the more chances of being married
    - The happier you are, the more amicable you are, and thus are more likely to have another person marry you
    - Here, **age does not influence happiness**, they are independent

    Below is a janky version of the simulation in the lecture. Rather than running the temporal simulation, starting at age 18,
    and sampling marital status for each happiness level at each point in time, we just do the whole simulation in one sample, modeling the probability of being married as a combination of both Age $A$ and Happiness $H$: $p_{married} = \text{invlogit}(\beta_H H + \beta_A (A - 58))$, with $\beta_H = 20$ and $\beta_A = 1$
    """)
    return


@app.cell
def age_happiness_marriage_plot():
    np.random.seed(123)
    _age = np.arange(1, 71)
    _happiness = np.linspace(-2, 2, 40)
    _age_grid, _happiness_grid = np.meshgrid(_age, _happiness)

    # Arbitrary
    _beta_happiness = 20
    _beta_age = 1
    _min_marriage_age = 18

    _age_grid = _age_grid.ravel()
    _happiness_grid = _happiness_grid.ravel()
    _married_grid = []
    for _a, _h in zip(_age_grid, _happiness_grid):
        _p_marriage = utils.invlogit(_h * _beta_happiness + (_a - 40 - 18) * _beta_age)
        _married_grid.append(stats.bernoulli(p=_p_marriage).rvs())

    _married_grid = np.array(_married_grid).astype(int)

    _fig, _ax = plt.subplots()
    utils.plot_scatter(_age_grid[_married_grid==0], _happiness_grid[_married_grid==0], s=20, alpha=1, color='lightgray', label='Unmarried')
    utils.plot_scatter(_age_grid[_married_grid==1], _happiness_grid[_married_grid==1], s=20, alpha=1, color='C0', label='Married')
    plt.xlabel("Age")
    plt.ylabel("Happiness (std)")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we were to stratify by married folks only ($M=1$), we would conclude that Age and Happiness are negatively associated, despite them actually being independent in this simulation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Collider Example: Restaurant Success
    """)
    return


@app.cell(hide_code=True)
def restaurant_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("F", "S"),
            ("L", "S")
        ],
        node_props={
            "F": {"label": "Food Quality, F"},
            "L": {"label": "Location Quality, L"},
            "S": {"label": "Restaurant Successful, S"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulate Restaurant Success

    $$
    \begin{align*}
    S &\sim \text{Bernoulli}(p_S) \\
    p_S &= \text{invlogit}(\alpha+ \beta_F F + \beta_L L) \\
    \end{align*}
    $$
    """)
    return


@app.cell
def simulate_restaurants():
    np.random.seed(123)
    _n_restaurants = 1000
    _food_quality = stats.norm.rvs(size=_n_restaurants)  # standardized food quality
    _location_quality = stats.norm.rvs(size=_n_restaurants)  # standardized quality of restaurant location

    _beta_food_quality = 20  # good food increases probability of success
    _beta_location_quality = 20  # good location also increases probability of success
    _intercept = -40  # the average restaurant will not succeed

    _p_success = utils.invlogit(
        _food_quality * _beta_food_quality + _location_quality * _beta_location_quality + _intercept
    )
    _stays_in_business = stats.bernoulli.rvs(_p_success)

    restaurant_data = pd.DataFrame(
        np.vstack(
            [_location_quality, _food_quality, _stays_in_business]
        ).T,
        columns=['location_quality', 'food_quality', 'stays_in_business']
    )
    return (restaurant_data,)


@app.cell
def restaurant_fits(restaurant_data):
    restaurant_models = {}
    restaurant_models['unstratified'] = smf.ols("food_quality ~ location_quality", data=restaurant_data).fit()

    # Stratified Models (post-hoc selection based on successful business)
    restaurant_models['business_failed'] = smf.ols(
        "food_quality ~ location_quality", data=restaurant_data[restaurant_data.stays_in_business==0]
    ).fit()
    restaurant_models['business_succeeded'] = smf.ols(
        "food_quality ~ location_quality", data=restaurant_data[restaurant_data.stays_in_business==1]
    ).fit()
    return (restaurant_models,)


@app.function
def plot_sms_restaurant_model_fit(model, xs, label, color):
    """Helper function to plot linear models"""
    params = model.params
    ys = params.Intercept + params.location_quality * xs
    return utils.plot_line(xs, ys, label=label, color=color)


@app.cell(hide_code=True)
def restaurant_plot(restaurant_data, restaurant_models):
    _xs = np.linspace(-4, 4, 20)

    _fig, _ax = plt.subplots()
    for _sib, _label in enumerate(["business_failed", "business_succeeded"]):
        _scatter_data = restaurant_data[restaurant_data.stays_in_business==_sib]
        utils.plot_scatter(
            _scatter_data.location_quality,
            _scatter_data.food_quality,
            color=f'C{_sib}',
            label="successful" if _sib else "failed"
        )

        plot_sms_restaurant_model_fit(restaurant_models[_label], _xs, color=f'C{_sib}', label=None)

    plot_sms_restaurant_model_fit(restaurant_models['unstratified'], _xs, label='unstratified', color='black')

    plt.xlabel("location quality (standardized)")
    plt.ylabel("food quality (standardized)")

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    By looking only at successful restaurants, we would mislead ourselves and infer that lower-quality locations have better food, when in fact there is no relationship between location and food quality.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Descendant
    Takes on a diluted behavior of the parent.
    - If the parent forms a Collider, the descendant acts as a weak collider. Same for Pipe and Fork descendants
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discrete Example: Pipe Descendant
    In this example the descendant branches off of a Pipe. Therefore we should observe the following:
    - $X \not \perp Y$
    - $X \perp Y | Z$
    - $X \not \perp Y | A$, but correlation between X and Y should be reduced
    """)
    return


@app.cell(hide_code=True)
def pipe_descendant_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "A"),
            ("Z", "Y")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    X &\sim \text{Bernoulli}(0.5) \\
    Z &\sim \text{Bernoulli}(p^*) \\
    A &\sim \text{Bernoulli}(q^*) \\
    Y &\sim \text{Bernoulli}(r^*) \\
    p^* &= 0.9 X + 0.1 (1 - X) \\
    q^* &= 0.9 Z + 0.1 (1 - Z) \\
    r^* &= 0.9 Z + 0.1 (1 - Z) \\
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def pipe_descendant_labelled_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "A"),
            ("Z", "Y")
        ],
        edge_props={
            ("X", "Z"): {"label": "p*"},
            ("Z", "A"): {"label": "q*"},
            ("Z", "Y"): {"label": "r*"}
        },
        graph_direction="LR"
    )
    return


@app.cell
def simulate_pipe_descendant():
    np.random.seed(1234)
    _n_samples = 1000

    _X = stats.bernoulli.rvs(p=.5, size=_n_samples)

    _p_star = .9 * _X + .1 * (1 - _X)
    _Z = stats.bernoulli.rvs(p=_p_star, size=_n_samples)

    _q_star = .9 * _Z + .1 * (1 - _Z)
    _A = stats.bernoulli.rvs(p=_q_star, size=_n_samples)

    _r_star = .9 * _Z + .1 * (1 - _Z)
    _Y = stats.bernoulli.rvs(p=_r_star, size=_n_samples)
    discrete_descendant = dict(X=_X, Y=_Y, A=_A)
    return (discrete_descendant,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Demonstrate that

    - $X \not \perp Y$
    - $X \not \perp Y | A$, but the correlation is weaker than in the total sample
    """)
    return


@app.cell
def pipe_descendant_summary(discrete_descendant):
    summarize_discrete_confound_simulation(
        discrete_descendant["X"], discrete_descendant["Y"], discrete_descendant["A"], z_name="A"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Collider Descendant
    In this example the descendant branches off of a Collider. Therefore we should observe the following:
    - $X \perp Y$
    - $X \not \perp Y | Z$
    - $X \not \perp Y | A$
    """)
    return


@app.cell(hide_code=True)
def collider_descendant_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "A"),
            ("Y", "Z")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    X &\sim \text{Normal}(0, 1) \\
    Y &\sim \text{Normal}(0, 1) \\
    Z &\sim \text{Bernoulli}(p^*) \\
    A &\sim \text{Bernoulli}(q^*) \\
    p^* &= \begin{cases}
        0.9 ,& \text{if } X + Y \gt 2\\
        0.1,              & \text{otherwise}
    \end{cases} \\
    q^* &= 0.95 Z + 0.05 (1 - Z)
    \end{align*}
    $$
    """)
    return


@app.cell
def simulate_collider_descendant():
    np.random.seed(123)
    _n_samples = 1000

    _X = stats.norm.rvs(size=_n_samples)
    _Y = stats.norm.rvs(size=_n_samples)

    _p_star = np.where(_X + _Y > 2, .9, .1)
    _Z = stats.bernoulli.rvs(p=_p_star, size=_n_samples)

    _q_star = .95 * _Z + .05 * (1 - _Z)
    _A = stats.bernoulli.rvs(p=_q_star, size=_n_samples)
    collider_descendant = dict(X=_X, Y=_Y, A=_A)
    return (collider_descendant,)


@app.cell
def collider_descendant_correlations(collider_descendant):
    _X, _Y, _A = collider_descendant["X"], collider_descendant["Y"], collider_descendant["A"]
    print(f'Correlation between X and Y: {np.corrcoef(_X, _Y)[0, 1]:1.2}')
    for _a in [0, 1]:
        print(f'Correlation between X and Y conditioned on A={_a}: {np.corrcoef(_X[_A==_a], _Y[_A==_a])[0, 1]:1.2}')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Descendant strength
    The descendant $A$ copies its parent $Z$ with probability $P(A = Z)$. At $0.5$, $A$ is pure noise and conditioning on it changes nothing; at $1$, $A$ is $Z$, and conditioning on $A$ is the same as conditioning on $Z$. In between, $A$ acts as a diluted version of $Z$.
    """)
    return


@app.cell(hide_code=True)
def descendant_strength_controls():
    descendant_parent = mo.ui.radio(["Pipe", "Collider"], value="Pipe", label="Parent", inline=True)
    descendant_fidelity = mo.ui.slider(0.5, 1, step=0.05, value=0.9, label="$P(A = Z)$", show_value=True)
    return descendant_fidelity, descendant_parent


@app.cell(hide_code=True)
def descendant_strength_explorer(descendant_fidelity, descendant_parent):
    np.random.seed(1234)
    _n = 2000
    _f = descendant_fidelity.value
    if descendant_parent.value == "Pipe":
        _X = stats.bernoulli.rvs(p=.5, size=_n)
        _Z = stats.bernoulli.rvs(p=.9 * _X + .1 * (1 - _X))
        _Y = stats.bernoulli.rvs(p=.9 * _Z + .1 * (1 - _Z))
    else:
        _X = stats.norm.rvs(size=_n)
        _Y = stats.norm.rvs(size=_n)
        _Z = stats.bernoulli.rvs(p=np.where(_X + _Y > 2, .9, .1))
    _A = stats.bernoulli.rvs(p=_f * _Z + (1 - _f) * (1 - _Z))

    _groups = {"total": np.ones(_n, dtype=bool)}
    for _name, _values in [("Z", _Z), ("A", _A)]:
        for _v in [0, 1]:
            _groups[f"{_name} = {_v}"] = _values == _v
    _corrs = {_k: np.corrcoef(_X[_mask], _Y[_mask])[0, 1] for _k, _mask in _groups.items()}

    _fig, _ax = plt.subplots(figsize=(7, 3))
    plt.bar(list(_corrs), list(_corrs.values()), color=["k", "C0", "C0", "C1", "C1"], alpha=.7)
    plt.axhline(0, color="k", linewidth=1)
    plt.ylim(-1, 1)
    plt.ylabel("corr(X, Y)")
    plt.title(f"{descendant_parent.value} descendant, $P(A = Z) = {_f:.2f}$")
    mo.vstack([
        mo.hstack([descendant_parent, descendant_fidelity], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Descendants are everywhere
    - Many measurements are **proxies** for what we want to measure

    ## Unobserved Confounds
    - Confounds are everywhere, and can ruin your day
    - Some controls are better than others
    - Often, trying to control for some variables can open up paths to unobserved confounds; we must always be aware of this possibility
    """)
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
