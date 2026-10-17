import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import inspect

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    import statsmodels.formula.api as smf
    from scipy import stats
    from matplotlib import pyplot as plt
    from patsy import dmatrix

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 04 - Categories & Curves](https://youtu.be/F0N4b7K_iYQ?si=GNw_RAI6LDdrpXNQ)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Linear Regression & Drawing Inferences
    - Can be used to approximate most anything, even nonlinear phenomena (e.g. GLMs)
    - We need to incorporate causal thinking into
      - How we compose statistical models
      - Process and interpret results
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Categories
    - non-continuous causes
    - discrete, unordered types
    - stratifying by category: fit a separate regression (e.g. line) to each

    ## Revisiting the Howell dataset
    """)
    return


@app.cell
def load_howell():
    HOWELL = utils.load_data("Howell1")

    # Adult data
    ADULT_HOWELL = HOWELL[HOWELL.age >= 18]

    # Split by the Sex Category
    SEX = ['women', 'men']
    return ADULT_HOWELL, HOWELL, SEX


@app.cell(hide_code=True)
def adult_howell_scatter(ADULT_HOWELL, SEX):
    _fig, _ax = plt.subplots(figsize=(5, 4))
    for _ii, _label in enumerate(SEX):
        utils.plot_scatter(
            ADULT_HOWELL[ADULT_HOWELL.male == _ii].height,
            ADULT_HOWELL[ADULT_HOWELL.male == _ii].weight,
            color=f"C{_ii}",
            label=_label,
        )
    plt.ylim([30, 65])
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def sex_height_weight_dag():
    # Draw the mediation graph
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "H"),
            ("S", "W")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Think scientifically first

    - How are height, weight, and sex **causally** related?
    - How are height, weight, and sex **statistically** related?

    ### The causes aren't in the data

    Height should affect weight, not vice versa
    - ✅ $H \rightarrow W$
    - ❌ $H \leftarrow W$

    Sex should affect height, not vice versa
    - ❌ $H \rightarrow S$
    - ✅ $H \leftarrow S$
    """)
    return


@app.cell
def height_weight_distribution_plotter(SEX):
    # Split height by the Sex Category
    def plot_height_weight_distributions(data):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        plt.sca(axs[0])

        for ii, label in enumerate(SEX):
            utils.plot_scatter(
                data[data.male == ii].height,
                data[data.male == ii].weight,
                color=f"C{ii}",
                label=label,
            )
        plt.xlabel("height (cm)")
        plt.ylabel("weight (kg)")
        plt.legend()
        plt.title("height vs weight")

        for vv, var in enumerate(['height', 'weight']):
            plt.sca(axs[vv+1])
            for ii in range(2):
                utils.plot_dist(
                    data.loc[data.male == ii, var].values,
                    color=f"C{ii}",
                    label=SEX[ii],
                    bw=1,
                    plot_kwargs=dict(linewidth=3, alpha=.6),
                )
            plt.title(f"{var} split by sex")
            plt.xlabel(f"{var} ({'cm' if var == 'height' else 'kg'})")
            plt.legend()

    return (plot_height_weight_distributions,)


@app.cell(hide_code=True)
def adult_howell_distributions(ADULT_HOWELL, plot_height_weight_distributions):
    plot_height_weight_distributions(ADULT_HOWELL)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Causal graph defines a set of functional relationships

    $$
    \begin{align*}
    H &= f_H(S) \\
    W &= f_W(H, S)
    \end{align*}
    $$

    Could also include unobservable causal influences (see below graph):

    $$
    \begin{align*}
    H &= f_H(S, U) \\
    W &= f_W(H, S, V) \\
    S &= f_S(T)
    \end{align*}
    $$

    > Note: we use $T$ as an unobserved variable, rather than $W$ to avoid replication in the lecture.
    """)
    return


@app.cell(hide_code=True)
def unobserved_causes_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "H"),
            ("S", "W"),
            ("V", "W"),
            ("U", "H"),
            ("T", "S")
        ],
        node_props={
            "T": {"style": "dashed"},
            "U": {"style": "dashed"},
            "V": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Synthetic People
    """)
    return


@app.cell(hide_code=True)
def synthetic_people_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "H"),
            ("S", "W"),
            ("V", "W"),
            ("U", "H"),
            ("T", "S")
        ],
        node_props={
            "T": {"style": "dashed", "color": "lightgray"},
            "U": {"style": "dashed", "color": "lightgray"},
            "V": {"style": "dashed", "color": "lightgray"},
            "unobserved": {"style": "dashed", "color": "lightgray"}
        },
        edge_props={
            ("T", "S"): {"color": "lightgray"},
            ("U", "H"): {"color": "lightgray"},
            ("V", "W"): {"color": "lightgray"}
        },
        graph_direction="LR"
    )
    return


@app.function
def simulate_sex_height_weight(
    S: np.ndarray,
    beta: np.ndarray = np.array([1, 1]),
    alpha: np.ndarray = np.array([0, 0])
) -> np.ndarray:
    """
    Generative model for the effect of Sex on height / weight

    S: np.array[int]
        The 0/1 indicator variable sex. 1 means 'male'
    beta: np.array[float]
        Length 2 slope coefficient for each sex
    alpha: np.array[float]
        Length 2 intercept for each sex
    """
    N = len(S)
    H = np.where(S, 160, 150) + stats.norm(0, 5).rvs(size=N)
    W = alpha[S] + beta[S] * H + stats.norm(0, 5).rvs(size=N)

    return pd.DataFrame({"height": H, "weight": W, "male": S})


@app.cell
def synthetic_people_sample():
    synthetic_sex = stats.bernoulli(p=.5).rvs(size=100).astype(int)
    synthetic_people = simulate_sex_height_weight(S=synthetic_sex)
    synthetic_people.head()
    return (synthetic_people,)


@app.cell(hide_code=True)
def synthetic_people_distributions(
    plot_height_weight_distributions,
    synthetic_people,
):
    plot_height_weight_distributions(synthetic_people)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore the generative model
    Change the sex-specific intercepts and slopes of the generative model. Height is $150$ cm (F) or $160$ cm (M) plus $\text{Normal}(0, 5)$ noise, and weight is $W = \alpha_S + \beta_S H + \text{Normal}(0, 5)$, with $\alpha_F = 0$.
    """)
    return


@app.cell(hide_code=True)
def synthetic_controls():
    synthetic_alpha_m = mo.ui.slider(-20, 20, step=1, value=0, label=r"Male weight boost $\alpha_M$ (kg)", show_value=True)
    synthetic_beta_f = mo.ui.slider(0, 1.5, step=0.05, value=1, label=r"$\beta_F$", show_value=True)
    synthetic_beta_m = mo.ui.slider(0, 1.5, step=0.05, value=1, label=r"$\beta_M$", show_value=True)
    synthetic_n = mo.ui.slider(steps=[20, 50, 100, 200, 500, 1000], value=100, label="People", show_value=True)
    return synthetic_alpha_m, synthetic_beta_f, synthetic_beta_m, synthetic_n


@app.cell(hide_code=True)
def synthetic_explorer(
    plot_height_weight_distributions,
    synthetic_alpha_m,
    synthetic_beta_f,
    synthetic_beta_m,
    synthetic_n,
):
    np.random.seed(12)
    _sex = stats.bernoulli(p=.5).rvs(size=synthetic_n.value).astype(int)
    _people = simulate_sex_height_weight(
        S=_sex,
        beta=np.array([synthetic_beta_f.value, synthetic_beta_m.value]),
        alpha=np.array([0, synthetic_alpha_m.value]),
    )
    plot_height_weight_distributions(_people)
    mo.vstack([
        mo.hstack([synthetic_alpha_m, synthetic_beta_f, synthetic_beta_m, synthetic_n], justify="start", wrap=True),
        plt.gcf(),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Think scientifically first

    Different causal questions require different statistical models:

    - Q: Causal effect of H on W?
    - Q: (Total) Causal effect of S on W?
    - Q: *Direct* Causal effect of S on W?

    Answering the last two questions requires different statistical models, but both will need stratification by S
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### From estimand to estimate
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Causal effect of H on W
    """)
    return


@app.cell(hide_code=True)
def height_on_weight_estimand_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "W"),
            ("S", "H")
        ],
        node_props={
            "H": {"color": "red"},
            "W": {"color": "red"}
        },
        edge_props={
            ("H", "W"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### **Total** Causal effect of S on W
    """)
    return


@app.cell(hide_code=True)
def total_effect_estimand_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "W"),
            ("S", "H")
        ],
        node_props={
            "S": {"color": "red"},
            "W": {"color": "red"}
        },
        edge_props={
            ("S", "H"): {"color": "red"},
            ("H", "W"): {"color": "red"},
            ("S", "W"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### **Direct** Causal effect of S on W
    """)
    return


@app.cell(hide_code=True)
def direct_effect_estimand_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "W"),
            ("S", "H")
        ],
        node_props={
            "S": {"color": "red"},
            "W": {"color": "red"}
        },
        edge_props={
            ("S", "W"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Stratify by S**: recover a different estimate for each value that $S$ can take

    ## Drawing the Causal Owl 🦉

    Implement Categories via **Indicator Variables**

    - generalizes code: can extend to any number of categories
    - better for prior specification
    - facilitates multi-level model specification

    For categories $C = [C_1, C_2, ... C_D]$

    $$
    \begin{align*}
    \alpha &= [\alpha_1, \alpha_2, ... \alpha_D] \\
    y_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha_{C[i]}
    \end{align*}
    $$

    For sex $S\in \{M, F\}$, we can model sex-specific weight $W$ as

    $$
    \begin{align*}
    W_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha_{S[i]} \\
    \alpha &= [\alpha_F, \alpha_M] \\
    \alpha_j &\sim \text{Normal}(60, 10) \\
    \sigma &\sim \text{Uniform}(0, 10)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Testing
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Total Causal Effect of Sex on Weight
    """)
    return


@app.cell(hide_code=True)
def total_effect_stratified_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "W"),
            ("S", "H")
        ],
        node_props={
            "S": {"color": "blue"},
            "W": {"color": "red"},
            "stratified": {"color": "blue"}
        },
        edge_props={
            ("S", "H"): {"color": "red"},
            ("H", "W"): {"color": "red"},
            ("S", "W"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell
def simulate_total_effect():
    np.random.seed(12345)
    _n_simulations = 100
    simulated_females = simulate_sex_height_weight(
        S=np.zeros(_n_simulations).astype(int),
        beta=np.array((0.5, 0.6))
    )

    simulated_males = simulate_sex_height_weight(
        S=np.ones(_n_simulations).astype(int),
        beta=np.array((0.5, 0.6))
    )

    _simulated_delta = simulated_males - simulated_females
    _mean_simulated_delta = _simulated_delta.mean()
    _fig, _ax = plt.subplots()
    utils.plot_dist(_simulated_delta['weight'].values)
    plt.axvline(
        _mean_simulated_delta['weight'],
        linestyle='--',
        color='k',
        label="Mean difference" + f"={_mean_simulated_delta['weight']:1.2f}"
    )
    plt.xlabel("M - F")
    plt.legend()
    plt.title("simulated difference")
    _fig
    return simulated_females, simulated_males


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Run estimating model on the synthetic sample
    """)
    return


@app.function
def fit_total_effect_model(data):

    sex_indicator, sex = pd.factorize(np.array(["M" if s else "F" for s in data["male"].values]))

    with pm.Model(coords={"sex": sex}) as model:
        # Data
        S = pm.Data("S", sex_indicator)

        # Priors
        sigma = pm.Uniform("sigma", 0, 10)
        alpha = pm.Normal("alpha", 60, 10, dims="sex")

        # Likelihood
        mu = alpha[S]
        pm.Normal("W_obs", mu, sigma, observed=data["weight"])

        inference = cached_sample()

    return inference, model


@app.cell
def simulated_total_effect_fit(simulated_females, simulated_males):
    # Concatenate simulations and code sex
    simulated_people = pd.concat([simulated_females, simulated_males])
    simulated_total_effect_inference, simulated_total_effect_model = fit_total_effect_model(simulated_people)
    return simulated_people, simulated_total_effect_inference


@app.cell
def simulated_total_effect_summary(simulated_total_effect_inference):
    simulated_summary = az.summary(simulated_total_effect_inference, var_names=['alpha', 'sigma'])
    simulated_delta = (simulated_summary.iloc[1] - simulated_summary.iloc[0])['mean']
    print(f"Delta in average sex-specific weight: {simulated_delta:1.2f}")

    simulated_summary
    return


@app.function
def plot_model_posterior(inference, effect_type: str = "Total"):
    np.random.seed(123)
    sex = ['F', 'M']
    posterior = inference.posterior

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    # Posterior mean
    plt.sca(axs[0][0])
    for ii, s in enumerate(sex):
        posterior_mean = posterior["alpha"].sel(sex=s).mean(dim='chain')
        utils.plot_dist(posterior_mean, color=f"C{ii}", label=s, plot_kwargs=dict(linewidth=3))

    plt.xlabel("posterior mean weight (kg)")
    plt.ylabel("density")
    plt.legend()
    plt.title("Posterior $\\alpha_S$")

    # Posterior Predictive
    plt.sca(axs[0][1])
    posterior_prediction_std = posterior["sigma"].mean(dim=['chain'])
    posterior_prediction = {}

    for ii, s in enumerate(sex):
        posterior_prediction_mean = posterior["alpha"].sel(sex=s).mean(dim=['chain'])
        posterior_prediction[s] = stats.norm.rvs(posterior_prediction_mean, posterior_prediction_std)
        utils.plot_dist(posterior_prediction[s], color=f"C{ii}", label=s, plot_kwargs=dict(linewidth=3))

    plt.xlabel("posterior predicted weight (kg)")
    plt.ylabel("density")
    plt.legend()
    plt.title("Posterior Predictive")


    # Plot Contrasts
    ## Posterior Contrast
    plt.sca(axs[1][0])
    posterior_contrast = posterior["alpha"].sel(sex='M') - posterior["alpha"].sel(sex='F')
    utils.plot_dist(posterior_contrast, color='k', plot_kwargs=dict(linewidth=3))
    plt.xlabel("$\\alpha_M$ - $\\alpha_F$ posterior mean weight contrast")
    plt.ylabel("density")
    plt.title("Posterior Contrast")

    ## Posterior Predictive Contrast
    plt.sca(axs[1][1])
    posterior_predictive_contrast = posterior_prediction['M'] - posterior_prediction['F']
    n_draws = len(posterior_predictive_contrast)
    kde_ax = utils.plot_dist(posterior_predictive_contrast, color='k', bw=1, plot_kwargs=dict(linewidth=3))

    # Shade underneath posterior predictive contrast
    kde_x, kde_y = kde_ax.get_lines()[0].get_data()

    # Proportion of PPD contrast below zero
    neg_idx = kde_x < 0
    neg_prob = 100 * np.sum(posterior_predictive_contrast < 0) / n_draws
    plt.fill_between(
        x=kde_x[neg_idx],
        y1=np.zeros(sum(neg_idx)),
        y2=kde_y[neg_idx],
        color='C0',
        label=f"{neg_prob:1.0f}%"
    )

    # Proportion of PPD contrast above zero (inclusive)
    pos_idx = kde_x >= 0
    pos_prob = 100 * np.sum(posterior_predictive_contrast >= 0) / n_draws
    plt.fill_between(
        x=kde_x[pos_idx],
        y1=np.zeros(sum(pos_idx)),
        y2=kde_y[pos_idx],
        color='C1',
        label=f"{pos_prob:1.0f}%"
    )

    plt.xlabel("(M - F)\nposterior prediction contrast")
    plt.ylabel("density")
    plt.legend()
    plt.title("Posterior\nPredictive Contrast")
    plt.suptitle(f"{effect_type} Causal Effect of Sex on Weight", fontsize=18)


@app.function
def plot_posterior_lines(data, inference, centered=False):
    plt.subplots(figsize=(6, 6))

    min_height = data.height.min()
    max_height = data.height.max()
    xs = np.linspace(min_height, max_height, 10)
    for ii, s in enumerate(['F', 'M']):
        sex_idx = data.male == ii
        utils.plot_scatter(
            xs=data[sex_idx].height,
            ys=data[sex_idx].weight,
            color=f"C{ii}",
            label=s
        )

        posterior_mean = inference.posterior.sel(sex=s).mean(dim=('chain', 'draw'))
        posterior_mean_alpha = posterior_mean['alpha'].values
        posterior_mean_beta = getattr(posterior_mean, 'beta', pd.Series([0])).values

        if centered:
            pred_x = xs - data.height.mean()
        else:
            pred_x = xs

        ys = posterior_mean_alpha + posterior_mean_beta * pred_x
        utils.plot_line(xs, ys, label=None, color=f"C{ii}")

    # Model fit to both sexes simultaneously
    global_model = smf.ols("weight ~ height", data=data).fit()
    ys = global_model.params.Intercept + global_model.params.height * xs
    utils.plot_line(xs, ys, color="k", label='Unstratified\nModel')

    plt.axvline(
        data["height"].mean(),
        label='Average H',
        linewidth=0.5, linestyle='--', color='black'
    )
    plt.axhline(
        data["weight"].mean(),
        label='Average W',
        linewidth=1, linestyle='--', color='black'
    )
    plt.legend()
    plt.xlabel("height (cm), H")
    plt.ylabel("weight (kg), W")


@app.cell(hide_code=True)
def simulated_total_effect_lines(
    simulated_people,
    simulated_total_effect_inference,
):
    plot_posterior_lines(simulated_people, simulated_total_effect_inference, centered=True)
    plt.gcf()
    return


@app.cell(hide_code=True)
def simulated_total_effect_posterior(simulated_total_effect_inference):
    plot_model_posterior(simulated_total_effect_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analyze real sample
    """)
    return


@app.cell
def howell_total_effect_fit(ADULT_HOWELL):
    adult_howell_total_effect_inference, adult_howell_total_effect_model = fit_total_effect_model(ADULT_HOWELL)
    return (adult_howell_total_effect_inference,)


@app.cell
def howell_total_effect_summary(adult_howell_total_effect_inference):
    adult_howell_total_effect_summary = az.summary(adult_howell_total_effect_inference, var_names=['alpha'])
    adult_howell_total_effect_delta = (adult_howell_total_effect_summary.iloc[1] - adult_howell_total_effect_summary.iloc[0])['mean']
    print(f"Delta in average sex-specific weight: {adult_howell_total_effect_delta:1.2f}")

    adult_howell_summary = az.summary(adult_howell_total_effect_inference)
    adult_howell_summary
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Always be contrasting
    - need compare the **contrast** between categories
    - **never valid to calculate overlap** in distributions
      - this means **no comparing confidence intervals for p-values**
    - Compute the difference of distributions -- the **contrast distribution**
    """)
    return


@app.cell(hide_code=True)
def howell_total_effect_posterior(adult_howell_total_effect_inference):
    plot_model_posterior(adult_howell_total_effect_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def howell_total_effect_lines(
    ADULT_HOWELL,
    adult_howell_total_effect_inference,
):
    plot_posterior_lines(ADULT_HOWELL, adult_howell_total_effect_inference, True)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### **Direct** causal effect of $S$ on $W$?
    We need another model/estimator for this estimand
    """)
    return


@app.cell(hide_code=True)
def direct_effect_stratified_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "W"),
            ("S", "H")
        ],
        node_props={
            "S": {"color": "blue"},
            "W": {"color": "red"},
            "H": {"color": "blue"},
            "stratified": {"color": "blue"}
        },
        edge_props={
            ("S", "W"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    W_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha_{S[i]} + \beta_{S[i]} (H_i - \bar H)
    \end{align*}
    $$

    Where we've **centered the height**, meaning that
    - $\beta$ scales the difference of $H_i$ from the average height
    - $\alpha$ is the weight when a person is the average height
    - Global model fit to all data lies at the intersection of global average height and weight
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulate some more people
    """)
    return


@app.cell
def simulate_boosted_people():
    np.random.seed(1234)
    _n_synthetic_people = 200

    _synthetic_sex = stats.bernoulli.rvs(p=.5, size=_n_synthetic_people)
    boosted_people = simulate_sex_height_weight(
        S=_synthetic_sex,
        beta=np.array([0.5, 0.5]), # Same relationship between height & weight
        alpha=np.array([0.0, 10])  # 10kg "boost for Males"
    )
    return (boosted_people,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the synthetic people
    """)
    return


@app.function
def fit_direct_effect_weight_model(data):

    sex_indicator, sex = pd.factorize(np.array(["M" if s else "F" for s in data["male"].values]))

    with pm.Model(coords={"sex": sex}) as model:
        # Data
        S = pm.Data("S", sex_indicator, dims="obs_ids")
        H = pm.Data("H", data["height"].values, dims="obs_ids")
        Hbar = pm.Data("Hbar", data["height"].mean())

        # Priors
        sigma = pm.Uniform("sigma", 0, 10)
        alpha = pm.Normal("alpha", 60, 10, dims="sex")
        beta = pm.Uniform("beta", 0, 1, dims="sex")  # postive slopes only

        # Likelihood
        mu = alpha[S] + beta[S] * (H - Hbar)
        pm.Normal("W_obs", mu, sigma, observed=data["weight"].values, dims="obs_ids")

        inference = cached_sample()

    return inference, model


@app.cell
def direct_effect_simulated_fit(boosted_people):
    direct_effect_simulated_inference, direct_effect_simulated_model = fit_direct_effect_weight_model(boosted_people)
    return direct_effect_simulated_inference, direct_effect_simulated_model


@app.cell(hide_code=True)
def direct_effect_simulated_lines(
    boosted_people,
    direct_effect_simulated_inference,
):
    plot_posterior_lines(boosted_people, direct_effect_simulated_inference, centered=True)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - **Indirect effect**: M & F have specific slopes - in this simulation, they are the same slope, thus parallel lines
    - **Direct effect**: There will be a delta, no matter the slope. -- in this simulation M are always 10kg heavier, thus blue is always above red
    """)
    return


@app.cell(hide_code=True)
def direct_effect_simulated_posterior(direct_effect_simulated_inference):
    plot_model_posterior(direct_effect_simulated_inference, "Direct")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analyze the real sample
    """)
    return


@app.cell
def direct_effect_howell_fit(ADULT_HOWELL):
    direct_effect_howell_inference, direct_effect_howell_model = fit_direct_effect_weight_model(ADULT_HOWELL)
    return direct_effect_howell_inference, direct_effect_howell_model


@app.cell(hide_code=True)
def direct_effect_howell_lines(ADULT_HOWELL, direct_effect_howell_inference):
    plot_posterior_lines(ADULT_HOWELL, direct_effect_howell_inference, True)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Contrasts
    """)
    return


@app.cell(hide_code=True)
def direct_effect_howell_posterior(direct_effect_howell_inference):
    plot_model_posterior(direct_effect_howell_inference, "Direct")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Contrast at each height
    """)
    return


@app.function
def plot_heightwise_contrast(model, inference):
    heights = np.linspace(130, 190, 100)
    ppds = {}
    for ii, s in enumerate(['F', 'M']):
        with model:
            pm.set_data(
                {
                    "S": np.ones_like(heights).astype(int) * ii,
                    "H": heights,  # keep the training-data Hbar
                }
            )
            ppds[s] = pm.sample_posterior_predictive(inference, extend_inferencedata=False).posterior_predictive["W_obs"]

    ppd_contrast = ppds['M'] - ppds['F']

    # Plot contours
    for prob in [.5, .75, .95, .99]:
        utils.plot_hdi(heights, ppd_contrast, hdi_prob=prob, color='gray')

    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel("height, H (cm)")
    plt.ylabel("weight W contrast (M-F)")
    plt.xlim([130, 190])


@app.cell(hide_code=True)
def howell_heightwise_contrast(
    direct_effect_howell_inference,
    direct_effect_howell_model,
):
    _fig, _ax = plt.subplots()
    plot_heightwise_contrast(direct_effect_howell_model, direct_effect_howell_inference)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When stratifying by Height, we see that Sex has very little, if any causal effect on weight. i.e. **a lion's share of the causal effect on weight comes via height.**
    """)
    return


@app.cell(hide_code=True)
def simulated_heightwise_contrast(
    direct_effect_simulated_inference,
    direct_effect_simulated_model,
):
    # Try on the simulated data
    _fig, _ax = plt.subplots()
    plot_heightwise_contrast(direct_effect_simulated_model, direct_effect_simulated_inference)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    we can see that in the simulated data, men are consistently heavier than women, which is aligned with the simulation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Curves from lines
    Not all relationships are linear
    - e.g. in the Howell dataset, we can see that if we include all ages, the relationship between Height and Weight is nonlinear
    - linear models can fit curves
    - **still not a mechanistic model**
    """)
    return


@app.cell(hide_code=True)
def howell_all_ages_scatter(HOWELL):
    _fig, _ax = plt.subplots(figsize=(4, 4))
    HOWELL.plot(x='height', y='weight', kind='scatter', ax=_ax)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Polynomial Linear Models

    $$\mu_i = \alpha + \sum_{d=1}^D \beta_d x^d $$

    ### Issues
    - symmetric -- strange edge anomalies
    - global models, so no local interpolation

    ### Quadratic Polynomial Model

    $$\mu_i = \alpha + \beta_1 x + \beta_2 x^2 $$
    """)
    return


@app.function
def plot_polynomial_sample(degree, random_seed=123):
    np.random.seed(random_seed)
    xs = np.linspace(-1, 1, 100)
    ys = 0
    for d in range(1, degree+1):
        beta_d = np.random.randn()
        ys += beta_d * xs ** d

    utils.plot_line(xs, ys, color=f"C{degree}", label=f"Degree: {degree}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$\\mu$")


@app.cell(hide_code=True)
def polynomial_prior_samples():
    _fig, _ax = plt.subplots()
    for _degree in [1, 2, 3, 4]:
        plot_polynomial_sample(_degree)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore random polynomials
    Draw random polynomials $\mu = \sum_{d=1}^D \beta_d x^d$ with $\beta_d \sim \text{Normal}(0, 1)$. As the degree grows, the curves stay flat near $x = 0$ and swing wildly near the edges.
    """)
    return


@app.cell(hide_code=True)
def polynomial_prior_controls():
    polynomial_degree = mo.ui.slider(1, 6, value=2, label="Degree $D$", show_value=True)
    polynomial_n_draws = mo.ui.slider(1, 20, value=10, label="Draws", show_value=True)
    polynomial_resample = mo.ui.button(value=0, on_click=lambda count: count + 1, label="New draws")
    return polynomial_degree, polynomial_n_draws, polynomial_resample


@app.cell(hide_code=True)
def polynomial_prior_explorer(
    polynomial_degree,
    polynomial_n_draws,
    polynomial_resample,
):
    _rng = np.random.default_rng(polynomial_resample.value)
    _xs = np.linspace(-1, 1, 100)
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _draw in range(polynomial_n_draws.value):
        _betas = _rng.normal(0, 1, size=polynomial_degree.value)
        _ys = sum(_beta * _xs ** (_d + 1) for _d, _beta in enumerate(_betas))
        plt.plot(_xs, _ys, color="C0", alpha=.6)
    plt.xlabel("x")
    plt.ylabel("$\\mu$")
    plt.title(f"Random degree-{polynomial_degree.value} polynomials")
    mo.vstack([
        mo.hstack([polynomial_degree, polynomial_n_draws, polynomial_resample], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulate Bayesian Updating for Quadratic Polynomial Model
    For the following simulation, we'll use a custom utility function `utils.simulate_2_parameter_bayesian_learning` for simulating general Bayesian posterior update simulation. Here's the API for that function (for more details see `utils.py`)
    """)
    return


@app.cell
def bayesian_learning_docs():
    mo.plain_text(inspect.getdoc(utils.simulate_2_parameter_bayesian_learning))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Functions required for Bayesian learning simulation
    """)
    return


@app.function
def quadratic_polynomial_model(x, beta_1, beta_2):
    return beta_1 * x + beta_2 * x ** 2


@app.function
def quadratic_polynomial_regression_posterior(x_obs, y_obs, beta_1_grid, beta_2_grid, likelihood_prior_std = 1.0):

    beta_1_grid = beta_1_grid.ravel()
    beta_2_grid = beta_2_grid.ravel()

    log_prior_beta_1 = stats.norm(0, 1).logpdf(beta_1_grid)
    log_prior_beta_2 = stats.norm(0, 1).logpdf(beta_2_grid)

    log_likelihood = np.array(
        [
            stats.norm(
                b1 * x_obs + b2 * x_obs ** 2,
                likelihood_prior_std
            ).logpdf(y_obs) for b1, b2 in zip(beta_1_grid, beta_2_grid)
        ]
    ).sum(axis=1)

    log_posterior = log_likelihood + log_prior_beta_1 + log_prior_beta_2

    return np.exp(log_posterior - log_posterior.max())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Run the simulation
    """)
    return


@app.cell
def quadratic_grid_data():
    np.random.seed(123)
    _RESOLUTION = 100
    QUAD_N_DATA_POINTS = 64
    QUAD_BETA_1 = 2
    QUAD_BETA_2 = -2
    _INTERCEPT = 0

    # Generate observations
    quad_x = stats.norm().rvs(size=QUAD_N_DATA_POINTS)
    quad_y = _INTERCEPT + QUAD_BETA_1 * quad_x + QUAD_BETA_2 * quad_x ** 2 + stats.norm.rvs(size=QUAD_N_DATA_POINTS) * .5

    quad_beta_1_grid = np.linspace(-3, 3, _RESOLUTION)
    quad_beta_2_grid = np.linspace(-3, 3, _RESOLUTION)
    return (
        QUAD_BETA_1,
        QUAD_BETA_2,
        quad_beta_1_grid,
        quad_beta_2_grid,
        quad_x,
        quad_y,
    )


@app.cell(hide_code=True)
def quadratic_learning_controls():
    quad_n_samples = mo.ui.slider(steps=[0, 2, 4, 8, 16, 32, 64], value=0, label="Data points used", show_value=True)
    return (quad_n_samples,)


@app.cell
def quadratic_learning(
    QUAD_BETA_1,
    QUAD_BETA_2,
    quad_beta_1_grid,
    quad_beta_2_grid,
    quad_n_samples,
    quad_x,
    quad_y,
):
    # Vary the sample size to show how the posterior adapts to more and more data
    utils.simulate_2_parameter_bayesian_learning(
        x_obs=quad_x[:quad_n_samples.value],
        y_obs=quad_y[:quad_n_samples.value],
        param_a_grid=quad_beta_1_grid,
        param_b_grid=quad_beta_2_grid,
        true_param_a=QUAD_BETA_1,
        true_param_b=QUAD_BETA_2,
        model_func=quadratic_polynomial_model,
        posterior_func=quadratic_polynomial_regression_posterior,
        param_labels=['$\\beta_1$', '$\\beta_2$'],
        data_range_x=(-2, 3), data_range_y=(-3, 3)
    )
    mo.vstack([quad_n_samples, plt.gcf()])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fitting N-th Order Polynomials to Height / Weight Data
    """)
    return


@app.function
def fit_nth_order_polynomial(data, n=3):
    with pm.Model() as model:
        # Data
        H_std = pm.Data("H", utils.standardize(data.height.values), dims="obs_ids")

        # Priors
        sigma = pm.Uniform("sigma", 0, 10)
        alpha = pm.Normal("alpha", 0, 60)
        betas = []
        for ii in range(n):
            betas.append(pm.Normal(f"beta_{ii+1}", 0, 5))

        # Likelihood
        mu = alpha
        for ii, beta in enumerate(betas):
            mu += beta * H_std ** (ii + 1)

        mu = pm.Deterministic('mu', mu)

        pm.Normal("W_obs", mu, sigma, observed=data.weight.values, dims="obs_ids")

        inference = cached_sample(target_accept=.95)

    return model, inference


@app.cell
def polynomial_fits(HOWELL):
    polynomial_models = {}
    for _order in [2, 4, 6]:
        polynomial_models[_order] = fit_nth_order_polynomial(HOWELL, n=_order)
    return (polynomial_models,)


@app.cell(hide_code=True)
def polynomial_order_controls(polynomial_models):
    polynomial_order = mo.ui.radio({str(_order): _order for _order in polynomial_models}, value="2", label="Polynomial order", inline=True)
    return (polynomial_order,)


@app.function
def plot_polynomial_model_posterior_predictive(model, inference, data, order):

    # Sample the posterior predictive for regions outside of the training data
    prediction_heights = np.linspace(30, 200, 100)
    with model:
        std_heights = (prediction_heights - data.height.mean()) / data.height.std(ddof=0)  # match utils.standardize
        pm.set_data({"H": std_heights})
        ppd = pm.sample_posterior_predictive(inference, extend_inferencedata=False)

    plt.subplots(figsize=(4, 4))
    plt.scatter(data.height, data.weight)
    utils.plot_hdi(prediction_heights, ppd.posterior_predictive['W_obs'], color='k', fill_kwargs=dict(alpha=.1))

    # Hack: use 5% HDI as proxy for posterior predictive mean
    utils.plot_hdi(prediction_heights, ppd.posterior_predictive['W_obs'], hdi_prob=.05, color='k', fill_kwargs=dict(alpha=1))
    terms = "+".join([f"\\beta_{o} H_i^{o}" for o in range(1, order + 1)])
    plt.title(f"$\\mu_i = \\alpha + {terms}$")


@app.cell(hide_code=True)
def polynomial_order_explorer(HOWELL, polynomial_models, polynomial_order):
    _model, _inference = polynomial_models[polynomial_order.value]
    plot_polynomial_model_posterior_predictive(_model, _inference, HOWELL, polynomial_order.value)
    mo.vstack([polynomial_order, plt.gcf()])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Thinking vs Fitting
    - Linear models can fit anything (geocentric)
    - Better off to use domain expertise to build more biologically plausible model e.g.

    $$
    \begin{align*}
    \log W_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta (H - \bar H)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Splines
    - "Wiggles" built from locally-fit smooth functions
    - good alternative when you have little domain knowledge of the problem

    $$
    \mu_i = \alpha_0 + \alpha_1 B_1 + \alpha_2 B_2 + ... \alpha_K B_K
    $$

    where $B$ is a set of $K$ local kernel functions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Cherry Blossom Blooms
    """)
    return


@app.cell
def load_blossoms():
    BLOSSOMS = utils.load_data("cherry_blossoms")
    BLOSSOMS.dropna(subset=["doy"], inplace=True)
    BLOSSOMS.head()
    return (BLOSSOMS,)


@app.cell(hide_code=True)
def blossoms_scatter(BLOSSOMS):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    plt.scatter(
        x=BLOSSOMS.year,
        y=BLOSSOMS.doy
    )
    plt.xlabel("year")
    plt.ylabel("day of first blossom")
    _fig
    return


@app.function
def generate_spline_basis(data, xdim='year', degree=2, n_bases=10):
    n_knots = n_bases - 1
    knots = np.quantile(data[xdim], np.linspace(0, 1, n_knots))
    return dmatrix(
        f"bs({xdim}, knots=knots, degree={degree}, include_intercept=True) - 1",
        {xdim: data[xdim], "knots": knots[1:-1]}
    )


@app.cell(hide_code=True)
def demo_spline_basis():
    # 4 spline basis for demo
    _demo_data = pd.DataFrame({"x": np.arange(100)})
    _demo_basis = generate_spline_basis(_demo_data, "x", n_bases=4)

    _fig, _axs = plt.subplots(2, 1, figsize=(10, 8))
    plt.sca(_axs[0])
    for _bi in range(_demo_basis.shape[1]):
        plt.plot(_demo_data.x, _demo_basis[:, _bi], color=f"C{_bi}", label=f'Basis{_bi+1}')
    plt.legend()
    plt.title("Demo Spline Basis")

    # Arbitrarily-set weights for demo
    _basis_weights = [1, 2, -1, 0]

    plt.sca(_axs[1])
    _resulting_curve = np.zeros_like(_demo_data.x)
    for _bi in range(_demo_basis.shape[1]):
        _weighted_basis = _demo_basis[:, _bi] * _basis_weights[_bi]
        _resulting_curve = _resulting_curve + _weighted_basis
        plt.plot(_demo_data.x, _weighted_basis, color=f"C{_bi}", label=f'{_basis_weights[_bi]} x Basis {_bi+1}')
    plt.plot(_demo_data.x, _resulting_curve, label='Sum', color='k', linewidth=4)
    plt.xlabel("x")
    plt.legend()
    plt.title("Sum of Weighted Bases")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Spline basis playground
    Change the number of bases, the degree of each basis function, and the weight of each basis. The black curve is the weighted sum of the bases. Each weight only bends the curve where its basis is nonzero.
    """)
    return


@app.cell(hide_code=True)
def spline_basis_controls():
    spline_n_bases = mo.ui.slider(4, 12, value=4, label="`n_bases`", show_value=True)
    spline_degree = mo.ui.slider(1, 3, value=2, label="Degree", show_value=True)
    return spline_degree, spline_n_bases


@app.cell(hide_code=True)
def spline_weight_controls(spline_degree, spline_n_bases):
    spline_demo_data = pd.DataFrame({"x": np.arange(100)})
    spline_demo_basis = np.asarray(
        generate_spline_basis(spline_demo_data, "x", degree=spline_degree.value, n_bases=spline_n_bases.value)
    )
    _default_weights = [1, 2, -1, 0]
    spline_weights = mo.ui.array([
        mo.ui.slider(
            -3, 3, step=0.25,
            value=_default_weights[_bi] if _bi < len(_default_weights) else 0,
            orientation="vertical", label=f"B{_bi + 1}", show_value=True,
        )
        for _bi in range(spline_demo_basis.shape[1])
    ])
    return spline_demo_basis, spline_demo_data, spline_weights


@app.cell(hide_code=True)
def spline_basis_explorer(
    spline_degree,
    spline_demo_basis,
    spline_demo_data,
    spline_n_bases,
    spline_weights,
):
    _n_bases = spline_demo_basis.shape[1]
    _weights = np.array(spline_weights.value)
    _fig, _axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plt.sca(_axs[0])
    for _bi in range(_n_bases):
        plt.plot(spline_demo_data.x, spline_demo_basis[:, _bi], color=f"C{_bi}")
    plt.title(f"{_n_bases} spline bases of degree {spline_degree.value}")

    plt.sca(_axs[1])
    for _bi in range(_n_bases):
        plt.plot(spline_demo_data.x, _weights[_bi] * spline_demo_basis[:, _bi], color=f"C{_bi}")
    plt.plot(spline_demo_data.x, spline_demo_basis @ _weights, label='Sum', color='k', linewidth=4)
    plt.xlabel("x")
    plt.legend()
    plt.title("Sum of Weighted Bases")
    mo.vstack([
        mo.hstack([spline_n_bases, spline_degree], justify="start", gap=2),
        mo.hstack(list(spline_weights), justify="start", wrap=True),
        _fig,
    ])
    return


@app.cell
def blossom_spline_basis(BLOSSOMS):
    # 10 spline basis for modeling blossoms data
    blossom_basis = generate_spline_basis(BLOSSOMS)

    _fig, _ax = plt.subplots(figsize=(10, 3))

    for _bi in range(blossom_basis.shape[1]):
        _ax.plot(BLOSSOMS.year, blossom_basis[:, _bi], color=f"C{_bi}", label=f'Basis{_bi+1}')
    plt.legend()
    plt.title("Basis functions, B for the Cherry Blossoms Dataset")
    _fig
    return (blossom_basis,)


@app.cell
def spline_prior_samples(BLOSSOMS, blossom_basis):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _n_samples = 5
    _spline_prior_sigma = 10
    _spline_prior = stats.norm(0, _spline_prior_sigma)
    for _s in range(_n_samples):
        _sample = 0
        for _bi in range(blossom_basis.shape[1]):
            _sample += _spline_prior.rvs() * blossom_basis[:, _bi]
        _ax.plot(BLOSSOMS.year, _sample)
    plt.title("Prior,  $\\alpha \\sim Normal(0, 10)$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore the spline prior
    Each prior draw weights the cherry blossom bases with $\alpha_k \sim \text{Normal}(0, \sigma)$. A larger $\sigma$ allows larger wiggles.
    """)
    return


@app.cell(hide_code=True)
def spline_prior_controls():
    spline_prior_sigma = mo.ui.slider(1, 50, value=10, label=r"Prior $\sigma$", show_value=True)
    spline_prior_n_draws = mo.ui.slider(1, 20, value=5, label="Draws", show_value=True)
    return spline_prior_n_draws, spline_prior_sigma


@app.cell(hide_code=True)
def spline_prior_explorer(
    BLOSSOMS,
    blossom_basis,
    spline_prior_n_draws,
    spline_prior_sigma,
):
    _rng = np.random.default_rng(4)
    _weights = _rng.normal(0, spline_prior_sigma.value, size=(blossom_basis.shape[1], spline_prior_n_draws.value))
    _samples = np.asarray(blossom_basis) @ _weights
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.plot(BLOSSOMS.year, _samples)
    plt.ylim([-100, 100])
    plt.title(f"Prior,  $\\alpha \\sim Normal(0, {spline_prior_sigma.value})$")
    mo.vstack([
        mo.hstack([spline_prior_sigma, spline_prior_n_draws], justify="start", gap=2),
        _fig,
    ])
    return


@app.function
def fit_spline_model(data, xdim, ydim, n_bases=10):
    basis_set = generate_spline_basis(data, xdim, n_bases=n_bases).base
    with pm.Model() as spline_model:

        # Priors
        sigma = pm.Exponential("sigma", 1)
        alpha = pm.Normal("alpha", data[ydim].mean(), data[ydim].std())
        beta = pm.Normal("beta", 0, 25, shape=n_bases)

        # Likelihood
        mu = pm.Deterministic("mu", alpha + pm.math.dot(basis_set, beta.T))
        pm.Normal("ydim_obs", mu, sigma, observed=data[ydim].values)

        spline_inference = cached_sample(target_accept=.95)

    _, ax = plt.subplots(figsize=(10, 3))
    plt.scatter(
        x=data[xdim],
        y=data[ydim]
    )
    utils.plot_hdi(
        data[xdim],
        spline_inference.posterior["mu"],
        color='k',
        hdi_prob=.89,
        fill_kwargs=dict(alpha=.3, label='Posterior Mean')
    )
    plt.legend(loc='lower right')
    plt.xlabel(f"{xdim}")
    plt.ylabel(f"{ydim}")

    return spline_model, spline_inference, basis_set


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Cherry Blossoms Model
    """)
    return


@app.cell
def blossom_spline_fit(BLOSSOMS):
    blossom_model, blossom_inference, blossom_model_basis = fit_spline_model(BLOSSOMS, 'year', 'doy', n_bases=20)
    plt.gcf()
    return blossom_inference, blossom_model_basis


@app.cell
def blossom_spline_summary(blossom_inference):
    az.summary(blossom_inference, var_names=['alpha', 'beta'])
    return


@app.cell(hide_code=True)
def blossom_weighted_bases(BLOSSOMS, blossom_inference, blossom_model_basis):
    _beta_spline_mean = blossom_inference.posterior['beta'].mean(dim=('chain', 'draw')).values
    _resulting_fit = np.zeros_like(BLOSSOMS.year)
    _fig, _ax = plt.subplots()
    for _bi, _beta in enumerate(_beta_spline_mean):
        _weighted_basis = _beta * blossom_model_basis[:, _bi]
        plt.plot(BLOSSOMS.year, _weighted_basis, color=f"C{_bi}")
        _resulting_fit = _resulting_fit + _weighted_basis
    plt.plot(BLOSSOMS.year, _resulting_fit, label='Resulting Fit (excluding intercept term)', color='k', linewidth=4)
    plt.legend()
    plt.title("weighted spline bases\nfit to Cherry Blossoms dataset")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Choose the number of bases
    Refit the cherry blossom model with a different number of bases. A new choice samples the model first (this can take a minute); later runs load it from the cache.
    """)
    return


@app.cell(hide_code=True)
def blossom_refit_controls():
    blossom_n_bases = mo.ui.slider(steps=[5, 10, 15, 20, 30, 40], value=20, label="`n_bases`", show_value=True)
    blossom_fit_button = mo.ui.run_button(label="Fit model")
    return blossom_fit_button, blossom_n_bases


@app.cell
def blossom_refit(BLOSSOMS, blossom_fit_button, blossom_n_bases):
    _controls = mo.hstack([blossom_n_bases, blossom_fit_button], justify="start", gap=2)
    mo.stop(not blossom_fit_button.value, _controls)

    fit_spline_model(BLOSSOMS, 'year', 'doy', n_bases=blossom_n_bases.value)
    mo.vstack([_controls, plt.gcf()])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Height as a function of age
    """)
    return


@app.cell
def howell_height_spline(HOWELL):
    # Fit spline model to Howell height data
    fit_spline_model(HOWELL, 'age', 'height', n_bases=10)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Weight as a function of age
    """)
    return


@app.cell
def howell_weight_spline(HOWELL):
    # While we're at it, let's fit spline model to Howell weight data
    fit_spline_model(HOWELL, 'age', 'weight', n_bases=10)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Full Luxury Bayes
    - Program the whole generative shebang into a single model
    - Includes multiple submodels (likelihoods)

    ### Why would we do this?
    - Can run simulations (interventions) from full generative model to look at causal estimates.
    """)
    return


@app.cell(hide_code=True)
def flb_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("H", "W"),
            ("S", "H"),
            ("S", "W")
        ],
        graph_direction="LR"
    )
    return


@app.cell
def flb_fit(ADULT_HOWELL):
    _sex_indicator, _sex = pd.factorize(np.array(["M" if s else "F" for s in ADULT_HOWELL["male"].values]))
    with pm.Model(coords={"sex": _sex}) as flb_model:

        # Data
        _S = pm.Data("S", _sex_indicator, dims="obs_ids")
        _H = pm.Data("H", ADULT_HOWELL.height.values, dims="obs_ids")
        _Hbar = pm.Data("Hbar", ADULT_HOWELL.height.mean())

        # Height Model
        _tau = pm.Uniform("tau", 0, 10)
        _h = pm.Normal("h", 160, 10, dims='sex')
        _nu = _h[_S]
        pm.Normal("H_obs", _nu, _tau, observed=ADULT_HOWELL.height.values, dims="obs_ids")

        # Weight Model
        _alpha = pm.Normal("alpha", 60, 10, dims="sex")
        _beta = pm.Uniform("beta", 0, 1, dims="sex")
        _sigma = pm.Uniform("sigma", 0, 10)
        _mu = _alpha[_S] + _beta[_S] * (_H - _Hbar)
        pm.Normal("W_obs", _mu, _sigma, observed=ADULT_HOWELL.weight.values, dims="obs_ids")

        flb_inference = cached_sample()
    return (flb_inference,)


@app.cell
def flb_summary(flb_inference):
    az.summary(flb_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulate interventions
    """)
    return


@app.cell
def flb_contrast_plotters(ADULT_HOWELL, flb_inference):
    def plot_posterior_mean_contrast(contrast_type='weight'):
        Hbar = ADULT_HOWELL.height.mean()
        means = az.summary(flb_inference)['mean']


        H_F = stats.norm(means['h[F]'], means['tau']).rvs(1000)
        H_M = stats.norm(means['h[M]'], means['tau']).rvs(1000)

        W_F = stats.norm(means['alpha[F]'] + means['beta[F]'] * (H_F - Hbar), means['sigma']).rvs(1000)
        W_M = stats.norm(means['alpha[M]'] + means['beta[M]'] * (H_M - Hbar), means['sigma']).rvs(1000)
        contrast = H_M - H_F if contrast_type == 'height' else W_M - W_F

        utils.plot_dist(contrast, color='k')
        plt.xlabel(f"Posterior mean {contrast_type} contrast")


    def plot_causal_intervention_contrast(contrast_type):
        Hbar = ADULT_HOWELL.height.mean()

        # Causal Intervention
        F_posterior = flb_inference.posterior.sel(sex="F")
        M_posterior = flb_inference.posterior.sel(sex="M")

        H_F = stats.norm.rvs(F_posterior['h'], F_posterior['tau'])
        H_M = stats.norm.rvs(M_posterior['h'], M_posterior['tau'])

        W_F = stats.norm.rvs(F_posterior['alpha'] + F_posterior['beta'] * (H_F - Hbar), F_posterior['sigma'])
        W_M = stats.norm.rvs(M_posterior['alpha'] + M_posterior['beta'] * (H_M - Hbar), M_posterior['sigma'])

        contrast = H_M - H_F if contrast_type == 'height' else W_M - W_F

        pos_prob = 100 * np.sum(contrast >= 0) / np.prod(contrast.shape)
        neg_prob = 100 - pos_prob

        kde_ax = utils.plot_dist(contrast, color='k', plot_kwargs=dict(linewidth=3), bw=.5)

        # Shade underneath posterior predictive contrast
        kde_x, kde_y = kde_ax.get_lines()[0].get_data()

        # Proportion of PPD contrast below zero
        neg_idx = kde_x < 0
        plt.fill_between(
            x=kde_x[neg_idx],
            y1=np.zeros(sum(neg_idx)),
            y2=kde_y[neg_idx],
            color='C0',
            label=f"{neg_prob:1.0f}%"
        )

        pos_idx = kde_x >= 0
        plt.fill_between(
            x=kde_x[pos_idx],
            y1=np.zeros(sum(pos_idx)),
            y2=kde_y[pos_idx],
            color='C1',
            label=f"{pos_prob:1.0f}%"
        )

        plt.axvline(0, color='k')
        plt.legend(title='Proportion M>F')
        plt.xlabel(f"Posterior {contrast_type} contrast")


    def plot_flb_contrasts(contrast_type='weight', figsize=(8, 4)):
        _, axs = plt.subplots(1, 2, figsize=figsize)
        plt.sca(axs[0])
        plot_posterior_mean_contrast(contrast_type)

        plt.sca(axs[1])
        plot_causal_intervention_contrast(contrast_type)

    return (plot_flb_contrasts,)


@app.cell(hide_code=True)
def flb_weight_contrast(plot_flb_contrasts):
    plot_flb_contrasts('weight')
    plt.gcf()
    return


@app.cell(hide_code=True)
def flb_height_contrast(plot_flb_contrasts):
    plot_flb_contrasts('height')
    plt.gcf()
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
