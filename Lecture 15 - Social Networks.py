import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from itertools import combinations

    import marimo as mo
    import networkx as nx
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
    # [Lecture 15 - Social Networks](https://youtu.be/hnYhJzYAQ60?si=Y9bnH_DopygCafIr)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # What Motivates Sharing?

    ## Koster & Leckie (2014) Arang Dak dataset
    - year of food transfers between 25 households
    - 300 dyads, i.e. $\binom{25}{2} = 300$
    - 2871 observations of food transfers ("gifts")

    ### Scientific Questions: Estimand(s)
    - How much sharing is explained by **reciprocity?**
    - How much by **generalized giving?**
    """)
    return


@app.cell
def _():
    SHARING = utils.load_data('KosterLeckie')
    SHARING.head()
    return (SHARING,)


@app.cell(hide_code=True)
def _(SHARING):
    utils.plot_scatter(
        SHARING.giftsAB,
        SHARING.giftsBA,
        color='C0',
        label='dyad',
        alpha=.5
    )
    plt.plot((0, 120), (0, 120), linestyle='--', color='gray')
    plt.axis('square')
    plt.xlabel("A gives B")
    plt.ylabel("B gives A")
    plt.xlim([-5, 120])
    plt.ylim([-5, 120])
    plt.title("What not to do with this data")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Improve analysis with a causal graph
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("H_A", "G_AB"),
            ("H_B", "G_AB"),
            ("H_A", "T_AB"),
            ("H_B", "T_AB"),
            ("H_A", "T_BA"),
            ("H_B", "T_BA"),
            ("T_AB", "G_AB"),
            ("T_BA", "G_AB"),
        ],
        node_props={
            "H_A": {"label": "household A, H_A"},
            "H_B": {"label": "household B, H_B"},
            "G_AB": {"label": "A gives to B, G_AB"},
            "T_AB": {"label": "Social tie from A to B, T_AB", "style": "dashed"},
            "T_BA": {"label": "Social tie from B to A, T_BA", "style": "dashed"},
            "unobserved": {"style": "dashed"},
        }

    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Social Network Analysis
    - $T_{AB}$ and $T_{BA}$ are not observable
    - **Social network** is Pattern of direct exchange
    - Social network is **an abstraction, not data**
    - what's a principled approach?
      - bad approach -> Null network analysis
        - no correct way to permute
        - we care about causality, not testing a null hypothesis


    ## Drawing the Social Owl 🦉
    1. Estimand(s): Reciprocity & what explains it
    2. Generative model
    3. Statistical model
    4. Analyze data

    We'll loop between 2 and 3 often as we build the complexity of our model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1) Estimand
    ### Starting Simpler, ignoring household effects for now
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("H_A", "G_AB"),
            ("H_B", "G_AB"),
            ("H_A", "T_AB"),
            ("H_B", "T_AB"),
            ("H_A", "T_BA"),
            ("H_B", "T_BA"),
            ("T_AB", "G_AB"),
            ("T_BA", "G_AB"),
        ],
        node_props={
            "H_A": {"color": "blue"},
            "H_B": {"color": "blue"},
            "T_AB": {"style": "dashed", "color": "red"},
            "T_BA": {"style": "dashed", "color": "red"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("T_AB", "G_AB"): {"color": "red"},
            ("T_BA", "G_AB"): {"color": "red"},
            ("H_A", "G_AB"): {"color": "blue"},
            ("H_B", "G_AB"): {"color": "blue", "label": " backdoor\npaths", "fontcolor": "blue"},
            ("H_A", "T_AB"): {"color": "blue"},
            ("H_A", "T_BA"): {"color": "blue"},
            ("H_B", "T_AB"): {"color": "blue"},
            ("H_B", "T_BA"): {"color": "blue"},
        }

    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    At first, we'll ignore the backdoor paths to get a good flow, and get the model running, then add them in later
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2) Generative Model
    ### Simulating a Social Network
    """)
    return


@app.cell
def _():
    np.random.seed(123)
    N = 25
    _dyads = list(combinations(np.arange(N), 2))

    # convert to numpy for np.where
    DYADS = np.array(_dyads)
    N_DYADS = len(DYADS)

    print(f"N dyads: {N_DYADS}")
    print(_dyads[:91])

    # Simulate "friendship", in which ties are shared
    _P_FRIENDSHIP = 0.1
    FRIENDSHIP = stats.bernoulli(p=_P_FRIENDSHIP).rvs(size=N_DYADS)

    # Simulate directed ties. Note: there can be ties that are not reciprocal
    _ALPHA = -3. # base rate has low probability
    _BASE_TIE_PROBABILITY = utils.invlogit(_ALPHA)

    print(f"\nBase non-friend social tie probability: {_BASE_TIE_PROBABILITY:.2}")

    def _get_dyad_index(source, target):
        # dyads are symmetric, but ordered by definition,
        # so ensure valid lookup by sorting
        ii, jj = sorted([source, target])
        return np.where((DYADS[:, 0] == ii) & (DYADS[:, 1] == jj))[0][0]

    # Simulate gift-giving
    TIES = np.zeros((N, N)).astype(int)
    for _source in range(N):
        for _target in range(N):
            if _source != _target:
                _dyad_index = _get_dyad_index(_source, _target)

                # Sample directed edge -- friends always share ties,
                # but there's also a base rate of sharing ties w/o friendship
                _is_friend = FRIENDSHIP[_dyad_index]
                _p_tie = _is_friend + (1 - _is_friend) * _BASE_TIE_PROBABILITY
                TIES[_source, _target] = stats.bernoulli(_p_tie).rvs()
    return DYADS, FRIENDSHIP, N, N_DYADS, TIES


@app.cell(hide_code=True)
def _(TIES):
    plt.matshow(TIES, cmap='gray')
    plt.ylabel("Household A")
    plt.xlabel("Household B")
    plt.title("Simulated Social Ties\nAdjacency Matrix")
    plt.gcf()
    return


@app.cell
def _(TIES):
    TIES_LAYOUT_POSITION = utils.plot_graph(TIES)
    plt.title("Simulated Social Ties Network")
    plt.gcf()
    return (TIES_LAYOUT_POSITION,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulate Gift-giving from social net
    """)
    return


@app.cell
def _(DYADS, N_DYADS, TIES):
    np.random.seed(123)
    _giftsAB = np.zeros(N_DYADS)
    _giftsBA = np.zeros(N_DYADS)
    _lam = np.log([0.5, 2])

    for _ii, (_A, _B) in enumerate(DYADS):
        _lambdaAB = np.exp(_lam[TIES[_A, _B]])
        _giftsAB[_ii] = stats.poisson(mu=_lambdaAB).rvs()

        _lambdaBA = np.exp(_lam[TIES[_B, _A]])
        _giftsBA[_ii] = stats.poisson(mu=_lambdaBA).rvs()

    ## Put simulation into a dataframe for fitting function
    simulated_gifts = pd.DataFrame({
        'giftsAB': _giftsAB.astype(int),
        'giftsBA': _giftsBA.astype(int),
        'did': np.arange(N_DYADS).astype(int)
    })
    simulated_gifts
    return (simulated_gifts,)


@app.cell(hide_code=True)
def _(simulated_gifts):
    plt.hist(simulated_gifts.giftsAB, bins=simulated_gifts.giftsAB.max(), width=.25)
    plt.title("Gifting from A to B")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(simulated_gifts):
    plt.hist(simulated_gifts.giftsBA, bins=simulated_gifts.giftsBA.max(), width=.25)
    plt.title("Gifting from B to A")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3) Statistical Model

    #### Likelihood
    $$
    \begin{align*}
        G_{AB} &\sim \text{Poisson}(\lambda_{AB}) \\
        G_{BA} &\sim \text{Poisson}(\lambda_{BA}) \\
        \log(\lambda_{AB}) &= \alpha + T_{AB} \\
        \log(\lambda_{BA}) &= \alpha + T_{BA} \\
    \end{align*}
    $$

    #### Global gift-giving prior
    $$
    \alpha \sim \text{Normal}(0, 1) \\
    $$

    #### Correlated Social Ties prior
    $$
    \begin{align*}
        \begin{pmatrix}
        T_{AB} \\
        T_{BA}
        \end{pmatrix} &= \text{MVNormal}
        \left(
            \begin{bmatrix}
                0 \\
                0
            \end{bmatrix},
            \begin{bmatrix}
                \sigma^2 & \rho \sigma^2 \\
                \rho \sigma^2 & \sigma^2
            \end{bmatrix}
        \right) \\
        \rho &\sim \text{LKJCorr}(\eta) \\
        \sigma &\sim \text{Exponential}(1)
    \end{align*}
    $$

    - Symmetric Likelihood
    - Global intercept, but tie-specific offset
    - We model the (symmetric) correlation between social $T_{AB}, T_{BA}$
    - Priors give us **partial pooling**, allowing us to share information from ties that have a lot of activity to those ties that have less activity
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fitting the social ties model
    ##### Notes
    - We use `LKJCholeskyCov` instead of `LKJCorr` for numerical stability/efficiency
    """)
    return


@app.function
def fit_social_ties_model(data, eta=4):
    n_dyads = len(data)

    # ensure zero-indexed IDs
    dyad_id = data.did.values.astype(int)
    if np.min(dyad_id) == 1:
        dyad_id -= 1

    n_correlated_features = 2

    with pm.Model() as model:

        # Single, global alpha
        alpha = pm.Normal("alpha", 0, 1)

        # Single, global sigma
        sigma = pm.Exponential.dist(1)
        chol, corr, stds = pm.LKJCholeskyCov(
            "rho",
            eta=eta,
            n=n_correlated_features,
            sd_dist=sigma
        )

        # Record quantities for reporting
        pm.Deterministic("corrcoef_T", corr[0, 1])
        pm.Deterministic("std_T", stds)

        z = pm.Normal("z", 0, 1, shape=(n_dyads, n_correlated_features))
        T = pm.Deterministic("T", chol.dot(z.T).T)

        # Likelihood(s)
        lambda_AB = pm.Deterministic("lambda_AB", pm.math.exp(alpha + T[dyad_id, 0]))
        lambda_BA = pm.Deterministic("lambda_BA", pm.math.exp(alpha + T[dyad_id, 1]))

        G_AB = pm.Poisson("G_AB", lambda_AB, observed=data.giftsAB)
        G_BA = pm.Poisson("G_BA", lambda_BA, observed=data.giftsBA)

        inference = cached_sample(target_accept=.9)

    return model, inference


@app.cell
def _(simulated_gifts):
    simulated_social_ties_model, simulated_social_ties_inference = fit_social_ties_model(simulated_gifts)
    return (simulated_social_ties_inference,)


@app.cell
def _(simulated_social_ties_inference):
    az.summary(simulated_social_ties_inference, var_names='rho_corr')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posterior Ties
    """)
    return


@app.cell
def _(FRIENDSHIP):
    def plot_posterior_reciprocity(inference, ax=None):
        utils.plot_dist(inference.posterior['corrcoef_T'], ax=ax);
        posterior_mean = inference.posterior['corrcoef_T'].mean().values
        plt.axvline(posterior_mean, label=f'posterior mean={posterior_mean:0.2f}', color='k', linestyle='--')
        plt.xlim([-1, 1])
        plt.xlabel("correlation amongst dyads")
        plt.ylabel("density")
        plt.legend()


    def plot_posterior_household_ties(inference, color_friends=False, ax=None):
        T = inference.posterior.mean(dim=('chain', 'draw'))['T']
        T_AB = T[:, 0]
        T_BA = T[:, 1]

        if color_friends:
            colors = ["black", "C4"]
            labels = [None, 'friends']
            for is_friends in [0, 1]:
                mask = FRIENDSHIP == is_friends
                utils.plot_scatter(
                    T_AB[mask],
                    T_BA[mask],
                    color=colors[is_friends],
                    alpha=.5,
                    label=labels[is_friends]
                )
        else:
            utils.plot_scatter(T_AB, T_BA, color="C0", label="dyadic ties")

        plt.xlabel("$T_{AB}$")
        plt.ylabel("$T_{BA}$");
        plt.legend();

    return plot_posterior_household_ties, plot_posterior_reciprocity


@app.cell(hide_code=True)
def _(
    plot_posterior_household_ties,
    plot_posterior_reciprocity,
    simulated_social_ties_inference,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))

    plt.sca(_axs[0])
    plot_posterior_reciprocity(simulated_social_ties_inference, ax=_axs[0])

    plt.sca(_axs[1])
    plot_posterior_household_ties(simulated_social_ties_inference, color_friends=True, ax=_axs[1])
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4) Analyze Data
    Run the model on the real data samples
    """)
    return


@app.cell
def _(SHARING):
    social_ties_model, social_ties_inference = fit_social_ties_model(SHARING)
    return (social_ties_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posterior correlation
    """)
    return


@app.cell
def _(social_ties_inference):
    az.summary(social_ties_inference, var_names='rho_corr')
    return


@app.cell(hide_code=True)
def _(plot_posterior_reciprocity, social_ties_inference):
    plot_posterior_reciprocity(social_ties_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introducing Household Giving/Receiving Confounds
    ## 2) Generative Model
    ### Simulate Including Unmeasured Household Wealth
    We use the same social ties network, but augment gift-giving behavior based on (unmeasured) household wealth:
    - Wealthier households give more
    - Poorer households recieve more
    """)
    return


@app.cell
def _(DYADS, N, N_DYADS, TIES):
    np.random.seed(123)
    _giftsAB = np.zeros(N_DYADS)
    _giftsBA = np.zeros(N_DYADS)

    _LAMBDA = np.log([0.5, 2])
    BETA_WG = 0.5  #  Effect of (standardized) household wealth on giving -> the wealthy give more
    BETA_WR = -1  # Effect of household wealth on receiving -> weathy recieve less

    WEALTH = stats.norm.rvs(size=N)  # standardized wealth

    for _ii, (_A, _B) in enumerate(DYADS):
        _lambdaAB = np.exp(_LAMBDA[TIES[_A, _B]] + BETA_WG * WEALTH[_A] + BETA_WR * WEALTH[_B])
        _giftsAB[_ii] = stats.poisson(mu=_lambdaAB).rvs()

        _lambdaBA = np.exp(_LAMBDA[TIES[_B, _A]] + BETA_WG * WEALTH[_B] + BETA_WR * WEALTH[_A])
        _giftsBA[_ii] = stats.poisson(mu=_lambdaBA).rvs()

    ## Put simulation into a dataframe for fitting function
    simulated_wealth_gifts = pd.DataFrame({
        'giftsAB': _giftsAB.astype(int),
        'giftsBA': _giftsBA.astype(int),
        'hidA': DYADS[:, 0],
        'hidB': DYADS[:, 1],
        'did': np.arange(N_DYADS).astype(int)
    })
    simulated_wealth_gifts
    return BETA_WG, BETA_WR, WEALTH, simulated_wealth_gifts


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3) Statistical Model

    - $G_{A,B}$ - househould $A,B$'s generalized _giving_
    - $R_{A,B}$ - househould $A,B$'s generalized _receiving_
    - Model Giving and Receiving covariance vian $\text{MVNormal}$

    #### Likelihood
    $$
    \begin{align*}
        G_{AB} &\sim \text{Poisson}(\lambda_{AB}) \\
        G_{BA} &\sim \text{Poisson}(\lambda_{BA}) \\
        \log(\lambda_{AB}) &= \alpha + T_{AB} + G_{A} + R_{B} \\
        \log(\lambda_{BA}) &= \alpha + T_{BA} + G_{B} + R_{A}
    \end{align*}
    $$

    #### Global gift-giving prior
    $$
    \alpha \sim \text{Normal}(0, 1)
    $$

    #### Correlated Social Ties prior
    $$
    \begin{align*}
        \begin{pmatrix}
        T_{AB} \\
        T_{BA}
        \end{pmatrix} &= \text{MVNormal}
        \left(
            \begin{bmatrix}
                0 \\
                0
            \end{bmatrix},
            \begin{bmatrix}
                \sigma^2 & \rho \sigma^2 \\
                \rho \sigma^2 & \sigma^2
            \end{bmatrix}
        \right) \\
        \rho &\sim \text{LKJCorr}(\eta) \\
        \sigma &\sim \text{Exponential}(1)
    \end{align*}
    $$

    #### Correlated Giving/Recieving prior
    $$
    \begin{align*}
        \begin{pmatrix}
            G_{A,B} \\
            R_{A,B}
        \end{pmatrix} &= \text{MVNormal}
        \left(
            \begin{bmatrix}
                0 \\
                0
            \end{bmatrix},
            \textbf{R}_{GR}, \textbf{S}_{GR}
        \right) \\
        \textbf{R}_{GR} &\sim \text{LKJCorr}(\eta) \\
        \textbf{S}_{GR} &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Wealth Gifting model on simulated data (validation)
    """)
    return


@app.function
def fit_giving_receiving_model(data, eta=2):
    n_dyads = len(data)
    n_correlated_features = 2

    dyad_id = data.did.values.astype(int)
    household_A_id = data.hidA.values.astype(int)
    household_B_id = data.hidB.values.astype(int)

    # Data are 1-indexed
    if np.min(dyad_id) == 1:
        dyad_id -= 1
        household_A_id -= 1
        household_B_id -= 1

    n_households = np.max([household_A_id, household_B_id]) + 1

    with pm.Model() as model:

        # single, global alpha
        alpha = pm.Normal("alpha", 0, 1)

        # Social ties interaction; shared sigma
        sigma_T = pm.Exponential.dist(1)
        chol_T, corr_T, std_T = pm.LKJCholeskyCov(
            "rho_T", eta=eta, n=n_correlated_features, sd_dist=sigma_T
        )
        z_T = pm.Normal("z_T", 0, 1, shape=(n_dyads, n_correlated_features))
        T = pm.Deterministic("T", chol_T.dot(z_T.T).T)

        # Giving-receiving interaction; full covariance
        sigma_GR = pm.Exponential.dist(1, shape=n_correlated_features)
        chol_GR, corr_GR, std_GR = pm.LKJCholeskyCov(
            "rho_GR", eta=eta, n=n_correlated_features, sd_dist=sigma_GR
        )
        z_GR = pm.Normal("z_GR", 0, 1, shape=(n_households, n_correlated_features))
        GR = pm.Deterministic("GR", chol_GR.dot(z_GR.T).T)

        lambda_AB = pm.Deterministic(
            "lambda_AB",
            pm.math.exp(alpha + T[dyad_id, 0] + GR[household_A_id, 0] + GR[household_B_id, 1])
        )
        lambda_BA = pm.Deterministic(
            "lambda_BA",
            pm.math.exp(alpha + T[dyad_id, 1] + GR[household_B_id, 0] + GR[household_A_id, 1])
        )

        # Record quantities for reporting
        pm.Deterministic("corrcoef_T", corr_T[0, 1])
        pm.Deterministic("std_T", std_T)
        pm.Deterministic("corrcoef_GR", corr_GR[0, 1])
        pm.Deterministic("std_GR", std_GR)

        G_AB = pm.Poisson("G_AB", lambda_AB, observed=data.giftsAB)
        G_BA = pm.Poisson("G_BA", lambda_BA, observed=data.giftsBA)

        inference = cached_sample(target_accept=.9)
        inference = pm.compute_log_likelihood(inference, extend_inferencedata=True)

    return model, inference


@app.cell
def _(simulated_wealth_gifts):
    simulated_gr_model, simulated_gr_inference = fit_giving_receiving_model(simulated_wealth_gifts)
    return (simulated_gr_inference,)


@app.cell
def _(simulated_gr_inference):
    az.summary(simulated_gr_inference, var_names=['corrcoef_T', 'corrcoef_GR'])
    return


@app.function
def plot_wealth_gifting_posterior(inference, data_range=None):

    posterior_alpha = inference.posterior['alpha']
    posterior_GR = inference.posterior['GR']
    log_posterior_giving = posterior_alpha + posterior_GR[:, :, :, 0]  # Household A
    log_posterior_receiving = posterior_alpha + posterior_GR[:, :, :, 1]  # Household B

    posterior_receiving = np.exp(log_posterior_giving.mean(dim=('chain', 'draw')))
    posterior_giving = np.exp(log_posterior_receiving.mean(dim=('chain', 'draw')))

    # Household giving/receiving
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(axs[0])
    utils.plot_scatter(
        posterior_receiving,
        posterior_giving,
        color='C0',
        label='households'
    )

    if data_range is None:
        axis_max = np.max(posterior_receiving.values.ravel()) * 1.05
        data_range = (0, axis_max)

    plt.xlabel("giving")
    plt.ylabel("receiving")
    plt.legend();

    plt.plot(data_range, data_range, color='k', linestyle='--')
    plt.xlim(data_range)
    plt.ylim(data_range)

    # Giving/receiving correlation distribution
    plt.sca(axs[1])
    utils.plot_dist(inference.posterior['corrcoef_GR'], color='C1', plot_kwargs={"linewidth": 3})
    cc_mean = inference.posterior['corrcoef_GR'].mean()
    plt.axvline(cc_mean, color='k', linestyle='--', label=f'posterior mean: {cc_mean:1.2}')
    plt.xlim([-1, 1])
    plt.legend()
    plt.ylabel("density")
    plt.xlabel("correlation, giving-receiving");


@app.function
def plot_dyadic_ties(inference):
    T = inference.posterior.mean(dim=('chain', 'draw'))['T']
    utils.plot_scatter(T[:, 0], T[:, 1], color='C0', label='dyadic ties')
    plt.xlabel("Household A")
    plt.ylabel("Household B")
    plt.legend()


@app.cell(hide_code=True)
def _(simulated_gr_inference):
    plot_wealth_gifting_posterior(simulated_gr_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(
    plot_posterior_household_ties,
    plot_posterior_reciprocity,
    simulated_gr_inference,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))

    plt.sca(_axs[0])
    plot_posterior_household_ties(simulated_gr_inference, color_friends=True)

    plt.sca(_axs[1])
    plot_posterior_reciprocity(simulated_gr_inference)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit on real data
    """)
    return


@app.cell
def _(SHARING):
    gr_model, gr_inference = fit_giving_receiving_model(SHARING)
    return (gr_inference,)


@app.cell
def _(gr_inference):
    az.summary(gr_inference, var_names=['corrcoef_T', 'corrcoef_GR'])
    return


@app.cell(hide_code=True)
def _(gr_inference):
    plot_wealth_gifting_posterior(gr_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(gr_inference, plot_posterior_household_ties, plot_posterior_reciprocity):
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))

    plt.sca(_axs[0])
    plot_posterior_household_ties(gr_inference)

    plt.sca(_axs[1])
    plot_posterior_reciprocity(gr_inference)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Plot the posterior mean social graph
    """)
    return


@app.cell
def _(DYADS, TIES, TIES_LAYOUT_POSITION):
    def plot_posterior_mean_graph(inference, rescale_probs=False, edge_colormap='gray', title=None, **plot_graph_kwargs):
        """Plot the simulated ties graph, weighting each tie connection's edge color by
        a model's posterior mean probability of T_AB.
        """

        T = inference.posterior['T']
        mean_lambda_ties = np.exp(T.mean(dim=('chain', 'draw'))[:, 0])

        _, axs = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[20, 1])
        plt.sca(axs[0])
        plt.set_cmap(edge_colormap)

        A, B = np.nonzero(TIES)
        G = nx.DiGraph()
        edge_color = []
        for ii, (A, B) in enumerate(DYADS):
            ii, jj = sorted([A, B])
            dyad_idx = np.where((DYADS[:, 0] == A) & (DYADS[:, 1] == B))[0][0].astype(int)
            weight = mean_lambda_ties[dyad_idx]

            # include edges that predict one or more social ties
            if weight >= 1:
                G.add_edge(A, B, weight=weight)
                edge_color.append(weight)

        edge_color = np.log(np.array(edge_color))

        utils.plot_graph(G, edge_color=edge_color, pos=TIES_LAYOUT_POSITION, **plot_graph_kwargs)
        plt.title(title)

        # Fake axis to hold colorbar
        plt.sca(axs[1])
        axs[1].set_aspect(.0001)
        axs[1].set_visible(False)
        img = plt.imshow(np.array([[edge_color.min(), edge_color.max()]]))
        img.set_visible(False)

        clb = plt.colorbar( orientation="vertical", fraction=0.8, pad=0.1)
        clb.set_label('log(# ties)', rotation=90)

    return (plot_posterior_mean_graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior Mean social network from gifting model
    """)
    return


@app.cell(hide_code=True)
def _(plot_posterior_mean_graph, social_ties_inference):
    plot_posterior_mean_graph(social_ties_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior mean social network for model that also incorporates Wealth
    """)
    return


@app.cell(hide_code=True)
def _(gr_inference, plot_posterior_mean_graph):
    plot_posterior_mean_graph(gr_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reminder: Social Networks Don't Exist
    - They are abstraction for things we can't measure directly
      - if we can account for all confounds, the network becomes less useful
    - Networks models aren't point estimates (though we plot the mean below)
      - Like all other posteriors, any quantities calculated from the graph model (e.g. centrality, degree, etc) should also use distributions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Including Predictive Household features

    Now we'll add predictor featurs to the model. Specifically, we'll add GLM parameters for
    - an association feature $A_{AB}$ for each diad (e.g. friendship), $\beta_A$.
    - the effect of household wealth $W_{A,B}$ on giving, $\beta_G$
    - the effect of household wealth $W_{A,B}$ on receiving, $\beta_R$

    #### Likelihood
    $$
    \begin{align*}
        G_{AB} &\sim \text{Poisson}(\lambda_{AB}) \\
        G_{BA} &\sim \text{Poisson}(\lambda_{BA}) \\
        \log(\lambda_{AB}) &= \alpha + \mathcal{T}_{AB} + \mathcal{G}_{A} + \mathcal{R}_{B} \\
        \log(\lambda_{BA}) &= \alpha + \mathcal{T}_{BA} + \mathcal{G}_{B} + \mathcal{R}_{A} \\
        \mathcal{T}_{AB} &= T_{AB} + \beta_A A_{AB} \\
        \mathcal{G}_{A} &= G_{A} + \beta_G W_{A} \\
        \mathcal{R}_{B} &= R_{A} + \beta_R W_{B}
    \end{align*}
    $$

    #### Global  priors
    $$
    \begin{align*}
        \alpha &\sim \text{Normal}(0, 1) \\
        \beta_{A, G, R} &\sim \text{Normal}(0, 1)
    \end{align*}
    $$

    #### Correlated Social Ties prior
    $$
    \begin{align*}
        \begin{pmatrix}
            T_{AB} \\
            T_{BA}
        \end{pmatrix} &= \text{MVNormal}
        \left(
            \begin{bmatrix}
                0 \\
                0
            \end{bmatrix},
            \begin{bmatrix}
                \sigma^2 & \rho \sigma^2 \\
                \rho \sigma^2 & \sigma^2
            \end{bmatrix}
        \right) \\
        \rho &\sim \text{LKJCorr}(\eta) \\
        \sigma &\sim \text{Exponential}(1)
    \end{align*}
    $$

    #### Correlated Giving/Recieving prior
    $$
    \begin{align*}
        \begin{pmatrix}
            G_{A} \\
            R_{A}
        \end{pmatrix} &= \text{MVNormal}
        \left(
            \begin{bmatrix}
                0 \\
                0
            \end{bmatrix},
            \textbf{R}_{GR}, \textbf{S}_{GR}
        \right) \\
        \textbf{R}_{GR} &\sim \text{LKJCorr}(\eta) \\
        \textbf{S}_{GR} &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Add observed confounds variables to simulated dataset
    """)
    return


@app.cell
def _(FRIENDSHIP, N, WEALTH, simulated_wealth_gifts):
    # Add **observed** association feature
    # Add **observed** wealth feature
    simulated_wealth_features = simulated_wealth_gifts.assign(
        association=FRIENDSHIP,
        wealthA=simulated_wealth_gifts.hidA.map({ii: WEALTH[ii] for ii in range(N)}),
        wealthB=simulated_wealth_gifts.hidB.map({ii: WEALTH[ii] for ii in range(N)}),
    )
    return (simulated_wealth_features,)


@app.function
def fit_giving_receiving_features_model(data, eta=2):
    n_dyads = len(data)
    n_correlated_features = 2

    dyad_id = data.did.values.astype(int)
    household_A_id = data.hidA.values.astype(int)
    household_B_id = data.hidB.values.astype(int)
    association_AB = data.association.values.astype(float)
    wealthA = data.wealthA.values.astype(float)
    wealthB = data.wealthB.values.astype(float)

    # Data are 1-indexed
    if np.min(dyad_id) == 1:
        dyad_id -= 1
        household_A_id -= 1
        household_B_id -= 1

    n_households = np.max([household_A_id, household_B_id]) + 1

    with pm.Model() as model:

        # Priors
        # single, global alpha
        alpha = pm.Normal("alpha", 0, 1)

        # global association and giving-receiving params
        beta_A = pm.Normal("beta_A", 0, 1)
        beta_G = pm.Normal("beta_G", 0, 1)
        beta_R = pm.Normal("beta_R", 0, 1)

        # Social ties interaction; shared sigma
        sigma_T = pm.Exponential.dist(1)
        chol_T, corr_T, std_T = pm.LKJCholeskyCov(
            "rho_T", eta=eta, n=n_correlated_features, sd_dist=sigma_T
        )
        z_T = pm.Normal("z_T", 0, 1, shape=(n_dyads, n_correlated_features))
        T = pm.Deterministic("T", chol_T.dot(z_T.T).T)

        T_AB = T[dyad_id, 0] + beta_A * association_AB
        T_BA = T[dyad_id, 1] + beta_A * association_AB

        # Giving-receiving interaction; full covariance
        sigma_GR = pm.Exponential.dist(1, shape=n_correlated_features)
        chol_GR, corr_GR, std_GR = pm.LKJCholeskyCov(
            "rho_GR", eta=eta, n=n_correlated_features, sd_dist=sigma_GR
        )

        z_GR = pm.Normal("z_GR", 0, 1, shape=(n_households, n_correlated_features))
        GR = pm.Deterministic("GR", chol_GR.dot(z_GR.T).T)

        G_A = GR[household_A_id, 0] + beta_G * wealthA
        G_B = GR[household_B_id, 0] + beta_G * wealthB

        R_A = GR[household_A_id, 1] + beta_R * wealthA
        R_B = GR[household_B_id, 1] + beta_R * wealthB

        lambda_AB = pm.Deterministic(
            "lambda_AB",
            pm.math.exp(alpha + T_AB + G_A + R_B)
        )
        lambda_BA = pm.Deterministic(
            "lambda_BA",
            pm.math.exp(alpha + T_BA + G_B + R_A)
        )

        # Record quantities for reporting
        pm.Deterministic("corrcoef_T", corr_T[0, 1])
        pm.Deterministic("std_T", std_T)
        pm.Deterministic("corrcoef_GR", corr_GR[0, 1])
        pm.Deterministic("std_GR", std_GR)
        pm.Deterministic("giving", beta_G)
        pm.Deterministic("receiving", beta_R)

        G_AB = pm.Poisson("G_AB", lambda_AB, observed=data.giftsAB)
        G_BA = pm.Poisson("G_BA", lambda_BA, observed=data.giftsBA)

        inference = cached_sample(target_accept=.9)

        # Include log-likelihood for model comparison
        inference = pm.compute_log_likelihood(inference, extend_inferencedata=True)

    return model, inference


@app.cell
def _(simulated_wealth_features):
    simulated_grf_model, simulated_grf_inference = fit_giving_receiving_features_model(simulated_wealth_features)
    return (simulated_grf_inference,)


@app.cell(hide_code=True)
def _(simulated_gr_inference, simulated_grf_inference):
    utils.plot_dist(simulated_gr_inference.posterior['std_T'], color='C0', label='without Association')
    utils.plot_dist(simulated_grf_inference.posterior['std_T'], color='C1', label='with Association')
    plt.xlabel("std T")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can see that including parameters for giving and receiving reduces posterior standard deviation associated with social ties. This is expected because, we're explaining away more variance with those additional parameters.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model coefficients
    """)
    return


@app.cell(hide_code=True)
def _(BETA_WG, BETA_WR, simulated_grf_inference):
    _fig, _ax = plt.subplots()
    plt.sca(_ax)
    _posterior = simulated_grf_inference.posterior
    utils.plot_dist(_posterior['beta_G'], color='C0', label='$\\beta_G$')
    plt.axvline(BETA_WG, color='C0', linestyle='--', label='True $\\beta_G$')

    utils.plot_dist(_posterior['beta_R'], color='C1', label='$\\beta_R$')
    plt.axvline(BETA_WR, color='C1', linestyle='--', label='True $\\beta_R$')

    utils.plot_dist(_posterior['beta_A'], color='C2', label='$\\beta_A$')
    plt.axvline(1, color='C2', linestyle='--', label='True $\\beta_A$')

    plt.xlabel("posterior, $\\beta$")
    plt.ylabel("density")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using this model--which is highly aligned with the data simulation--we're able to recover the coefficients from the underlying generative model.
    - Association (in this case friendship) gets positive coefficient $\beta_A$. All friendship associations result in giving
    - Wealth positively affects giving, $\beta_G$
    - Wealth negatively affects receiving $\beta_R$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Comparison
    Controlling for the correct confounds provides a better model of the data, in terms of cross-validation scores
    """)
    return


@app.cell
def _(simulated_gr_inference, simulated_grf_inference):
    az.compare(
        {
            "with confounds": simulated_grf_inference,
            "without confounds": simulated_gr_inference
        },
        var_name='G_AB'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Accounting for the confounding features makes social network abstraction less important
    """)
    return


@app.cell(hide_code=True)
def _(simulated_grf_inference):
    plot_wealth_gifting_posterior(simulated_grf_inference, data_range=(.4, .6))
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Giving/receiving is mostly explained by friendship and/or household wealth, so after accounting for those variables, the giving receiving dynamics defined in the dyads has less signal
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Social ties become more independent when controlling for correct predictors
    """)
    return


@app.cell(hide_code=True)
def _(
    plot_posterior_household_ties,
    plot_posterior_reciprocity,
    simulated_grf_inference,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))

    plt.sca(_axs[0])
    plot_posterior_household_ties(simulated_grf_inference, color_friends=True)

    plt.sca(_axs[1])
    plot_posterior_reciprocity(simulated_grf_inference)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The x-shape in the joint is indicative of an independent set of variables. m
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### When accounting for confounds, a majority of the connection probabilities are around 0.5
    This indicates social ties are more-or-less random after accounting for freindship and wealth
    """)
    return


@app.cell(hide_code=True)
def _(plot_posterior_mean_graph, simulated_grf_inference):
    plot_posterior_mean_graph(simulated_grf_inference)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fitting Giving-receiving Features Model to Real Data

    In the lecture McElreath reports results on the real data, but given the version of the dataset I have in hand, it's somewhat unclear how to move forward.
    """)
    return


@app.cell
def _(SHARING):
    SHARING.head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To fit the model we need to know which columns in the real dataset are associated with

    - The association metric $A_{AB}$
    - Wealth of Household A $W_A$
    - Wealth of Household B $W_B$

    Looking at the dataset, it's not entirely clear which columns we should/could associate with each of those variables to replicate the figure in lecture. That said, if we DID know those columns--or how to derive them--it would be easy to fit the model via

    ```python
    >>> grf_model, grf_inference = fit_giving_receiving_features_model(SHARING)
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Additional Structure: Triangle Closures
    - Relationships tend to come in triads
    - **Block models** -- social ties are more common within a social group
      - families
      - classrooms
      - actual city blocks
    - Adds additional confounds
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("H_A", "G_AB"),
            ("H_B", "G_AB"),
            ("H_A", "T_AB"),
            ("H_B", "T_AB"),
            ("H_A", "T_BA"),
            ("H_B", "T_BA"),
            ("T_AB", "G_AB"),
            ("T_BA", "G_AB"),
            ("H_A", "K_A"),
            ("H_B", "K_B"),
            ("K_A", "T_AB"),
            ("K_B", "T_AB"),
            ("K_A", "T_BA"),
            ("K_B", "T_BA"),
    
        ],
        node_props={
            "T_AB": {"style": "dashed"},
            "T_BA": {"style": "dashed"},
            "K_A": {"color": "red", "label": "A's block membership"},
            "K_B": {"color": "red", "label": "B's block membership"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("K_A", "T_AB"): {"color": "red"},
            ("K_A", "T_BA"): {"color": "red"},
            ("K_B", "T_AB"): {"color": "red"},
            ("K_B", "T_BA"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posterior Network is regularized
    - Social networks try to express **regularities** in the observations
    - Inferred networks are **regularized**

    > Blocks and clusters are still discrete subgroups, what about "continuous clusters" like age or spatial distance? The goal of next lecture on Gaussian Processes
    """)
    return


@app.cell
def _(DYADS, TIES_LAYOUT_POSITION):
    def plot_gifting_graph(data, title=None, edge_colormap='gray_r', **plot_graph_kwargs):
        G = nx.DiGraph()
        edge_weights = []
        for ii, (A, B) in enumerate(DYADS):
            row = data.iloc[ii]
            if row.giftsAB > 0:
                A_, B_ = A, B
                weight = row.giftsAB
            if row.giftsBA > 0:
                A_, B_ = B, A
                weight = row.giftsBA

            G.add_edge(A, B, weight=weight)
            edge_weights.append(weight)

        edge_weights = np.array(edge_weights)
        edge_color = np.log(edge_weights)

        _, axs = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[20, 1])
        plt.sca(axs[0])
        plt.set_cmap(edge_colormap)
        utils.plot_graph(G, edge_color=edge_color, pos=TIES_LAYOUT_POSITION, **plot_graph_kwargs)
        plt.title(title);

        # Fake axis to hold colorbar
        plt.sca(axs[1])
        axs[1].set_aspect(.0001)
        axs[1].set_visible(False)
        img = plt.imshow(np.array([[edge_color.min(), edge_color.max()]]))
        img.set_visible(False)

        clb = plt.colorbar( orientation="vertical", fraction=0.8, pad=0.1)
        clb.set_label('log(# gifts)', rotation=90)

    return (plot_gifting_graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Comparing Gifting Observations to Model Trained on those observations
    ⚠️ The example below is using the simulated data
    """)
    return


@app.cell(hide_code=True)
def _(
    plot_gifting_graph,
    plot_posterior_mean_graph,
    simulated_grf_inference,
    simulated_wealth_gifts,
):
    # Data that is modeled
    _edge_weight_cmap = 'gray'  # switch colorscale to highlight sparsity
    plot_gifting_graph(
        simulated_wealth_gifts,
        edge_colormap=_edge_weight_cmap,
        alpha=1,
        title = 'Observed Gifting'
    )
    _observed_fig = plt.gcf()

    # Resulting model is sparser
    plot_posterior_mean_graph(
        simulated_grf_inference,
        edge_colormap=_edge_weight_cmap,
        alpha=1,
        title="Model Posterior Social Ties"
    )
    mo.vstack([_observed_fig, plt.gcf()])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The observed gifting network is denser than the social ties network estimated from the data, indicating that the model's pooling is adding regularization (compression) to the network's of social ties.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Varying effects as technology
    - Social nets try to express regularities in observed data
    - Inferred nets are thus regularized, capturing those structured regular effects
    - What happens when clusters are not discrete?
      - Age, distance, spatial location
      - We need a way to stratify or perform local poolling by "continuous clusters"
        - This is where Gaussian Processes come in next lecture.
        - Allows us to attack problems that require phylogenic and spatial models
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Constructed Variables $\neq$ Stratification
    - Outcomes that are deterministic functions of e.g.
      - Body Mass Index: $BMI = \frac{mass}{height^2}$
      - "*per capita", "*per unit time"
      - % changes, difference from reference value
    - **It's a common misunderstanding that dividing or rescaling by a variable is equivalent to controlling for that variable.**
    - Causal inference can provide a means to combine variables in a more principled manner.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: Dividing GDP by population
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("P", "GDP/P"),
            ("GDP", "GDP/P"),
            ("P", "GDP")
        ],
        node_props={
            "P": {"label": "population, P"},
            "GDP": {"label": "gross domestic product, GDP"},
        },
        edge_props={
            ("P", "GDP"): {"color": "red", "label": "assumed to be linear\n(not realistic)", "fontcolor": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - makes the assumption that population scales GDP linearly
    - dividing by population is not equivalent to stratifying by population
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("P", "GDP/P"),
            ("GDP", "GDP/P"),
            ("P", "GDP"),
            ("P", "X"),
            ("X", "GDP")
        ],
        node_props={
            "X": {"color": "red", "label": "cause of interest, X"},
            "GDP": {"color": "red"},
        },
        edge_props={
            ("P", "X"): {"color": "blue", "label": "backdoor\npath", "fontcolor": "blue"},
            ("X", "GDP"): {"color": "red", "label": "Causal Path", "fontcolor": "red"},
            ("P", "GDP/P"): {"color": "blue"},
            ("GDP", "GDP/P"): {"color": "red"},
    
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    it gets worse, though. In the scenario below, where we want to estimate the causal effect of $X$ on GDP/P, the fork created by $P$ isn't removed by simply calculating GDP/P
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Another Example: Rates
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("T", "Y/T"),
            ("Y", "Y/T"),
            ("T", "Y"),
            ("X", "Y")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Rates are often calculated as $\frac{\text{\# events}}{\text{unit time}}$  and modeled as outcomes
    - Does not consider the varying precision for different amounts of time
      - assuming all Y/T has the same precision, despite larger Y generally having more time to occur, and thus having higher precision
      - collapsing to point estimates removes our ability to talk about uncertainty in rates (e.g. distributions)
      - datapoints with less time/less prcision are given as much credibility to the estimate as data with longer/better precision
    - **Division by time does not control for time**
    - If rates are the focus of scientific question, **model the counts (e.g. Poisson regression) to estimate the rate parameter**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Another Example: Difference scores
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("H0", "H1-H0"),
            ("H1", "H1-H0"),
            ("H0", "H1"),
            ("X", "H1")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For example the plant growth experiment, where H0 and H1 are the starting and ending heights of the plant, X is the antifungal treatment.

    - difference score H1-H0 makes strong assumptions about the effect of H0 on H1, namely that
      - growth effects are constant for all starting heights
      - there is no floor or ceiling effects on plant height
    - **need to model H0 on the right side of the GLM (linear regression), as it is a cause of H1**. This is what we did in the plant growth example
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Review: Constructed Variables are Bad
    - **arithmetic is not stratification**
    - implicity assume fixed functional relationships amongst causes; you should be estimating these functional relationships
    - generally ignores uncertainty
    - using residuals as new data does not control for variables; don't use in causal inference (though it is often common to do so in predictive settings)

    ### Adhockery
    - adhoc procedures that have intuitive justifications
      - "we expect to see a correlation".
        - Why do you expect to see correlation (make assumptions explicit)?
        - Also, if you don't see a correlation, why does that mean some NULL contingent on that correlation is disproven?
    - if an adhoc procedure _does_ work (they sometimes can be correct by chance), it needs to be justified by causal logic and testing
    - **Simple rule: Model what you measure**
      - don't try to model new metrics that are derived from measures
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
