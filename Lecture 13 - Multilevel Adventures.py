import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
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
    # [Lecture 13 - Multilevel Adventures](https://www.youtube.com/watch?v=sgqMkZeslxA)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Review: Drawing the Bayesian Owl 🦉

    1. Establish **Estimand**
    2. Build **Scientific model(s)** (i.e. Causal model), depending on 1.
    3. Use 1 & 2 to build a **Statistical Model**
    4. **Simulate** data from 2 and **Validate** you can recover from 3
    5. **Analyze Real Data** using 3.

    In real life, it's never a linear path; you are jumping back and forth, iterating on 2-5, much like a branching path/choose your own adventure book.

    # Multi-level Adventures
    Similarly there is no one-size-fits all approach to applying the methods in this course. In order to optimize for success when applying these methods, McElreath suggests a few strategies ("paths") moving forward:

    - **Return to the start** -- McElreath suggests to return to the beginning of the course, reviewing the material now that you've observed a lion's share of the building blocks.
      - **It turns out that the notes presented in this repo are after the _Third Pass of the course_. I can't recommend strongly enough to take McElreath's advice and review the material from the beginning.** It's a lot of material, but I was flabbergasted with how much I had forgotten in the short time between this lecture and the earlier ones. Similarly, I was surprised by how much easier it was to soak up the material the 2nd time around--a real testament to McElreath's outstanding teaching style.
    - **Skim & Index** -- don't sweat the details, just acquaint yourself with the possibilities.
      - One thing that I've found useful is to compile each of the model classes discussed in the course into a "recipe book" or "toolbox" of plug-and-play models that can be reused for different applications


    # Clusters vs Features

    $$
    \begin{align*}
    \textbf{Clusters}& && &\textbf{Features} \\
    tanks& &\longrightarrow& &survival \\
    stories& &\longrightarrow& & treatment\;effect \\
    individuals& &\longrightarrow& & average\;response \\
    departments& &\longrightarrow& & admission\;rate \\
    \end{align*}
    $$

    - **Clusters**: subgroups in the data (e.g. tanks, participants, stories, departments)
      - **Adding clusters is fairly simple**
        - requires more index variables; more population priors
    - **Features**: aspects of the model (i.e. parameters) that vary by cluster (e.g. survival, average response rate, admission rate, etc.)
      - **Adding features requires more complexity**
        - more parameters, particularly dimensions in each population prior

    ## Varying effects as confounds
    - **Varying effect strategy**: using **repeat observations** and **partial pooling** to estimate unmeasured features of clusters that have left an imprint on the data
    - **Predictive perspective**: regularization
    - **Causal Perspective**: unobserved confounds are terrifying, but leveraging repeat observations give us some hope at more accurate inference

    ### Previous Examples:
    #### Grandparents & Education
    """)
    return


@app.cell(hide_code=True)
def grandparents_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "P"),
            ("G", "C"),
            ("P", "C"),
            ("U", "P"),
            ("U", "C")
        ],
        node_props={
            "G": {"label": "Grandparents Ed, G"},
            "P": {"label": "Parents Ed, P"},
            "C": {"label": "Children's Ed, C"},
            "U": {"label": "Neighborhood, U", "color": "blue"}
        },
        edge_props={
            ("U", "P"): {"color": "blue"},
            ("U", "C"): {"color": "blue"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Neighborhood is a backdoor path confound that blocks mediation analysis of the **direct effect** of $G$ on $C$
    - but having repeat observations for neighborhoods U allows us to estimate the effects of this confound
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Trolley Problem Example
    """)
    return


@app.cell(hide_code=True)
def trolley_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "R"),
            ("S", "R"),
            ("E", "R"),
            ("G", "R"),
            ("Y", "R"),
            ("U", "R"),
            ("G", "E"),
            ("Y", "E"),
            ("E", "P"),
            ("U", "P")
        ],
        node_props={
            "X": {"label": "Treatment, X"},
            "R": {"label": "Response, R"},
            "P": {"label": "Participation, P", "style": "dashed"},
            "U": {"label": "Individual Traits, U", "color": "blue"},
            "unobserved": {"style": "dashed"}
        },
        edge_props={
            ("U", "P"): {"color": "blue"},
            ("U", "R"): {"color": "blue"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Individuals vary on how they react to the response scale, adding noise to our estimates
    - However, given that each participant has repeat observations, we use the repeats to estimate this noise.
    - Similarly, individual traits may cause sampling bias through an unobserved participation node; we can use mixed effects to help address this sampling bias.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fixed Effect Approach
    - rather than partial pooling, **no pooling**.
    - very few benefits to using fixed effects over varying effects.
      - e.g. less efficient
    - **focus on getting the story straight** (generative model, causal graph), you can worry about the details of estimator efficiency, etc. later

    ### Practical Difficulties

    - **Varying effects models are always a good default**, but
      1. how do you use more than one cluster
      2. Predictions are at the level of the hierarchy now, which level do we care about
      3. Sampling efficiency -- e.g. centered/non-centered priors
      4. Group-level confounding -- e.g. Full Luxury Bayes or Mundlak Machines. For details, see the BONUS section of [Lecture 12 - Multilevel Models](<Lecture 12 - Multilevel Models.py>)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fertility & Behavior in Bangladesh
    - 1989 Fertility Survey
    - 1934 women, 61 districts (district 54 has no data)
    - **Outcome, $C$:** contraceptive use (binary variable)
    - **Predictors**: age $A$, # of living children $K$, urban/rural location $U$
    - **Potential (unobserved) confounds**: Family traits, $F$
    - **District ID**: $D$
    """)
    return


@app.cell
def fertility_data():
    FERTILITY = utils.load_data("bangladesh")
    FERTILITY.head()
    return (FERTILITY,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Competing causes
    """)
    return


@app.cell(hide_code=True)
def competing_causes_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "C"),
            ("K", "C"),
            ("U", "C"),
            ("D", "C"),
            ("D", "U"),
            ("U", "K"),
            ("A", "K"),
            ("D", "K"),
            ("F", "C"),
            ("F", "K")
        ],
        node_props={
            "A": {"label": "age, A"},
            "K": {"label": "# kids, K"},
            "U": {"label": "urbanity, U"},
            "D": {"label": "district, D"},
            "C": {"label": "contraceptive use, C"},
            "F": {"label": "family traits, F", "style": "dashed"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def district_sample_sizes_plot(FERTILITY):
    _district_counts = FERTILITY.groupby('district').count()['woman']
    plt.bar(_district_counts.index, _district_counts)
    plt.xlabel("district")
    plt.ylabel("# of women")
    plt.title("Variation in district-level sampling")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Start simple: varying districts
    - **Estimand**: contraceptive use in each district; partially pooled
    - **Model**:
      - varying intercept/offset for each district
    """)
    return


@app.cell(hide_code=True)
def district_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("D", "C"),
        ],
        node_props={
            "D": {"label": "district, D"},
            "C": {"label": "contraceptive use, C"},
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    C_i &\sim \text{Bernoulli}(p_i) &\text{Contraception use is 0/1 outcome} \\
    \text{logit}(p_i) &= \alpha_{D[i]} &\text{log odds of C=1} \\
    \alpha_j &\sim \text{Normal}(\bar \alpha, \sigma) &\text{prior for districts; regularizer} \\
    \bar \alpha &\sim \text{Normal}(0, 1) &\text{average district} \\
    \sigma &\sim \text{Exponential}(1) &\text{std. dev amongst districts}
    \end{align*}
    $$
    """)
    return


@app.cell
def district_fit(FERTILITY):
    USES_CONTRACEPTION = FERTILITY['use.contraception'].values.astype(int)
    DISTRICT = np.arange(1, 62).astype(int)  # note: district 54 has no data, but still gets a parameter
    DISTRICT_ID = FERTILITY.district.values - 1  # index into DISTRICT, so district d is entry d - 1

    with pm.Model(coords={"district": DISTRICT}) as district_model:
        # Priors
        ## Global priors
        _sigma = pm.Exponential('sigma', 1)  # variation amongst districts
        _alpha_bar = pm.Normal("alpha_bar", 0, 1)  # the average district

        # District-level priors
        _alpha = pm.Normal('alpha', _alpha_bar, _sigma, dims="district")

        # p(contraceptive)
        _p_C = pm.Deterministic("p_C", pm.math.invlogit(_alpha), dims='district')

        # Likelihood
        _p = pm.math.invlogit(_alpha[DISTRICT_ID])
        _C = pm.Bernoulli("C", p=_p, observed=USES_CONTRACEPTION)

        district_inference = cached_sample()
    return DISTRICT, DISTRICT_ID, USES_CONTRACEPTION, district_inference


@app.cell
def district_posterior_plot_helper(DISTRICT, FERTILITY):
    def plot_district_posterior(
        inference, sigma=None, color='C0', var='p_C', hdi_prob=.89, data_filter=None, title=None, ax=None
    ):

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        # Filter the dataset for urban/rural if requested
        if data_filter == 'urban':
            data_mask = (FERTILITY.urban).astype(bool)
        elif data_filter == 'rural':
            data_mask = (1 - FERTILITY.urban).astype(bool)
        else:
            data_mask = np.ones(len(FERTILITY)).astype(bool)

        plot_data = FERTILITY[data_mask]
        district_counts = plot_data.groupby('district').count()['woman']
        contraceptive_counts = plot_data.groupby('district').sum()['use.contraception']
        proportion_contraceptive = contraceptive_counts / district_counts

        plt.sca(ax)
        utils.plot_scatter(
            xs=proportion_contraceptive.index,
            ys=proportion_contraceptive.values,
            color='k', s=50, zorder=3, alpha=.8, label='raw proportions'
        )

        # Posterior per-district mean probability of contraceptive use. The model
        # indexes districts by label, so entry j is district DISTRICT[j]
        posterior_mean = inference.posterior[var].mean(dim=('chain', 'draw')).values

        utils.plot_scatter(DISTRICT, posterior_mean, color=color, zorder=50, alpha=.8, label='posterior means')

        # Posterior HDI error bars
        hdis = az.hdi(
            inference.posterior,
            var_names=var,
            prob=hdi_prob
        )[var]

        error_upper = hdis.sel(ci_bound='upper').values - posterior_mean
        error_lower = posterior_mean - hdis.sel(ci_bound='lower').values
        utils.plot_errorbar(
            xs=DISTRICT,
            ys=posterior_mean,
            error_lower=error_lower,
            error_upper=error_upper,
            colors=color,
            error_width=8
        )

        # Add empirical mean
        empirical_mean = FERTILITY[data_mask]['use.contraception'].mean()
        plt.axhline(y=empirical_mean, c='k', linestyle="--", label='global mean')

        plt.ylim([-0.05, 1.05])
        plt.xlabel("district")
        plt.ylabel("prob. use contraception")
        plt.title(title)
        plt.legend()


    return (plot_district_posterior,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### District-only model posterior predictions
    """)
    return


@app.cell(hide_code=True)
def district_posterior_plot(district_inference, plot_district_posterior):
    plot_district_posterior(district_inference, title='District Model')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Studying the posterior graph

    Districts with small sample sizes (e.g. district 3) have

    - larger error bars -- exhibiting more uncertainty estimate
    - exhibit more shrinkage
      - posteriors are pulled toward the global mean (dashed line)
      - red circles are far from black circles, because the model is less confident

    Districts with large sample sizes (e.g. district 1) have

    - smaller error bars -- more certainty about estimates
    - less shrinkage
      - posteriors are closer to the empirical observations for the district

    Districts with no data (district 54) still have posteriors

    - informed posterior from partial pooling
    - mean is near the global mean
    - error bars larger than for other districts
    """)
    return


@app.cell
def shrinkage_grid_helper(DISTRICT, DISTRICT_ID, USES_CONTRACEPTION):
    def grid_district_pooling(
        sigma,
        alpha_grid=np.linspace(-6, 6, 481),
        alpha_bar_grid=np.linspace(-3, 3, 121),
    ):
        """Posterior mean probability of contraceptive use in each district under

            alpha_j ~ Normal(alpha_bar, sigma), alpha_bar ~ Normal(0, 1)

        with sigma held fixed, computed exactly on a grid instead of with MCMC:
        each district's alpha_j is integrated over `alpha_grid`, and alpha_bar
        over `alpha_bar_grid`.
        """
        users = np.bincount(DISTRICT_ID, weights=USES_CONTRACEPTION, minlength=len(DISTRICT))
        n = np.bincount(DISTRICT_ID, minlength=len(DISTRICT))
        p = utils.invlogit(alpha_grid)

        # Likelihood of each district at each alpha, scaled per district (the scale cancels below)
        log_lik = users[:, None] * np.log(p) + (n - users)[:, None] * np.log1p(-p)
        lik = np.exp(log_lik - log_lik.max(axis=1, keepdims=True))  # (districts, alphas)

        # Prior density of alpha_j for every alpha_bar, times the grid step
        prior = stats.norm.pdf(alpha_grid[:, None], alpha_bar_grid, sigma) * (alpha_grid[1] - alpha_grid[0])

        # Integrate out each alpha_j: p(data_j | alpha_bar) and E[p_j * likelihood]
        marginal = lik @ prior  # (districts, alpha_bars)
        weighted = (lik * p) @ prior

        # Posterior over alpha_bar
        log_post = stats.norm.logpdf(alpha_bar_grid, 0, 1) + np.log(np.maximum(marginal, 1e-300)).sum(axis=0)
        weights = np.exp(log_post - log_post.max())
        weights /= weights.sum()

        p_district = np.divide(weighted, marginal, out=np.zeros_like(weighted), where=marginal > 0)
        return p_district @ weights

    return (grid_district_pooling,)


@app.cell(hide_code=True)
def shrinkage_controls(district_inference):
    _sigma_steps = np.round(np.geomspace(0.05, 5, 41), 3).tolist()
    _fitted_sigma = float(district_inference.posterior["sigma"].mean())
    shrinkage_sigma = mo.ui.slider(
        steps=_sigma_steps,
        value=min(_sigma_steps, key=lambda s: abs(s - _fitted_sigma)),
        label="σ, spread between districts",
        show_value=True,
        full_width=True,
    )
    mo.vstack([
        mo.md(rf"""
    #### Explore: how much pooling?
    Hold $\sigma$ fixed and see where each district's estimate lands. A small $\sigma$ says districts are alike, so every estimate is pulled to the global mean (complete pooling). A large $\sigma$ says districts are unrelated, so each estimate stays at its raw proportion (no pooling). The district model estimated $\sigma \approx {_fitted_sigma:.2f}$, which the slider starts at.

    The estimates are computed exactly on a grid (see `grid_district_pooling`), so they update instantly.
    """),
        shrinkage_sigma,
    ])
    return (shrinkage_sigma,)


@app.cell(hide_code=True)
def shrinkage_explorer(
    DISTRICT,
    DISTRICT_ID,
    USES_CONTRACEPTION,
    grid_district_pooling,
    shrinkage_sigma,
):
    _n = np.bincount(DISTRICT_ID, minlength=len(DISTRICT))
    _users = np.bincount(DISTRICT_ID, weights=USES_CONTRACEPTION, minlength=len(DISTRICT))
    _has_data = _n > 0
    _raw = np.divide(_users, _n, out=np.full(len(_n), np.nan), where=_has_data)
    _pooled = grid_district_pooling(shrinkage_sigma.value)

    _fig, (_ax_districts, _ax_shrinkage) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})

    _ax_districts.vlines(DISTRICT[_has_data], _raw[_has_data], _pooled[_has_data], color="gray", alpha=.6)
    _ax_districts.scatter(
        DISTRICT[_has_data], _raw[_has_data], s=10 + 1.5 * _n[_has_data], color="k", alpha=.5,
        label="raw proportions (size: # of women)",
    )
    _ax_districts.scatter(DISTRICT[_has_data], _pooled[_has_data], s=25, color="C0", zorder=3, label="posterior means")
    _ax_districts.scatter(DISTRICT[~_has_data], _pooled[~_has_data], s=60, marker="x", color="C0", zorder=3, label="no data (district 54)")
    _ax_districts.axhline(USES_CONTRACEPTION.mean(), color="k", linestyle="--", label="global mean")
    _ax_districts.set_ylim([-0.05, 1.05])
    _ax_districts.set_xlabel("district")
    _ax_districts.set_ylabel("prob. use contraception")
    _ax_districts.set_title(f"σ = {shrinkage_sigma.value:g}")
    _ax_districts.legend(fontsize=8, loc="upper right")

    _ax_shrinkage.scatter(_n[_has_data], np.abs(_pooled - _raw)[_has_data], color="C0", alpha=.7)
    _ax_shrinkage.set_xscale("log")
    _ax_shrinkage.set_ylim([0, 0.65])
    _ax_shrinkage.set_xlabel("# of women in district")
    _ax_shrinkage.set_ylabel("|posterior mean − raw proportion|")
    _ax_shrinkage.set_title("shrinkage vs sample size")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Varying districts + urban
    """)
    return


@app.cell(hide_code=True)
def urban_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "C"),
            ("K", "C"),
            ("U", "C"),
            ("D", "C"),
            ("D", "U"),
            ("U", "K"),
            ("A", "K"),
            ("D", "K"),
            ("F", "C"),
            ("F", "K")
        ],
        node_props={
            "A": {"color": "lightgray"},
            "K": {"color": "lightgray"},
            "F": {"label": "family traits, F", "style": "dashed"},
        },
        edge_props={
            ("A", "K"): {"color": "lightgray"},
            ("A", "C"): {"color": "lightgray"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - What is the effect of urban living?
    - Beware:
      - district features have potential group-level confounds
      - Total effect of $U$ passes through $K$
      - Do not stratify by $K$:
        - it's a mediator on $U \rightarrow K \rightarrow C$, so stratifying by $K$ removes part of the total effect of $U$
        - it's a collider on $U \rightarrow K \leftarrow F \rightarrow C$, so stratifying by $K$ opens a path through the unobserved family traits $F$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Statistical Model
    """)
    return


@app.cell(hide_code=True)
def district_urban_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("D", "C"),
            ("U", "C"),
        ],
        node_props={
            "D": {"label": "district, D"},
            "C": {"label": "contraceptive use, C"},
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align*}
    C_i &\sim \text{Bernoulli}(p_i) \\
    \text{logit}(p_i) &= \alpha_{D[i]} + \beta_{D[i]} U_i  &\text{include urban effect}\\
    \alpha_j &\sim \text{Normal}(\bar \alpha, \sigma) &\text{regularizing prior for district}\\
    \bar \alpha &\sim \text{Normal}(0, 1)  &\text{district average prior} \\
    \sigma &\sim \text{Exponential}(1) &\text{district std prior} \\
    \beta_j &\sim \text{Normal}(\bar \beta, \tau) &\text{regularizing prior for urban effect} \\
    \bar \beta &\sim \text{Normal}(0, 1) &\text{urban average prior} \\
    \tau &\sim \text{Exponential}(1) &\text{urban std prior}
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the district-urban model
    We use the non-centered prior version here -- details about non-centered will be discussed later
    """)
    return


@app.cell
def district_urban_fit(DISTRICT, DISTRICT_ID, FERTILITY, USES_CONTRACEPTION):
    URBAN_CODED, URBAN = pd.factorize(FERTILITY.urban, sort=True)

    with pm.Model(coords={"district": DISTRICT}) as district_urban_model:

        # Mutable data
        _urban = pm.Data("urban", URBAN_CODED)

        # Priors
        # District offset
        _alpha_bar = pm.Normal("alpha_bar", 0, 1)  # the average district
        _sigma = pm.Exponential('sigma', 1)  # variation amongst districts

        # Uncentered parameterization
        _z_alpha = pm.Normal("z_alpha", 0, 1, dims='district')
        _alpha = _alpha_bar + _z_alpha * _sigma

        # District / urban interaction
        _beta_bar = pm.Normal("beta_bar", 0, 1)  # the average urban effect
        _tau = pm.Exponential('tau', 1)  # variation amongst urban

        # Uncentered parameterization
        _z_beta = pm.Normal("z_beta", 0, 1, dims='district')
        _beta = _beta_bar + _z_beta * _tau

        # Record p(contraceptive)
        _p_C_urban = pm.Deterministic("p_C_urban", pm.math.invlogit(_alpha + _beta), dims="district")
        _p_C_rural = pm.Deterministic("p_C_rural", pm.math.invlogit(_alpha), dims="district")

        # Likelihood
        _p = pm.math.invlogit(_alpha[DISTRICT_ID] + _beta[DISTRICT_ID] * _urban)
        _C = pm.Bernoulli("C", p=_p, observed=USES_CONTRACEPTION)

        district_urban_inference = cached_sample(target_accept=.95)
    return URBAN_CODED, district_urban_inference


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Summarize the urban-district posterior
    """)
    return


@app.cell
def district_urban_summary(district_urban_inference):
    az.summary(district_urban_inference, var_names=['alpha_bar', 'beta_bar', 'tau', 'sigma'])
    return


@app.cell(hide_code=True)
def centering_controls():
    centered_fit_button = mo.ui.run_button(label="Fit centered model")
    mo.vstack([
        mo.md(r"""
    #### Explore: centered vs non-centered priors
    The model above uses the non-centered form, $\alpha_j = \bar \alpha + z_{\alpha,j} \sigma$ with $z_{\alpha,j} \sim \text{Normal}(0, 1)$, and the same for $\beta_j$. The centered form writes the same model directly as $\alpha_j \sim \text{Normal}(\bar \alpha, \sigma)$ and $\beta_j \sim \text{Normal}(\bar \beta, \tau)$. Click the button to fit the centered version and compare how well the two sample. The first fit takes a few seconds, then it loads from the cache. [Lecture 14](<Lecture 14 - Correlated Features.py>) explains why the parameterization matters.
    """),
        centered_fit_button,
    ])
    return (centered_fit_button,)


@app.cell
def centered_fit(
    DISTRICT,
    DISTRICT_ID,
    URBAN_CODED,
    USES_CONTRACEPTION,
    centered_fit_button,
):
    mo.stop(not centered_fit_button.value, mo.md("Click **Fit centered model** above to fit and compare."))

    with pm.Model(coords={"district": DISTRICT}) as centered_district_urban_model:
        _urban = pm.Data("urban", URBAN_CODED)

        # District offset, centered
        _alpha_bar = pm.Normal("alpha_bar", 0, 1)
        _sigma = pm.Exponential("sigma", 1)
        _alpha = pm.Normal("alpha", _alpha_bar, _sigma, dims="district")

        # District / urban interaction, centered
        _beta_bar = pm.Normal("beta_bar", 0, 1)
        _tau = pm.Exponential("tau", 1)
        _beta = pm.Normal("beta", _beta_bar, _tau, dims="district")

        # Likelihood
        _p = pm.math.invlogit(_alpha[DISTRICT_ID] + _beta[DISTRICT_ID] * _urban)
        _C = pm.Bernoulli("C", p=_p, observed=USES_CONTRACEPTION)

        centered_district_urban_inference = cached_sample(target_accept=.95)
    return (centered_district_urban_inference,)


@app.cell(hide_code=True)
def centering_comparison_plot(
    centered_district_urban_inference,
    district_urban_inference,
):
    _funnel_district = 11  # no urban women, so its urban effect is informed only by the prior
    _posteriors = {
        "non-centered": (
            district_urban_inference,
            district_urban_inference.posterior["beta_bar"]
            + district_urban_inference.posterior["z_beta"].sel(district=_funnel_district) * district_urban_inference.posterior["tau"],
        ),
        "centered": (
            centered_district_urban_inference,
            centered_district_urban_inference.posterior["beta"].sel(district=_funnel_district),
        ),
    }

    _rows = []
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for _ax, (_label, (_idata, _beta)) in zip(_axs, _posteriors.items()):
        _summary = az.summary(_idata, var_names=["sigma", "tau"])
        _divergent = _idata.sample_stats["diverging"].values.ravel()
        _rows.append({
            "model": _label,
            "divergences": int(_divergent.sum()),
            "ESS bulk, σ": int(_summary.loc["sigma", "ess_bulk"]),
            "ESS bulk, τ": int(_summary.loc["tau", "ess_bulk"]),
            "r̂, τ": _summary.loc["tau", "r_hat"],
        })
        _log_tau = np.log(_idata.posterior["tau"].values.ravel())
        _beta = _beta.values.ravel()
        _ax.scatter(_beta, _log_tau, s=3, alpha=.3, color="C1", label="draws")
        _ax.scatter(_beta[_divergent], _log_tau[_divergent], s=12, color="k", label="divergent")
        _ax.set_xlabel(f"β for district {_funnel_district}")
        _ax.set_title(_label)
        _ax.legend(fontsize=8)
    _axs[0].set_ylabel("log τ")
    _fig.tight_layout()

    mo.vstack([
        mo.md(f"The funnel: when $\\tau$ is small, every $\\beta_j$ is squeezed near $\\bar \\beta$; when it is large, they spread out. The centered sampler has to explore that changing shape directly, while the non-centered one samples $z_{{\\beta,j}}$, whose shape doesn't depend on $\\tau$. District {_funnel_district} has no urban women, so its $\\beta_j$ shows the funnel most clearly. The non-centered fit reaches deep into the narrow neck where $\\tau$ is small; the centered fit can't, so its divergences pile up at the edge it reaches, and the ESS of $\\tau$ collapses."),
        pd.DataFrame(_rows).set_index("model"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare posterior means for urban/rural from a single model fit jointly
    """)
    return


@app.cell(hide_code=True)
def rural_urban_posterior_plot(
    district_urban_inference,
    plot_district_posterior,
):
    _fig, _axs = plt.subplots(2, 1, figsize=(10, 8))
    for _ii, (_label, _var) in enumerate(zip(['rural', 'urban'], ["p_C_rural", "p_C_urban"])):
        plot_district_posterior(district_urban_inference, color=f'C{_ii}', var=_var, data_filter=_label, ax=_axs[_ii])
        plt.title(_label)

    # Save fig for reference in next lecture
    utils.savefig("fertility_posterior_means_rural_urban.png")
    _fig
    return


@app.cell(hide_code=True)
def district_inspector_controls(DISTRICT, DISTRICT_ID, URBAN_CODED):
    _n_urban = np.bincount(DISTRICT_ID, weights=URBAN_CODED, minlength=len(DISTRICT))
    _n_all = np.bincount(DISTRICT_ID, minlength=len(DISTRICT))

    def _district_label(district):
        _j = district - 1
        if _n_all[_j] == 0:
            return f"{district} (no data)"
        if _n_urban[_j] == 0:
            return f"{district} (no urban women)"
        if _n_urban[_j] == _n_all[_j]:
            return f"{district} (no rural women)"
        return f"{district}"

    district_choice = mo.ui.dropdown(
        options={_district_label(_d): int(_d) for _d in DISTRICT},
        value=_district_label(1),
        label="district",
    )
    mo.vstack([
        mo.md(r"""
    #### Explore: one district at a time
    Pick a district to compare its data with its posterior. Districts with no urban women (or no data at all) still get estimates, borrowed from the other districts through partial pooling.
    """),
        district_choice,
    ])
    return (district_choice,)


@app.cell(hide_code=True)
def district_inspector_explorer(
    FERTILITY,
    district_choice,
    district_inference,
    district_urban_inference,
):
    _district = district_choice.value
    _data = FERTILITY[FERTILITY.district == _district]
    _rows = []
    for _label, _mask in [("rural", _data.urban == 0), ("urban", _data.urban == 1)]:
        _women = int(_mask.sum())
        _users = int(_data.loc[_mask, "use.contraception"].sum())
        _rows.append({
            "": _label,
            "women": _women,
            "use contraception": _users,
            "observed proportion": f"{_users / _women:.2f}" if _women else "–",
        })
    _counts = pd.DataFrame(_rows).set_index("")

    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _select = {"district": _district}
    utils.plot_dist(
        district_inference.posterior["p_C"].sel(_select), color="gray", label="district-only model",
        plot_kwargs={"linestyle": "--"},
    )
    for _ii, _label in enumerate(["rural", "urban"]):
        utils.plot_dist(
            district_urban_inference.posterior[f"p_C_{_label}"].sel(_select), color=f"C{_ii}", label=f"{_label} (district-urban model)"
        )
        if _rows[_ii]["women"]:
            _ax.axvline(_rows[_ii]["use contraception"] / _rows[_ii]["women"], color=f"C{_ii}", linestyle=":")
    _ax.set_xlim([-0.02, 1.02])
    _ax.set_xlabel("prob. use contraception")
    _ax.set_ylabel("density")
    _ax.set_title(f"District {_district}: posteriors (dotted: observed proportions)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.hstack([_counts, _fig], justify="start", align="center", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior variances
    The plots above indicate that urban areas have
    - higher overall rates of contraceptive use
    - more variation between districts; the urban error bars are also wider, because fewer women live in urban areas (562, against 1372 in rural areas)

    The plot below re-confirms that urban districts vary more. $\sigma$ is the spread of the rural log-odds $\alpha_j$ between districts. $\tau$ is the spread of the urban *difference* $\beta_j$, so the urban log-odds $\alpha_j + \beta_j$ vary with a standard deviation of $\sqrt{\sigma^2 + \tau^2}$, which is larger than $\sigma$
    """)
    return


@app.cell(hide_code=True)
def posterior_std_plot(district_urban_inference):
    _sigma = district_urban_inference.posterior['sigma']
    _tau = district_urban_inference.posterior['tau']
    for _ii, (_label, _values) in enumerate(
        [
            ('rural districts, $\\sigma$', _sigma),
            ('urban difference, $\\tau$', _tau),
            ('urban districts, $\\sqrt{\\sigma^2 + \\tau^2}$', np.sqrt(_sigma**2 + _tau**2)),
        ]
    ):
        utils.plot_dist(_values, color=f"C{_ii}", label=_label)

    def _exponential_prior(x, lambda_=1):
        return lambda_ * np.exp(-lambda_ * x)

    _xs = np.linspace(0, 1.6)
    plt.plot(_xs, _exponential_prior(_xs), label='prior ($\\sigma$, $\\tau$)', color='k', linestyle='--')
    plt.xlim([0, 1.6])
    plt.xlabel("posterior std dev (log-odds)")
    plt.ylabel("density")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Urban and Rural Outcomes are positively correlated
    """)
    return


@app.cell(hide_code=True)
def rural_urban_correlation_plot(district_urban_inference):
    _posterior_mean = district_urban_inference.posterior.mean(dim=("chain", "draw"))
    utils.plot_scatter(_posterior_mean["p_C_rural"], _posterior_mean["p_C_urban"], color='C0', label='posterior means')
    _correlation = np.corrcoef(_posterior_mean["p_C_rural"], _posterior_mean["p_C_urban"])[0][1]
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed', label='urban = rural')
    plt.xlabel("prob C (rural)")
    plt.ylabel("prob C (urban)")
    plt.title(f"Correlation: {_correlation:1.2f}")
    plt.legend()
    plt.axis("square")

    # Save fig for reference in next lecture
    utils.savefig("fertility_p_C_rural_urban.png")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### A few observations
    - There's more contraceptive use in urban areas than in rural areas: all but one district lie above the dashed diagonal
    - There's a high amount of correlation (cc>0.7) between contraceptive use in rural and urban areas in each district.

    **We should be able to take advantage of this correlation information to make better estimates**. MORE TO COME!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Summary: Multilevel Adventures
    - **Clusters**: distinct groups in the data
    - **Features**: aspects of the model (e.g. parameters) that vary by cluster
    - There is useful information transferred across features
      - We can use partial pooling to efficiently estimate features, even in absence of data
    """)
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
