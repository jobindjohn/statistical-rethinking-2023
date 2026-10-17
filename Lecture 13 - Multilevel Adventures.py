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
    4. **Simulate** data from from 2 and **Validate** you can recover from 3
    5. **Analyze Real Data** using 3.

    In real life, it's never a linear path; you are jumping back and forth, iterating on 2-5, much like a branching path/choose your own adventure book.

    # Multi-level Adventures
    Similarly there is no one-size-fits all approach to applying the methods in this course. In order to optimize for success when applying these methods, McElreath suggests a few strategies ("paths") moving forward:

    - **Return to the start** -- McElreath suggests to return to the beginning of the course, reviewing the material now that you've observed a lion's share of the building blocks.
      - **It turns out that the notes presented in this repo are after the _Third Pass of the course_. I can't recommend strongly enough to take McElreath's advice and review the material from the beginning.** It's a lot of material, but I was flabbergasted with how much I had forgotten in the short time between this lecture and the earlier ones. Similarly, I was surprised by how much easier it was to soak up the material the 2nd time around--a real testament to McElreath's outstanding teaching style.
    - **Skim & Index** -- don't sweat the deatils, just aquaint yourself with the possiblities.
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
        - more parametrers, particularly dimensions in each population prior

    ## Varying effects as confounds
    - **Varying effect strategy**: using **repeat observations** and **partial pooling** to estimate unmeasured features of clusters that have left an imprent on the data
    - **Predictive perspective**: regularization
    - **Causal Perspective**: unobserved confounds are terrifying, but leveraging repeat observations give us some hope at more accurate inference

    ### Previous Examples:
    #### Grandparents & Education
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
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
    - Individuals vary on how they react to the response scale, adding noise to our estimatates
    - However, given that each participant has repeat observations, we use the repeats to estimate this noise.
    - Similarly, individual traits may cause sampling bias through an unobserved participation node; we can use mixed effects to help address this sampling bias.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fixed Effect Approach
    - rather than partial pooling, **no pooling**.
    - very few benefits to using fixed effects over varying reffects.
      - e.g. less efficient
    - **focus on getting the story straight** (generative model, causal graph), you can worry about the details of estimator efficiency, etc. later

    ### Practical Difficulties

    - **Varying effects models are always a good default**, but
      1. how do you use more than one cluster
      2. Predictions is at the level of the hierarchy now, which level do we care about
      3. Sampling efficiency -- e.g. centered/non-centered priors
      4. Group-level confounding -- e.g. Full Luxury Bayes or Mundlak Machines. For details, see the BONUS section of [Lecture 12 - Multilevel Models](<Lecture 12 - Multilevel Models.ipynb>)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fertility & Behavior in Bangladesh
    - 1989 Fertility Survey
    - 1924 women, 61 districts
    - **Outcome, $C$:** contraceptive use (binary variable)
    - **Predictors**: age, $A$ # of living children $K$, urban/rural location $U$
    - **Potential (unobserved) confounds**: Family traits, $F$
    - **District ID**: $D$
    """)
    return


@app.cell
def _():
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
def _(FERTILITY):
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
      - varying intecept/offset for each district
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _(FERTILITY):
    USES_CONTRACEPTION = FERTILITY['use.contraception'].values.astype(int)
    DISTRICT_ID, _ = pd.factorize(FERTILITY.district)
    DISTRICT = np.arange(1, 62).astype(int)  # note: district 54 has no data so we create it's dim by hand

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
def _(DISTRICT, FERTILITY):
    def plot_survival_posterior(
        inference, sigma=None, color='C0', var='p_C', hdi_prob=.89, data_filter=None, title=None, ax=None
    ):

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        def reorder_missing_district_param(vec):
            """
            It appears that pymc tacks on the estimates for district with no data
            (54) onto the end of the parameter vector, so we put it into the correct
            position with this closure
            """
            vec_ = vec.copy()
            end = vec_[-1]  # no data is district 54 (index 53)
            vec_ = np.delete(vec_, -1)
            vec_ = np.insert(vec_, 52, end)
            return vec_

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

        # Posterior per-district mean survival probability
        posterior_mean = inference.posterior.mean(dim=('chain', 'draw'))[var]
        posterior_mean = reorder_missing_district_param(posterior_mean.values)

        utils.plot_scatter(DISTRICT, posterior_mean, color=color, zorder=50, alpha=.8, label='posterior means')

        # Posterior HDI error bars
        hdis = az.hdi(
            inference.posterior,
            var_names=var,
            prob=hdi_prob
        )[var]


        error_upper = reorder_missing_district_param(hdis.sel(ci_bound='upper').values) - posterior_mean
        error_lower = posterior_mean - reorder_missing_district_param(hdis.sel(ci_bound='lower').values)
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
        plt.xlabel("district ")
        plt.ylabel("prob. use contraception")
        plt.title(title)
        plt.legend()

    return (plot_survival_posterior,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### District-only model posterior predictions
    """)
    return


@app.cell(hide_code=True)
def _(district_inference, plot_survival_posterior):
    plot_survival_posterior(district_inference, title='District Model')
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
      - red circles are far from black circles) b.c. the model is less confident

    Districts with large sample sizes (e.g. district 1) have

    - smaller error bars -- more certainty about estimates
    - less shrinkage
      - posteriors are closer to the empirical observations for the district

    Districts with no data (e.g. district 49) still have posteriors

    - informed posterior from partial pooling
    - mean is near globla mean
    - error bars larger than for other districts (looks like i may have an indexing bug in my errorbar code--need to look into that)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Varying districs + urban
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
            "A": {"color": "lightgray"},
            "K": {"color": "lightgray"},
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
      - Do not stratify by $K$ -- it's a collider, that opens up the district-level confound through $D$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Statistical Model
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _(DISTRICT, DISTRICT_ID, FERTILITY, USES_CONTRACEPTION):
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

        # Recored p(contraceptive)
        _p_C = pm.Deterministic("p_C", pm.math.invlogit(_alpha + _beta))
        _p_C_urban = pm.Deterministic("p_C_urban", pm.math.invlogit(_alpha + _beta))
        _p_C_rural = pm.Deterministic("p_C_rural", pm.math.invlogit(_alpha))

        # Likelihood
        _p = pm.math.invlogit(_alpha[DISTRICT_ID] + _beta[DISTRICT_ID] * _urban)
        _C = pm.Bernoulli("C", p=_p, observed=USES_CONTRACEPTION)

        district_urban_inference = cached_sample(target_accept=.95)
    return (district_urban_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Summarize the urban-district posterior
    """)
    return


@app.cell
def _(district_urban_inference):
    az.summary(district_urban_inference, var_names=['alpha_bar', 'beta_bar', 'tau', 'sigma'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare posterior predictive for urban/rural form single model fit jointly
    """)
    return


@app.cell(hide_code=True)
def _(district_urban_inference, plot_survival_posterior):
    _fig, _axs = plt.subplots(2, 1, figsize=(10, 8))
    for _ii, (_label, _var) in enumerate(zip(['rural', 'urban'], ["p_C_rural", "p_C_urban"])):
        plot_survival_posterior(district_urban_inference, color=f'C{_ii}', var=_var, data_filter=_label, ax=_axs[_ii])
        plt.title(_label)

    # Save fig for reference in next lecture
    utils.savefig("fertility_posterior_means_rural_urban.png")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Posterior variances
    The plots above indicate that urban areas have
    - higher overall rates of contraceptive use
    - The error bars higher variance in urban areas

    The plot below re-confirms that variance in contraceptive use is indeed larger in urban areas; the posterior standard deviation parameter for urban areas $\tau$ is larger than the parameter $\sigma$ for rural areas
    """)
    return


@app.cell(hide_code=True)
def _(district_urban_inference):
    for _ii, (_label, _var) in enumerate(zip(['rural, $\\sigma$', 'urban, $\\tau$'], ['sigma', 'tau'])):
        utils.plot_dist(district_urban_inference.posterior[_var], color=f"C{_ii}", label=_label)

    def _exponential_prior(x, lambda_=1):
        return lambda_ * np.exp(-lambda_ * x)

    _xs = np.linspace(0, 1.2)
    plt.plot(_xs, _exponential_prior(_xs), label='prior', color='k', linestyle='--')
    plt.xlim([0, 1.2])
    plt.xlabel("posterior std dev")
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
def _(district_urban_inference):
    _posterior_mean = district_urban_inference.posterior.mean(dim=("chain", "draw"))
    utils.plot_scatter(_posterior_mean["p_C_rural"], _posterior_mean["p_C_urban"], color='C0', label='posterior means')
    _correlation = np.corrcoef(_posterior_mean["p_C_rural"], _posterior_mean["p_C_urban"])[0][1]
    plt.axvline(.5, color='gray', linestyle='dashed')
    plt.axhline(.5, color='gray', linestyle='dashed')
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
    - There's more contraceptive use in urban areas than in rural areas--a majority of points are to the left of the vertical line
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
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
