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
    from matplotlib.lines import Line2D

    import utils
    import init_notebook
    from init_notebook import cached_sample

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 10 - Counts and Hidden Confounds](https://www.youtube.com/watch?v=jokxu18egu0)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Revisiting Generalized Linear Models
    - Expected value is **some function** of an additive combination of parameters
      - That function tends to be tied to the data Likelihood distribution -- e.g. Identity for the Normal distribution (linear regression) or the log odds for Bernoulli/Binomial distribution (logistic regression)
    - Uniform changes in predictors are not generally associated with uniform changes in outcomes
    - Predictor variables interact -- causal interpretation of coefficients (outside of simplest models) is fraught with misleading conclusions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Confounded Admissions
    """)
    return


@app.cell(hide_code=True)
def confounded_admissions_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A"),
            ("u", "D"),
            ("u", "A")
        ],
        node_props={
            "u": {"style": "dashed", "label": "Ability, u"},
            "G": {"label": "Gender, G"},
            "D": {"label": "Department, D"},
            "A": {"label": "Admission, A"},
            "unobserved": {"style": "dashed"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - We estimated Direct and Total effect of Gender on Admission rates in order to identify different flavors of gender discrimination in admissions
    - However, it's implausible that there are no unobserved confounds between variables,
      - e.g. Applicant ability could link Department to Admission rate
        - affects which students apply to each department (more ability biases department application)
        - also affects baseline admission rates (more ability leads to higher admission rates)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Generative Simulation
    """)
    return


@app.cell
def admissions_simulation():
    np.random.seed(1)

    # Number of applicants
    n_samples = 2000

    # Gender equally likely
    G = np.random.choice([0, 1], size=n_samples, replace=True)

    # Unobserved Ability -- 10% have High, everyone else is Average
    _u = stats.bernoulli.rvs(p=0.1, size=n_samples)

    # Choice of department
    # G0 applies to D1 only if they have high ability; G1 applies to D1 with 75% probability
    D = stats.bernoulli.rvs(p=np.where(G==0, _u * 1.0, 0.75))

    # Ability-based acceptance rates (ability x dept x gender)
    # D0 treats the genders equally; D1 favors G1 at both ability levels
    _p_u0_dg = np.array([[0.1, 0.1], [0.1, 0.3]])
    _p_u1_dg = np.array([[0.3, 0.3], [0.3, 0.5]])
    _p_udg = np.array([_p_u0_dg, _p_u1_dg])
    print("Acceptance Probabilities\n(ability x dept x gender):\n\n", _p_udg)

    # Simulate acceptance
    _p = _p_udg[_u, D, G]
    A = stats.bernoulli.rvs(p=_p)
    return A, D, G, n_samples


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Total Effect Estimator
    """)
    return


@app.cell(hide_code=True)
def total_effect_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A"),
            ("u", "D"),
            ("u", "A")
        ],
        node_props={
            "u": {"style": "dashed", "label": "Ability, u"},
            "G": {"label": "Gender, G"},
            "D": {"label": "Department, D"},
            "A": {"label": "Admission, A"},
            "unobserved": {"style": "dashed"}
        },
        edge_props={
            ("G", "A"): {"color": "red"},
            ("G", "D"): {"color": "red"},
            ("D", "A"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Estimating <span style="color:red">Total effect</span>  requires no adjustment set, model gender only in the GLM

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
    ### Fit the Total Effect Model
    """)
    return


@app.cell
def admissions_coords(D, G):
    # Define coordinates
    GENDER_ID, GENDER = pd.factorize(np.array(["G2" if g else "G1" for g in G]), sort=True)
    DEPARTMENT_ID, DEPARTMENT = pd.factorize(np.array(["D2" if d else "D1" for d in D]), sort=True)
    return DEPARTMENT, DEPARTMENT_ID, GENDER, GENDER_ID


@app.cell
def fit_total_effect(A, GENDER, GENDER_ID):
    # Gender-only model
    with pm.Model(coords={"gender": GENDER}) as total_effect_admissions_model:
        _alpha = pm.Normal("alpha", 0, 1, dims='gender')

        # Likelihood
        _p = pm.math.invlogit(_alpha[GENDER_ID])
        pm.Bernoulli("admitted", p=_p, observed=A)

        # Record the probability param for simpler reporting
        pm.Deterministic("p_admit", pm.math.invlogit(_alpha), dims="gender")

        total_effect_admissions_inference = cached_sample()
    return total_effect_admissions_inference, total_effect_admissions_model


@app.cell
def total_effect_graph(total_effect_admissions_model):
    pm.model_to_graphviz(total_effect_admissions_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Summarize the Total Effect Estimates
    """)
    return


@app.function
def summarize_posterior(inference, figsize=(5, 3)):
    """Helper function for displaying model fits"""
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    utils.plot_forest(inference, var_names='alpha', ax=axs[0])
    utils.plot_forest(inference, var_names='p_admit', ax=axs[1])
    return mo.vstack([az.summary(inference, var_names=["alpha", "p_admit"]), fig])


@app.cell
def summary_total_effect(total_effect_admissions_inference):
    summarize_posterior(total_effect_admissions_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Direct Effect Estimator (now confounded due to common ability cause)
    """)
    return


@app.cell(hide_code=True)
def direct_effect_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A"),
            ("u", "D"),
            ("u", "A")
        ],
        node_props={
            "u": {"style": "dashed", "label": "Ability, u"},
            "G": {"label": "Gender, G"},
            "D": {"label": "Department, D", "style": "filled"},
            "A": {"label": "Admission, A"},
            "Direct effect\nadjustment set": {"style": "filled"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("G", "A"): {"color": "red"},
            ("G", "D"): {"color": "blue"},
            ("u", "D"): {"color": "blue"},
            ("u", "A"): {"color": "blue"},
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Estimating <span style="color:red">Direct effect</span> includes Department in the adjustment set
    - However, stratifying by Department (collider)  <span style="color:blue">opens a confounder backdoor path </span> through unobserved ability, u

    $$
    \begin{align*}
    A_i &\sim \text{Bernoulli}(p=p_i) \\
    \text{logit}(p_i) &= \alpha_{[D_i, G_i]} \\
    \alpha &= \begin{bmatrix} \alpha_{0,0} & \alpha_{0,1} \\ \alpha_{1,0} & \alpha_{1,1} \end{bmatrix} \\
    \alpha_{j,k} &\sim \text{Normal}(0, 1)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the (confounded) Direct Effect Model
    """)
    return


@app.cell
def fit_direct_effect(A, DEPARTMENT, DEPARTMENT_ID, GENDER, GENDER_ID):
    with pm.Model(coords={"gender": GENDER, "department": DEPARTMENT}) as direct_effect_admissions_model:
        # Prior
        _alpha = pm.Normal("alpha", 0, 1, dims=["department", "gender"])

        # Likelihood
        _p = pm.math.invlogit(_alpha[DEPARTMENT_ID, GENDER_ID])
        pm.Bernoulli("admitted", p=_p, observed=A)

        # Record the acceptance probability parameter for reporting
        pm.Deterministic("p_admit", pm.math.invlogit(_alpha), dims=["department", "gender"])

        direct_effect_admissions_inference = cached_sample(tune=2000)

    pm.model_to_graphviz(direct_effect_admissions_model)
    return (direct_effect_admissions_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Summarize the (confounded) Direct Effect Estimates
    """)
    return


@app.cell
def summary_direct_effect(direct_effect_admissions_inference):
    summarize_posterior(direct_effect_admissions_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpreting the (confounded) Direct Effect Estimates
    """)
    return


@app.function
def plot_department_gender_admissions(inference, title):
    plt.subplots(figsize=(8, 3))
    for ii, dept in enumerate(["D1", "D2"]):
        for jj, gend in enumerate(["G1", "G2"]):
            # note inverse link function applied
            post = inference.posterior
            post_p_accept = post.sel(department=dept, gender=gend)['p_admit']
            label=f'{dept}({"biased" if ii else "unbiased"})-G{jj+1}'
            color=f'C{np.abs(ii - 1)}'  # flip colorscale to match lecture
            linestyle= '--' if jj else "-"
            utils.plot_dist(
                post_p_accept,
                color=color,
                label=label,
                plot_kwargs=dict(linestyle=linestyle),
            )
    plt.xlim([0, .5])
    plt.xlabel("admission probability")
    plt.ylabel("density")
    plt.title(title)


@app.cell(hide_code=True)
def plot_direct_effect(direct_effect_admissions_inference):
    plot_department_gender_admissions(direct_effect_admissions_inference, 'direct effect, confounded model')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Interpreting the Confounded Direct Effect Model
    - In D1, G1 appears to be disadvantaged, with a lower admission rate. However, we know this isn't true. What's happening is that all the higher-ability G1 applicants are being sent to D2, thus artificially lowering the G1 acceptance rate in D1
    - We know that there is bias in D2, however, we see little evidence for discrimination. This is due to higher-ability G1 applications offsetting this bias by having higher-than-average acceptance
    - We can see that G1 estimates for D2 have higher variance; this is due to there being only 10% of applicants having high ability, thus fewer G1 applicants overall apply to D2.

    #### You guessed it: Collider Bias
    - **This is due to _collider bias_**
      - stratifying by Department--which forms a collider with Gender and ability--opens a path through the ability to acceptance.
      - **You CANNOT estimate _Direct effect_ of $G$ on $A$**
      - **You CAN estimate the _Total effect_**
    - **sorting can mask or accentuate bias**
    """)
    return


@app.cell(hide_code=True)
def sorting_controls():
    sorting_params = mo.ui.dictionary({
        "share_high": mo.ui.slider(0.0, 0.5, step=0.05, value=0.1, label="share of high-ability applicants", show_value=True),
        "sort_g1": mo.ui.slider(0.0, 1.0, step=0.05, value=1.0, label="P(D2 | G1, high ability)", show_value=True),
        "boost": mo.ui.slider(0.0, 0.5, step=0.05, value=0.2, label="acceptance boost for high ability", show_value=True),
        "bias": mo.ui.slider(0.0, 0.3, step=0.05, value=0.2, label="D2 bias against G1", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: how sorting by ability biases the stratified estimates
        The same setup as the simulation above. G1 applicants apply to D2 only if they have high ability (by default), and G2 applicants apply to D2 75% of the time, whatever their ability. Average-ability applicants are accepted at 0.1, plus the D2 bias for G2 applicants in D2; high-ability applicants get an extra boost. The sliders start at the values of the simulation.

        The filled bars are the acceptance rates within each department and gender: what the confounded direct-effect model estimates, given enough data. The outlined bars are the true rates for an average-ability applicant. Everything is computed exactly, without simulating applicants.
        """),
        mo.hstack([
            mo.vstack([mo.md("**Who applies where**"), sorting_params["share_high"], sorting_params["sort_g1"]]),
            mo.vstack([mo.md("**Acceptance**"), sorting_params["boost"], sorting_params["bias"]]),
        ], justify="start", gap=4),
    ])
    return (sorting_params,)


@app.cell(hide_code=True)
def sorting_explorer(sorting_params):
    _v = sorting_params.value
    _p_ability = np.array([1 - _v["share_high"], _v["share_high"]])  # average, high

    # P(D2 | gender, ability): rows are genders (G1, G2), columns are abilities (average, high)
    _p_d2 = np.array([[0.0, _v["sort_g1"]], [0.75, 0.75]])

    # Acceptance rates (ability x department x gender)
    _rates = np.full((2, 2, 2), 0.1)
    _rates[:, 1, 1] += _v["bias"]
    _rates[1] += _v["boost"]
    _rates = np.clip(_rates, 0, 1)

    # Within-department rates: average over the abilities of the applicants in each department (Bayes' rule)
    _observed = np.full((2, 2), np.nan)
    _share_high = np.full((2, 2), np.nan)
    for _d in range(2):
        for _g in range(2):
            _joint = _p_ability * (_p_d2[_g] if _d else 1 - _p_d2[_g])
            if _joint.sum() > 0:
                _p_ability_given_dg = _joint / _joint.sum()
                _observed[_d, _g] = (_p_ability_given_dg * _rates[:, _d, _g]).sum()
                _share_high[_d, _g] = _p_ability_given_dg[1]
    _true = _rates[0]

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for _d, _ax in enumerate(_axs):
        for _g in range(2):
            _color = f"C{1 - _g}"
            if np.isnan(_observed[_d, _g]):
                _ax.text(_g - 0.2, 0.02, "no\napplicants", ha="center", fontsize=8)
            else:
                _ax.bar(_g - 0.2, _observed[_d, _g], width=0.38, color=_color, alpha=0.8)
                _ax.text(_g - 0.2, _observed[_d, _g] + 0.01, f"{_share_high[_d, _g]:.0%}\nhigh ability", ha="center", fontsize=8)
            _ax.bar(_g + 0.2, _true[_d, _g], width=0.38, facecolor="none", edgecolor=_color, linewidth=2)
        _gap_observed = _observed[_d, 1] - _observed[_d, 0]
        _gap_true = _true[_d, 1] - _true[_d, 0]
        _ax.set_xticks([0, 1], ["G1", "G2"])
        _ax.set_title(f"D{_d + 1}: G2 − G1 gap within the department {_gap_observed:+.2f}\ntrue gap for average ability {_gap_true:+.2f}")
    _axs[0].set_ylabel("acceptance rate")
    _axs[0].set_ylim([0, 1])
    _axs[1].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.8),
            plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="gray", linewidth=2),
        ],
        labels=["within the department (what the model sees)", "true, average ability"],
        fontsize=8,
        loc="upper right",
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analogous Example: NAS Membership & Citations

    Two papers
    - same data
    - find drastically different conclusions about Gender and its effect on admission to the National Academy of Sciences (NAS)
      - One found Women are strongly advantaged
      - The other found women strongly disadvantaged
    - How can both conclusions be true?
    """)
    return


@app.cell(hide_code=True)
def nas_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "C"),
            ("G", "M"),
            ("C", "M"),
            ("q", "C"),
            ("q", "M")
        ],
        node_props={
            "q": {"style": "dashed", "label": "Researcher quality, q"},
            "G": {"label": "Gender, G"},
            "C": {"label": "Citations, C"},
            "M": {"label": "NAS Member, M"},
            "unobserved": {"style": "dashed"},
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - There are likely latent researcher quality differences
    - Stratifying by number of citations opens up a collider bias with unobserved researcher quality
    - Citations is a **post-treatment variable**
    - Citation-stratification provides misleading conclusions
    - e.g. if there is discrimination in publication/citation, one gender may get elected at a higher rate just because they will have higher quality on average for any citation level

    ### No Causes in, no causes out

    These papers suffer from a number of shortcomings

    - vague estimands
    - unwise adjustment sets
    - requires stronger assumptions than presented
    - collider bias could affect policy design in a bad way
    - qualitative data can be useful in these circumstances
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sensitivity Analysis: Modeling latent ability confound variable

    **What are the implications of things we can't measure?**

    Similar to Direct Effect scenario

    - Estimating <span style="color:red">Direct effect</span> includes Department in the adjustment set
    - However, stratifying by Department (collider)  <span style="color:blue">opens a confounder backdoor path </span> through unobserved ability, u
    """)
    return


@app.cell(hide_code=True)
def sensitivity_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "D"),
            ("G", "A"),
            ("D", "A"),
            ("u", "D"),
            ("u", "A")
        ],
        node_props={
            "u": {"label": "Ability, u", "style": "dashed"},
            "G": {"label": "Gender, G",},
            "D": {"label": "Department, D", "style": "filled"},
            "A": {"label": "Admission, A"},
            "Sensitivity analysis\nadjustment set": {"style": "filled"},
            "Modeled as\nrandom variable": {"style": "dashed"},
        },
        edge_props={
            ("G", "A"): {"color": "red"},
            ("G", "D"): {"color": "blue"},
            ("u", "D"): {"color": "green", "label": "gamma", "fontcolor": "green"},
            ("u", "A"): {"color": "green", "label": "beta", "fontcolor": "green"},
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Though we can't directly measure a potential confound, what we can do is _simulate_ the degree of the effect of a potential confound. Specifically, we can set up a simulation where we create a random variable associated with the potential confound, then weight the amount of contribution that confound has on generating the observed data.

    In this particular example, we can simulate the degree of effect of an ability random variable $U \sim \text{Normal}(0, 1)$, by adding a linearly weighted contribution of that variable to the log odds of Acceptance and selecting a department (because u affects both D and A in our causal graph):

    #### Department submodel
    $$
    \begin{align*}
    D_i &\sim \text{Bernoulli}(q_i) \\
    logit(q_i) &= \delta[G_i] + \gamma_{G[i]}  u_i \\
    \delta[G_i] &\sim \text{Normal}(0, 1) \\
    u_j &\sim \text{Normal}(0, 1)
    \end{align*}
    $$

    #### Acceptance submodel
    $$
    \begin{align*}
    A_i &\sim \text{Bernoulli}(p_i) \\
    logit(p_i) &= \alpha[D_i, G_i] + \beta_{G[i]} u_i \\
    \alpha[D_i, G_i] &\sim \text{Normal}(0, 1) \\
    u_j &\sim \text{Normal}(0, 1)
    \end{align*}
    $$

    Where we <span style="color:green">**manually set the values of $\beta_{G[i]}$ and $\gamma_{G[i]}$ by hand**</span> to perform the simulation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the latent ability model
    """)
    return


@app.cell
def fit_latent_ability(
    A,
    D,
    DEPARTMENT,
    DEPARTMENT_ID,
    GENDER,
    GENDER_ID,
    n_samples,
):
    # Gender-specific counterfactual parameters
    # Ability confound affects admission rates equally for genders
    _BETA = np.array([1., 1.])

    # Ability confound affects department application differentially
    # for genders (as is the case in generative data process)
    _GAMMA = np.array([1., 0.])

    _coords = {"gender": GENDER, "department": DEPARTMENT, "obs": np.arange(n_samples)}

    with pm.Model(coords=_coords) as latent_ability_model:

        # Latent ability variable, one for each applicant
        _U = pm.Normal("u", 0, 1, dims="obs")

        # Department application submodel
        _delta = pm.Normal('delta', 0, 1, dims='gender')
        _q = pm.math.invlogit(_delta[GENDER_ID] + _GAMMA[GENDER_ID] * _U)

        pm.Bernoulli('d', p=_q, observed=D)

        # Acceptance submodel
        _alpha = pm.Normal("alpha", 0, 1, dims=["department", "gender"])
        _p = pm.math.invlogit(_alpha[DEPARTMENT_ID, GENDER_ID] + _BETA[GENDER_ID] * _U)
        pm.Bernoulli('accepted', p=_p, observed=A)

        # Record p(A | D, G) for reporting
        pm.Deterministic("p_admit", pm.math.invlogit(_alpha), dims=['department', 'gender'])

        latent_ability_inference = cached_sample()

    pm.model_to_graphviz(latent_ability_model)
    return (latent_ability_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Summarize the latent ability estimate
    """)
    return


@app.cell
def summary_latent_ability(latent_ability_inference):
    summarize_posterior(latent_ability_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpreting the Effect of modeling the confound
    """)
    return


@app.cell(hide_code=True)
def plot_direct_effect_for_comparison(direct_effect_admissions_inference):
    plot_department_gender_admissions(direct_effect_admissions_inference, 'direct effect, confounded model')
    plt.gcf()
    return


@app.cell(hide_code=True)
def plot_latent_ability(latent_ability_inference):
    plot_department_gender_admissions(latent_ability_inference, "direct effect, latent ability model")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    By adding sensitivity analysis that is aligned with the data-generating process, we are able to identify gender bias in department 2

    ## Review of Sensitivity Analysis
    - Confounds exist, even if we can't measure them directly -- don't simply pretend that they don't exist
    - Address the question: **What are the implications of what we _don't know_?**
    - SA is somewhere between simulation and analysis
      - Hard-coding what we don't know, and let the rest play out.
      - Vary the confound over a range (e.g. std deviations) and show how that change affects the estimate
    - More honest than pretending that confounds do not exist.

    ## Note on number of parameters 🤯
    - Sensitivity Analysis model has 2006 free parameters
    - Only 2000 observations
    - No biggie in Bayesian Analysis
      - The minimum sample size is 0, where we just fall back on the prior
    """)
    return


@app.cell(hide_code=True)
def sensitivity_controls():
    sensitivity_beta = mo.ui.slider(steps=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0], value=1.0, label="$\\beta$: ability → admission", show_value=True)
    sensitivity_gamma = mo.ui.slider(steps=[0.0, 0.5, 1.0, 2.0, 3.0], value=1.0, label="$\\gamma_{G1}$: ability → department choice of G1", show_value=True)
    sensitivity_button = mo.ui.run_button(label="Fit the latent ability model")
    mo.vstack([
        mo.md(r"""
        #### Explore: sensitivity analysis
        Vary the assumed strength of the confound and refit the latent ability model. $\beta$ is the same for both genders and $\gamma_{G2}$ stays at 0, as in the model above, so $\beta = 1, \gamma_{G1} = 1$ reproduces it. With $\beta = 0$, ability has no effect on admission, and we're back to the confounded direct-effect model. A new setting takes about 20 seconds to sample; it is cached after that.

        The plots show the contrast $p_{admit}[G2] - p_{admit}[G1]$ in each department, for an applicant of average ability ($u = 0$). The dashed line is the true gap for an average-ability applicant in the simulation.
        """),
        mo.hstack([sensitivity_beta, sensitivity_gamma, sensitivity_button], justify="start", gap=2),
    ])
    return sensitivity_beta, sensitivity_button, sensitivity_gamma


@app.cell
def sensitivity_fit(
    A,
    D,
    DEPARTMENT,
    DEPARTMENT_ID,
    GENDER,
    GENDER_ID,
    n_samples,
    sensitivity_beta,
    sensitivity_button,
    sensitivity_gamma,
):
    mo.stop(not sensitivity_button.value, mo.md("Pick $\\beta$ and $\\gamma_{G1}$ and click **Fit the latent ability model**."))

    # Same model as above, with the confound strengths set by the sliders
    _BETA = np.array([sensitivity_beta.value, sensitivity_beta.value])
    _GAMMA = np.array([sensitivity_gamma.value, 0.])

    _coords = {"gender": GENDER, "department": DEPARTMENT, "obs": np.arange(n_samples)}
    with pm.Model(coords=_coords) as sensitivity_model:
        _U = pm.Normal("u", 0, 1, dims="obs")

        _delta = pm.Normal('delta', 0, 1, dims='gender')
        _q = pm.math.invlogit(_delta[GENDER_ID] + _GAMMA[GENDER_ID] * _U)
        pm.Bernoulli('d', p=_q, observed=D)

        _alpha = pm.Normal("alpha", 0, 1, dims=["department", "gender"])
        _p = pm.math.invlogit(_alpha[DEPARTMENT_ID, GENDER_ID] + _BETA[GENDER_ID] * _U)
        pm.Bernoulli('accepted', p=_p, observed=A)

        pm.Deterministic("p_admit", pm.math.invlogit(_alpha), dims=['department', 'gender'])

        sensitivity_inference = cached_sample()
    return (sensitivity_inference,)


@app.cell(hide_code=True)
def sensitivity_explorer(
    direct_effect_admissions_inference,
    sensitivity_beta,
    sensitivity_gamma,
    sensitivity_inference,
):
    def _gender_contrast(inference, dept):
        _p = inference.posterior["p_admit"].sel(department=dept)
        return _p.sel(gender="G2") - _p.sel(gender="G1")


    # True gaps for average ability, from the acceptance rates in the simulation
    _true_gaps = {"D1": 0.0, "D2": 0.2}

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.5))
    for _ax, _dept in zip(_axs, ["D1", "D2"]):
        plt.sca(_ax)
        utils.plot_dist(_gender_contrast(direct_effect_admissions_inference, _dept), color="gray", label="confounded direct-effect model")
        utils.plot_dist(
            _gender_contrast(sensitivity_inference, _dept),
            color="C0" if _dept == "D2" else "C1",
            label=f"latent ability, $\\beta$={sensitivity_beta.value:g}, $\\gamma_{{G1}}$={sensitivity_gamma.value:g}",
        )
        plt.axvline(_true_gaps[_dept], color="k", linestyle="--", label="true gap, average ability")
        plt.xlim([-0.4, 0.4])
        plt.xlabel("$p_{admit}[G2] - p_{admit}[G1]$")
        plt.ylabel("density")
        plt.title(f"{_dept} ({'biased' if _dept == 'D2' else 'unbiased'})")
    for _ax in _axs:
        _ax.legend(fontsize=8, loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Counts and Poisson Regression
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Kline & Boyd Oceanic Technology Dataset

    How is technological complexity in a society related to population size?

    **Estimand**: Influence of **population size** and **contact** on **total tools**
    """)
    return


@app.cell
def kline_data():
    # Load the data
    KLINE = utils.load_data("Kline")
    KLINE
    return (KLINE,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conceptual Ideas
    - The more innovation the more tools
    - The more people, the more innovation
    - The more contact between cultures, the more innovation
    - Innovations (tools) are also forgotten over time, or become obsolete
    """)
    return


@app.cell(hide_code=True)
def tools_concept_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("Population", "Innovation"),
            ("Innovation", "Tools Developed"),
            ("Contact Level", "Innovation"),
            ("Tools Developed", "Tool Loss"),
            ("Tool Loss", "Total Tools Observed")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Scientific Model
    """)
    return


@app.cell(hide_code=True)
def tools_scientific_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("C", "T"),
            ("L", "T"),
            ("L", "P"),
            ("L", "C"),
            ("P", "C"),
            ("P", "T")
        ],
        node_props={
            "L": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Population is treatment
    - Tools is outcome
    - Contact level moderates effect of Population (Pipe)
    - Location is unobserved confound
      - better materials
      - proximity to other cultures
      - can support larger populations
      - _we'll ignore for now_


    Adjustment set for **Direct effect of Population on Tools**
      - Location, if it were observed
      - Also stratify by contact to study interactions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modeling total tools
    - There's no upper limit on tools --> can't use Binomial
    - The **Binomial distribution** approaches the **Poisson distribution** for large $N$ (approaching infinity) and low $p$ (approaching 0)

    ### Poisson GLM

    $$
    \begin{align*}
    Y_i &\sim \text{Poisson}(\lambda_i) \\
    \log(\lambda_i) &= \alpha + \beta x_i
    \end{align*}
    $$

    - link function is $\log(\lambda)$
    - inverse link function is $\exp(\alpha + \beta x_i)$
    - strictly positive $\lambda$ (due to exponential)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Poisson Priors

    - **Be careful with Exponential scaling, it can give shocking results!** Usually long tails result
    - Easier to shift the location (e.g. the mean of a Normal prior), and keep tight variances
    - Prior variances generally need to be quite tight, on the order of 0.1 - 0.5
    """)
    return


@app.cell
def poisson_intercept_priors():
    np.random.seed(123)
    _num_tools = np.arange(1, 100)
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4))

    plt.sca(_axs[0])

    _normal_alpha_prior_params = [
        (0, 10),  # flat, centered prior
        (3, .5)   # offset, tight prior
    ]

    for _prior_mu, _prior_sigma in _normal_alpha_prior_params:
        # since log(lambda) = alpha, if
        # alpha is Normally distributed, therefore lambda is log-normal
        _lambda_prior_dist = stats.lognorm(s=_prior_sigma, scale=np.exp(_prior_mu))
        _label=f"$\\alpha \\sim \\mathcal{{N}}{_prior_mu, _prior_sigma}$"
        _pdf = _lambda_prior_dist.pdf(_num_tools)
        plt.plot(_num_tools, _pdf, label=_label, linewidth=3)

    plt.xlabel("expected # of tools")
    plt.ylabel("density")
    plt.legend()
    plt.title("Prior over $\\lambda$")

    plt.sca(_axs[1])

    _n_prior_samples = 30
    for _ii, (_prior_mu, _prior_sigma) in enumerate(_normal_alpha_prior_params):

        # Sample lambdas from the prior
        _prior_dist = stats.norm(_prior_mu, _prior_sigma)
        _alphas = _prior_dist.rvs(_n_prior_samples)
        _lambdas = np.exp(_alphas)
        _mean_lambda = _lambdas.mean()

        for _sample_idx, _lambda in enumerate(_lambdas):
            _pmf = stats.poisson(_lambda).pmf(_num_tools)

            _label=f"$\\alpha \\sim \\mathcal{{N}}{_prior_mu, _prior_sigma}$" if _sample_idx == 1 else None
            _color=f"C{_ii}"
            plt.plot(_num_tools, _pmf, color=_color, label=_label, alpha=.1)

        _mean_lambda_ = np.exp(_alphas).mean()
        _pmf = stats.poisson(_mean_lambda_).pmf(_num_tools)
        plt.plot(_num_tools, _pmf, color=_color, label=f"Mean $\\lambda$: {_mean_lambda:1,.0f}", alpha=1, linewidth=4)

    plt.xlabel("# of tools")
    plt.ylabel("density")
    plt.ylim([0, .1])
    plt.legend()
    plt.title("Resulting Poisson Samples")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Adding a Slope to the mix $\log(\lambda_i) = \alpha + \beta x_i$
    """)
    return


@app.cell
def poisson_slope_priors():
    np.random.seed(123)
    _normal_alpha_prior_params = [
        (0, 1),  # wide, centered prior
        (3, .5)   # offset, tight prior
    ]

    _normal_beta_prior_params = [
        (0, 10),  # flat, centered prior
        (0, .2)   # tight, centered
    ]

    _n_prior_samples = 10
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4))
    _xs = np.linspace(-2, 2, 100)[:, None]
    for _ii, (_a_params, _b_params) in enumerate(zip(_normal_alpha_prior_params, _normal_beta_prior_params)):
        plt.sca(_axs[_ii])
        _alpha_prior_samples = stats.norm(*_a_params).rvs(_n_prior_samples)
        _beta_prior_samples = stats.norm(*_b_params).rvs(_n_prior_samples)
        _lambda_samples = np.exp(_alpha_prior_samples + _xs * _beta_prior_samples)

        utils.plot_line(_xs, _lambda_samples, label=None, color='C0')
        plt.ylim([0, 100])
        plt.title(f"$\\alpha \\sim Normal{_a_params}$\n$\\beta \\sim Normal{_b_params}$")
        plt.xlabel("x value")
        plt.ylabel("expected count")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Expected Count functions drawn from two different types of priors
    - Left: wide priors
    - Right: tight priors
    """)
    return


@app.cell(hide_code=True)
def poisson_prior_controls():
    poisson_prior_params = mo.ui.dictionary({
        "alpha_mu": mo.ui.slider(-1, 5, step=0.5, value=3, label="$\\mu_\\alpha$", show_value=True),
        "alpha_sigma": mo.ui.slider(steps=[0.1, 0.25, 0.5, 1, 2, 5, 10], value=0.5, label="$\\sigma_\\alpha$", show_value=True),
        "beta_sigma": mo.ui.slider(steps=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10], value=0.2, label="$\\sigma_\\beta$", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: Poisson priors
        Set the priors $\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha)$ and $\beta \sim \mathcal{N}(0, \sigma_\beta)$ of $\log(\lambda) = \alpha + \beta x$. Because $\lambda = \exp(\alpha + \beta x)$, small changes on the log scale are large changes in the expected count: watch how quickly the prior spills past 100 tools as the standard deviations grow.
        """),
        mo.hstack([poisson_prior_params["alpha_mu"], poisson_prior_params["alpha_sigma"], poisson_prior_params["beta_sigma"]], justify="start", gap=2),
    ])
    return (poisson_prior_params,)


@app.cell(hide_code=True)
def poisson_prior_explorer(poisson_prior_params):
    _v = poisson_prior_params.value
    _rng = np.random.default_rng(123)
    _lambda_prior = stats.lognorm(s=_v["alpha_sigma"], scale=np.exp(_v["alpha_mu"]))

    _fig, (_ax_lambda, _ax_curves) = plt.subplots(1, 2, figsize=(11, 4))
    _tools = np.linspace(0.1, 150, 500)
    _ax_lambda.plot(_tools, _lambda_prior.pdf(_tools), color="C1", linewidth=3)
    _ax_lambda.set_xlabel("expected # of tools, $\\lambda$")
    _ax_lambda.set_ylabel("density")
    _ax_lambda.set_title(f"prior over $\\lambda$ at x = 0\n{_lambda_prior.sf(100):.0%} of the prior is above 100 tools")

    _n_curves = 30
    _xs = np.linspace(-2, 2, 100)[:, None]
    _alphas = _rng.normal(_v["alpha_mu"], _v["alpha_sigma"], _n_curves)
    _betas = _rng.normal(0, _v["beta_sigma"], _n_curves)
    _lambdas = np.exp(_alphas + _xs * _betas)
    _ax_curves.plot(_xs, _lambdas, color="C0", alpha=0.5)
    _ax_curves.set_ylim([0, 100])
    _ax_curves.set_xlabel("x value")
    _ax_curves.set_ylabel("expected count")
    _ax_curves.set_title(f"{_n_curves} prior expected-count curves\n{(_lambdas.max(axis=0) > 100).mean():.0%} go above 100 for some -2 < x < 2")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Direct Effect Estimator

    - Estimating <span style="color:red">Direct effect of population P on number of tools T</span> includes contact level, C in the adjustment set and location, L (currently unobserved; we will ignore for now)
    """)
    return


@app.cell(hide_code=True)
def tools_direct_effect_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("L", "C"),
            ("P", "C"),
            ("C", "T"),
            ("L", "T"),
            ("L", "P"),
            ("P", "T")
        ],
        node_props={
            "L": {"label": "location, L", "style": "dashed,filled"},
            "C": {"label": "contact, C", "style": "filled"},
            "P": {"label": "population, P"},
            "T": {"label": "number of tools, T"},
            "unobserved": {"style": "dashed"},
            "Direct Effect\nadjustment set": {"style": "filled"}
        },
        edge_props={
            ("P", "T"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparing Models

    We'll estimate a couple of models in order to practice model comparison
    - Model A) A simple, global intercept Poisson GLM
    - Model B) A Poisson GLM that includes intercept and parameter for the standardized log-population, both of which are stratified by contact level

    ### Model A - Global Intercept model
    Here we model the tool count as a Poisson random variable. The Poisson rate is the exponential of a linear model, and this linear model has only a single global intercept, shared by low- and high-contact populations.

    $$
    \begin{align*}
    T_i &\sim \text{Poisson}(\lambda_i) \\
    \log(\lambda_i) &= \alpha \\
    \alpha &\sim \text{Normal}(3, 0.5) \\
    \end{align*}
    $$
    """)
    return


@app.cell
def tools_data(KLINE):
    # Set up data and coords
    TOOLS = KLINE.total_tools.values.astype(float)
    return (TOOLS,)


@app.cell
def fit_global_intercept(TOOLS):
    with pm.Model() as global_intercept_model:

        # Prior on global intercept
        _alpha = pm.Normal('alpha', 3.0, .5)

        # Likelihood
        _lam = pm.math.exp(_alpha)
        pm.Poisson("tools", _lam, observed=TOOLS)

        global_intercept_inference = cached_sample()
        global_intercept_inference = pm.compute_log_likelihood(global_intercept_inference)

    pm.model_to_graphviz(global_intercept_model)
    return (global_intercept_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model B  - Interaction model

    Here we stratify both the intercept and the population regression coefficient by contact level

    $$
    \begin{align*}
    T_i &\sim \text{Poisson}(\lambda_i) \\
    \log(\lambda_i) &= \alpha_{C[i]} + \beta_{C[i]} \tilde{P}_i \\
    \alpha &\sim \text{Normal}(3, 0.5) \\
    \beta &\sim \text{Normal}(0, 0.2) \\
    \end{align*}
    $$

    where $\tilde{P}_i$ is the standardized log population of society $i$.
    """)
    return


@app.cell
def fit_interaction(KLINE, TOOLS):
    # contact-level
    CONTACT_LEVEL, CONTACT = pd.factorize(KLINE.contact)

    # Standardized log population
    POPULATION = KLINE.population.values.astype(float)
    STD_LOG_POPULATION = utils.standardize(np.log(POPULATION))
    N_CULTURES = len(KLINE)
    OBS_ID = np.arange(N_CULTURES).astype(int)

    with pm.Model(coords={"contact": CONTACT}) as interaction_model:

        # Set up mutable data for predictions
        _std_log_population = pm.Data("population", STD_LOG_POPULATION, dims="obs_id")
        _contact_level = pm.Data("contact_level", CONTACT_LEVEL, dims="obs_id")

        # Priors
        _alpha = pm.Normal("alpha", 3, .5, dims="contact")  # intercept
        _beta = pm.Normal("beta", 0, .2, dims="contact")    # linear interaction with std(log(Population))

        # Likelihood
        _lamb = pm.math.exp(_alpha[_contact_level] + _beta[_contact_level] * _std_log_population)
        pm.Poisson("tools", _lamb, observed=TOOLS, dims="obs_id")

        interaction_inference = cached_sample()

        interaction_inference = pm.compute_log_likelihood(interaction_inference)

    pm.model_to_graphviz(interaction_model)
    return (
        CONTACT,
        CONTACT_LEVEL,
        POPULATION,
        STD_LOG_POPULATION,
        interaction_inference,
        interaction_model,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Comparisons
    """)
    return


@app.cell
def compare_two_models(global_intercept_inference, interaction_inference):
    # Compare the intercept-only and interaction models
    _compare_dict = {
        'global intercept': global_intercept_inference,
        'interaction': interaction_inference
    }
    az.compare(_compare_dict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > pPSIS discussed in the lecture is analogous to `p_loo` in the output above
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posterior Predictions
    """)
    return


@app.cell
def tools_ppd_plot_helper(
    CONTACT_LEVEL,
    POPULATION,
    STD_LOG_POPULATION,
    TOOLS,
):
    def plot_tools_model_posterior_predictive(model, inference, title, input_natural_scale=False, plot_natural_scale=True, resolution=100):

        # Set up population grid based on model input scale
        if input_natural_scale:
            ppd_population_grid = np.linspace(0, np.max(POPULATION) * 1.05, resolution)

        # input is standardized log scale
        else:
            ppd_population_grid = np.linspace(
                STD_LOG_POPULATION.min() * 1.05,
                STD_LOG_POPULATION.max() * 1.05,
                resolution
            )

        # Set up contact-level counterfactuals
        with model:
            # Predictions for low contact
            pm.set_data(
                {
                    'contact_level':np.array([0] * resolution),
                    'population': ppd_population_grid
                }
            )

            low_contact_ppd = pm.sample_posterior_predictive(
                inference,
                var_names=["tools"],
                predictions=True,
                random_seed=1,
            )["predictions"]["tools"]

            # Predictions for high contact
            pm.set_data(
                {
                    'contact_level':np.array([1] * resolution),
                    'population': ppd_population_grid
                }
            )

            hi_contact_ppd = pm.sample_posterior_predictive(
                inference,
                var_names=["tools"],
                predictions=True,
                random_seed=1,
            )["predictions"]["tools"]

            low_contact_ppd_mean = low_contact_ppd.mean(["chain", "draw"])
            hi_contact_ppd_mean = hi_contact_ppd.mean(["chain", "draw"])

            colors = ["C0" if c else "C1" for c in CONTACT_LEVEL]

            # Set up visualization scale
            if plot_natural_scale:
                if input_natural_scale:
                    population_grid = ppd_population_grid
                else:
                    population_grid = np.exp(ppd_population_grid * np.log(POPULATION).std() + np.log(POPULATION).mean())

                scatter_population = POPULATION
                xlabel = 'population'
            # visualize in log scale
            else:
                if input_natural_scale:
                    population_grid = np.log(ppd_population_grid)
                else:
                    population_grid = ppd_population_grid

                scatter_population = STD_LOG_POPULATION
                xlabel = 'population (standardized log)'

            marker_size = 50 + 400 * POPULATION / POPULATION.max()
            utils.plot_scatter(xs=scatter_population, ys=TOOLS, color=colors, s=marker_size, facecolors=None, alpha=1)

            # low-contact posterior predictive
            utils.plot_hdi(
                x=population_grid,
                y=low_contact_ppd, color="C1",
                hdi_prob=.89,
                fill_kwargs={"alpha": 0.2}
            )
            plt.plot(population_grid, low_contact_ppd_mean, color="C1", label='low contact')

            # high-contact posterior predictive
            utils.plot_hdi(
                x=population_grid,
                y=hi_contact_ppd,
                color="C0",
                hdi_prob=.89,
                fill_kwargs={"alpha": 0.2}
            )
            plt.plot(population_grid, hi_contact_ppd_mean, color="C0", label='hi contact')
            plt.legend()

            plt.xlabel(xlabel)
            plt.ylabel('total tools')
            plt.title(title)

    return (plot_tools_model_posterior_predictive,)


@app.cell(hide_code=True)
def plot_interaction_log_scale(
    interaction_inference,
    interaction_model,
    plot_tools_model_posterior_predictive,
):
    _fig, _ax = plt.subplots()
    plot_tools_model_posterior_predictive(
        interaction_model,
        interaction_inference,
        title="Interaction Model\nLog Population Scale",
        plot_natural_scale=False
    )
    _fig
    return


@app.cell(hide_code=True)
def plot_interaction_natural_scale(
    interaction_inference,
    interaction_model,
    plot_tools_model_posterior_predictive,
):
    _fig, _ax = plt.subplots()
    plot_tools_model_posterior_predictive(
        interaction_model,
        interaction_inference,
        title="Interaction Model\nNatural Population Scale",
        plot_natural_scale=True
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The model above is "wack" for the following reasons:

    1. The low-contact mean intersects with the high-contact mean (around population of 150k). This makes little sense logically
    2. A population of 0 should have close to 0 tools, but the curves start at around 20 (low contact) and 30 (high contact) tools for the smallest populations.
    3. Tonga, the largest high-contact society, sits at the upper edge of the high-contact 89% HDI
    4. Error for hi-contact group is absurdly large for a majority of the population range

    Can we do better?

    ## Improving the estimator with a better scientific model

    There are two immediate ways to improve the model, including:

    1. Robust regression model -- in this case a gamma-Poisson (neg-binomial) model
    2. **Use a more principled scientific model**

    ## Scientific model that includes innovation and technology loss
    Recall this DAG from earlier that highlights the general conceptual idea of how observed tools can arise:
    """)
    return


@app.cell(hide_code=True)
def tools_concept_dag_repeat():
    utils.draw_causal_graph(
        edge_list=[
            ("Population", "Innovation"),
            ("Innovation", "Tools Developed"),
            ("Contact Level", "Innovation"),
            ("Tools Developed", "Tool Loss"),
            ("Tool Loss", "Total Tools Observed")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Why not develop a scientific model that does just that?**

    ### Using the difference equation: $\Delta T = \alpha P^\beta - \gamma T$

    - $\Delta T$ is the change in # of tools given the current number of tools. Here T can be thought of as # of tools at current generation
    - $\alpha$ is the innovation rate for a population of size $P$
    - $\beta$ is the elasticity, and can be thought of as a saturation rate, or "diminishing returns" factor; if we constrain $0 < \beta < 1$
    - $\gamma$ is the attrition / technology loss rate at time T

    Furthermore we can parameterize such an equation by the class of contact rate, $C$ as $\Delta T = \alpha_C P^{\beta_C} - \gamma T$

    Now we leverage the notion of **equilibrium** to identify the steady state # of tools that are eventually obtained. At this point $\Delta T = 0$, and we can solve for the resulting $\hat T$ using algebra:

    $$
    \begin{align*}
    \Delta T &= \alpha P^\beta - \gamma \hat T = 0 \\
    \gamma \hat T &= \alpha P^\beta\\
    \hat T &= \frac{\alpha_C P^{\beta_C}}{\gamma}
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulate the difference equation for various societies
    """)
    return


@app.function
def simulate_tools(alpha=.25, beta=.25, P=1e3, gamma=.25, n_generations=30, color='C0'):
    """Simulate the Tools data as additive difference process equilibrium condition"""
    def difference_equation(t):
        return  alpha * P ** beta - gamma * t

    # Run the simulation
    tools = [0]
    generations = range(n_generations)
    for g in generations:
        t = tools[-1]
        tools.append(t + difference_equation(t))

    t_equilibrium = (alpha * P ** beta) / gamma

    # Plot it
    plt.plot(generations, tools[:-1], color=color)
    plt.axhline(
        t_equilibrium,
        color=color,
        linestyle='--',
        label=f'equilibrium, P={int(P):,}'
    )
    plt.legend()


@app.cell
def difference_equation_simulation():
    _fig, _ax = plt.subplots()
    simulate_tools(P=1e3, color='C0')
    simulate_tools(P=1e4, color='C1')
    simulate_tools(P=300_000, color='C2')
    plt.xlabel("Generation")
    plt.ylabel("# Tools")
    _fig
    return


@app.cell(hide_code=True)
def innovation_controls():
    innovation_params = mo.ui.dictionary({
        "alpha": mo.ui.slider(0.05, 1, step=0.05, value=0.25, label="$\\alpha$: innovation rate", show_value=True),
        "beta": mo.ui.slider(0.05, 1, step=0.05, value=0.25, label="$\\beta$: elasticity", show_value=True),
        "gamma": mo.ui.slider(0.05, 1, step=0.05, value=0.25, label="$\\gamma$: loss rate", show_value=True),
        "population": mo.ui.slider(steps=[100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000], value=1_000, label="population, P", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: innovation and loss
        Each generation adds $\alpha P^\beta$ new tools and loses a fraction $\gamma$ of the existing ones, so the tool count settles at $\hat T = \alpha P^\beta / \gamma$. The left panel follows one society over the generations. The right panel shows the equilibrium for every population size, with the Kline data. Can you find values that run through the low-contact or the high-contact societies?
        """),
        mo.hstack([
            mo.vstack([innovation_params["alpha"], innovation_params["beta"]]),
            mo.vstack([innovation_params["gamma"], innovation_params["population"]]),
        ], justify="start", gap=4),
    ])
    return (innovation_params,)


@app.cell(hide_code=True)
def innovation_explorer(CONTACT_LEVEL, POPULATION, TOOLS, innovation_params):
    _v = innovation_params.value
    _alpha, _beta, _gamma, _P = _v["alpha"], _v["beta"], _v["gamma"], _v["population"]

    _n_generations = 40
    _tools = [0.0]
    for _ in range(_n_generations):
        _tools.append(_tools[-1] + _alpha * _P ** _beta - _gamma * _tools[-1])
    _equilibrium = _alpha * _P ** _beta / _gamma

    _fig, (_ax_time, _ax_pop) = plt.subplots(1, 2, figsize=(12, 4))
    _ax_time.plot(range(_n_generations + 1), _tools, color="C2")
    _ax_time.axhline(_equilibrium, color="C2", linestyle="--", label=f"equilibrium $\\hat T$ = {_equilibrium:.1f}")
    _ax_time.set_xlabel("generation")
    _ax_time.set_ylabel("# tools")
    _ax_time.set_title(f"P = {_P:,}")
    _ax_time.legend(fontsize=8)

    _population_grid = np.linspace(0, 300_000, 500)
    _ax_pop.plot(_population_grid, _alpha * _population_grid ** _beta / _gamma, color="C2", label="equilibrium $\\hat T$")
    for _level, _label, _color in [(0, "low contact", "C1"), (1, "high contact", "C0")]:
        _is_level = CONTACT_LEVEL == _level
        _ax_pop.scatter(POPULATION[_is_level], TOOLS[_is_level], color=_color, label=_label, zorder=3)
    _ax_pop.axvline(_P, color="gray", linestyle=":")
    _ax_pop.set_ylim([0, 120])
    _ax_pop.set_xticks([0, 100_000, 200_000, 300_000], ["0", "100k", "200k", "300k"])
    _ax_pop.set_xlabel("population")
    _ax_pop.set_ylabel("total tools")
    _ax_pop.legend(fontsize=8, loc="lower right")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Innovation / Loss Statistical Model
    - Use $\hat T = \lambda$ as the expected number of tools, i.e. $T \sim \text{Poisson}(\hat T)$
    - **Note**: we must constrain $\lambda$ to be positive, which we can do in a couple of ways:
      1. Exponentiate variables
      2. Use appropriate priors that constrain the variables to be positive (we'll use this approach)

    $$
    \begin{align*}
    T &\sim \text{Poisson}(\hat T_i) \\
    \hat T_i &= \frac{\alpha_{C[i]} P^{\beta_{C[i]}}}{\gamma} \\
    \alpha_j, \beta_j, \gamma &\sim \text{Exponential}(\eta)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Determine good prior hyperparams
    We'll use an Exponential distribution as a prior on the difference equation parameters $\alpha, \beta, \gamma$. We thus need to identify a good rate hyperparameter $\eta$ for those priors.

    Reasonable values for all parameters were approximately 0.25 in the simulation above. We would thus like to choose the Exponential rate parameter $\eta$ whose mean $1/\eta$ is 0.25, i.e. $\eta = 4$.
    """)
    return


@app.cell
def fit_innovation_loss(CONTACT, CONTACT_LEVEL, POPULATION, TOOLS):
    _ETA = 4
    with pm.Model(coords={"contact": CONTACT}) as innovation_loss_model:

        # Note: raw population here, not log/standardized
        _population = pm.Data("population", POPULATION, dims="obs_id")
        _contact_level = pm.Data("contact_level", CONTACT_LEVEL, dims="obs_id")

        # Priors -- we use Exponential for all.
        # Note that in the lecture: McElreath uses a Normal for alpha
        # then applies a exp(alpha) to enforce positive contact-level
        # innovation rate
        _alpha = pm.Exponential("alpha", _ETA, dims="contact")

        # contact-level elasticity
        _beta = pm.Exponential("beta", _ETA, dims="contact")

        # global technology loss rate
        _gamma = pm.Exponential("gamma", _ETA)

        # Likelihood using difference equation equilibrium as mean Poisson rate
        _T_hat = (_alpha[_contact_level] * (_population ** _beta[_contact_level])) / _gamma
        pm.Poisson("tools", _T_hat, observed=TOOLS, dims="obs_id")

        innovation_loss_inference = cached_sample(tune=2000, target_accept=.98)
        innovation_loss_inference = pm.compute_log_likelihood(innovation_loss_inference)


    pm.model_to_graphviz(innovation_loss_model)
    return innovation_loss_inference, innovation_loss_model


@app.cell
def summary_innovation_loss(innovation_loss_inference):
    az.summary(innovation_loss_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The posterior means are about 0.3 for both $\alpha$s, 0.26 (low contact) and 0.30 (high contact) for $\beta$, and 0.12 for $\gamma$. $\alpha$ and $\gamma$ have wide intervals because the equilibrium only depends on their ratio $\alpha_C / \gamma$, so the data can't tell them apart and the prior has to. We could reparameterize the model in terms of that ratio, or use a tighter prior on $\gamma$, but meh.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posterior predictions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Replot the contact model for comparison
    """)
    return


@app.cell(hide_code=True)
def plot_interaction_for_comparison(
    interaction_inference,
    interaction_model,
    plot_tools_model_posterior_predictive,
):
    _fig, _ax = plt.subplots()
    plot_tools_model_posterior_predictive(
        interaction_model,
        interaction_inference,
        title="Interaction Model\nNatural Population Scale",
        plot_natural_scale=True
    )
    _fig
    return


@app.cell(hide_code=True)
def plot_innovation_loss(
    innovation_loss_inference,
    innovation_loss_model,
    plot_tools_model_posterior_predictive,
):
    _fig, _ax = plt.subplots()
    plot_tools_model_posterior_predictive(
        innovation_loss_model,
        innovation_loss_inference,
        title="Innovation Model\nNatural Population Scale",
        input_natural_scale=True,
        plot_natural_scale=True
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice the following improvements over the basic interaction model
    - No weird crossover of low/high contact trends
    - zero population now associated with zero tools
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Comparisons
    """)
    return


@app.cell
def compare_three_models(
    global_intercept_inference,
    innovation_loss_inference,
    interaction_inference,
):
    _compare_dict = {
        'global intercept': global_intercept_inference,
        'interaction': interaction_inference,
        'innovation loss' : innovation_loss_inference
    }
    az.compare(_compare_dict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The innovation / loss model ranks first and gets the most weight (about 0.6), but it is only about 1 point of elpd ahead of the interaction model, well within the standard error of that difference. Both are far ahead of the global-intercept model. The real gain of the scientific model is that its predictions make sense.

    ### Take-homes
    - Generally best to have a domain-informed scientific model
    - We still have the unobserved location confound to deal with
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Review: Count GLMs
    - MaxEnt priors
      - Binomial
      - Poisson & Extensions
      - `log` link function; `exp` inverse link function


    - Robust Regression
      - Beta-Binomial
      - Gamma-Poisson
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: Simpson's Pandora's Box

    **The reversal of some measured/estimated association when groups are either combined or separated.**

    - There is nothing particularly interesting or Paradoxical about Simpson's paradox
    - It is simply a statistical phenomenon
    - There are any number of causal phenomena that can create SP
      - Pipes and Forks can cause one flavor of SP -- namely stratifying destroys trend / associations
      - Collider can cause the inverse flavor -- namely stratifying elicits a trend / association
    - You can't say one way or the other which direction of "reversal" is correct without making explicit causal claims

    ### Classic example is UC Berkeley Admissions
    - If you do not stratify/condition on Department, you find that Females are Admitted at a lower rate
    - If you stratify/condition on Department, you find that Females are Admitted at a slightly higher rate (see the admissions analyses in Lecture 09)
    - Which is correct? Could be explained by either:
      - a mediator/pipe (department)
      - a collider + confound (unobserved ability)

    **For examples of how Pipes, Forks, and Colliders can "lead to" Simpson's paradox, see [Lecture 05 - Elemental Confounds](<Lecture 05 - Elemental Confounds.py>)**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nonlinear Haunting

    Though $Z$ is not a confound, it is a competing cause of $Y$. If the causal model is nonlinear and we stratify by $Z$ to get the direct causal effect of the treatment on the outcome, this can cause some strange outcomes akin to Simpson's paradox.
    """)
    return


@app.cell(hide_code=True)
def haunting_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("treatment, X", "outcome, Y"),
            ("covariate, Z", "outcome, Y")
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: Base Rate Differences

    Here we simulate data where $X$ and $Z$ are independent, but $Z$ has a nonlinear causal effect on $Y$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Generative Simulation
    """)
    return


@app.cell
def haunting_simulation():
    np.random.seed(123)
    _n_simulations = 1000

    X = stats.norm.rvs(size=_n_simulations)
    Z = stats.bernoulli.rvs(p=.5, size=_n_simulations)

    # encode a nonlinear effect of Z on Y
    _BETA_XY = 1
    _BETA_Z0Y = 5
    _BETA_Z1Y = -1

    _p = utils.invlogit(X * _BETA_XY + np.where(Z, _BETA_Z1Y, _BETA_Z0Y))
    Y = stats.bernoulli.rvs(p=_p)

    _fig, _ax = plt.subplots(figsize=(6, 3))
    plt.hist(_p, bins=25)
    plt.xlabel("p(Y = 1)")
    plt.title("Probabilities")
    _fig
    return X, Y, Z


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Unstratified Model -- $\text{logit}(p_i) = \alpha + \beta X_i$
    """)
    return


@app.cell
def fit_unstratified(X, Y):
    # Unstratified model
    with pm.Model() as unstratified_model:

        # MutableData for PPDs
        _x = pm.Data("X", X)

        # Global params
        _alpha = pm.Normal("alpha", 0, 1)
        _beta = pm.Normal("beta", 0, 1)

        # record p for plotting predictions
        _p = pm.Deterministic("p", pm.math.invlogit(_alpha + _beta * _x))
        pm.Bernoulli("Y", p=_p, observed=Y)

        unstratified_inference = cached_sample()

    pm.model_to_graphviz(unstratified_model)
    return unstratified_inference, unstratified_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Partially Stratified Model -- $\text{logit}(p) = \alpha + \beta_{Z[i]} X_i$
    """)
    return


@app.cell
def fit_partially_stratified(X, Y, Z):
    # Partially stratified model
    with pm.Model() as partially_stratified_model:

        # Mutable data for PPDs
        _x = pm.Data("X", X)
        _z = pm.Data("Z", Z)

        _alpha = pm.Normal("alpha", 0, 1)
        _beta = pm.Normal("beta", 0, 1, shape=2)

        # record p for plotting predictions
        _p = pm.Deterministic("p", pm.math.invlogit(_alpha + _beta[_z] * _x))
        pm.Bernoulli("Y", p=_p, observed=Y)

        partially_stratified_inference = cached_sample()


    pm.model_to_graphviz(partially_stratified_model)
    return partially_stratified_inference, partially_stratified_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Unstratified model posterior predictions
    """)
    return


@app.cell
def predict_unstratified(unstratified_inference, unstratified_model):
    # Get predictions -- unstratified model
    RESOLUTION = 100
    xs = np.linspace(-3, 3, RESOLUTION)

    with unstratified_model:
        pm.set_data({'X':xs})
        unstratified_ppd = pm.sample_posterior_predictive(
            unstratified_inference,
            var_names=["p"]
        )["posterior_predictive"]["p"]
    return RESOLUTION, unstratified_ppd, xs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Partially Stratified model posterior predictions
    """)
    return


@app.cell
def predict_partially_stratified(
    RESOLUTION,
    partially_stratified_inference,
    partially_stratified_model,
    xs,
):
    # Get predictions -- partially stratified model
    partially_stratified_predictions = {}
    with partially_stratified_model:
        for _z in [0, 1]:
            # Predictions for this value of Z
            pm.set_data({'X':xs})
            pm.set_data({'Z':[_z] * RESOLUTION})
            partially_stratified_predictions[_z] = pm.sample_posterior_predictive(
                partially_stratified_inference,
                var_names=["p"]
            )["posterior_predictive"]["p"]
    return (partially_stratified_predictions,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Plot the effect of stratification
    """)
    return


@app.cell(hide_code=True)
def plot_stratification_effect(
    partially_stratified_predictions,
    unstratified_ppd,
    xs,
):
    _n_posterior_samples = 20
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(_axs[0])

    # Plot predictions -- unstratified model
    plt.plot(xs, unstratified_ppd.sel(chain=0)[:_n_posterior_samples].T, color='C0')
    plt.xlabel("treatment, X")
    plt.ylabel("p(Y)")
    plt.ylim([0, 1])
    plt.title("$logit(p_i) = \\alpha + \\beta x_i$")

    # For tagging conditions in legend
    legend_data = [
        Line2D([0], [0], color="C0", lw=4),
        Line2D([0], [0], color="C1", lw=4),
    ]


    plt.sca(_axs[1])
    for _z in [0, 1]:
        _ys = partially_stratified_predictions[_z].sel(chain=0)[:_n_posterior_samples].T
        plt.plot(xs,_ys, color=f'C{_z}', label=f"Z={_z}")

    plt.xlabel("treatment, X")
    plt.ylabel("p(Y)")
    plt.ylim([0, 1])

    plt.legend(legend_data, ['Z=0', 'Z=1'])

    plt.title("$logit(p_i) = \\alpha + \\beta_{Z[i]}x_i$")
    _fig
    return (legend_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When stratifying only on the X coefficient, and thus sharing a common intercept, we can see that for Z=0, the predictions saturate at about 0.62–0.66. This is due to the +5 added to the log odds of Y|Z=0 in the logistic regression model. Because of this saturation, it's difficult to tell if the treatment affects the outcome for that group.
    """)
    return


@app.cell(hide_code=True)
def plot_partially_stratified_posterior(partially_stratified_inference):
    _fig, _axs = plt.subplots(figsize=(5, 3))
    for _z in [0, 1]:
        _post = partially_stratified_inference.posterior.sel(chain=0)['beta'][:, _z]
        utils.plot_dist(_post, color=f"C{_z}", label=f'Z={_z}')
    plt.xlabel("$\\beta_Z$")
    plt.ylabel("density")
    plt.legend()
    plt.title("Partially Stratified Posterior")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Try a fully-stratified model -- $\text{logit}(p_i) = \alpha_{Z[i]} + \beta_{Z[i]}X_i$

    Include a separate intercept for each group
    """)
    return


@app.cell
def fit_fully_stratified(X, Y, Z):
    # Fully stratified model
    with pm.Model() as fully_stratified_model:

        _x = pm.Data("X", X)
        _z = pm.Data("Z", Z)

        # Stratify intercept by Z as well
        _alpha = pm.Normal("alpha", 0, 1, shape=2)
        _beta = pm.Normal("beta", 0, 1, shape=2)

        # Log p for plotting predictions
        _p = pm.Deterministic('p', pm.math.invlogit(_alpha[_z] + _beta[_z] * _x))
        pm.Bernoulli("Y", p=_p, observed=Y)
        fully_stratified_inference = cached_sample()


    pm.model_to_graphviz(fully_stratified_model)
    return fully_stratified_inference, fully_stratified_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fully Stratified Model posterior predictions
    """)
    return


@app.cell
def predict_fully_stratified(
    RESOLUTION,
    fully_stratified_inference,
    fully_stratified_model,
    xs,
):
    # Get predictions -- fully stratified model
    fully_stratified_predictions = {}
    with fully_stratified_model:
        for _z in [0, 1]:
            # Predictions for this value of Z
            pm.set_data({'X': xs})
            pm.set_data({'Z': [_z] * RESOLUTION})
            fully_stratified_predictions[_z] = pm.sample_posterior_predictive(
                fully_stratified_inference,
                var_names=["p"]
            )["posterior_predictive"]["p"]
    return (fully_stratified_predictions,)


@app.cell(hide_code=True)
def plot_full_vs_partial_predictions(
    fully_stratified_predictions,
    legend_data,
    partially_stratified_predictions,
    xs,
):
    _n_posterior_samples = 20
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(_axs[0])

    for _z in [0, 1]:
        _ys = partially_stratified_predictions[_z].sel(chain=0)[:_n_posterior_samples].T
        plt.plot(xs,_ys, color=f'C{_z}', label=f"Z={_z}")

    plt.xlabel("treatment, X")
    plt.ylabel("p(Y)")
    plt.ylim([0, 1])

    plt.legend(legend_data, ['Z=0', 'Z=1'])
    plt.title("$logit(p_i) = \\alpha + \\beta_{Z[i]}X_i$")

    plt.sca(_axs[1])
    for _z in [0, 1]:
        _ys = fully_stratified_predictions[_z].sel(chain=0)[:_n_posterior_samples].T
        plt.plot(xs,_ys, color=f'C{_z}', label=f"Z={_z}")

    plt.xlabel("treatment, X")
    plt.ylabel("p(Y)")
    plt.ylim([0, 1])

    plt.legend(legend_data, ['Z=0', 'Z=1'])

    plt.title("$logit(p_i) = \\alpha_{Z[i]} + \\beta_{Z[i]}X_i$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we can see that with a fully stratified model, one in which we include a group-level intercept, the predictions for Z=0 shift up even higher toward one, though the predictions remain mostly flat across all values of the treatment X
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare Posteriors
    """)
    return


@app.cell(hide_code=True)
def compare_beta_posteriors(
    fully_stratified_inference,
    partially_stratified_inference,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 3))

    plt.sca(_axs[0])

    for _z in [0, 1]:
        _post = partially_stratified_inference.posterior.sel(chain=0)['beta'][:, _z]
        utils.plot_dist(_post, color=f"C{_z}", label=f'Z={_z}')
    plt.xlabel("$\\beta_Z$")
    plt.ylabel("density")
    plt.legend()
    plt.title("Partially Stratified Posterior")


    plt.sca(_axs[1])
    for _z in [0, 1]:
        _post = fully_stratified_inference.posterior.sel(chain=0)['beta'][:, _z]
        utils.plot_dist(_post, color=f"C{_z}", label=f'Z={_z}')
    plt.xlabel("$\\beta_Z$")
    plt.ylabel("density")
    plt.legend()
    plt.title("Fully Stratified Posterior")
    _fig
    return


@app.cell(hide_code=True)
def haunting_controls():
    haunting_params = mo.ui.dictionary({
        "beta_xy": mo.ui.slider(0, 3, step=0.25, value=1, label="effect of X, $\\beta_{XY}$", show_value=True),
        "offset_z0": mo.ui.slider(-3, 6, step=0.5, value=5, label="log-odds offset for Z = 0", show_value=True),
        "offset_z1": mo.ui.slider(-3, 6, step=0.5, value=-1, label="log-odds offset for Z = 1", show_value=True),
    })
    mo.vstack([
        mo.md(r"""
        #### Explore: nonlinear haunting
        Simulate the data above with your own settings, then fit the partially and the fully stratified models. To keep it instant, each model is fit by finding its posterior mode (MAP), with the same $\mathcal{N}(0, 1)$ priors, instead of sampling. The dashed lines are the true curves. When one group's offset is large, its probabilities saturate near 1, and a shared intercept can't fit both groups: the slope for that group gets pulled toward 0.
        """),
        mo.hstack([haunting_params["beta_xy"], haunting_params["offset_z0"], haunting_params["offset_z1"]], justify="start", gap=2),
    ])
    return (haunting_params,)


@app.cell(hide_code=True)
def haunting_explorer(haunting_params):
    from scipy.optimize import minimize as _minimize

    _v = haunting_params.value
    _rng = np.random.default_rng(123)
    _n = 1000
    _x = _rng.normal(size=_n)
    _z = _rng.binomial(1, 0.5, size=_n)
    _offsets = np.array([_v["offset_z0"], _v["offset_z1"]])
    _y = _rng.binomial(1, utils.invlogit(_v["beta_xy"] * _x + _offsets[_z]))


    def _fit_map(design):
        """Posterior mode of a logistic regression with Normal(0, 1) priors on all coefficients"""
        def neg_log_posterior(theta):
            eta = design @ theta
            return np.sum(np.logaddexp(0, eta) - _y * eta) + 0.5 * np.sum(theta ** 2)

        def gradient(theta):
            return design.T @ (utils.invlogit(design @ theta) - _y) + theta

        return _minimize(neg_log_posterior, np.zeros(design.shape[1]), jac=gradient, method="BFGS").x


    _is_z = np.column_stack([_z == 0, _z == 1]).astype(float)
    _alpha, _b0, _b1 = _fit_map(np.column_stack([np.ones(_n), _x[:, None] * _is_z]))
    _a0, _a1, _c0, _c1 = _fit_map(np.column_stack([_is_z, _x[:, None] * _is_z]))

    _xs = np.linspace(-3, 3, 100)
    _fits = {
        "partially stratified": ([_alpha, _alpha], [_b0, _b1], "$logit(p_i) = \\alpha + \\beta_{Z[i]}X_i$"),
        "fully stratified": ([_a0, _a1], [_c0, _c1], "$logit(p_i) = \\alpha_{Z[i]} + \\beta_{Z[i]}X_i$"),
    }
    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4))
    for _ax, (_name, (_intercepts, _slopes, _formula)) in zip(_axs, _fits.items()):
        for _zz in [0, 1]:
            _ax.plot(_xs, utils.invlogit(_v["beta_xy"] * _xs + _offsets[_zz]), color=f"C{_zz}", linestyle="--")
            _ax.plot(_xs, utils.invlogit(_intercepts[_zz] + _slopes[_zz] * _xs), color=f"C{_zz}", linewidth=3)
        _ax.set_ylim([0, 1])
        _ax.set_xlabel("treatment, X")
        _ax.set_ylabel("p(Y)")
        _ax.set_title(f"{_formula}\n$\\beta_{{Z=0}}$ = {_slopes[0]:.2f}, $\\beta_{{Z=1}}$ = {_slopes[1]:.2f} (true {_v['beta_xy']:g})")
    _axs[0].legend(
        [Line2D([0], [0], color="C0", lw=3), Line2D([0], [0], color="C1", lw=3), Line2D([0], [0], color="k", linestyle="--")],
        ["Z=0, MAP fit", "Z=1, MAP fit", "true curves"],
        fontsize=8,
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simpson's Paradox Summary
    - No paradox, almost anything can produce SP
    - Coefficient reversals have little interpretive value outside of causal framework
    - **Don't focus on coefficients: push predictions through model to compare**
    - Random note: you can't accept the NULL, you can only reject it.
      - Just because a distribution overlaps 0 doesn't mean it's zero
    """)
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
