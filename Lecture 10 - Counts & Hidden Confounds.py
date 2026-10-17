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
    - Predictor variables interact -- causal intererpretation of coefficients (outside of simplest models) is fraught with misleading conclusions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Confounded Admissions
    """)
    return


@app.cell(hide_code=True)
def _():
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
    - However, it's implausible that there are no uobserved confounds between variables,
      - e.g. Applicant ability could link Department to Admission rate
        - affects which students apply to each department (more ability biases department application)
        - also affect baseline admission rates (more ability leads to higher admission rates)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Generative Simulation
    """)
    return


@app.cell
def _():
    np.random.seed(1)

    # Number of applicants
    n_samples = 2000

    # Gender equally likely
    G = np.random.choice([0, 1], size=n_samples, replace=True)

    # Unobserved Ability -- 10% have High, everyone else is Average
    _u = stats.bernoulli.rvs(p=0.1, size=n_samples)

    # Choice of department
    # G0 applies to D0 with 75% probability else D1 with 1% or 0% based on ability
    D = stats.bernoulli.rvs(p=np.where(G==0, _u * 1.0, 0.75))

    # Ability-based acceptance rates (ability x dept x gender)
    _p_u0_dg = np.array([[0.1, 0.1], [0.1, 0.3]])
    _p_u1_dg = np.array([[0.3, 0.5], [0.3, 0.5]])
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
def _():
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
def _(D, G):
    # Define coordinates
    GENDER_ID, GENDER = pd.factorize(np.array(["G2" if g else "G1" for g in G]), sort=True)
    DEPTARTMENT_ID, DEPARTMENT = pd.factorize(np.array(["D2" if d else "D1" for d in D]), sort=True)
    return DEPARTMENT, DEPTARTMENT_ID, GENDER, GENDER_ID


@app.cell
def _(A, GENDER, GENDER_ID):
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
def _(total_effect_admissions_model):
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
def _(total_effect_admissions_inference):
    summarize_posterior(total_effect_admissions_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Direct Effect Estimator (now confounded due to common ability cause)
    """)
    return


@app.cell(hide_code=True)
def _():
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
    \alpha &= \begin{bmatrix} \alpha_{0,0}, \alpha_{0,1} \\  \alpha_{1,0}, \alpha_{1,1} \end{bmatrix} \\
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
def _(A, DEPARTMENT, DEPTARTMENT_ID, GENDER, GENDER_ID):
    with pm.Model(coords={"gender": GENDER, "department": DEPARTMENT}) as direct_effect_admissions_model:
        # Prior
        _alpha = pm.Normal("alpha", 0, 1, dims=["department", "gender"])

        # Likelihood
        _p = pm.math.invlogit(_alpha[DEPTARTMENT_ID, GENDER_ID])
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
def _(direct_effect_admissions_inference):
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
def _(direct_effect_admissions_inference):
    plot_department_gender_admissions(direct_effect_admissions_inference, 'direct effect, confounded model')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Interpreting the Confounded Direct Effect Model
    - In D1, G1 appears to be disatvantaged, with a lower admission rate. However, we know this isn't true. What's happening is that all the higher-ability G1 applicants are being sent to D2, thus artificially lowering the G1 acceptance rate in D1
    - We know that there is bias in D2, however, we see little evidence for discrimination. This is due to higher-ability G1 applicatins ofsetting this bias by having higher-than-average acceptance
    - We can see that G1 estimates for D2 have higher variance; this is due to there being only 10% of applicants having high ability, thus fewer G1 applicants overall apply to D2.

    #### You guessed it: Collider Bias
    - **This is due to _collider bias_**
      - stratifying by Department--which forms a colider with Gender and ability--opens a path through the ability to acceptance.
      - **You CANNOT estimate _Direct effect_ of $D$ on $G$**
      - **You CAN estimate the _Total effect_**
    - **sorting can mask or accentuate bias**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analogous Example: NAS Membership & Citations

    Two papers
    - same data
    - find drastically different conclusions about Gender and its effect on admission to the National Academy of Sciences (NAS)
      - One found Women are strongly advantaged
      - The other found woemen strongly disadvantaged
    - How can both conclusions be true?
    """)
    return


@app.cell(hide_code=True)
def _():
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
    - There are likely latent Researcher quality difference
    - Stratifying by number of citations opens up a collider bias with unobserved Researcher quality
      Citations is a **post-treatment variable**
    - Citation-stratification provide misleading conclusions
    - e.g. if there is discrimination in publication/citation, one gender may get elected at a higher rate just because they will have higher quality on on average for any citation level

    ### No Causes in, no causes out

    These papers suffere from a number of shortcomings

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

    - Estimatinge <span style="color:red">Direct effect</span> includes Department in the adjustment set
    - However, stratifying by Department (collider)  <span style="color:blue">opens a confounder backdoor path </span> through unobserved ability, u
    """)
    return


@app.cell(hide_code=True)
def _():
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

    In this particular example, we can simulate the degree of effect of an ability random variable $U \sim \text{Normal}(0, 1)$, by adding a linearly weighted contribution of that variable to the log odds of Acceptance and selecting a department (this is because we u affect both D and A in our causal graph):

    #### Department submodel
    $$
    \begin{align*}
    D_i &\sim \text{Bernouilli}(q_i) \\
    logit(q_i) &= \delta[G_i] + \gamma_{G[i]}  u_i \\
    \delta[G_i] &\sim \text{Normal}(0, 1) \\
    u_j &\sim \text{Normal}(0, 1)
    \end{align*}
    $$

    #### Acceptance submodel
    $$
    \begin{align*}
    A_i &\sim \text{Bernouilli}(p_i) \\
    logit(p_i) &= \alpha[G_i, D_i] + \beta_{G[i]} u_i \\
    \alpha[G_i, D_i] &\sim \text{Normal}(0, 1) \\
    u_j &\sim \text{Normal}(0, 1)
    \end{align*}
    $$

    Where we <span style="color:green"> **manually set the value of $\beta_{G[i]}$ and $\gamma_{G[i]}$ by hand</span>  to perform the simulation**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the latent ability model
    """)
    return


@app.cell
def _(A, D, DEPARTMENT, GENDER, GENDER_ID, n_samples):
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

        _selected_department = pm.Bernoulli('d', p=_q, observed=D)

        # Acceptance submodel
        _alpha = pm.Normal("alpha", 0, 1, dims=["department", "gender"])
        _p = pm.math.invlogit(_alpha[GENDER_ID, _selected_department] + _BETA[GENDER_ID] * _U)
        pm.Bernoulli('accepted', p=_p, observed=A)

        # Record p(A | D, G) for reporting
        _p_admit = pm.Deterministic("p_admit", pm.math.invlogit(_alpha), dims=['department', 'gender'])

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
def _(latent_ability_inference):
    summarize_posterior(latent_ability_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpreting the Effect of modeling the confound
    """)
    return


@app.cell(hide_code=True)
def _(direct_effect_admissions_inference):
    plot_department_gender_admissions(direct_effect_admissions_inference, 'direct effect, confounded model')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(latent_ability_inference):
    plot_department_gender_admissions(latent_ability_inference, "direct effect, latent ability model")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    By adding sensitivity analysis that is aligned with the data-generating process, we are able to identify gender bias in department 2

    ## Review of Sensitivity Analysis
    - Confounds exist, event if we can't measure them directly -- don't simply pretend that they don't exist
    - Address the question: **What are the implications of what we _don't know_?**
    - SA is somewhere between simulation and analysis
      - Hard-coding what we don't know, and let the rest play out.
      - Vary the confound over a range (e.g. std deviations) and show how that change effects the estimate
    - More honest than pretending that confounds do not exist.

    ## Note on number of parameters 🤯
    - Sensitivity Analysis model has 2006 free parameters
    - Only 2000 observations
    - No biggie in Bayesian Analysis
      - The minimum sample size is 0, where we just fall back on the prior
    """)
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
def _():
    # Load the data
    KLINE = utils.load_data("Kline")
    KLINE
    return (KLINE,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conceptual Ideas
    - The more innnovation the more tools
    - The more people, the more innovation
    - The more contact between cultures, the more innovation
    - Innovations (tools) are also forgotten over time, or become obsolete
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
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
    - Poplulation is treatment
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
    - **Poisson Distribution** approaches Binomial for large $N$ (approaching infinity) and low $p$ (approching 0)

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
    - Easier to shift the location (e.g. a the mean of a Normal prior), and keep tight variances
    - Prior variances generally need to be quite tight, on the order of 0.1 - 0.5
    """)
    return


@app.cell
def _():
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
def _():
    np.random.seed(123)
    _normal_alpha_prior_params = [
        (0, 1),  # flat, centered prior
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
def _():
    mo.md(r"""
    ## Direct Effect Estimator

    - Estimatinge <span style="color:red">Direct effect of population P on number of tools T</span> includes contact level, C in the adjustment set and location, L (currently unobserved; we will ignore for now)
    """)
    return


@app.cell(hide_code=True)
def _():
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
    Here we model tools count as a Poisson random variable. The poisson rate parameter is the exponent of a linear model. In this linear model, we include only an offset for low- or high- contact populations.

    $$
    \begin{align*}
    T_i &\sim \text{Poisson}(\lambda_i) \\
    \lambda_i &= \alpha_{C[i]} \\
    \alpha &\sim \text{Normal}(3, 0.5) \\
    \end{align*}
    $$
    """)
    return


@app.cell
def _(KLINE):
    # Set up data and coords
    TOOLS = KLINE.total_tools.values.astype(float)
    return (TOOLS,)


@app.cell
def _(TOOLS):
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

    Here we bracket both the intercept and the population regression coefficient by contact level

    $$
    \begin{align*}
    T_i &\sim \text{Poisson}(\lambda_i) \\
    \lambda_i &= \alpha_{C[i]} + \beta_{C[i]} \log(P) \\
    \alpha &\sim \text{Normal}(3, 0.5) \\
    \beta &\sim \text{Normal}(0, 0.2) \\
    \end{align*}
    $$
    """)
    return


@app.cell
def _(KLINE, TOOLS):
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

        # NOTE: For compute_log_likelihood to work for models that contain variables
        # with dims but no coords (e.g. dims="obs_ids"), we need the
        # following PR to be merged https://github.com/pymc-devs/pymc/pull/6882
        # (I've implemented the fix locally in pymc/stats/log_likelihood.py to
        # get analysis to execute)
        #
        # TODO: Once merged, update pymc version in conda environment
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
def _(global_intercept_inference, interaction_inference):
    # Compare the intercept-only and interation models
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
def _(CONTACT_LEVEL, POPULATION, STD_LOG_POPULATION, TOOLS):
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
                predictions=True
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
                predictions=True
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
def _(
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
def _(
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

    1. The low-contact mean intersects with the high-contact mean (around population of 150k). This makes little since logically
    2. The intercept for population = 0 should be very near 0. It's instead around 20 and 30 for both groups.
    3. Tonga isn't even included in the 89% HDI for the hi contact group
    4. Error for hi-contact group is absurdly large for a majority of the population range

    Can we do better?

    ## Improving the estimator with a better scientific model

    There are two immediate to improve the model, including:

    1. Robust regression model -- in this case a gamma-Poisson (neg-binomial) model
    2. **Use a more prinicipled scientific model**

    ## Scientific model that includes innovation and technology loss
    Recall from earlier this DAG that highlights the general conceptual idea of how observeed tools can arise:
    """)
    return


@app.cell(hide_code=True)
def _():
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
    **Why not develope a sicentific model that does just that?**

    ### Using the difference equation: $\Delta T = \alpha P^\beta - \gamma T$

    - $\Delta T$ is the change in # of tools given the current number of tools. Here T can be thought of as # of tools at current generation
    - $\alpha$ is the innovation rate for a population of size $P$
    - $\beta$ is the elasticity, and can be thought of as a saturation rate, or "diminishing returns" factor; if we constrain $0 > \beta < 1$
    - $\gamma$ is the attritions / technology loss rate at time T

    Furthermore we can parameterize such an equation by the class of contact rate, $C$ as $\Delta T = \alpha_C P^{\beta_C} - \gamma T$

    Now we leverage the notioin of **equilibrium** identify the steady state # of tools that are eventually obtained. At this point $\Delta T = 0$, and we can solve for the resulting $\hat T$ using algebra:

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
def _():
    _fig, _ax = plt.subplots()
    simulate_tools(P=1e3, color='C0')
    simulate_tools(P=1e4, color='C1')
    simulate_tools(P=300_000, color='C2')
    plt.xlabel("Generation")
    plt.ylabel("# Tools")
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
    We'll use an Exponential distribution as a prior on the difference equation parameters $\alpha, \beta, \gamma$. We thus need to identify a good rate hypoerparmeter $\eta$ for those priors.

    Reasonable values for all parameters were were approximately 0.25 in the simulation above. We would thus like to identify the the Exponential rate parameter that covers 0.25 = 1/4.
    """)
    return


@app.cell
def _(CONTACT, CONTACT_LEVEL, POPULATION, TOOLS):
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
        # NOTE: For compute_log_likelihood to work for models that contain variables
        # with dims but no coords (e.g. dims="obs_ids"), we need the
        # following PR to be merged https://github.com/pymc-devs/pymc/pull/6882
        # (I've implemented the fix locally in pymc/stats/log_likelihood.py to
        # get analysis to execute)
        #
        # TODO: Once merged, update pymc version in conda environment
        innovation_loss_inference = pm.compute_log_likelihood(innovation_loss_inference)


    pm.model_to_graphviz(innovation_loss_model)
    return innovation_loss_inference, innovation_loss_model


@app.cell
def _(innovation_loss_inference):
    az.summary(innovation_loss_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can see that for the $\alpha$ and $\beta$ parameters, the optimal value was around 0.23-0.28. For gamma, it was a bit smaller, at around 0.09. We could potentially re-parameterize our model to have a tigher prior for the Gamma variable, but meh.
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
def _(
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
def _(
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
def _(
    global_intercept_inference,
    innovation_loss_inference,
    interaction_inference,
):
    _compare_dict = {
        'global intercept': global_intercept_inference,
        'contact': interaction_inference,
        'innovation loss' : innovation_loss_inference
    }
    az.compare(_compare_dict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can also see that the innovation / loss model is far superior (weight=.94) in terms of LOO prediction.

    ### Take-homes
    - Generally best to have a domain-informed scientific model
    - We still have the unobserved location confound to deal with
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Review: Count GLMS
    - MaxEnt priors
      - Binomal
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
    # BONUS: Simpons's Pandora's Box

    **The reversal of some measured/estimated association when groups are either combined or separated.**

    - There is nothing particularly interesting or Paradoxical about Simpson's paradox
    - It is simply a statistical phenomena
    - There are any number of causal phenoena that can create SP
      - Pipes and Forks can cause one flavor of SP -- namely stratifying destroys trend / associations
      - Collider can cause the inverse flavor -- nameley stratifying ellicits a trend / association
    - You can't say one way or the other which direction of "reversal" is correct without making explicit causal claims

    ### Classic example is UC Berkeley Admissions
    - If you do not stratify/condition on Department, you find that Females are Admitted at a lower rate
    - If you stratify/condition on Department, you find that Females are Admitted at a slightly higher rate (see above Admissions analyses)
    - Which is correct? Could be explained by either:
      - a mediator/pipe (department)
      - a collider + confound (unobserved ability)

    **For examples of how Pipes, Forks, and Colliders can "lead to" Simpson's paradox, see [Leture 05 -- Elemental Confounds](<Lecture 05 - Elemental Confounds.ipynb>) **
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nonlinear Haunting

    Though $Z$ is not a confound, it is an competing cause of $Y$. If the causal model is nonlinear and we stratify by $Z$ to get the direct causa effect of the treatment on the outcome, this can cause some strange outcomes akin to Simpson's paradox.
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
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
def _(X, Y):
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
def _(X, Y, Z):
    # Partially statified Model
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
def _(unstratified_inference, unstratified_model):
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
def _(
    RESOLUTION,
    partially_stratified_inference,
    partially_stratified_model,
    xs,
):
    # Get predictions -- partially stratified model
    partially_stratified_predictions = {}
    with partially_stratified_model:
        for _z in [0, 1]:
            # Z = 0 predictions
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
def _(partially_stratified_predictions, unstratified_ppd, xs):
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
    When stratifying only on the X coefficient, and thus sharing a common intercept, we can see that for Z=0, there is a saturation around 0.6. This is due to the +5 added to the log odds of Y|Z=0 in the logistic regression model. Because of this saturation, it's difficult to tell if the treatment affects the outcome for that group.
    """)
    return


@app.cell(hide_code=True)
def _(partially_stratified_inference):
    _fig, _axs = plt.subplots(figsize=(5, 3))
    for _z in [0, 1]:
        _post = partially_stratified_inference.posterior.sel(chain=0)['beta'][:, _z]
        utils.plot_dist(_post, color=f"C{_z}", label=f'Z={_z}')
    plt.xlabel("beta_Z")
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
def _(X, Y, Z):
    # Fully statified Model
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
    ### Fullly Stratified Model posterior predictions
    """)
    return


@app.cell
def _(RESOLUTION, fully_stratified_inference, fully_stratified_model, xs):
    # Get predictions -- partially stratified model
    fully_stratified_predictions = {}
    with fully_stratified_model:
        for _z in [0, 1]:
            # Z = 0 predictions
            pm.set_data({'X': xs})
            pm.set_data({'Z': [_z] * RESOLUTION})
            fully_stratified_predictions[_z] = pm.sample_posterior_predictive(
                fully_stratified_inference,
                var_names=["p"]
            )["posterior_predictive"]["p"]
    return (fully_stratified_predictions,)


@app.cell(hide_code=True)
def _(
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

    plt.xlabel("Treatment, X")
    plt.ylabel("p(Y)")
    plt.ylim([0, 1])

    plt.legend(legend_data, ['Z=0', 'Z=1'])

    plt.title("$logit(p_i) = \\alpha_{Z[i]} + \\beta_{Z[i]}X_i$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we can see that with a fully stratified model, one in which we include a group-level intercept, the predicitions for Z=0 shift up even higher toward one, though the predictions remain mostly flat across all values of the treatment X
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare Posteriors
    """)
    return


@app.cell(hide_code=True)
def _(fully_stratified_inference, partially_stratified_inference):
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 3))

    plt.sca(_axs[0])

    for _z in [0, 1]:
        _post = partially_stratified_inference.posterior.sel(chain=0)['beta'][:, _z]
        utils.plot_dist(_post, color=f"C{_z}", label=f'Z={_z}')
    plt.xlabel("beta_Z")
    plt.ylabel("density")
    plt.legend()
    plt.title("Partially Stratified Posterior")


    plt.sca(_axs[1])
    for _z in [0, 1]:
        _post = fully_stratified_inference.posterior.sel(chain=0)['beta'][:, _z]
        utils.plot_dist(_post, color=f"C{_z}", label=f'Z={_z}')
    plt.xlabel("beta_Z")
    plt.ylabel("density")
    plt.legend()
    plt.title("Fully Stratified Posterior")
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
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
