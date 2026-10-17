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
    # [Lecture 18 - Missing Data](https://www.youtube.com/watch?v=Oeq6GChHOzc)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Missing Data, Found
    - Observed data is a special case
      - we trick ourselves into believing there is no error
      - no/little error is rare

    Most data are missing
      - but there's still information about the missing data. no data is fully missing with a generative model
      1. constraints - e.g. heights are never negative, and bounded
      2. relationships to other variables - e.g. weather affects our behavior, our behavior does not affect the weather
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dealing with missing data is a workflow

    #### What to do with missing data?

    - **Complete case analysis**: drop cases missing values
      - sometimes the right thing to do
      - depends on the context and causal assumptions
    - **Imputation**: fill the missing values with reasonable values that are based on the present values
      - often necessary and benefitial for accurate estimation

    ## Dog ate my homework: looking a different missing data scenarios
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("S", "H"),
            ("H", "H*"),
            ("D", "H*")
        ],
        node_props={
            "S": {"color": "red"},
            "H*": {"color": "red"},
            "H": {"style": "dashed"},
            "unobserved": {"style": "dashed"}

        },
        edge_props={
            ("S", "H"): {"color": "red"},
            ("H", "H*"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Student ability/knowledge state $S$
    - Students would receive score on homework $H$
    - We only get to observe the homework the teacher sees $H^*$
      - we don't get to observe the real score, because homework may not make it to teacher
      - e.g. the family dog $D$ randomly eats homework (missingness mechanism)

    ### Dog usually benign
    """)
    return


@app.function
def plot_regression_line(x, y, color, label, **plot_kwargs):
    valid_idx = ~np.isnan(y)

    X = np.vstack((np.ones_like(x[valid_idx]), x[valid_idx])).T
    intercept, slope = np.linalg.lstsq(X, y[valid_idx])[0]

    xs = np.linspace(x.min(), x.max(), 10)
    ys = xs * slope + intercept
    utils.plot_line(xs, ys, color=color, label=label, **plot_kwargs)


@app.function
def plot_dog_homework(S, H, Hstar, title=None):

    utils.plot_scatter(
        S, H, color='k', alpha=1, label='total', s=10
    )
    plot_regression_line(S, H, label='total trend', color='k', alpha=.5)

    utils.plot_scatter(
        S, Hstar, color='C0', alpha=.8, label='incomplete'
    )
    plot_regression_line(S, Hstar, label='incomplete trend', color='C0', alpha=.5)

    plt.xlabel("S")
    plt.ylabel("H")
    plt.title(title)
    plt.legend();


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulate random homework eating
    """)
    return


@app.cell
def _():
    np.random.seed(123)
    _n_homework = 100
    # Student knowledge
    _S = stats.norm.rvs(size=_n_homework)

    # Homework score
    _mu_score = _S * 0.5
    _H = stats.norm.rvs(_mu_score)

    # Dog eats 50% of of homework _at random_
    _D = stats.bernoulli(0.5).rvs(size=_n_homework)
    _Hstar = _H.copy()
    _Hstar[_D==1] = np.nan

    plot_dog_homework(_S, _H, _Hstar, title="Random missing data\ncauses loss of precision; little/no bias")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - When losing outcomes randomly, we obtain a similar linear fit with the similar slope, and no little/no bias.
    - Thus dropping complete cases ok, but lose efficiency
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Dog eats homework based on cause of homework (student ability)

    Now the student's ability effects the dog's behavior (e.g. the student studies too much and doesn't feed the dog, and the dog is hungry)
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("S", "H"),
            ("H", "H*"),
            ("D", "H*"),
            ("S", "D")
        ],
        node_props={
            "S": {"color": "blue"},
            "H*": {"color": "red"},
            "H": {"style": "dashed"},
            "unobserved": {"style": "dashed"}

        },
        edge_props={
            ("S", "H"): {"color": "red"},
            ("H", "H*"): {"color": "red"},
            ("S", "D"): {"color": "blue", "label": "possible biasing path", "fontcolor": "blue"},
            ("D", "H*"): {"color": "blue"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulate data where treatment effects oucome linearly
    """)
    return


@app.cell
def _():
    np.random.seed(12)
    _n_homework = 100
    # Student knowledge
    _S = stats.norm.rvs(size=_n_homework)

    # Linear association between student ability and homework score
    _mu_score = _S * 0.5
    _H = stats.norm.rvs(_mu_score)

    # Dog eats based on the student's ability
    _p_dog_eats_homework = np.where(_S > 0, 0.9, 0)
    _D = stats.bernoulli.rvs(p=_p_dog_eats_homework)
    _Hstar = _H.copy()
    _Hstar[_D==1] = np.nan

    plot_dog_homework(_S, _H, _Hstar, title="Missing data conditioned on common cause\nmay work for linear relationships (rare) ")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When the association between student ability $S$ and homework sckor $H$ is linear, we can still get fairly similar linear fits from the total and incomplete sample, you just lose precision on one the upper end of the $S$ dimenions in this simulation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulate data where treatment effects oucome nonlinearly
    """)
    return


@app.cell
def _():
    np.random.seed(1)
    _n_homework = 100
    # Student knowledge
    _S = stats.norm.rvs(size=_n_homework)

    # Nonlinear association between student ability and homework score
    _mu_score = 1 - np.exp(-0.7 * _S)
    _H = stats.norm.rvs(_mu_score)

    # Dog eats all the homework of above-average students
    _p_dog_eats_homework = np.where(_S > 0, 1, 0)
    _D = stats.bernoulli.rvs(p=_p_dog_eats_homework)
    _Hstar = _H.copy()
    _Hstar[_D==1] = np.nan

    plot_dog_homework(_S, _H, _Hstar, title="Missing data based on common cause\nvery bady for non-linear relationships (common)")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - When the association between student ability $S$ and homework sckor $H$ is nonlinear, we now get very different linear fits from the total and incomplete sample
      - on nonlinearity: this is why we use generative models with scientifically-motivated functions: **functions matter**
    - Therefore need to correctly condition on the cause
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Dog eats homework conditioned on the _state_ of homework itself
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("S", "H"),
            ("H", "H*"),
            ("D", "H*"),
            ("H", "D")
        ],
        node_props={
            "H*": {"color": "red"},
            "H": {"style": "dashed"},
            "unobserved": {"style": "dashed"}

        },
        edge_props={
            ("S", "H"): {"color": "red"},
            ("H", "H*"): {"color": "red"},
            ("H", "D"): {"color": "blue", "label": "biasing path", "fontcolor": "blue"},
            ("D", "H*"): {"color": "blue"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulate some linear data
    """)
    return


@app.cell
def _():
    np.random.seed(1)
    _n_homework = 100
    # Student knowledge
    _S = stats.norm.rvs(size=_n_homework)

    # Linear association between ability and score
    _mu_score = _S * 0.5
    _H = stats.norm.rvs(_mu_score)

    # Dog eats 90% of homework that is below average
    _p_dog_eats_homework = np.where(_H < 0, 0.9, 0)
    _D = stats.bernoulli.rvs(p=_p_dog_eats_homework)
    _Hstar = _H.copy()
    _Hstar[_D==1] = np.nan

    plot_dog_homework(_S, _H, _Hstar, title="Missing data conditioned on outcome state\nusually not benign")
    plt.gcf()
    return


@app.cell
def _():
    np.random.seed(1)
    _n_homework = 100
    # Student knowledge
    _S = stats.norm.rvs(size=_n_homework)

    # Nonlinear association between student ability and homework score
    _mu_score = 1 - np.exp(-0.9 * _S)
    _H = stats.norm.rvs(_mu_score)

    # Dog eats 90% of homework that is below average
    _p_dog_eats_homework = np.where(_H < 0, 0.9, 0)
    _D = stats.bernoulli.rvs(p=_p_dog_eats_homework)
    _Hstar = _H.copy()
    _Hstar[_D==1] = np.nan

    plot_dog_homework(_S, _H, _Hstar, title="Missing data conditioned on outcome state\nsimilar for nonlinear relationships")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - without knowing the causal relationship between the state and the data loss, and the functional forms of how $S$ is associated with $H$, it's difficult to account for this scenario
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stereotypical cases

    1. Data loss is **random and indpendent of causes**
       - "Dog eats homework randomly"
       - Dropping complete cases ok, but lose efficiency
    3. Data loss is **conditioned on the cause**
       - "Dog eats homework based on the student's study habits"
       - Need to correctly condition on cause
    5. Data loss is **conditioned on the outcome**
       - "Dog eats homeowork based on the score of the homework"
       - This is usually hopeless unless we can model the causal process of how state affects data loss (i.e. the dog's behavior in this simulation)
         - e.g. survival analysis and censored data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bayesian Imputation

    - Works for **Stereotypical cases** 1), 2) and subsets of 3) (e.g. survival analysis)
    - Having a joint (Bayesian) model for all variables (observed data and parameters) provides a means for inferring reasonable bounds on the values that are missing

    ### Imputing or Marginalizing
    - **Imputation**: compute a posterior probability distribution of the missing values
      - usually involves partial pooling
      - we've already done this to some extent with the missing district in the Bangladesh dataset
    - **Marginalizing unknowns**: average over the distribution of missing values using the posteriors for the other variables
      - no need to calculate posteriors for missing values

    #### When to impute? When to marginalize?
    - Sometimes imputation is unnecessary -- e.g. discrete variables
      - Marginalization works here
    - It's sometimes easier to impute
      - e.g. when the data loss process is well understood as in censoring in survival analysis
      - marginalization may be difficult for more exotic models
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Revisiting Phylogenetic Regression
    """)
    return


@app.cell
def _():
    PRIMATES301 = utils.load_data("Primates301")
    PRIMATES301.head()
    return (PRIMATES301,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - 301 Primate profiles
    - Life history traits
      - Body Mass $M$
      - Brain size $B$
      - Social Group Size $G$
    - Issues
      - Lots of **missing data**
        - Complete case analysis only includes 151 observations
      - measurement error
      - unobserved confounding

    ## Imputing Primates

    - **Key Idea**: missing values have probability distributions
      - Express a causal model for each partially-observed variable
      - Replace each missing value with a parameter, and let the model proceed as normal
        - Duality of parameters and data:
          - In Bayes there is no distinction between params and data: missing value is just a parameter, observed value is data
      - e.g. one variable that _is_ available can inform us of reasonable values for missing variable associated with the observation
    - **Conceptually weired, technically awkward**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Revisiting previous causal model
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("M", "B"),
            ("M", "G"),
            ("u", "M"),
            ("u", "G"),
            ("u", "B"),
            ("H", "u")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - where $M,G,B$ are defined above
    - shared history $H$, which leads to
    - historical-based confounds $u$, which we try to capture with a phylogeny

    #### Let's now include how missing values may occur
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("M", "B"),
            ("M", "G"),
            ("u", "M"),
            ("u", "G"),
            ("u", "B"),
            ("H", "u"),
            ("G", "G*"),
            ("mG", "G*")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - We observe group size $G^*$, which has missing values for some observations/species
    - What influences the cause of missingness $m_G$?
      - often the assumption is that it's totally random, but this is highly unlikely
      - there are **many possible ways that missingness could occur**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Some hypotheses on what causes missing data...
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("M", "B"),
            ("M", "G"),
            ("u", "M"),
            ("u", "G"),
            ("u", "B"),
            ("H", "u"),
            ("G", "G*"),
            ("mG", "G*"),
            ("M", "mG"),
            ("u", "mG"),
            ("G", "mG")
        ],
        edge_props={
            ("u", "mG"): {"color": "blue", "fontcolor": "blue", "label": "anthro-narcisism"},
            ("M", "mG"): {"color": "darkcyan", "fontcolor": "darkcyan", "label": "larger species"},
            ("G", "mG"): {"color": "red", "fontcolor": "red", "label": "solitary species"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - All the colored arrows, and assumptions/hypotheses, are potentially in play
      - <span style="color:blue">**anthro-narcisism**: humans are more likely to study primates that are like them</span>
      - <span style="color:darkcyan">**larger species** with larger body mass easier to count: e.g. some species might live in trees and are harder to observe</span>
      - <span style="color:red">some species have little or no social group; these **solitary species** are difficult to observe and gather data on</span>
    - Whatever the assumption, the goal is to **use causal model to infer the probability distribution of each missing value**
    - Uncertainty in each missing value will cascade through the model
      - again, no need to be clever
      - trust the axioms of probability
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modeling missing data for multiple variables simultaneously

    ### $M, G, B$ all have missing values $M^*, G^*, B^*$
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("M", "G"),
            ("M", "B"),
            ("G", "B"),
            ("M", "M*"),
            ("G", "G*"),
            ("B", "B*"),
            ("mM", "M*"),
            ("mB", "B*"),
            ("mG", "G*"),
            ("H", "u"),
            ("u", "G"),
            ("u", "M"),
            ("u", "B"),
        ],
        node_props={
            "G": {"style": "dashed"},
            "M": {"style": "dashed"},
            "B": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Dealing with missing values in Brain Size $B$

    It turns out we already handled this using Gaussian processes in the original Phylogenetic Regression example. By leveraging local pooling offered by the Gaussian process, we fill in missing brain size $B$ values using information from species that are similar those species with missing values, but _do_ have values for $B$.

    The equivalent causal-sub-graph used to modele Brain Size in the "Phylogenetic Regression" section of [Lecture 16 - Gaussian Processes](<Lecture 16 - Gaussian Processes.ipynb>)

    #### Brain Size $B$ Model

    Assuming missingness is totally at random
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "B"),
            ("M", "B"),
            ("u, D", "B"),
            ("H", "u, D")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align}
        \textbf{B} &\sim \text{MVNormal}(\mu_B, \textbf{K}_B) \\
        \mu_{B,i} &= \alpha_B + \beta_{GB} G_i + \beta_{MB} M_i \\
        \mathbf{K}_B &= \eta_B^2 \exp(-\rho_B D_{jk})\\
        \alpha_B &\sim \text{Normal}(0, 1) \\
        \beta_{GB, MB} &\sim \text{Normal}(0, 1) \\
        \eta_B^2 &\sim \text{HalfNormal}(1, 0.25)\\
        \rho_B &\sim \text{HalfNormal}(3, 0.25)
    \end{align}
    $$

    We can use a simular approach to simultaneously formulate parallel models for the other variables $G,M$ that may have missing values, a' la Full Luxury Bayes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Group Size $G$ Model
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("M", "G"),
            ("u, D", "G"),
            ("H", "u, D")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align}
        \textbf{G} &\sim \text{MVNormal}(\mu_G, \textbf{K}_G) \\
        \mu_{G,i} &= \alpha_G + \beta_{MG} M_i \\
        \mathbf{K}_G &= \eta_G^2 \exp(-\rho_G D_{jk})\\
        \alpha_G &\sim \text{Normal}(0, 1) \\
        \beta_{MG} &\sim \text{Normal}(0, 1) \\
        \eta_G^2 &\sim \text{HalfNormal}(1, 0.25)\\
        \rho_G &\sim \text{HalfNormal}(3, 0.25)
    \end{align}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Body Mass $M$ model
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("u, D", "M"),
            ("H", "u, D")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align}
        \textbf{M} &\sim \text{MVNormal}(\mathbf{0}, \textbf{K}_M) \\
        \mathbf{K}_M &= \eta_M^2 \exp(-\rho_M D_{jk})\\
        \eta_M^2 &\sim \text{HalfNormal}(1, 0.25)\\
        \rho_M &\sim \text{HalfNormal}(3, 0.25)
    \end{align}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Drawing the missing owl 🦉

    Building a model containing three simultaneous Gaussian processes has a lot of working parts. It's better to build up model complexity in small steps.

    1. Ignore cases with missing $B$ values (for now)
       - $B$ is the outcome, so any imputation would just be prediction
    2. Impute $G,M$ ignoring models for each
       - wrong model, correct process
       - allows us to see the consequences of adding complexity to model
    3. Impute $G$ using a $G$-specific submodel
    4. Impute $B,G,M$ using a submodel for each
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1. Ignore cases with missing $B$ values
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Filter out observations missing brain volume measurments
    """)
    return


@app.cell
def _(PRIMATES301):
    PRIMATES = PRIMATES301[PRIMATES301.brain.notna()]
    PRIMATES
    return (PRIMATES,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Cross-tabulate missingess for $G,M$ on the resulting dataset
    """)
    return


@app.cell
def _(PRIMATES):
    PRIMATES.group_size.isna().sum()
    return


@app.cell
def _(PRIMATES):
    print("Pattern of missingness:\n")
    print(utils.crosstab(PRIMATES.body.notna(), PRIMATES.group_size.notna()))
    print(f"\nCorrelation of missing values: {np.corrcoef(PRIMATES.body.isna(), PRIMATES.group_size.isna())[0][1]:0.2}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - True: Observed
    - False: Missing
    - Group size $G$ missing a lot of values (33)
    - Some correlation between the presence/absence of each variable
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2. Impute $G,M$ naively, ignoring models for each

    Idea here is to treat $G, M$ as independent random variables with no causes, e.g. standard Normals.

    $$
    M_i, G_i \sim \text{Normal}(0, 1) \\
    $$

    - This is not the correct model, but the correct starting place to build up the model gradually
    - Interpretation:
      - When $G_i$ is observed, the distribution is the likelihood for a standardized variable
      - When $G_i$ is missing, the distribution is the prior for the "parameter"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Fit the impuatation model that ignores the submodels for $G,M$
    ##### Load the phlogenetic distance matrix
    """)
    return


@app.cell
def _(PRIMATES):
    PRIMATES_DISTANCE_MATRIX = utils.load_data("primates301_distance_matrix").values

    # Filter out cases without $B$ values from the raw distance matrix
    PRIMATES_DISTANCE_MATRIX = PRIMATES_DISTANCE_MATRIX[PRIMATES.index][:, PRIMATES.index]

    # Rescale distances
    D_mat = (PRIMATES_DISTANCE_MATRIX / PRIMATES_DISTANCE_MATRIX.max())
    assert D_mat.max() == 1
    return (D_mat,)


@app.cell
def _(PRIMATES):
    # Preprocess the data
    G_obs = utils.standardize(np.log(PRIMATES.group_size.values))
    N_G_missing = np.isnan(G_obs).sum()
    M_obs = utils.standardize(np.log(PRIMATES.body.values))
    N_M_missing = np.isnan(M_obs).sum()
    B_obs = utils.standardize(np.log(PRIMATES.brain.values))
    N_obs = len(B_obs)
    return B_obs, G_obs, M_obs, N_G_missing, N_M_missing


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Fit the naive imputation model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an implementation that uses PyMC's Gaussian process module.
    """)
    return


@app.cell
def _(PRIMATES):
    PRIMATE_ID, PRIMATE = pd.factorize(PRIMATES['name'], sort=False)
    coords = {'primate': PRIMATE}

    class MeanBodyMassSocialGroupSize(pm.gp.mean.Linear):
        """
        Custom mean function that separates Social Group Size and
        Body Mass effects from phylogeny
        """
        def __init__(self, alpha, beta_G, beta_M, G, M):
            self.alpha = alpha
            self.beta_G = beta_G
            self.beta_M = beta_M
            self.G = G
            self.M = M

        def __call__(self, X):
            return self.alpha + self.beta_G * self.G + self.beta_M * self.M

    # with pm.Model(coords=coords) as naive_imputation_model:

    #     # Priors
    #     alpha = pm.Normal("alpha", 0, 1)
    #     beta_G = pm.Normal("beta_G", 0, 0.5)
    #     beta_M = pm.Normal("beta_M", 0, 0.5)
    #     sigma = pm.Exponential("sigma", 1)

    #     # Naive imputation for G, M
    #     G = pm.Normal("G", 0, 1, observed=G_obs, dims='primate')
    #     M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

    #     mean_func = MeanBodyMassSocialGroupSize(alpha, beta_G, beta_M, G, M)

    #     # Phylogenetic distance covariance prior, L1-kernel function
    #     eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.01)
    #     rho = pm.TruncatedNormal("rho", 3, .25, lower=.01)
    #     cov_func = eta_squared * pm.gp.cov.Exponential(1, ls=rho)

    #     gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
    #     gp.marginal_likelihood("B", X=D_mat, y=B_obs, sigma=sigma)

    #     naive_imputation_inference = pm.sample(cores=1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the results from the lecture.
    """)
    return


@app.function
def generate_L1_kernel_matrix(D, eta_squared, rho, smoothing=0.01):
    K = eta_squared * pm.math.exp(-rho * D)
    # Smooth the diagonal of the covariance matrix
    N = D.shape[0]
    K += np.eye(N) * smoothing
    return K


@app.cell
def _(B_obs, D_mat, G_obs, M_obs, PRIMATES):
    _coords = {'primate': PRIMATES['name'].values}
    with pm.Model(coords=_coords) as naive_imputation_model:

        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _beta_G = pm.Normal("beta_G", 0, 0.5)
        _beta_M = pm.Normal("beta_M", 0, 0.5)

        # Phylogenetic distance covariance prior, L1-kernel function
        _eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.001)
        _rho = pm.TruncatedNormal("rho", 3, .25, lower=.001)
        # K = pm.Deterministic('K', generate_L1_kernel_matrix(D_mat, eta_squared, rho))
        _K = pm.Deterministic('K', _eta_squared * pm.math.exp(-_rho * D_mat))

        # Naive imputation for G, M
        _G = pm.Normal("G", 0, 1, observed=G_obs, dims='primate')
        _M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

        # Likelihood for B
        _mu = _alpha + _beta_G * _G + _beta_M * _M
        pm.MvNormal("B", mu=_mu, cov=_K, observed=B_obs)

        naive_imputation_inference = cached_sample()
    return (naive_imputation_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Demonstrate the effect of imputation
    ##### $M$ Imputation
    """)
    return


@app.cell
def _(naive_imputation_inference):
    # Now all $M$ have imputed values
    az.summary(naive_imputation_inference, var_names=['M'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Prior values for $M$
    """)
    return


@app.cell
def _(N_M_missing, naive_imputation_inference):
    print("Number of missing M values:", N_M_missing)
    az.summary(naive_imputation_inference, var_names=['M_unobserved'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### $G$ Imputation
    """)
    return


@app.cell
def _(naive_imputation_inference):
    # Now all $G$ have imputed values
    az.summary(naive_imputation_inference, var_names=['G'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Prior Values for $G$
    """)
    return


@app.cell
def _(N_G_missing, naive_imputation_inference):
    print("Number of missing G values:", N_G_missing)
    az.summary(naive_imputation_inference, var_names=['G_unobserved'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Plot posterior imputation
    """)
    return


@app.cell
def _(B_obs, G_obs, M_obs):
    def plot_posterior_imputation(xvar, yvar, inference, impute_color='C0', impute_x=False, impute_y=False, title=None):
        plot_data = pd.DataFrame(
            {
                "G": G_obs,
                "M": M_obs,
                "B": B_obs
            }
        )
        # Plot observed
        utils.plot_scatter(
            plot_data[xvar],
            plot_data[yvar],
            color='k',
            label='observed'
        )

        # Get and plot imputed
        impute_x_values = None
        impute_y_values = None
        if impute_x and f"{xvar}_unobserved" in inference.posterior:
            x_impute_idx = plot_data[plot_data[xvar].isnull()].index.values
            impute_x_values = az.summary(inference, var_names=[f"{xvar}_unobserved"])['mean'].values

        if impute_y and f"{yvar}_unobserved" in inference.posterior:
            y_impute_idx = plot_data[plot_data[yvar].isnull()].index.values
            impute_y_values = az.summary(inference, var_names=[f"{yvar}_unobserved"])['mean'].values

        if impute_x_values is None:
            impute_x_values = plot_data.loc[y_impute_idx, xvar].values

        if impute_y_values is None:
            impute_y_values = plot_data.loc[x_impute_idx, yvar].values

        utils.plot_scatter(
            impute_x_values,
            impute_y_values,
            color=impute_color,
            label='imputed',
            alpha=1,
            s=100
        )

        plt.xlabel(f"${xvar}$ (standardized)")
        plt.ylabel(f"${yvar}$ (standardized)")
        plt.legend()
        plt.title(title)

    return (plot_posterior_imputation,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Looking at imputed values
    """)
    return


@app.cell(hide_code=True)
def _(naive_imputation_inference, plot_posterior_imputation):
    plot_posterior_imputation("M", "B", naive_imputation_inference, impute_x=True, title="Naviely Imputed $M$ values with $B$")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Because $M$ is strongly associated with $B$, the imputed $M$ values follow the linear relationship
    """)
    return


@app.cell(hide_code=True)
def _(naive_imputation_inference, plot_posterior_imputation):
    plot_posterior_imputation("M", "G", naive_imputation_inference, impute_y=True, title="Naively Imputed $M$ values with $G$")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Because the association between $M$ and $G$ isn't modeled, imputed values doesn't follow the linear trend between $M$ and $G$
    - We've left some information on the table; this is the result of ignoring the full generative model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Complete case model for comparison

    Here we'll fit a complet-cases-only model where there's no need for imputation, but we throw away data.
    """)
    return


@app.cell
def _(PRIMATES301):
    # Complete Case
    PRIMATES_CC = PRIMATES301.query(
        "brain.notnull() and body.notnull() and group_size.notnull()",
        engine='python'
    )

    G_CC = utils.standardize(np.log(PRIMATES_CC.group_size.values))
    M_CC = utils.standardize(np.log(PRIMATES_CC.body.values))
    B_CC = utils.standardize(np.log(PRIMATES_CC.brain.values))

    # Get complete case distance matrix
    PRIMATES_DISTANCE_MATRIX_CC = utils.load_data("primates301_distance_matrix").values

    # Filter out incomplete cases
    PRIMATES_DISTANCE_MATRIX_CC  = PRIMATES_DISTANCE_MATRIX_CC[PRIMATES_CC.index][:, PRIMATES_CC.index]

    # Rescale distances to [0, 1]
    D_mat_CC = (PRIMATES_DISTANCE_MATRIX_CC / PRIMATES_DISTANCE_MATRIX_CC.max())
    assert D_mat_CC.max() == 1
    return B_CC, D_mat_CC, G_CC, M_CC


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an implementation that uses PyMC's Gaussian process module.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the results from the lecture.
    """)
    return


@app.cell
def _(B_CC, D_mat_CC, G_CC, M_CC):
    # with pm.Model() as complete_case_model:

    #     # Priors
    #     alpha = pm.Normal("alpha", 0, 1)
    #     beta_G = pm.Normal("beta_GB", 0, 0.5)
    #     beta_M = pm.Normal("beta_MB", 0, 0.5)
    #     sigma = pm.Exponential("sigma", 1)

    #     # Phylogenetic distance covariance
    #     mean_func = MeanBodyMassSocialGroupSize(alpha, beta_G, beta_M, G_CC, M_CC)

    #     # Phylogenetic distance covariance prior, L1-kernel function
    #     eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.01)
    #     rho = pm.TruncatedNormal("rho", 3, .25, lower=.01)
    #     cov_func = eta_squared * pm.gp.cov.Exponential(1, ls=rho)

    #     gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
    #     gp.marginal_likelihood("B", X=D_mat_CC, y=B_CC, sigma=sigma)

    #     complete_case_inference = pm.sample(target_accept=.95)

    with pm.Model() as complete_case_model:

        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _beta_G = pm.Normal("beta_G", 0, 0.5)
        _beta_M = pm.Normal("beta_M", 0, 0.5)

        # Phylogenetic distance covariance
        _eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.001)
        _rho = pm.TruncatedNormal("rho", 3, .25, lower=.001)

        _K = pm.Deterministic('K', generate_L1_kernel_matrix(D_mat_CC, _eta_squared, _rho, smoothing=.001))
        # K = pm.Deterministic('K', eta_squared * pm.math.exp(-rho * D_mat_CC))

        # Likelihood
        _mu = _alpha + _beta_G * G_CC + _beta_M * M_CC
        pm.MvNormal("B", mu=_mu, cov=_K, observed=B_CC)
        complete_case_inference = cached_sample()
    return (complete_case_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare posteriors for naively-imputed and complete case model
    """)
    return


@app.cell(hide_code=True)
def _(complete_case_inference, naive_imputation_inference):
    utils.plot_dist(complete_case_inference.posterior['beta_G'], label='complete case', color='k')
    utils.plot_dist(naive_imputation_inference.posterior['beta_G'], label='imputed', color='C0')
    plt.axvline(0, linestyle='--', color='k', label='no effect')
    plt.xlabel("effect of Social Group size on Brain size, $\\beta_G$")
    plt.ylabel("density")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - we can see that the posterior for the imputed model attenuates the effect of Group size on Brain Size
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3. Impute $G$ using a $G$-specific submodels

    Add the generative model for Group Size. But, again, we'll do this in baby steps, slowly building complexity

    1) Model that only models effect of body mass on group size $M \rightarrow G$
    2) Model that only includes Social group phylogentic interactions
    3) Model that combines phylogeny and $M \rightarrow G$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1. Model that only models effect of body mass on group size $M \rightarrow G$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an implementation that uses PyMC's Gaussian process module.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the results from the lecture.
    """)
    return


@app.cell
def _(B_obs, D_mat, G_obs, M_obs, PRIMATES):
    # PRIMATE_ID, PRIMATE = pd.factorize(PRIMATES['name'].values)
    # coords = {'primate': PRIMATE}

    # with pm.Model(coords=coords) as G_body_mass_model:

    #     # Priors
    #     alpha_G = pm.Normal("alpha_G", 0, 1)
    #     beta_MG = pm.Normal("beta_MG", 0, 0.5)
    #     sigma_G = pm.Exponential("sigma_G", 1)

    #     alpha_B = pm.Normal("alpha_B", 0, 1)
    #     beta_GB = pm.Normal("beta_GB", 0, 0.5)
    #     beta_MB = pm.Normal("beta_MB", 0, 0.5)
    #     sigma_B = pm.Exponential("sigma_B", 1)

    #     # Naive imputation for M
    #     M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

    #     # G model M->G (performs imputation)
    #     mu_G = alpha_G + beta_MG * M
    #     G = pm.Normal("G", mu_G, sigma_G, observed=G_obs)

    #     # B Model
    #     eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.01)
    #     rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.01)
    #     cov_func_B = eta_squared_B * pm.gp.cov.Exponential(1, ls=rho_B)
    #     mean_func_B = MeanBodyMassSocialGroupSize(alpha_B, beta_GB, beta_MB, G, M)

    #     # Gaussian Process
    #     gp_B = pm.gp.Marginal(mean_func=mean_func_B, cov_func=cov_func_B)
    #     gp_B.marginal_likelihood("B", X=D_mat, y=B_obs, sigma=sigma_B)

    #     G_body_mass_inference = pm.sample(target_accept=.95, cores=1)

    _PRIMATE_ID, _PRIMATE = pd.factorize(PRIMATES['name'].values)
    _coords = {'primate': _PRIMATE}

    with pm.Model(coords=_coords) as G_body_mass_model:

        # Priors
        _alpha_B = pm.Normal("alpha_B", 0, 1)
        _beta_GB = pm.Normal("beta_GB", 0, 0.5)
        _beta_MB = pm.Normal("beta_MB", 0, 0.5)

        _alpha_G = pm.Normal("alpha_G", 0, 1)
        _beta_MG = pm.Normal("beta_MG", 0, 0.5)
        _sigma_G = pm.Exponential("sigma_G", 1)

        # Naive imputation for M
        _M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

        # Body-mass only, no interactions
        _mu_G = _alpha_G + _beta_MG * _M
        _G = pm.Normal("G", _mu_G, _sigma_G, observed=G_obs)

        # B Model
        _eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.001)
        _rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.001)
        _K_B = pm.Deterministic('K_B', _eta_squared_B * pm.math.exp(-_rho_B * D_mat))

        # Likelihood for B
        _mu_B = _alpha_B + _beta_GB * _G + _beta_MB * _M
        pm.MvNormal("B", mu=_mu_B, cov=_K_B, observed=B_obs)

        G_body_mass_inference = cached_sample()
    return (G_body_mass_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Look at $M \rightarrow G$-only imputation
    """)
    return


@app.cell(hide_code=True)
def _(G_body_mass_inference, plot_posterior_imputation):
    plot_posterior_imputation("M", "G", G_body_mass_inference, impute_y=True, title="$M \\rightarrow G$ model")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - By modeling the effect of $M$ on $G$, we're able to capture the linear trend for; imputed variables lie along this trend
    - However, by ignoring phylogenetic similarity, we aren't able to capture nonlinearities in the data
      - e.g. some species with small group size lying along the lower left of the plot do not follow the linear trend
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 2. Model that only includes Social group phylogentic interactions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an implementation that uses PyMC's Gaussian process module.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the results from lecture.
    """)
    return


@app.cell
def _(B_obs, D_mat, G_obs, M_obs, PRIMATES):
    # PRIMATE_ID, PRIMATE = pd.factorize(PRIMATES['name'].values)
    # coords = {'primate': PRIMATE}

    # with pm.Model(coords=coords) as G_phylogeny_model:

    #     # Priors
    #     alpha_G = pm.Normal("alpha_G", 0, 1)
    #     sigma_G = pm.Exponential("sigma_G", 1)

    #     alpha_B = pm.Normal("alpha_B", 0, 1)
    #     beta_GB = pm.Normal("beta_GB", 0, 0.5)
    #     beta_MB = pm.Normal("beta_MB", 0, 0.5)
    #     sigma_B = pm.Exponential("sigma_B", 1)

    #     # Naive imputation for M
    #     M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

    #     # G model, interactions only
    #     mean_func_G = MeanBodyMassSocialGroupSize(alpha_G, 0, 0, 0, 0)

    #     eta_squared_G = pm.TruncatedNormal("eta_squared_G", 1, .25, lower=.01)
    #     rho_G = pm.TruncatedNormal("rho_G", 3, .25, lower=.01)
    #     cov_func_G = eta_squared_G * pm.gp.cov.Exponential(1, ls=rho_G)

    #     gp_G = pm.gp.Marginal(mean_func=mean_func_G, cov_func=cov_func_G)
    #     G = gp_G.marginal_likelihood("G", X=D_mat, y=G_obs, sigma=sigma_G)

    #     # B Model
    #     mean_func_B = MeanBodyMassSocialGroupSize(alpha_B, beta_GB, beta_MB, G, M)

    #     eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.01)
    #     rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.01)
    #     cov_func_B = eta_squared_B * pm.gp.cov.Exponential(1, ls=rho_B)

    #     gp_B = pm.gp.Marginal(mean_func=mean_func_B, cov_func=cov_func_B)
    #     gp_B.marginal_likelihood("B", X=D_mat, y=B_obs, sigma=sigma_B)

    #     G_phylogeny_inference = pm.sample(target_accept=.95, cores=1)

    _coords = {'primate': PRIMATES['name'].values}
    with pm.Model(coords=_coords) as G_phylogeny_model:

        # Priors
        _alpha_B = pm.Normal("alpha_B", 0, 1)
        _beta_GB = pm.Normal("beta_GB", 0, 0.5)
        _beta_MB = pm.Normal("beta_MB", 0, 0.5)

        _alpha_G = pm.Normal("alpha_G", 0, 1)
        _beta_MG = pm.Normal("beta_MG", 0, 0.5)

        # Naive imputation for M
        _M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

        # G Model Imputation, only phylogenetic interaction
        _eta_squared_G = pm.TruncatedNormal("eta_squared_G", 1, .25, lower=.001)
        _rho_G = pm.TruncatedNormal("rho_G", 3, .25, lower=.001)
        _K_G = pm.Deterministic('K_G', _eta_squared_G * pm.math.exp(-_rho_G * D_mat))

        _mu_G = pm.math.zeros_like(B_obs)  # no linear model
        _G = pm.MvNormal("G", mu=_mu_G, cov=_K_G, observed=G_obs)

        # B Model
        _eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.001)
        _rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.001)
        _K_B = pm.Deterministic('K_B', _eta_squared_B * pm.math.exp(-_rho_B * D_mat))

        # Likelihood for B
        _mu_B = _alpha_B + _beta_GB * _G + _beta_MB * _M
        pm.MvNormal("B", mu=_mu_B, cov=_K_B, observed=B_obs)

        G_phylogeny_inference = cached_sample()
    return (G_phylogeny_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Look at phylogeny-only imputation
    """)
    return


@app.cell(hide_code=True)
def _(G_phylogeny_inference, plot_posterior_imputation):
    plot_posterior_imputation(
        "M", "G",
        G_phylogeny_inference,
        impute_y=True,
        impute_color='C1',
        title="Phylogeny-only Model"
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ⚠️ for some reason, after trying two different implementation of GPs using PyMCs `gp` module, as well as using `MVNormal` likelihood (both give similar results), I'm not able to replicate the imputation for solo primates presented in lecture.

    **Comments and suggestions are welcome on what I may be doing incorrectly here**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 3. Fit model that combines phylogeny and $M \rightarrow G$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an implementation that implements the model using PyMC's Gaussian process module.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the results from lecture.
    """)
    return


@app.cell
def _(B_obs, D_mat, G_obs, M_obs, PRIMATES):
    # PRIMATE_ID, PRIMATE = pd.factorize(PRIMATES['name'].values)
    # coords = {'primate': PRIMATE}

    # with pm.Model(coords=coords) as G_imputation_model:

    #     # Priors
    #     alpha_G = pm.Normal("alpha_G", 0, 1)
    #     beta_MG = pm.Normal("beta_MG", 0, 1)
    #     sigma_G = pm.Exponential("sigma_G", 1)

    #     alpha_B = pm.Normal("alpha_B", 0, 1)
    #     beta_GB = pm.Normal("beta_GB", 0, 0.5)
    #     beta_MB = pm.Normal("beta_MB", 0, 0.5)
    #     sigma_B = pm.Exponential("sigma_B", 1)

    #     # Naive imputation for M
    #     M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

    #     # G model, interactions only
    #     mean_func_G = MeanBodyMassSocialGroupSize(alpha_G, 0, beta_MG, 0, M)

    #     eta_squared_G = pm.TruncatedNormal("eta_squared_G", 1, .25, lower=.01)
    #     rho_G = pm.TruncatedNormal("rho_G", 3, .25, lower=.01)
    #     cov_func_G = eta_squared_G * pm.gp.cov.Exponential(1, ls=rho_G)

    #     gp_G = pm.gp.Marginal(mean_func=mean_func_G, cov_func=cov_func_G)
    #     G = gp_G.marginal_likelihood("G", X=D_mat, y=G_obs, sigma=sigma_G)

    #     # B Model
    #     mean_func_B = MeanBodyMassSocialGroupSize(alpha_B, beta_GB, beta_MB, G, M)

    #     eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.01)
    #     rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.01)
    #     cov_func_B = eta_squared_B * pm.gp.cov.Exponential(1, ls=rho_B)

    #     gp_B = pm.gp.Marginal(mean_func=mean_func_B, cov_func=cov_func_B)
    #     gp_B.marginal_likelihood("B", X=D_mat, y=B_obs, sigma=sigma_B)

    #     G_imputation_inference = pm.sample(target_accept=.95, cores=1)

    _coords = {'primate': PRIMATES['name'].values}
    with pm.Model(coords=_coords) as G_imputation_model:

        # Priors
        _alpha_B = pm.Normal("alpha_B", 0, 1)
        _beta_GB = pm.Normal("beta_GB", 0, 0.5)
        _beta_MB = pm.Normal("beta_MB", 0, 0.5)

        _alpha_G = pm.Normal("alpha_G", 0, 1)
        _beta_MG = pm.Normal("beta_MG", 0, 0.5)

        # Naive imputation for M
        _M = pm.Normal("M", 0, 1, observed=M_obs, dims='primate')

        # G Model Imputation
        _eta_squared_G = pm.TruncatedNormal("eta_squared_G", 1, .25, lower=.001)
        _rho_G = pm.TruncatedNormal("rho_G", 3, .25, lower=.001)

        _K_G = pm.Deterministic('K_G', _eta_squared_G * pm.math.exp(-_rho_G * D_mat))
        _mu_G = _alpha_G + _beta_MG * _M
        _G = pm.MvNormal("G", mu=_mu_G, cov=_K_G, observed=G_obs)

        # B Model
        _eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.001)
        _rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.001)
        _K_B = pm.Deterministic('K_B', _eta_squared_B * pm.math.exp(-_rho_B * D_mat))

        # Likelihood for B
        _mu_B = _alpha_B + _beta_GB * _G + _beta_MB * _M
        pm.MvNormal("B", mu=_mu_B, cov=_K_B, observed=B_obs)

        G_imputation_inference = cached_sample()
    return (G_imputation_inference,)


@app.cell(hide_code=True)
def _(G_imputation_inference, plot_posterior_imputation):
    plot_posterior_imputation(
        "M", "G",
        G_imputation_inference,
        impute_y=True,
        impute_color='C4',
        title="Phylogeny + $M \\rightarrow G$ model"
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Purple points move toward the regression line

    ⚠️ for some reason, after trying two different implementation of GPs using PyMCs `gp` module, as well as using `MVNormal` likelihood (both give similar results), I'm not able to replicate the imputation for solo primates presented in lecture.

    **Comments and suggestions are welcome on what I may be doing incorrectly here**
    """)
    return


@app.cell(hide_code=True)
def _(
    G_body_mass_inference,
    G_imputation_inference,
    G_phylogeny_inference,
    complete_case_inference,
):
    utils.plot_dist(G_body_mass_inference.posterior['beta_GB'], color='C0', label='$M \\rightarrow G$ only')
    utils.plot_dist(G_phylogeny_inference.posterior['beta_GB'], color='C1', label='phylogeny only')
    utils.plot_dist(G_imputation_inference.posterior['beta_GB'], color='C4', label='phylogeny + $M \\rightarrow G$')
    utils.plot_dist(complete_case_inference.posterior['beta_G'], color='k', label='observed')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4. Impute $B,G,M$ using a submodel for each
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an implementation that uses PyMC's Gaussian process module.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the results from lecture.
    """)
    return


@app.cell
def _(B_obs, D_mat, G_obs, M_obs, PRIMATES):
    # with pm.Model() as full_model:

    #     # Priors
    #     sigma_M = pm.Exponential("sigma_M", 1

    #     alpha_G = pm.Normal("alpha_G", 0, 1)
    #     beta_MG = pm.Normal("beta_MG", 0, 1)
    #     sigma_G = pm.Exponential("sigma_G", 1)

    #     alpha_B = pm.Normal("alpha_B", 0, 1)
    #     beta_GB = pm.Normal("beta_GB", 0, 0.5)
    #     beta_MB = pm.Normal("beta_MB", 0, 0.5)
    #     sigma_B = pm.Exponential("sigma_B", 1))

    #     # Naive imputation for M
    #     eta_squared_M = pm.TruncatedNormal("eta_squared_M", 1, .25, lower=.01)
    #     rho_M = pm.TruncatedNormal("rho_M", 3, .25, lower=.01)

    #     cov_func_M = eta_squared_M * pm.gp.cov.Exponential(1, ls=rho_M)
    #     mean_func_M = pm.gp.mean.Zero()

    #     gp_M = pm.gp.Marginal(mean_func=mean_func_M, cov_func=cov_func_M)
    #     M = gp_M.marginal_likelihood("M", X=D_mat, y=M_obs, sigma=sigma_M)

    #     # G model, interactions only
    #     mean_func_G = MeanBodyMassSocialGroupSize(alpha_G, 0, beta_MG, 0, M)

    #     eta_squared_G = pm.TruncatedNormal("eta_squared_G", 1, .25, lower=.01)
    #     rho_G = pm.TruncatedNormal("rho_G", 3, .25, lower=.01)
    #     cov_func_G = eta_squared_G * pm.gp.cov.Exponential(1, ls=rho_G)

    #     gp_G = pm.gp.Marginal(mean_func=mean_func_G, cov_func=cov_func_G)
    #     G = gp_G.marginal_likelihood("G", X=D_mat, y=G_obs, sigma=sigma_G)

    #     # B Model
    #     mean_func_B = MeanBodyMassSocialGroupSize(alpha_B, beta_GB, beta_MB, G, M)

    #     eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.01)
    #     rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.01)
    #     cov_func_B = eta_squared_B * pm.gp.cov.Exponential(1, ls=rho_B)

    #     gp_B = pm.gp.Marginal(mean_func=mean_func_B, cov_func=cov_func_B)
    #     gp_B.marginal_likelihood("B", X=D_mat, y=B_obs, sigma=sigma_B)

    #     full_inference = pm.sample(target_accept=.95, cores=1)

    _coords = {'primate': PRIMATES['name'].values}
    with pm.Model(coords=_coords) as full_model:

        # Priors
        _alpha_B = pm.Normal("alpha_B", 0, 1)
        _beta_GB = pm.Normal("beta_GB", 0, 0.5)
        _beta_MB = pm.Normal("beta_MB", 0, 0.5)

        _alpha_G = pm.Normal("alpha_G", 0, 1)
        _beta_MG = pm.Normal("beta_MG", 0, 0.5)

        # M model (imputation)
        _eta_squared_M = pm.TruncatedNormal("eta_squared_M", 1, .25, lower=.001)
        _rho_M = pm.TruncatedNormal("rho_M", 3, .25, lower=.001)

        _K_M = pm.Deterministic('K_M', _eta_squared_M * pm.math.exp(-_rho_M * D_mat))
        _mu_M = pm.math.zeros_like(M_obs)
        _M = pm.MvNormal("M", mu=_mu_M, cov=_K_M, observed=M_obs)

        # G Model (imputation)
        _eta_squared_G = pm.TruncatedNormal("eta_squared_G", 1, .25, lower=.001)
        _rho_G = pm.TruncatedNormal("rho_G", 3, .25, lower=.001)

        _K_G = pm.Deterministic('K_G', _eta_squared_G * pm.math.exp(-_rho_G * D_mat))
        _mu_G = _alpha_G + _beta_MG * _M
        _G = pm.MvNormal("G", mu=_mu_G, cov=_K_G, observed=G_obs)

        # B Model
        _eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, .25, lower=.001)
        _rho_B = pm.TruncatedNormal("rho_B", 3, .25, lower=.001)
        _K_B = pm.Deterministic('K_B', _eta_squared_B * pm.math.exp(-_rho_B * D_mat))

        # Likelihood for B
        _mu_B = _alpha_B + _beta_GB * _G + _beta_MB * _M
        pm.MvNormal("B", mu=_mu_B, cov=_K_B, observed=B_obs)

        full_inference = cached_sample()
    return (full_inference,)


@app.cell(hide_code=True)
def _(full_inference, plot_posterior_imputation):
    plot_posterior_imputation(
        "M", "G",
        full_inference,
        impute_y=True,
        impute_color='C5',
        title="Full model, including M"
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Imputation changes little from the previous model that does not directly model the causes of $M$. I would think that this is because there are only two missing $M$ values
    """)
    return


@app.cell(hide_code=True)
def _(
    G_body_mass_inference,
    G_imputation_inference,
    G_phylogeny_inference,
    complete_case_inference,
    full_inference,
):
    utils.plot_dist(G_body_mass_inference.posterior['beta_GB'], color='C0', label='$M \\rightarrow G$ only')
    utils.plot_dist(G_phylogeny_inference.posterior['beta_GB'], color='C1', label='phylogeny only')
    utils.plot_dist(G_imputation_inference.posterior['beta_GB'], color='C4', label='phylogeny + $M \\rightarrow G$')
    utils.plot_dist(full_inference.posterior['beta_GB'], color='C5', label='full model, including M')
    utils.plot_dist(complete_case_inference.posterior['beta_G'], color='k', label='observed')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    we also get very similare effects to the phylogeny + $M \rightarrow G$ model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Review: Imputing Primates

    - **Key Idea**: missing values have probability distributions
    - Think like a graph, not like a regression
    - Blind imputation without relationships among predictorrs may be risky
      - take advantage of partial pooling
    - Even if imputation doesn't change results, it's scientific duty
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
