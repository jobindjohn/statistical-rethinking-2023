import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
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
    # [Lecture 06 - Good & Bad Controls](https://youtu.be/uanZZLlzKHw?si=ODJXflnZndT_SMe9)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Avoid being clever at all costs
    Being clever is
    - unreliable
    - opaque

    Using explicit causal models allows one to:
    - derive implications using logic
    - verify work & assumptions
    - facilitates peer review & verification
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Confounds
    ## Review
    - The Fork $X \leftarrow Z \rightarrow Y$
      - $X$ and $Y$ are associated unless we stratify by $Z$
    - The Pipe $X \rightarrow Z \rightarrow Y$
      - $X$ and $Y$ are associated unless we stratify by $Z$
    - The Collider $X \rightarrow Z \leftarrow Y$
      - $X$ and $Y$ are not associated unless we stratify by $Z$
    - The Descendant $Z \rightarrow A$
      - Descendant $A$ takes on behavior of parent $Z$
    """)
    return


@app.cell(hide_code=True)
def confounded_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("U", "X"),
            ("U", "Y"),
            ("X", "Y")
        ],
        node_props={
            "X": {"label": "treatment, X"},
            "Y": {"label": "outcome, Y"},
            "U": {"label": "confound, U", "style": "dashed"},
            "unmeasured": {"style": "dashed"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def randomized_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("U", "X"),
            ("U", "Y"),
            ("X", "Y"),
            ("R", "X")
        ],
        node_props={
            "X": {"label": "treatment, X"},
            "Y": {"label": "outcome, Y"},
            "U": {"label": "confound, U", "style": "dashed"},
            "R": {"label": "randomization, R"},
            "unmeasured": {"style": "dashed"}
        },
        edge_props={
            ("U", "X"): {"color": 'lightgray'}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The gold standard is **Randomization**. However, randomization often isn't possible:
    - impossible
    - pragmatism
    - ethical concerns
    - unmeasured confounds

    ## Causal Thinking
    - We would like a procedure $do(X)$ that intervenes on $X$ in such a way that it can "mimic" the effect of randomization.
    - Such a procedure would transform the Confounded graph:

    #### Without randomization
    """)
    return


@app.cell(hide_code=True)
def confounded_dag_plain():
    utils.draw_causal_graph(
        [
            ("U", "X"),
            ("U", "Y"),
            ("X", "Y"),
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    in such a way that all the non-causal arrows entering X have been removed
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### With "randomization" induced by $do(X)$
    """)
    return


@app.cell(hide_code=True)
def do_x_dag():
    utils.draw_causal_graph(
        [
            ("U", "Y"),
            ("X", "Y"),
        ],
        edge_props={
            ("X", "Y"): {"label": "do(X)   "}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It turns out that we can analyze graph structure to determine if there is such a procedure that exists
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Simple Confound
    """)
    return


@app.cell(hide_code=True)
def simple_confound_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("U", "Y"),
            ("U", "X"),
            ("X", "Y")
        ],
        node_props={
            "U": {"color": "red"},
        },
        edge_props={
            ("U", "X"): {"color": "red"},
            ("U", "Y"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the Fork example, we've shown that stratifying by the confound, we "close" the fork by conditioning on U, thus removing any of the causal effect of U on X, thus allowing us to isolate the treatment's effect on Y.

    This procedure is part of what is known as Do-calculus. The operator `do(X)` tends to mean intervening on X (i.e. setting it to a specific value that is independent of the confound)

    $$
    p(Y | do(X)) = \sum_U p(Y | X, U)p(U) = \mathbb{E}_U[p(Y | X, U)]
    $$

    i.e. **the procedure that gives us the intervention on X is equivalent to the distribution of Y, stratified by the treatment X and the confound U, averaged over the distribution of the confound.**

    > Note that when we use linear regression estimator for each X, we are **implicitly marginalizing and averaging** over our treatment and confound (e.g. in the model form $Y \sim \mathcal{N}(\alpha + \beta_X X + \beta_U U, \sigma^2)$)

    - it is generally **not the estimated coefficient** in the model that relates X to Y
    - it is the distribution of Y when we change X, **averaged over the distribution** defined by the control/confound variables (i.e. U)

    ## Do-calculus
    - Applied to DAGs, provides a set of rules for identifying $p(Y | do(X))$
    - **Informs what is possible** before picking functions or distributions
    - **Justifies graphical analysis**
    - If do-calculus claims that inference is possible no further special assumptions are required for inference
      - often additional assumptions can make the inference even stronger

    ## Backdoor Criterion
    **Shortcut for applying Do-calculus graphically with your eyeballs.** General rule for finding the minimal sufficient adjustment set of variables to condition on.

    1. Identify **all** paths connecting treatment X to outcome Y, including those entering and exiting X (association can be directed/undirected, causation is directed)
    2. Any of those paths **entering X** are backdoor (non-causal) paths
    3. Find the **adjustment set** of variables that, once conditioned on, "closes/blocks" all the backdoor paths identified in step 2

    #### Backdoor Criterion Example
    """)
    return


@app.cell(hide_code=True)
def backdoor_example_dag():
    utils.draw_causal_graph(
        [
            ("U", "Z"),
            ("Z", "X"),
            ("U", "Y"),
            ("X", "Y")
        ],
        node_props={
            "Z": {"color": "red"},
            "U": {"style": "dashed"}
        },
        edge_props={
            ("U", "Z"): {"color": "red"},
            ("Z", "X"): {"color": "red"},
            ("U", "Y"): {"color": "red"}
        },
        graph_direction="LR"

    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Backdoor path highlighted in red.

    - If we could measure $U$ we could just stratify by $U$; however, it is unobserved.
    - However, we can block the backdoor path by conditioning on $Z$, despite not being able to measure $U$.
    - This works because $Z$ "knows" everything we need to know about association between $X$, $Y$ that is due to the unmeasured confound $U$.

    #### Resulting graph after stratifying by $Z$
    - The $U \rightarrow Z \rightarrow X$ Pipe has now been broken, dissociating $X$ from the confound $U$
    - Note: this doesn't remove the confound's effect on $Y$
    """)
    return


@app.cell(hide_code=True)
def backdoor_blocked_dag():
    utils.draw_causal_graph(
        [
            ("U", "Z"),
            ("Z", "X"),
            ("U", "Y"),
            ("X", "Y")
        ],
        node_props={
            "Z": {"color": "red"},
            "U": {"style": "dashed"}
        },
        edge_props={
            ("U", "Z"): {"color": "red"},
            ("Z", "X"): {"color": "none"},
            ("U", "Y"): {"color": "red"}
        },
        graph_direction="LR"

    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Validate through simulation

    Here we simulate a situation where Y is caused by X and an unmeasured confound U that also affects X through Z. (We could also prove mathematically, but simulation is quite convincing as well--for me anyways)

    $$
    \begin{align*}
    U &\sim \text{Bernoulli}(0.5) \\
    Z &\sim \text{Normal}(\beta_{UZ}U, 1) \\
    X &\sim \text{Normal}(\beta_{ZX}Z, 1) \\
    Y &\sim \text{Normal}(\alpha + \beta_{XY}X + \beta_{UY}U, 1) \\
    \end{align*}
    $$
    """)
    return


@app.cell
def simulate_backdoor():
    np.random.seed(123)
    n_samples = 200

    alpha = 0
    beta_XY = 0
    beta_UY = -1
    beta_UZ = -1
    beta_ZX = 1

    U = stats.bernoulli.rvs(p=.5, size=n_samples)
    Z = stats.norm.rvs(loc=beta_UZ * U)
    X = stats.norm.rvs(loc=beta_ZX * Z)
    Y = stats.norm.rvs(loc=alpha + beta_XY * X + beta_UY * U)
    return X, Y, Z, beta_XY


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Unstratified (confounded) Model

    $$
    \begin{align*}
    Y &\sim \text{Normal}(\mu_Y, \sigma_Y) \\
    \mu_Y &= \alpha + \beta_{XY}X \\
    \alpha &\sim \text{Normal}(0, 1) \\
    \beta_{XY} &\sim \text{Normal}(0, 1) \\
    \sigma_Y &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Fit the unstratified model, ignoring Z (and U)
    """)
    return


@app.cell
def unstratified_fit(X, Y):
    with pm.Model() as unstratified_model:
        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _beta_XY = pm.Normal("beta_XY", 0, 1)
        _sigma = pm.Exponential("sigma", 1)

        # Likelihood
        _mu = _alpha + _beta_XY * X
        _Y = pm.Normal("Y", mu=_mu, sigma=_sigma, observed=Y)
        unstratified_model_inference = cached_sample()
    return (unstratified_model_inference,)


@app.cell
def unstratified_summary(unstratified_model_inference):
    az.summary(unstratified_model_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Stratifying by Z (unconfounded)

    $$
    \begin{align*}
    Y &\sim \text{Normal}(\mu_Y, \sigma_Y) \\
    \mu_Y &= \alpha + \beta_{XY}X + \beta_{Z}Z \\
    \alpha &\sim \text{Normal}(0, 1) \\
    \beta_{*} &\sim \text{Normal}(0, 1) \\
    \sigma_Y &\sim \text{Exponential}(1)
    \end{align*}
    $$
    """)
    return


@app.cell
def stratified_fit(X, Y, Z):
    # Fit the stratified Model
    with pm.Model() as stratified_model:
        # Priors
        _alpha = pm.Normal("alpha", 0, 1)
        _beta_XY = pm.Normal("beta_XY", 0, 1)
        _beta_Z = pm.Normal("beta_Z", 0, 1)
        _sigma = pm.Exponential("sigma", 1)

        # Likelihood (includes Z)
        _mu = _alpha + _beta_XY * X + _beta_Z * Z
        _Y = pm.Normal("Y", mu=_mu, sigma=_sigma, observed=Y)
        stratified_model_inference = cached_sample()
    return (stratified_model_inference,)


@app.cell
def stratified_summary(stratified_model_inference):
    az.summary(stratified_model_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **NOTE**: the model coefficient `beta_Z` means nothing in terms of causal effect of $Z$ on $Y$. In order to determine the causal effect of $Z$ on $Y$ you'd need a different estimator. In general, variables in the adjustment set are not interpretable. This is related to the "Table 2 Fallacy"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare stratified and unstratified models
    """)
    return


@app.cell(hide_code=True)
def stratified_vs_unstratified_plot(
    beta_XY,
    stratified_model_inference,
    unstratified_model_inference,
):
    _fig, _ax = plt.subplots(figsize=(8, 3))
    utils.plot_dist(unstratified_model_inference.posterior.beta_XY, color='k', label='Y|X', ax=_ax)
    utils.plot_dist(stratified_model_inference.posterior.beta_XY, color='C0', label='Y|X,Z', ax=_ax)
    plt.axvline(beta_XY, color='C0', linestyle='--', alpha=.5, label=f"Actual={beta_XY}")
    plt.xlim([-.5, .5])
    plt.xlabel("posterior beta_XY")
    plt.ylabel("density")
    plt.legend()
    _fig
    return


@app.function(hide_code=True)
def ols_first_slopes(y, *covariates):
    """Least-squares coefficient on the first covariate, fitted separately to
    each simulated dataset. `y` and each covariate have shape (n_simulations, n)."""
    design = np.stack([np.ones_like(y), *covariates], axis=-1)
    xtx = design.transpose(0, 2, 1) @ design
    xty = design.transpose(0, 2, 1) @ y[..., None]
    return np.linalg.solve(xtx, xty)[:, 1, 0]


@app.function(hide_code=True)
def plot_estimator_comparison(unstratified, stratified, actual):
    """Distributions of the Y ~ X and Y ~ X + Z estimates over many simulated datasets."""
    _, ax = plt.subplots(figsize=(8, 4))
    utils.plot_dist(unstratified, label="Y ~ X\n(ignore Z)", color="black", ax=ax)
    utils.plot_dist(stratified, label="Y ~ X + Z\n(stratify by Z)", color="C0", ax=ax)
    plt.axvline(actual, color="k", linestyle="--", label=f"actual={actual:g}")
    plt.legend(loc="upper left")
    plt.xlabel("estimated effect of X on Y")
    plt.ylabel("density")
    return ax


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Explore the backdoor
    Simulate 500 datasets from the generative model above and fit both estimators to each one (by least squares, for speed). Stratifying by $Z$ recovers $\beta_{XY}$ whatever the confound does. Ignoring $Z$ is biased unless the backdoor path is broken, i.e. one of $\beta_{UZ}$, $\beta_{ZX}$ or $\beta_{UY}$ is $0$.
    """)
    return


@app.cell(hide_code=True)
def backdoor_simulation_controls():
    backdoor_beta_uz = mo.ui.slider(-2, 2, step=0.25, value=-1, label=r"$\beta_{UZ}$", show_value=True)
    backdoor_beta_zx = mo.ui.slider(-2, 2, step=0.25, value=1, label=r"$\beta_{ZX}$", show_value=True)
    backdoor_beta_uy = mo.ui.slider(-2, 2, step=0.25, value=-1, label=r"$\beta_{UY}$", show_value=True)
    backdoor_beta_xy = mo.ui.slider(-1, 1, step=0.25, value=0, label=r"$\beta_{XY}$", show_value=True)
    backdoor_n_samples = mo.ui.slider(steps=[50, 100, 200, 500], value=200, label="Samples per dataset", show_value=True)
    return (
        backdoor_beta_uy,
        backdoor_beta_uz,
        backdoor_beta_xy,
        backdoor_beta_zx,
        backdoor_n_samples,
    )


@app.cell(hide_code=True)
def backdoor_simulation_explorer(
    backdoor_beta_uy,
    backdoor_beta_uz,
    backdoor_beta_xy,
    backdoor_beta_zx,
    backdoor_n_samples,
):
    _rng = np.random.default_rng(123)
    _shape = (500, backdoor_n_samples.value)
    _U = _rng.binomial(1, 0.5, size=_shape)
    _Z = _rng.normal(backdoor_beta_uz.value * _U)
    _X = _rng.normal(backdoor_beta_zx.value * _Z)
    _Y = _rng.normal(backdoor_beta_xy.value * _X + backdoor_beta_uy.value * _U)

    plot_estimator_comparison(ols_first_slopes(_Y, _X), ols_first_slopes(_Y, _X, _Z), actual=backdoor_beta_xy.value)
    plt.title("500 simulated datasets")
    mo.vstack([
        mo.hstack([backdoor_beta_uz, backdoor_beta_zx, backdoor_beta_uy, backdoor_beta_xy, backdoor_n_samples], justify="start", wrap=True),
        plt.gcf(),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## More Complicated Example
    """)
    return


@app.cell(hide_code=True)
def complex_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "Z"),
            ("A", "X"),
            ("Z", "X"),
            ("B", "Z"),
            ("B", "Y"),
            ("Z", "Y"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y")
        ],
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### All Paths Connecting X to Y
    """)
    return


@app.cell(hide_code=True)
def complex_dag_direct_path():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "Z"),
            ("A", "X"),
            ("Z", "X"),
            ("B", "Z"),
            ("B", "Y"),
            ("Z", "Y"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y")
        ],
        edge_props={
            ("X", "Y"): {"color": "blue", "label": "Direct Causal Path", "fontcolor": "blue"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $X \rightarrow Y$
    - Direct, causal path, leave open
    """)
    return


@app.cell(hide_code=True)
def complex_dag_xcy():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "Z"),
            ("A", "X"),
            ("Z", "X"),
            ("B", "Z"),
            ("B", "Y"),
            ("Z", "Y"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y")
        ],
        edge_props={
            ("C", "X"): {"color": "red", "label": "XCY", "fontcolor": "red"},
            ("C", "Y"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $X \leftarrow C \rightarrow Y$
    - Backdoor non-causal path
    - Block by **stratifying by $C$**
    """)
    return


@app.cell(hide_code=True)
def complex_dag_xzy():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "Z"),
            ("A", "X"),
            ("Z", "X"),
            ("B", "Z"),
            ("B", "Y"),
            ("Z", "Y"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y")
        ],
        edge_props={
            ("Z", "X"): {"color": "red", "label": "XZY", "fontcolor": "red"},
            ("Z", "Y"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $X \leftarrow Z \rightarrow Y$
    - Backdoor non-causal path
    - Block by **stratifying by $Z$**
    """)
    return


@app.cell(hide_code=True)
def complex_dag_xazby():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "Z"),
            ("A", "X"),
            ("Z", "X"),
            ("B", "Z"),
            ("B", "Y"),
            ("Z", "Y"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y")
        ],
        edge_props={
            ("A", "X"): {"color": "red", "label": "XAZBY", "fontcolor": "red"},
            ("A", "Z"): {"color": "red"},
            ("B", "Z"): {"color": "red"},
            ("B", "Y"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $X \leftarrow A \rightarrow Z \leftarrow B \rightarrow Y$
    - Backdoor non-causal path
    - Block by **stratifying by $A$ or $B$**; stratifying by $Z$ opens the path b.c. it's a collider
      - we're already stratifying by $Z$ for the $X,Z,Y$ backdoor path
    """)
    return


@app.cell(hide_code=True)
def complex_dag_xazy():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "Z"),
            ("A", "X"),
            ("Z", "X"),
            ("B", "Z"),
            ("B", "Y"),
            ("Z", "Y"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y")
        ],
        edge_props={
            ("A", "X"): {"color": "red", "label": "XAZY", "fontcolor": "red"},
            ("A", "Z"): {"color": "red"},
            ("Z", "Y"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $X \leftarrow A \rightarrow Z \rightarrow Y$
    - Backdoor non-causal path
    - Block by **stratifying by $Z$** (or $A$); on this path $Z$ is a pipe, not a collider
      - we're already stratifying by $Z$ for the $X \leftarrow Z \rightarrow Y$ backdoor path, so this path is already closed
    """)
    return


@app.cell(hide_code=True)
def complex_dag_xzby():
    utils.draw_causal_graph(
        edge_list=[
            ("A", "Z"),
            ("A", "X"),
            ("Z", "X"),
            ("B", "Z"),
            ("B", "Y"),
            ("Z", "Y"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y")
        ],
        edge_props={
            ("Z", "X"): {"color": "red", "label": "XZBY", "fontcolor": "red"},
            ("B", "Z"): {"color": "red"},
            ("B", "Y"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### $X \leftarrow Z \leftarrow B \rightarrow Y$
    - Backdoor non-causal path
    - Block by **stratifying by $Z$** (or $B$); on this path $Z$ is a pipe, not a collider
      - we're already stratifying by $Z$ for the $X \leftarrow Z \rightarrow Y$ backdoor path, so this path is already closed
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Resulting Minimum Adjustment set: Z, C, (A or B)
    Choosing B over A turns out to be more statistically efficient, though not causally different than choosing A
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Backdoor path finder
    Pick a DAG and tick the variables to condition on. Every path between the treatment and the outcome is listed as open or closed under the d-separation rules: conditioning on a pipe or fork blocks a path, and conditioning on a collider (or one of its descendants) opens it. In the graph, open causal paths are blue, open non-causal paths are red, conditioned variables are shaded, and unobserved variables are dashed (they can't be conditioned on).
    """)
    return


@app.cell(hide_code=True)
def backdoor_path_helpers():
    def find_paths(edges, start, end):
        """All simple paths from `start` to `end`, ignoring edge direction."""
        neighbors = {}
        for a, b in edges:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
        paths = []

        def extend(path):
            if path[-1] == end:
                paths.append(path)
                return
            for node in sorted(neighbors[path[-1]]):
                if node not in path:
                    extend(path + [node])

        extend([start])
        return paths


    def descendants(edges, node):
        children = {b for a, b in edges if a == node}
        found = set(children)
        for child in children:
            found |= descendants(edges, child)
        return found


    def path_is_open(edges, path, conditioned):
        """d-separation: check each node in the middle of `path`."""
        for prev, node, nxt in zip(path, path[1:], path[2:]):
            is_collider = (prev, node) in edges and (nxt, node) in edges
            if is_collider:
                if node not in conditioned and not descendants(edges, node) & conditioned:
                    return False
            elif node in conditioned:
                return False
        return True


    def path_label(edges, path):
        label = path[0]
        for a, b in zip(path, path[1:]):
            label += (r" \rightarrow " if (a, b) in edges else r" \leftarrow ") + b
        return f"${label}$"

    return find_paths, path_is_open, path_label


@app.cell(hide_code=True)
def backdoor_path_controls():
    BACKDOOR_DAGS = {
        "More complicated example": dict(
            edges=[("A", "Z"), ("A", "X"), ("Z", "X"), ("B", "Z"), ("B", "Y"),
                   ("Z", "Y"), ("X", "Y"), ("C", "X"), ("C", "Y")],
            treatment="X", outcome="Y", unobserved=[],
        ),
        "Backdoor through Z (U unobserved)": dict(
            edges=[("U", "Z"), ("Z", "X"), ("U", "Y"), ("X", "Y")],
            treatment="X", outcome="Y", unobserved=["U"],
        ),
        "Grandparents (u unobserved)": dict(
            edges=[("G", "P"), ("P", "C"), ("G", "C"), ("u", "P"), ("u", "C")],
            treatment="G", outcome="C", unobserved=["u"],
        ),
        "Bad control": dict(
            edges=[("u", "Z"), ("v", "Z"), ("u", "X"), ("X", "Y"), ("v", "Y")],
            treatment="X", outcome="Y", unobserved=["u", "v"],
        ),
        "Bad mediator": dict(
            edges=[("X", "Z"), ("Z", "Y"), ("u", "Z"), ("u", "Y")],
            treatment="X", outcome="Y", unobserved=["u"],
        ),
        "Table 2: smoking, age, HIV, stroke": dict(
            edges=[("X", "Y"), ("S", "Y"), ("S", "X"), ("A", "Y"), ("A", "X"), ("A", "S")],
            treatment="X", outcome="Y", unobserved=[],
        ),
    }
    backdoor_dag_choice = mo.ui.dropdown(list(BACKDOOR_DAGS), value="More complicated example", label="DAG")
    return BACKDOOR_DAGS, backdoor_dag_choice


@app.cell(hide_code=True)
def backdoor_adjustment_controls(BACKDOOR_DAGS, backdoor_dag_choice):
    _dag = BACKDOOR_DAGS[backdoor_dag_choice.value]
    _observed = sorted(
        {_node for _edge in _dag["edges"] for _node in _edge}
        - {_dag["treatment"], _dag["outcome"]}
        - set(_dag["unobserved"])
    )
    backdoor_adjustment = mo.ui.dictionary({_node: mo.ui.checkbox(label=_node) for _node in _observed})
    return (backdoor_adjustment,)


@app.cell(hide_code=True)
def backdoor_path_explorer(
    BACKDOOR_DAGS,
    backdoor_adjustment,
    backdoor_dag_choice,
    find_paths,
    path_is_open,
    path_label,
):
    _dag = BACKDOOR_DAGS[backdoor_dag_choice.value]
    _edges = set(_dag["edges"])
    _treatment, _outcome = _dag["treatment"], _dag["outcome"]
    _conditioned = {_node for _node, _ticked in backdoor_adjustment.value.items() if _ticked}

    _rows, _edge_colors = [], {}
    _causal_open, _noncausal_closed = True, True
    for _path in find_paths(_edges, _treatment, _outcome):
        _steps = list(zip(_path, _path[1:]))
        _forward = [_step in _edges for _step in _steps]
        if all(_forward):
            _kind = "causal"
        elif not _forward[0]:
            _kind = "backdoor"
        else:
            _kind = "non-causal"
        _open = path_is_open(_edges, _path, _conditioned)
        if _kind == "causal":
            _causal_open &= _open
            _status = "open ?" if _open else "closed ?"
        else:
            _noncausal_closed &= not _open
            _status = "open ?" if _open else "closed ?"
        if _open:
            for _a, _b in _steps:
                _edge = (_a, _b) if (_a, _b) in _edges else (_b, _a)
                if _kind != "causal":
                    _edge_colors[_edge] = "red"
                elif _edge_colors.get(_edge) != "red":
                    _edge_colors[_edge] = "blue"
        _rows.append(f"| {path_label(_edges, _path)} | {_kind} | {_status} |")

    if _causal_open and _noncausal_closed:
        _verdict = mo.callout(mo.md(f"This adjustment set identifies the **total causal effect** of ${_treatment}$ on ${_outcome}$."), kind="success")
    elif _noncausal_closed:
        _verdict = mo.callout(mo.md(f"Every non-causal path is closed, but the adjustment set also blocks part of the causal effect: the estimate is a partial (direct) effect of ${_treatment}$, not its total effect."), kind="warn")
    else:
        _verdict = mo.callout(mo.md(f"Some non-causal paths are open: the estimate of the effect of ${_treatment}$ on ${_outcome}$ is confounded."), kind="danger")

    _node_props = {_node: {"style": "dashed"} for _node in _dag["unobserved"]}
    _node_props.update({_node: {"style": "filled", "fillcolor": "lightgray"} for _node in _conditioned})
    _graph = utils.draw_causal_graph(
        _dag["edges"],
        node_props=_node_props,
        edge_props={_edge: {"color": _color, "penwidth": "2"} for _edge, _color in _edge_colors.items()},
        graph_direction="LR",
    )
    mo.vstack([
        mo.hstack([backdoor_dag_choice, mo.md("Condition on:"), backdoor_adjustment.hstack(justify="start")], justify="start", align="center", gap=1),
        mo.hstack([_graph, mo.vstack([mo.md("| Path | Type | Status |\n|---|---|---|\n" + "\n".join(_rows)), _verdict])], justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example with unobserved confounds
    """)
    return


@app.cell(hide_code=True)
def grandparents_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("G", "P"),
            ("P", "C"),
            ("G", "C"),
            ("u", "P"),
            ("u", "C")
        ],
        node_props={
            "G": {"color": "blue"},
            "C": {"color": "blue"},
            "P": {"color": "red", "label": "collider, P"},
            "u": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        },
        edge_props={
            ("G", "C"): {"color": "blue"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - $P$ is mediator between $G$ and $C$
    - $P$ is also a collider of $G$ and the unobserved $u$
    - If we want to estimate direct effect of $G \rightarrow C$, we'll need to stratify by $P$--close the Pipe
    - However, this will open up the Collider path to the unobserved confound.
    - **It's not possible to accurately estimate the _Direct Causal Effect_ of G on C**
    - It **is possible to estimate the _Total Causal Effect_**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Good and bad controls

    ## Common _incorrect_ heuristics for choosing control variables:

    - YOLO approach -- anything in the spreadsheet
    - Ignore highly collinear variables
      - false, no support for this
      - collinearity can arise through many different causal processes that _can be modeled accurately_
    - It's safe to add pre-treatment variables
      - false, pre-treatment, just like post-treatment variables can cause confounds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Good & Bad Controls Examples
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Bad control
    """)
    return


@app.cell(hide_code=True)
def bad_control_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("u", "Z"),
            ("v", "Z"),
            ("u", "X"),
            ("X", 'Y'),
            ("v", "Y")
        ],
        node_props={
            "u": {"style": "dashed"},
            "v": {"style": "dashed"},
            "X": {"color": "red"},
            "Y": {"color": "red"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("X", "Y"): {"color": "red"}
        },
        graph_direction="TD"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $Z$ is a collider for unobserved variables $u$ and $v$, which independently affect $X$ and $Y$
    #### List the paths
    - $X \rightarrow Y$
      - causal, leave open
    - $X \leftarrow u \rightarrow Z \leftarrow v \rightarrow Y$
      - backdoor, closed due to collider
      - $Z$ is a **bad control**: stratifying by $Z$ would open the backdoor path
      - $Z$ could be a pre-treatment variable -- not always good to stratify by pre-treatment variables; **draw your causal assumptions**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Bad mediator
    """)
    return


@app.cell(hide_code=True)
def bad_mediator_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "Y"),
            ("u", "Z"),
            ("u", "Y"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
            "u": {"style": "dashed"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("X", "Z"): {"color": "red"},
            ("Z", "Y"): {"color": "red"}
        },
        graph_direction='LR'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### List the paths
    - $X \rightarrow Z \rightarrow Y$
      - causal, leave open
    - $X \rightarrow Z \leftarrow u \rightarrow Y$
      - backdoor, but only if stratifying by Z
    - There **is no backdoor path, so no need to stratify by Z**
    - Can measure total effect of $X$ on $Y$, but not direct effect, because of Mediator $Z$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    No backdoor path here, so no need to control for any confounds. In fact, stratifying by `Z` (the bad mediator) will introduce bias in estimate because it introduces the causal effect of `u` that would otherwise be blocked.

    `Z` is often a post-treatment variable, e.g. "Happiness", which is affected by the treatment "Win Lottery"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Verify bad mediator with simulation:
    """)
    return


@app.cell(hide_code=True)
def bad_mediator_labelled_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "Y"),
            ("u", "Z"),
            ("u", "Y"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
            "u": {"style": "dashed"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("X", "Z"): {"color": "red", "label": "1"},
            ("Z", "Y"): {"color": "red", "label": "1"},
            ("u", "Z"): {"label": "1"},
            ("u", "Y"): {"label": "1"}
        },
        graph_direction='LR'
    )
    return


@app.function
def simulate_bad_mediator(beta_XZ, beta_ZY, n_samples=100):
    # independent variables
    u = stats.norm.rvs(size=n_samples)
    X = stats.norm.rvs(size=n_samples)

    # causal effect of X on Z
    mu_Z = X * beta_XZ + u
    Z = stats.norm.rvs(size=n_samples, loc=mu_Z)

    # Causal effect of Z on Y (including confound)
    mu_Y = Z * beta_ZY + u
    Y = stats.norm.rvs(size=n_samples, loc=mu_Y)

    # Put data into format for statsmodels
    data = pd.DataFrame(np.vstack([Y, X, Z, u]).T, columns=["Y", "X", "Z", "u"])

    unstratified_model = smf.ols("Y ~ X", data=data).fit()
    stratified_model = smf.ols("Y ~ X + Z", data=data).fit()

    return unstratified_model.params.X, stratified_model.params.X


@app.function
def run_bad_mediator_simulation(beta_XZ=1, beta_ZY=1, n_simulations=500, n_samples_per_simulation=100):
    beta_X = beta_XZ * beta_ZY

    simulations = np.array(
        [
            simulate_bad_mediator(beta_XZ=beta_XZ, beta_ZY=beta_ZY, n_samples=n_samples_per_simulation)
            for _ in range(n_simulations)
        ]
    )
    _, ax = plt.subplots(figsize=(8, 4))
    utils.plot_dist(simulations[:, 0], label='Y ~ X\ncorrect', color='black', ax=ax)
    utils.plot_dist(simulations[:, 1], label='Y ~ X + Z\nwrong', color="C0", ax=ax)
    plt.axvline(beta_X, color='k', linestyle='--', label=f'actual={beta_X}')
    plt.legend(loc='upper left')
    plt.xlabel("estimated effect of X on Y")
    plt.ylabel("density")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Run the simulation, $\beta_{XZ} = \beta_{ZY} = 1$
    """)
    return


@app.cell
def bad_mediator_simulation():
    np.random.seed(123)
    run_bad_mediator_simulation(beta_XZ=1, beta_ZY=1)
    plt.gcf()
    return


@app.cell(hide_code=True)
def bad_mediator_null_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Z"),
            ("Z", "Y"),
            ("u", "Z"),
            ("u", "Y"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
            "u": {"style": "dashed"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("X", "Z"): {"color": "red", "label": "1"},
            ("Z", "Y"): {"color": "red", "label": "0"},
            ("u", "Z"): {"label": "1"},
            ("u", "Y"): {"label": "1"}
        },
        graph_direction='LR'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Turn off Causal effect by changing $\beta_{ZY}$ to 0
    """)
    return


@app.cell
def bad_mediator_null_simulation():
    np.random.seed(123)
    run_bad_mediator_simulation(beta_XZ=1, beta_ZY=0)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    though there is no causal effect, you end up concluding a negative effect of X on Y
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Colliders & Descendants

    Generally, **Avoid the Collider**!

    Adding descendants of the target variable is almost always a terrible idea, because you're selecting groups based on the outcome. This is known as **Case Control Bias** (selection on outcome)
    """)
    return


@app.cell(hide_code=True)
def collider_on_outcome_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("X", "Z"),
            ("Y", "Z")
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"}
        },
        edge_props={
            ("X", "Y"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Colliders not always so obvious
    """)
    return


@app.cell(hide_code=True)
def hidden_collider_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("X", "Z"),
            ("u", "Y"),
            ("u", "Z"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
            "u": {"style": "dashed"},
            "unobserved": {"style": "dashed"}
        },
        edge_props={
            ("X", "Y"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Collider is formed by unobserved variable u
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Bad Descendant: Selection on Outcome (Case Control Bias)

    Stratifying on a variable affected by the outcome is a **very bad** practice.
    - reduces variation in $Y$ that could have been explained by $X$
    """)
    return


@app.cell(hide_code=True)
def case_control_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("Y", "Z"),
        ],
        node_props={
            "X": {"color": "red", "label": "education, X"},
            "Y": {"color": "red", "label": "occupation, Y"},
            "Z": {"label": "income, Z"},
        },
        edge_props={
            ("X", "Y"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Verify via simulation:
    """)
    return


@app.cell(hide_code=True)
def case_control_labelled_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("Y", "Z"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
        },
        edge_props={
            ("X", "Y"): {"color": "red", "label": "1"},
            ("Y", "Z"): {"label": "1"}
        },
        graph_direction="LR"
    )
    return


@app.function
def simulate_case_control_bias(beta_XY=1, beta_YZ=1, n_samples=100):
    # independent variables
    X = stats.norm.rvs(size=n_samples)

    # Causal effect of X on Y
    mu_Y = X * beta_XY
    Y = stats.norm.rvs(size=n_samples, loc=mu_Y)

    # Causal effect of Y on its descendant Z
    mu_Z = Y * beta_YZ
    Z = stats.norm.rvs(size=n_samples, loc=mu_Z)

    # Put data into format for statsmodels
    data = pd.DataFrame(np.vstack([Y, X, Z]).T, columns=["Y", "X", "Z"])

    unstratified_model = smf.ols("Y ~ X", data=data).fit()
    stratified_model = smf.ols("Y ~ X + Z", data=data).fit()

    return unstratified_model.params.X, stratified_model.params.X


@app.function
def run_case_control_simulation(beta_XY=1, beta_YZ=1, n_simulations=500, n_samples_per_simulation=100):
    beta_X = beta_XY

    simulations = np.array(
        [
            simulate_case_control_bias(beta_XY=beta_XY, beta_YZ=beta_YZ, n_samples=n_samples_per_simulation)
            for _ in range(n_simulations)
        ]
    )
    _, ax = plt.subplots(figsize=(8, 4))
    utils.plot_dist(simulations[:, 0], label='Y ~ X\ncorrect', color='black', ax=ax)
    utils.plot_dist(simulations[:, 1], label='Y ~ X + Z\nwrong', color="C0", ax=ax)
    plt.axvline(beta_X, color='k', linestyle='--', label=f'actual={beta_X}')
    plt.legend(loc='upper left')
    plt.xlabel("estimated effect of X on Y")
    plt.ylabel("density")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Descendant explains away some of the causal effect of $X$ on $Y$
    The estimated causal effect has been reduced because the descendant reduces the variation in $Y$ that can be explained by $X$
    """)
    return


@app.cell
def case_control_simulation():
    np.random.seed(123)
    run_case_control_simulation(beta_XY=1, beta_YZ=1)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Removing Descendant effect by setting $\beta_{YZ}=0$
    The descendant no longer has any effect here, so we should recover the same (correct) inference for both stratified and unstratified models
    """)
    return


@app.cell
def case_control_null_simulation():
    np.random.seed(123)
    run_case_control_simulation(beta_XY=1, beta_YZ=0)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Bad Ancestor (aka precision parasite)

    Now $Z$ is a parent of $X$

    - no backdoor path, $X$ is directly connected to $Y$
    - when stratifying by $Z$ you're explaining away variation in $X$, thus reducing the amount of causal information between $X$ and $Y$ that can be explained otherwise.
    - **Does not bias your estimate,** but it **reduces precision**, so estimates will have more uncertainty
    """)
    return


@app.cell(hide_code=True)
def bad_ancestor_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("Z", "X"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
        },
        edge_props={
            ("X", "Y"): {"color": "red", "label": "1"},
            ("Z", "X"): {"label": "1"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Verify via simulation
    """)
    return


@app.cell(hide_code=True)
def bad_ancestor_labelled_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("Z", "X"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
        },
        edge_props={
            ("X", "Y"): {"color": "red", "label": "1"},
            ("Z", "X"): {"label": "1"}
        },
        graph_direction="LR"
    )
    return


@app.function
def simulate_bad_ancestor(beta_ZX=1, beta_XY=1, unobserved_variance=None, n_samples=100):
    Z = stats.norm.rvs(size=n_samples)

    mu_X = Z * beta_ZX
    if unobserved_variance is not None:
        u = stats.norm.rvs(size=n_samples) * unobserved_variance
        mu_X += u

    X = stats.norm.rvs(size=n_samples, loc=mu_X)

    mu_Y = X * beta_XY
    if unobserved_variance is not None:
        mu_Y += u

    Y = stats.norm.rvs(size=n_samples, loc=mu_Y)

    data = pd.DataFrame(np.vstack([X, Y, Z]).T, columns=['X', 'Y', 'Z'])

    non_stratified_model = smf.ols("Y ~ X", data=data).fit()
    stratified_model = smf.ols("Y ~ X + Z", data=data).fit()

    return non_stratified_model.params.X, stratified_model.params.X


@app.function
def run_bad_ancestor_simulation(beta_ZX=1, beta_XY=1, n_simulations=500, unobserved_variance=None, n_samples_per_simulation=100):
    beta_X = beta_XY

    simulations = np.array(
        [
            simulate_bad_ancestor(
                beta_ZX=beta_ZX,
                beta_XY=beta_XY,
                unobserved_variance=unobserved_variance,
                n_samples=n_samples_per_simulation
            )
            for _ in range(n_simulations)
        ]
    )
    _, ax = plt.subplots(figsize=(8, 4))
    utils.plot_dist(simulations[:, 0], label='Y ~ X\n(ignore Z)', color='black', ax=ax)
    utils.plot_dist(simulations[:, 1], label='Y ~ X + Z\n(stratify by Z)', color="C0", ax=ax)
    plt.axvline(beta_X, color='k', linestyle='--', label=f'actual={beta_X}')
    plt.legend(loc='upper left')
    plt.xlabel("estimated effect of X on Y")
    plt.ylabel("density")


@app.cell
def bad_ancestor_simulation():
    np.random.seed(123)
    run_bad_ancestor_simulation()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Stratifying by Z doesn't add bias (it's centered on the correct value), but it does increase variance in estimator. This reduction in precision is proportional to the magnitude of the causal relationship between Z and X
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Increasing the relationship between Z and X further reduces precision
    """)
    return


@app.cell
def strong_ancestor_simulation():
    np.random.seed(123)
    run_bad_ancestor_simulation(beta_ZX=3)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bias Amplification

    Stratifying on an ancestor when there are other confounders, particularly unobserved forks. **This is like the Precision Parasite scenario, but it also adds bias.**
    """)
    return


@app.cell(hide_code=True)
def bias_amplification_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("Z", "X"),
            ("u", "X"),
            ("u", "Y")
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
            "u": {"style": "dashed"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("X", "Y"): {"color": "red"},
            ("Z", "X"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Verify via simulation
    """)
    return


@app.cell(hide_code=True)
def bias_amplification_labelled_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("Z", "X"),
            ("u", "X"),
            ("u", "Y")
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"},
            "u": {"style": "dashed"},
            "unobserved": {"style": "dashed"},
        },
        edge_props={
            ("X", "Y"): {"color": "red", "label": "0"},
            ("Z", "X"): {"color": "red", "label": "1"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Run simulation with no actual causal effect, $\beta_{XY} = 0$
    """)
    return


@app.cell
def bias_amplification_simulation():
    np.random.seed(123)
    run_bad_ancestor_simulation(beta_ZX=1, beta_XY=0, unobserved_variance=1)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Above we see that both estimators are biased -- even in the best case, we can't observe, and thus control for the confound $u$. But when stratifying by the ancestor, things are MUCH WORSE.

    Hand-wavy explanation:
    - in order for $X$ and $Y$ to be associated, their causes need to be associated
    - by stratifying by $Z$, we remove the amount of variation in $X$ that is caused by $Z$
    - this reduction in variation in $X$ makes the confound $u$ more important comparatively
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Discrete example of bias amplification
    Use the sliders to change the strength of $Z \rightarrow X$ and the true effect of $X$ on $Y$.
    """)
    return


@app.function
def simulate_discrete_bias_amplifications(beta_ZX=1, beta_XY=1, n_samples=1000):
    Z = stats.bernoulli.rvs(p=0.5, size=n_samples)
    u = stats.norm.rvs(size=n_samples)

    mu_X = beta_ZX * Z + u
    X = stats.norm.rvs(loc=mu_X, size=n_samples)

    mu_Y = X * beta_XY + u
    Y = stats.norm.rvs(loc=mu_Y, size=n_samples)

    data = pd.DataFrame(np.vstack([u, Z, X, Y]).T, columns=['u', 'Z', 'X', 'Y'])

    models = {}
    models['unstratified'] = smf.ols("Y ~ X", data=data).fit()
    models['z=0'] = smf.ols("Y ~ X", data=data[data.Z == 0]).fit()
    models['z=1'] = smf.ols("Y ~ X", data=data[data.Z == 1]).fit()

    return models, data


@app.cell(hide_code=True)
def bias_amplification_controls():
    bias_amplification_beta_zx = mo.ui.slider(0, 10, step=0.5, value=7, label=r"$\beta_{ZX}$", show_value=True)
    bias_amplification_beta_xy = mo.ui.slider(-1, 1, step=0.25, value=0, label=r"$\beta_{XY}$", show_value=True)
    return bias_amplification_beta_xy, bias_amplification_beta_zx


@app.cell
def bias_amplification_explorer(
    bias_amplification_beta_xy,
    bias_amplification_beta_zx,
):
    np.random.seed(123)
    _beta_ZX = bias_amplification_beta_zx.value
    _beta_XY = bias_amplification_beta_xy.value
    _models, _data = simulate_discrete_bias_amplifications(beta_ZX=_beta_ZX, beta_XY=_beta_XY)

    _fig, _axs = plt.subplots(figsize=(5, 5))

    def _plot_linear_model(model, color, label):
        params = model.params
        xs = np.linspace(_data.X.min(), _data.X.max(), 10)
        ys = params.Intercept + params.X * xs
        utils.plot_line(xs, ys, color=color, label=label)

    for _z in [0, 1]:
        _color = f'C{_z}'
        utils.plot_scatter(_data.X[_data.Z==_z], _data.Y[_data.Z==_z], color=_color)
        _model = _models[f"z={_z}"]
        _plot_linear_model(_model, color=_color, label=f"Z={_z}")

    _model = _models['unstratified']
    _plot_linear_model(_model, color="k", label=f"total sample")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    mo.vstack([
        mo.hstack([bias_amplification_beta_zx, bias_amplification_beta_xy], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - When ignoring Z (the ancestor), the estimate is still somewhat biased (i.e. the black slope is not flat, as it should be for $\beta_{XY}=0$, the default)
    - but it's not nearly as bad as the individual slopes (blue/red) when stratifying by Z.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Good & bad controls lab
    Pick a scenario from this section and set its two causal effects. Each run simulates 500 datasets and fits $Y \sim X$ and $Y \sim X + Z$ to each one (by least squares, for speed). Compare where each estimator lands with the true effect of $X$ on $Y$ (dashed line), and how spread out it is.
    """)
    return


@app.cell(hide_code=True)
def controls_lab_controls():
    lab_scenario = mo.ui.radio(
        ["Bad mediator", "Case control", "Precision parasite", "Bias amplification"],
        value="Bad mediator", label="Scenario", inline=True,
    )
    lab_n_samples = mo.ui.slider(steps=[50, 100, 200, 500], value=100, label="Samples per dataset", show_value=True)
    return lab_n_samples, lab_scenario


@app.cell(hide_code=True)
def controls_lab_coefficients(lab_scenario):
    _coefficients = {
        "Bad mediator": [("beta_XZ", r"\beta_{XZ}", 1), ("beta_ZY", r"\beta_{ZY}", 1)],
        "Case control": [("beta_XY", r"\beta_{XY}", 1), ("beta_YZ", r"\beta_{YZ}", 1)],
        "Precision parasite": [("beta_ZX", r"\beta_{ZX}", 1), ("beta_XY", r"\beta_{XY}", 1)],
        "Bias amplification": [("beta_ZX", r"\beta_{ZX}", 1), ("beta_XY", r"\beta_{XY}", 0)],
    }[lab_scenario.value]
    lab_betas = mo.ui.dictionary({
        _key: mo.ui.slider(-3, 3, step=0.25, value=_value, label=f"${_tex}$", show_value=True)
        for _key, _tex, _value in _coefficients
    })
    return (lab_betas,)


@app.cell(hide_code=True)
def controls_lab_explorer(lab_betas, lab_n_samples, lab_scenario):
    _rng = np.random.default_rng(123)
    _shape = (500, lab_n_samples.value)
    _b = lab_betas.value
    if lab_scenario.value == "Bad mediator":
        _u = _rng.normal(size=_shape)
        _X = _rng.normal(size=_shape)
        _Z = _rng.normal(_b["beta_XZ"] * _X + _u)
        _Y = _rng.normal(_b["beta_ZY"] * _Z + _u)
        _actual = _b["beta_XZ"] * _b["beta_ZY"]
        _edges = {("X", "Z"): _b["beta_XZ"], ("Z", "Y"): _b["beta_ZY"], ("u", "Z"): 1, ("u", "Y"): 1}
    elif lab_scenario.value == "Case control":
        _X = _rng.normal(size=_shape)
        _Y = _rng.normal(_b["beta_XY"] * _X)
        _Z = _rng.normal(_b["beta_YZ"] * _Y)
        _actual = _b["beta_XY"]
        _edges = {("X", "Y"): _b["beta_XY"], ("Y", "Z"): _b["beta_YZ"]}
    elif lab_scenario.value == "Precision parasite":
        _Z = _rng.normal(size=_shape)
        _X = _rng.normal(_b["beta_ZX"] * _Z)
        _Y = _rng.normal(_b["beta_XY"] * _X)
        _actual = _b["beta_XY"]
        _edges = {("Z", "X"): _b["beta_ZX"], ("X", "Y"): _b["beta_XY"]}
    else:
        _Z = _rng.normal(size=_shape)
        _u = _rng.normal(size=_shape)
        _X = _rng.normal(_b["beta_ZX"] * _Z + _u)
        _Y = _rng.normal(_b["beta_XY"] * _X + _u)
        _actual = _b["beta_XY"]
        _edges = {("Z", "X"): _b["beta_ZX"], ("X", "Y"): _b["beta_XY"], ("u", "X"): 1, ("u", "Y"): 1}

    _estimates = {"Y ~ X": ols_first_slopes(_Y, _X), "Y ~ X + Z": ols_first_slopes(_Y, _X, _Z)}
    plot_estimator_comparison(*_estimates.values(), actual=_actual)
    plt.title(f"{lab_scenario.value}: 500 simulated datasets")
    _summary = pd.DataFrame(
        {"mean": [_e.mean() for _e in _estimates.values()], "std": [_e.std() for _e in _estimates.values()]},
        index=list(_estimates),
    ).round(2)
    _dag = utils.draw_causal_graph(
        list(_edges),
        node_props={"u": {"style": "dashed"}} if ("u", "Y") in _edges else None,
        edge_props={_edge: {"label": f" {_value:g}"} for _edge, _value in _edges.items()},
        graph_direction="LR",
    )
    mo.vstack([
        mo.hstack([lab_scenario, lab_n_samples], justify="start", gap=2),
        lab_betas.hstack(justify="start"),
        mo.hstack([plt.gcf(), mo.vstack([_dag, _summary])], justify="start", align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Review: Good & Bad Controls
    - **Confound**: estimator design or sample that "confounds" or "confuses" our causal estimate
    - **Control**: variable added to the analysis so that a causal estimate is possible
    - Adding controls can often be worse than omitting them
    - Make assumptions explicit, and use backdoor criterion to verify those assumptions

    **You have to do scientific modeling to do scientific analysis**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BONUS: The Table 2 Fallacy
    - Not all coefficients represent causal effects, particularly those in the adjustment set
    - Those that _are_ causal effects tend to be partial effects, not total causal effects.
    - Table 2 actively _encourages_ misinterpretation
    - As mentioned multiple times: **Need different estimators for addressing different causal effects.**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Smoking, Age, HIV, and Stroke
    """)
    return


@app.cell(hide_code=True)
def table2_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("S", "Y"),
            ("S", "X"),
            ("A", "Y"),
            ("A", "X"),
            ("A", "S"),
        ],
        node_props={
            "X": {"label": "X, HIV", "color": "red"},
            "Y": {"label": "Y, Stroke", "color": "red"},
            "S": {"label": "S, Smoking"},
            "A": {"label": "A, Age"}
        },
        edge_props={
            ("X", "Y"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Identify paths via backdoor criterion
    - $X \rightarrow Y$ (front door)
    - $X \leftarrow S \rightarrow Y$ (backdoor, fork)
    - $X \leftarrow A \rightarrow Y$ (backdoor, fork)
    - $X \leftarrow A \rightarrow S \rightarrow Y$ (backdoor, fork and pipe)

    Adjustment set is $\{S, A\}$

    ### Conditional Statistical Model

    $$
    \begin{align*}
    Y_i &\sim \mathcal{N}(\mu_i, \sigma) \\
    \mu_i &= \alpha + \beta_X X_i + \beta_S S_i + \beta_A A_i
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Looking at the model "from the perspective of various variables"
    #### From perspective of X
    Conditioning on $A$ and $S$ essentially removes the arrows going into $X$, $\beta_X$ giving us the **direct effect** of $X$ on $Y$
      - we've removed all backdoor paths by stratifying by $S$ and $A$ through the coefficients $\beta_S, \beta_A$
    """)
    return


@app.cell(hide_code=True)
def table2_x_perspective_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("S", "Y"),
            ("S", "X"),
            ("A", "Y"),
            ("A", "X"),
            ("A", "S"),
        ],
        node_props={
            "X": {"color": "red"},
            "Y": {"color": "red"}
        },
        edge_props={
            ("X", "Y"): {"color": "red"},
            ("A", "X"): {"color": "none"},
            ("S", "X"): {"color": "none"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### From perspective of S
    """)
    return


@app.cell(hide_code=True)
def table2_s_perspective_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("S", "Y"),
            ("S", "X"),
            ("A", "Y"),
            ("A", "X"),
            ("A", "S"),
        ],
        node_props={
            "Y": {"color": "red"},
            "S": {"color": "red"}
        },
        edge_props={
            ("X", "Y"): {"color": "red"},
            ("S", "X"): {"color": "red"},
            ("S", "Y"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Adjusted graph with the full model
    """)
    return


@app.cell(hide_code=True)
def table2_s_adjusted_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("S", "Y"),
            ("S", "X"),
            ("A", "Y"),
            ("A", "X"),
            ("A", "S"),
        ],
        node_props={
            "Y": {"color": "red"},
            "S": {"color": "red"}
        },
        edge_props={
            ("S", "X"): {"color": "none"},
            ("A", "S"): {"color": "none"},
            ("S", "Y"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - In the unconditional model, the effect of $S$ on $Y$ is confounded by $A$, because it's a common cause of $S$ and $Y$ (and $X$)
    - Conditioning on $A$ & $X$ (via the same statistical model above), $\beta_{S}$ gives the **direct effect** of $S$ on $Y$.
      - Since we've blocked the path along $X$ in the linear regression, we no longer get the **total effect**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### From the perspective of A

    In the unconditional model, the **total causal effect** of $A$ on $Y$ flows through all paths:
    """)
    return


@app.cell(hide_code=True)
def table2_a_perspective_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("S", "Y"),
            ("S", "X"),
            ("A", "Y"),
            ("A", "X"),
            ("A", "S"),
        ],
        node_props={
            "Y": {"color": "red"},
            "A": {"color": "red"}
        },
        edge_props={
            ("X", "Y"): {"color": "red"},
            ("A", "X"): {"color": "red"},
            ("A", "Y"): {"color": "red"},
            ("A", "S"): {"color": "red"},
            ("S", "X"): {"color": "red"},
            ("S", "Y"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Conditioning on $S$ & $X$ (via the same statistical model above), $\beta_{A}$ gives the **direct effect** of $A$ on $Y$.
      - Since we've blocked the paths through $X$ and $S$ in the linear regression, we no longer get the **total effect**.
    """)
    return


@app.cell(hide_code=True)
def table2_a_adjusted_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("X", "Y"),
            ("S", "Y"),
            ("S", "X"),
            ("A", "Y"),
            ("A", "X"),
            ("A", "S"),
        ],
        node_props={
            "Y": {"color": "red"},
            "A": {"color": "red"}
        },
        edge_props={
            ("A", "X"): {"color": "none"},
            ("A", "S"): {"color": "none"},
            ("A", "Y"): {"color": "red"},
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This gets trickier if we consider unobserved confounds on variables!

    ## Summary: Table 2 Fallacy
    - Not all coefficients have the same interpretation
      - different estimands require different models
    - Do not present coefficients as if they are equal (i.e. in Table 2)
    - ...or, don't present coefficients at all, instead push out predictions.
    - Provide _explicit interpretation_ of each
    """)
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
