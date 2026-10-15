"""Shared helpers for the lecture notebooks.

This file is both a plain Python module (`import utils`) and a marimo notebook
that demonstrates every helper. Open the notebook with `pixi run utils`.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import os
    import pandas as pd
    import numpy as np
    import pymc as pm
    import arviz as az
    import xarray as xr
    import graphviz as gr
    import networkx as nx
    from arviz_stats.base import array_stats
    from matplotlib import pyplot as plt
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D
    from pathlib import Path
    from scipy import stats
    from scipy.signal import savgol_filter
    from scipy.special import logsumexp
    from typing import List, Union, Callable

    HERE = Path(".")

    # Dimensions that index posterior draws, as opposed to model dimensions
    SAMPLE_DIMS = ("chain", "draw", "sample")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `utils.py`: helpers for the lecture notebooks

    The lecture notebooks load this file with `import utils as utils` (via
    `init_notebook.py`). It is also a [marimo](https://marimo.io) notebook: each
    helper is followed by a short demo showing what it draws and why it exists.

    - **Run the demos:** `pixi run utils` (or `marimo edit utils.py`).
    - **Importing is unaffected:** the demo cells only run inside marimo, never on
      `import utils`.

    **Editing rule.** Every helper is a marimo *top-level function*
    (`@app.function`). That only works while the function uses its own arguments,
    the imports and constants in the setup block at the top, and other helpers.
    If a helper refers to a variable from a demo cell, marimo silently turns it
    into an ordinary cell and `utils.<helper>` disappears from the module. Don't
    start helper names with `_`: marimo treats those as private to their cell,
    so other helpers can't call them inside the notebook.
    """)
    return


@app.cell
def _():
    # Demo settings, matching init_notebook.py. marimo runs cells that don't depend
    # on each other in notebook order, so this runs before the demos below.
    from matplotlib import style

    style.use("statistical-rethinking-2023.mplstyle")
    az.rcParams["stats.ci_kind"] = "hdi"
    az.rcParams["stats.ci_prob"] = 0.89
    rng = np.random.default_rng(12)
    return (rng,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    `load_data` reads a CSV from the `data/` folder by name. The datasets from
    the book are `;`-separated.
    """)
    return


@app.function
def load_data(dataset, delimiter=";"):
    fname = f"{dataset}.csv"
    data_path = HERE / "data"
    data_file = data_path / fname
    return pd.read_csv(data_file, sep=delimiter)


@app.cell
def _():
    howell = load_data("Howell1")
    howell.head()
    return (howell,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `crosstab` counts how often each pair of values of two discrete variables
    occurs. Lecture 05 uses it to show that two binary variables are associated.
    """)
    return


@app.function
def crosstab(x: np.array, y: np.array, labels: list[str] = None):
    """Simple cross tabulation of two discrete vectors x and y"""
    ct = pd.crosstab(x, y)
    if labels:
        ct.index = labels
        ct.columns = labels
    return ct


@app.cell
def _(rng):
    _x = rng.binomial(1, 0.5, 200)
    _y = rng.binomial(1, np.where(_x == 1, 0.8, 0.2))  # Y depends on X
    crosstab(_x, _y, labels=["no", "yes"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transforming variables

    `center` subtracts the mean; `standardize` also divides by the standard
    deviation, so the result has mean 0 and standard deviation 1. Standardized
    predictors make default priors such as `Normal(0, 1)` sensible. Both ignore
    missing values.
    """)
    return


@app.function
def center(vals: np.ndarray) -> np.ndarray:
    return vals - np.nanmean(vals)


@app.function
def standardize(vals: np.ndarray) -> np.ndarray:
    centered_vals = center(vals)
    return centered_vals / np.nanstd(centered_vals)


@app.cell
def _(howell):
    _weight = howell.weight.values
    pd.DataFrame(
        {
            "raw": [_weight.mean(), _weight.std()],
            "center": [center(_weight).mean(), center(_weight).std()],
            "standardize": [standardize(_weight).mean(), standardize(_weight).std()],
        },
        index=["mean", "std"],
    ).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `convert_to_categorical` turns labels into integer codes (in sorted label
    order), ready to index a vector of parameters such as `alpha[code]`.
    """)
    return


@app.function
def convert_to_categorical(vals):
    return vals.astype("category").cat.codes.values


@app.cell
def _():
    _labels = pd.Series(["urban", "rural", "rural", "urban", "rural"])
    pd.DataFrame({"label": _labels, "code": convert_to_categorical(_labels)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `logit` maps a probability to log-odds, and `invlogit` maps log-odds back to
    a probability. Generalized linear models put the linear model on the
    log-odds scale and use `invlogit` to get probabilities.
    """)
    return


@app.function
def logit(p: float) -> float:
    return np.log(p / (1 - p))


@app.function
def invlogit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


@app.cell
def _():
    _log_odds = np.linspace(-6, 6, 100)
    _fig, _ax = plt.subplots(figsize=(5, 3))
    plot_line(_log_odds, invlogit(_log_odds), label="invlogit")
    _ax.set(xlabel="log-odds", ylabel="probability")
    print("logit(invlogit(x)) == x:", np.allclose(logit(invlogit(_log_odds)), _log_odds))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Causal graphs and networks

    `draw_causal_graph` draws a DAG with graphviz from a list of edges. Node and
    edge properties use graphviz attributes; the lectures draw unobserved
    variables dashed.
    """)
    return


@app.function
def draw_causal_graph(
    edge_list, node_props=None, edge_props=None, graph_direction="UD"
):
    """Utility to draw a causal (directed) graph"""
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props[e] if e in edge_props else {}
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


@app.cell
def _():
    draw_causal_graph(
        edge_list=[("X", "Y"), ("Z", "X"), ("Z", "Y"), ("U", "X"), ("U", "Y")],
        node_props={"U": {"style": "dashed"}},
        edge_props={("X", "Y"): {"color": "red", "label": "effect of interest"}},
        graph_direction="LR",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_graph` draws a network with networkx, from a `networkx.DiGraph` or an
    adjacency matrix. It returns the node positions, so later plots of the same
    network (Lecture 15) can reuse the layout via `pos=`.
    """)
    return


@app.function
def plot_graph(graph, **graph_kwargs):
    """Draw a network graph.

    graph: Union[networkx.DiGraph, np.ndarray]
        if ndarray, assume `graph` is an adjacency matrix defining
        a directed graph.

    """
    # convert to networkx.DiGraph, if needed
    G = (
        nx.from_numpy_array(graph, create_using=nx.DiGraph)
        if isinstance(graph, np.ndarray)
        else graph
    )

    # Set default styling
    np.random.seed(123)  # for consistent spring-layout
    if "layout" in graph_kwargs:
        graph_kwargs["pos"] = graph_kwargs["layout"](G)

    default_graph_kwargs = {
        "node_color": "C0",
        "node_size": 500,
        "arrowsize": 30,
        "width": 3,
        "alpha": 0.7,
        "connectionstyle": "arc3,rad=0.1",
        "pos": nx.kamada_kawai_layout(G),
    }
    for k, v in default_graph_kwargs.items():
        if k not in graph_kwargs:
            graph_kwargs[k] = v

    nx.draw(G, **graph_kwargs)
    # return the node layout for consistent graphing
    return graph_kwargs["pos"]


@app.cell
def _(rng):
    _ties = rng.binomial(1, 0.15, size=(12, 12))
    np.fill_diagonal(_ties, 0)
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(_axs[0])
    _layout = plot_graph(_ties)
    _axs[0].set_title("default style")
    plt.sca(_axs[1])
    plot_graph(_ties, pos=_layout, node_color="C1", edge_color="gray")
    _axs[1].set_title("same layout, custom colors")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matplotlib styling helpers

    `plot_scatter` draws open circles, and `plot_line` draws a line with a white
    border so it stands out on top of data or bands. Both draw on the current
    axes, like `plt.scatter` and `plt.plot`.
    """)
    return


@app.function
def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)"""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none"}

    for k, v in defaults.items():
        val = scatter_kwargs.get(k, v)
        scatter_kwargs[k] = val

    plt.scatter(xs, ys, **scatter_kwargs)


@app.function
def plot_line(xs, ys, **plot_kwargs):
    """Plot line with consistent style (e.g. bordered lines)"""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    # Copy settings for background
    background_plot_kwargs = {k: v for k, v in plot_kwargs.items()}
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    background_plot_kwargs.pop("label", None)  # no legend label for background

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)


@app.cell
def _(howell):
    _adults = howell[howell.age >= 18]
    _fit = np.polyfit(_adults.weight, _adults.height, 1)
    _xs = np.linspace(_adults.weight.min(), _adults.weight.max(), 10)
    _fig, _ax = plt.subplots(figsize=(6, 4))
    plot_scatter(_adults.weight, _adults.height, label="adults")
    plot_line(_xs, np.polyval(_fit, _xs), color="k", label="least squares")
    _ax.set(xlabel="weight (kg)", ylabel="height (cm)")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_errorbar` and `plot_x_errorbar` draw thick, translucent vertical or
    horizontal error bars, e.g. a point estimate with its interval, or a
    measurement with its standard error (Lecture 17). The bars extend
    `error_lower` below and `error_upper` above each point.
    """)
    return


@app.function
def plot_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    """Draw thick error bars with consistent style"""
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            yerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


@app.function
def plot_x_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    """Draw thick horizontal error bars with consistent style"""
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            xerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


@app.cell
def _(rng):
    _groups = np.arange(4)
    _means = rng.normal(0, 1, 4)
    _errors = rng.uniform(0.3, 1, 4)
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4))
    plt.sca(_axs[0])
    plot_errorbar(_groups, _means, _errors, _errors, colors=["C0", "C1", "C2", "C3"])
    plot_scatter(_groups, _means, color="k")
    _axs[0].set(title="plot_errorbar", xlabel="group", ylabel="estimate")
    plt.sca(_axs[1])
    plot_x_errorbar(_means, _groups, _errors, _errors, error_width=7)
    plot_scatter(_means, _groups, color="k")
    _axs[1].set(title="plot_x_errorbar", xlabel="estimate", ylabel="group")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_2d_function` evaluates `func(x, y)` on a grid and draws its contours,
    e.g. a bivariate density or a 2D prior (Lectures 09 and 14).
    """)
    return


@app.function
def plot_2d_function(xrange, yrange, func, ax=None, **countour_kwargs):
    """Evaluate the function `func` over the values of xrange and yrange and
    plot the resulting value contour over that range.

    Parameters
    ----------
    xrange : np.ndarray
        The horizontal values to evaluate/plot
    yrange : p.ndarray
        The horizontal values to evaluate/plot
    func : Callable
        function of two arguments, xs and ys. Should return a single value at
        each point.
    ax : matplotlib.Axis, optional
        An optional axis to plot the function, by default None

    Returns
    -------
    contour : matplotlib.contour.QuadContourSet
    """
    resolution = len(xrange)
    xs, ys = np.meshgrid(xrange, yrange)
    xs = xs.ravel()
    ys = ys.ravel()

    value = func(xs, ys)

    if ax is not None:
        plt.sca(ax)

    return plt.contour(
        xs.reshape(resolution, resolution),
        ys.reshape(resolution, resolution),
        value.reshape(resolution, resolution),
        **countour_kwargs,
    )


@app.cell
def _():
    _mvn = stats.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]])
    _grid = np.linspace(-3, 3, 100)
    _fig, _ax = plt.subplots(figsize=(5, 5))
    plot_2d_function(_grid, _grid, lambda x, y: _mvn.pdf(np.column_stack([x, y])), ax=_ax, cmap="Reds")
    _ax.set(xlabel="x", ylabel="y", title="correlated bivariate normal")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `create_variables_dataframe` collects equal-length arrays into a
    `DataFrame`, one column per keyword argument.
    """)
    return


@app.function
def create_variables_dataframe(**variables: np.ndarray) -> pd.DataFrame:
    """Combine equal-length arrays into a DataFrame, one column per keyword"""
    return pd.DataFrame({name: np.asarray(vals) for name, vals in variables.items()})


@app.cell
def _(rng):
    _x = rng.normal(size=5)
    create_variables_dataframe(x=_x, y=2 * _x + rng.normal(size=5)).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting posteriors on existing axes

    ArviZ 1 rebuilt plotting around `PlotCollection`: every `az.plot_*`
    function creates its **own** figure and has no `ax=` argument. The lectures
    often draw a posterior into a panel they have already set up, next to other
    curves, data or reference lines, which ArviZ 1 cannot do. `az.plot_hdi`
    is also gone. The helpers below fill these gaps with matplotlib and
    ArviZ's statistics functions, and keep the old call signatures.

    The demos use one model, fitted with nutpie: adult height regressed on
    weight, with a separate intercept for each sex (Lecture 03/04).
    """)
    return


@app.cell
def _(howell, rng):
    adults = howell[howell.age >= 18].reset_index(drop=True)
    weight_mean = adults.weight.mean()

    with pm.Model(coords={"sex": ["female", "male"]}) as height_model:
        _weight = pm.Data("weight", adults.weight.values - weight_mean)
        _alpha = pm.Normal("alpha", 160, 20, dims="sex")
        _beta = pm.LogNormal("beta", 0, 1)
        _sigma = pm.Uniform("sigma", 0, 50)
        _mu = _alpha[adults.male.values] + _beta * _weight
        pm.Normal("height", _mu, _sigma, observed=adults.height.values)

        idata = pm.sample(
            nuts_sampler="nutpie", progressbar=False, random_seed=rng.integers(2**31)
        )
        pm.compute_log_likelihood(idata)

    az.summary(idata, var_names=["alpha", "beta", "sigma"])
    return adults, idata, weight_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For comparison, native ArviZ 1 draws each variable in a figure of its own.
    That is the right tool for a standalone look at a posterior:
    """)
    return


@app.cell
def _(idata):
    az.plot_dist(idata, var_names=["alpha", "beta"])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `plot_dist`

    Draws the density of samples onto an axes of your choice (the current axes
    by default): a KDE for continuous samples, a histogram for integer samples,
    and filled 2D KDE contours when given two variables. Chains and draws are
    pooled. Here it overlays both posterior intercepts and a reference line on
    one panel, which a native ArviZ 1 plot can't do.
    """)
    return


@app.function
def default_ci_prob(hdi_prob):
    """`hdi_prob`, or the global interval probability from `az.rcParams` if None"""
    return az.rcParams["stats.ci_prob"] if hdi_prob is None else hdi_prob


@app.function
def flatten_samples(values) -> np.ndarray:
    """Flatten array-like / DataArray draws to a 1D array without NaNs"""
    values = np.asarray(values).ravel()
    return values[~np.isnan(values)] if values.dtype.kind == "f" else values


@app.function
def plot_dist(
    values,
    values2=None,
    ax=None,
    color=None,
    label=None,
    bw=None,
    plot_kwargs=None,
    contourf_kwargs=None,
):
    """Plot the distribution of samples on an existing axes; replaces the
    ArviZ 0.x `az.plot_dist`.

    Parameters
    ----------
    values : array-like or xarray.DataArray
        Samples to plot; all dimensions are flattened. Integer samples are
        plotted as a histogram, continuous samples as a KDE.
    values2 : array-like or xarray.DataArray, optional
        If provided, plot the 2D KDE of `values` vs `values2` as filled contours.
    ax : matplotlib.Axes, optional
        Axes to draw on, by default the current axes.
    color, label : str, optional
        Line color and legend label. A legend is added when `label` is given.
    bw : float or str, optional
        KDE bandwidth; a number is an absolute bandwidth in data units.
    plot_kwargs : dict, optional
        Extra `Line2D` kwargs (aliases like `lw` are accepted).
    contourf_kwargs : dict, optional
        Extra `contourf` kwargs for the 2D case.

    Returns
    -------
    ax : matplotlib.Axes
    """
    ax = plt.gca() if ax is None else ax
    values = flatten_samples(values)

    if values2 is not None:
        values2 = flatten_samples(values2)
        kde = stats.gaussian_kde(np.vstack([values, values2]))
        xs, ys = np.meshgrid(
            np.linspace(values.min(), values.max(), 128),
            np.linspace(values2.min(), values2.max(), 128),
        )
        density = kde(np.vstack([xs.ravel(), ys.ravel()])).reshape(xs.shape)
        ax.contourf(xs, ys, density, **(contourf_kwargs or {}))
        ax.contour(xs, ys, density, colors="k", linewidths=0.5, alpha=0.5)
        return ax

    line_kwargs = normalize_kwargs(plot_kwargs or {}, Line2D)
    if color is not None:
        line_kwargs["color"] = color

    if values.dtype.kind in "iub":
        bins = np.arange(values.min(), values.max() + 2) - 0.5
        ax.hist(values, bins=bins, density=True, label=label, **line_kwargs)
    else:
        kde_kwargs = {} if bw is None else {"bw": bw}
        grid, density, _ = array_stats.kde(values, **kde_kwargs)
        ax.plot(grid, density, label=label, **line_kwargs)

    if label is not None:
        ax.legend()
    return ax


@app.cell
def _(idata, rng):
    _alpha = idata["posterior"]["alpha"]
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 4))

    plot_dist(_alpha.sel(sex="female"), ax=_axs[0], color="C1", label="female", plot_kwargs={"lw": 3})
    plot_dist(_alpha.sel(sex="male"), ax=_axs[0], color="C0", label="male", plot_kwargs={"lw": 3})
    _axs[0].axvline(_alpha.mean(), color="k", linestyle="--")
    _axs[0].set(xlabel="alpha (cm at mean weight)", title="overlay, plus a reference line")

    plot_dist(_alpha.sel(sex="male"), idata["posterior"]["beta"], ax=_axs[1], contourf_kwargs={"cmap": "Reds"})
    _axs[1].set(xlabel="alpha[male]", ylabel="beta", title="two variables: 2D KDE")

    plot_dist(rng.poisson(4, 4000), ax=_axs[2], color="C2")
    _axs[2].set(xlabel="count", title="integer samples: histogram")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `plot_pymc_distribution`

    Draws samples from a PyMC distribution with the given parameters and plots
    them with `plot_dist`, a quick way to see what a prior looks like.
    """)
    return


@app.function
def plot_pymc_distribution(distribution: pm.Distribution, **distribution_params):
    """Plot a PyMC Distribution with specific distrubution parameters

    Parameters
    ----------
    distribution : pymc.Distribution
        The class of distribution to
    **distribution_params : dict
        Distribution-specific parameters.

    Returns
    -------
    ax : matplotlib.Axes
        The axes object associated with the plot.
    """
    with pm.Model() as _:
        d = distribution(name=distribution.__name__, **distribution_params)
        draws = pm.draw(d, draws=10_000)
    return plot_dist(draws)


@app.cell
def _():
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 3.5))
    for _ax, (_dist, _params) in zip(
        _axs,
        [
            (pm.LogNormal, {"mu": 0, "sigma": 1}),
            (pm.Exponential, {"lam": 1}),
            (pm.Poisson, {"mu": 3}),
        ],
    ):
        plt.sca(_ax)
        plot_pymc_distribution(_dist, **_params)
        _ax.set_title(f"{_dist.__name__}({', '.join(f'{k}={v}' for k, v in _params.items())})")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `plot_hdi`

    ArviZ 1 removed `az.plot_hdi`. This helper fills the HDI band of draws `y`
    at each value of `x` on an existing axes; by default the interval is the 89%
    HDI set in `az.rcParams`. Like ArviZ 0.x, it smooths the band by default.

    Here the bands show the posterior of the mean height for each sex.
    """)
    return


@app.function
def plot_hdi(x, y, hdi_prob=None, ax=None, color="C1", smooth=True, fill_kwargs=None):
    """Plot the HDI band of draws `y` over `x` on an existing axes; replaces the
    ArviZ 0.x `az.plot_hdi`.

    Parameters
    ----------
    x : array-like, shape (n,)
        Horizontal values.
    y : array-like or xarray.DataArray
        Draws for each `x`. Arrays have shape (n_samples, n) or (chain, draw, n);
        DataArrays reduce over their chain/draw/sample dims.
    hdi_prob : float, optional
        Interval probability, by default `az.rcParams["stats.ci_prob"]`.
    ax : matplotlib.Axes, optional
        Axes to draw on, by default the current axes.
    color : str
        Band color.
    smooth : bool
        Interpolate and Savitzky-Golay smooth the band, as ArviZ 0.x did.
    fill_kwargs : dict, optional
        Extra `fill_between` kwargs, e.g. `alpha` or `label`.

    Returns
    -------
    ax : matplotlib.Axes
    """
    ax = plt.gca() if ax is None else ax
    x = np.asarray(x, dtype=float).ravel()

    if isinstance(y, xr.DataArray):
        sample_dims = [d for d in y.dims if d in SAMPLE_DIMS]
        other_dims = [d for d in y.dims if d not in SAMPLE_DIMS]
        y = y.transpose(*sample_dims, *other_dims).values
    y = np.asarray(y, dtype=float).reshape(-1, x.size)

    hdi = array_stats.hdi(y, default_ci_prob(hdi_prob), axis=0)

    order = np.argsort(x)
    x, hdi = x[order], hdi[order]
    if smooth:
        x_smooth = np.linspace(x.min(), x.max(), 200)
        hdi = np.column_stack([np.interp(x_smooth, x, hdi[:, ii]) for ii in range(2)])
        hdi = savgol_filter(hdi, window_length=55, polyorder=2, axis=0)
        x = x_smooth

    kwargs = {"color": color, "alpha": 0.5, "linewidth": 0}
    kwargs.update(fill_kwargs or {})
    ax.fill_between(x, hdi[:, 0], hdi[:, 1], **kwargs)
    return ax


@app.cell
def _(adults, idata, weight_mean):
    _post = idata["posterior"]
    _weights = xr.DataArray(np.linspace(30, 63, 30), dims="weight")
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    plt.sca(_axs[0])
    for _sex, _color in [("female", "C1"), ("male", "C0")]:
        _is_sex = adults.male == (_sex == "male")
        _mu = _post["alpha"].sel(sex=_sex) + _post["beta"] * (_weights - weight_mean)
        plot_scatter(adults.weight[_is_sex], adults.height[_is_sex], color=_color, alpha=0.3)
        plot_hdi(_weights, _mu, color=_color, fill_kwargs={"label": f"{_sex}, 89% HDI"})
        plot_line(_weights, _mu.mean(("chain", "draw")), color=_color)
    _axs[0].set(xlabel="weight (kg)", ylabel="height (cm)", title="DataArray draws, smoothed")
    _axs[0].legend()

    _mu = (_post["alpha"].sel(sex="male") + _post["beta"] * (_weights - weight_mean)).values
    plot_hdi(_weights, _mu, ax=_axs[1], color="k", fill_kwargs={"alpha": 0.2, "label": "89%"})
    plot_hdi(_weights, _mu, hdi_prob=0.5, ax=_axs[1], color="C0", smooth=False, fill_kwargs={"label": "50%, unsmoothed"})
    _axs[1].set(xlabel="weight (kg)", title="numpy (chain, draw, n) draws, male")
    _axs[1].legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `plot_forest`

    For a standalone figure, use the native `az.plot_forest`:
    """)
    return


@app.cell
def _(idata):
    az.plot_forest(idata, var_names="alpha", combined=True)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_forest` draws the same summary into an existing axes, e.g. one panel
    of a subplot grid next to a related plot. Each row shows the posterior mean
    (open circle), interquartile range (thick line) and HDI (thin line), with
    chains combined.
    """)
    return


@app.function
def plot_forest(data, var_names=None, hdi_prob=None, ax=None, color="C0"):
    """Forest plot of posterior variables (chains combined) on an existing axes.

    Each row shows the posterior mean (marker), interquartile range (thick
    line), and HDI (thin line). For standalone figures use `az.plot_forest`.

    Parameters
    ----------
    data : xarray.DataTree
        Inference data with a `posterior` group.
    var_names : str or list[str], optional
        Variables to plot, by default all posterior variables.
    hdi_prob : float, optional
        Interval probability, by default `az.rcParams["stats.ci_prob"]`.
    ax : matplotlib.Axes, optional
        Axes to draw on, by default the current axes.
    color : str
        Color for all rows.

    Returns
    -------
    ax : matplotlib.Axes
    """
    ax = plt.gca() if ax is None else ax
    hdi_prob = default_ci_prob(hdi_prob)
    posterior = data["posterior"].to_dataset()
    if var_names is None:
        var_names = list(posterior.data_vars)
    elif isinstance(var_names, str):
        var_names = [var_names]

    labels, rows = [], []
    for var in var_names:
        samples = posterior[var].stack(__sample__=("chain", "draw"))
        element_dims = [d for d in samples.dims if d != "__sample__"]
        if not element_dims:
            labels.append(var)
            rows.append(samples.values)
            continue
        elements = samples.stack(__element__=element_dims).transpose("__element__", ...)
        for ii in range(elements.sizes["__element__"]):
            element = elements.isel(__element__=ii)
            coords = ", ".join(str(element[d].values) for d in element_dims)
            labels.append(f"{var}[{coords}]")
            rows.append(element.values)

    rows = np.asarray(rows)
    ys = np.arange(len(rows))[::-1]
    hdi = array_stats.hdi(rows, hdi_prob, axis=-1)
    quartiles = np.quantile(rows, [0.25, 0.75], axis=-1)

    ax.hlines(ys, hdi[:, 0], hdi[:, 1], color=color, linewidth=2)
    ax.hlines(ys, quartiles[0], quartiles[1], color=color, linewidth=5)
    ax.scatter(rows.mean(axis=-1), ys, color="white", edgecolor=color, zorder=3)
    ax.set_yticks(ys, labels)
    ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)
    ax.set_title(f"{hdi_prob:.0%} HDI")
    return ax


@app.cell
def _(idata):
    _post = idata["posterior"]
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 3.5))
    plot_forest(idata, var_names="alpha", ax=_axs[0])
    _axs[0].set_xlabel("alpha (cm at mean weight)")

    _contrast = (_post["alpha"].sel(sex="male") - _post["alpha"].sel(sex="female")).values.ravel()
    plot_dist(_contrast, ax=_axs[1], color="k")
    _axs[1].axvline(0, color="gray", linestyle="--")
    _axs[1].set(xlabel="alpha[male] - alpha[female] (cm)", title="contrast")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling diagnostics: `check_sampling`

    PyMC 6 samples with nutpie by default when it is installed. After a nutpie
    run, PyMC no longer prints its warnings about divergences, R-hat and
    effective sample size (ESS), so a problematic fit looks just like a good
    one. `check_sampling` runs the same checks with PyMC's thresholds and
    prints what it finds. The lectures call it where they discuss
    diagnostics.

    The demo checks the height model, then the devil's funnel (Lecture 14),
    a prior that is hard for HMC to sample.
    """)
    return


@app.function
def check_sampling(data, var_names=None, max_rhat=1.01, min_ess_per_chain=100):
    """Print warnings about sampling problems: divergences, R-hat and ESS.

    Uses PyMC's checks and thresholds, which PyMC itself skips after sampling
    with nutpie.

    Parameters
    ----------
    data : xarray.DataTree
        Inference data with `posterior` and `sample_stats` groups.
    var_names : str or list[str], optional
        Variables to check, by default all posterior variables.
    max_rhat : float
        Warn about variables with R-hat above this.
    min_ess_per_chain : int
        Warn about variables with bulk or tail ESS below this many draws per chain.
    """

    def _listed(labels, limit=5):
        labels = [str(label) for label in labels]
        return ", ".join(labels[:limit]) + (", ..." if len(labels) > limit else "")

    problems = []
    sample_stats = data["sample_stats"]
    if "diverging" in sample_stats:
        n_divergences = int(sample_stats["diverging"].sum())
        if n_divergences:
            problems.append(
                f"There were {n_divergences} divergences after tuning. "
                "Increase `target_accept` or reparameterize."
            )

    diagnostics = az.summary(data, var_names=var_names, kind="diagnostics", round_to="none")
    high_rhat = diagnostics.index[diagnostics["r_hat"] > max_rhat]
    if len(high_rhat):
        problems.append(f"R-hat is larger than {max_rhat} for {_listed(high_rhat)}.")

    n_chains = data["posterior"].sizes["chain"]
    min_ess = diagnostics[["ess_bulk", "ess_tail"]].min(axis=1)
    low_ess = diagnostics.index[min_ess < min_ess_per_chain * n_chains]
    if len(low_ess):
        problems.append(
            f"The effective sample size per chain is smaller than {min_ess_per_chain} "
            f"for {_listed(low_ess)}."
        )

    for problem in problems:
        print(f"WARNING: {problem}")
    if not problems:
        print(
            f"No sampling problems: no divergences, R-hat <= {max_rhat}, "
            f"ESS >= {min_ess_per_chain} per chain."
        )


@app.cell
def _(idata):
    with pm.Model():
        _v = pm.Normal("v", 0, 3)
        pm.Normal("x", 0, pm.math.exp(_v))
        _funnel_idata = pm.sample(progressbar=False, random_seed=1)

    print("Height model:")
    check_sampling(idata)
    print("\nDevil's funnel:")
    check_sampling(_funnel_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model comparison: `waic`

    ArviZ 1 dropped WAIC in favour of PSIS-LOO (`az.loo`). Lecture 07 compares
    the two, so `waic` computes WAIC on the same ELPD scale as `az.loo`. It needs
    a `log_likelihood` group, which PyMC 6 adds with `pm.compute_log_likelihood`.
    """)
    return


@app.function
def waic(data, var_name=None, pointwise=False) -> pd.Series:
    """Widely Applicable Information Criterion, on the log (ELPD) scale.

    ArviZ 1 dropped WAIC in favour of LOO; this keeps WAIC available for
    comparing against PSIS-LOO. Requires a `log_likelihood` group, e.g. from
    `pm.compute_log_likelihood`.

    Parameters
    ----------
    data : xarray.DataTree
        Inference data with a `log_likelihood` group.
    var_name : str, optional
        Observed variable, required if the group has more than one.
    pointwise : bool
        Also return the pointwise values as `waic_i`.

    Returns
    -------
    pd.Series
        `elpd_waic`, `se`, `p_waic`, and with `pointwise=True`, `waic_i`: the
        pointwise ELPD as a DataArray over the observation dims.
    """
    log_likelihood = data["log_likelihood"].to_dataset()
    if var_name is None:
        if len(log_likelihood.data_vars) != 1:
            raise ValueError(
                f"Specify var_name, one of {list(log_likelihood.data_vars)}"
            )
        var_name = next(iter(log_likelihood.data_vars))

    log_lik = log_likelihood[var_name].stack(__sample__=("chain", "draw"))
    n_samples = log_lik.sizes["__sample__"]
    lppd_i = xr.apply_ufunc(
        logsumexp, log_lik, input_core_dims=[["__sample__"]], kwargs={"axis": -1}
    ) - np.log(n_samples)
    p_waic_i = log_lik.var(dim="__sample__", ddof=1)
    waic_i = lppd_i - p_waic_i

    result = {
        "elpd_waic": float(waic_i.sum()),
        "se": float(np.sqrt(waic_i.size * waic_i.var())),
        "p_waic": float(p_waic_i.sum()),
    }
    if pointwise:
        result["waic_i"] = waic_i
    return pd.Series(result)


@app.cell
def _(idata, mo):
    _waic = waic(idata, pointwise=True)
    _loo = az.loo(idata, pointwise=True)

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4))
    _axs[0].scatter(np.asarray(_loo.elpd_i).ravel(), _waic.waic_i.values, s=10)
    _axs[0].axline((0, 0), slope=1, color="k", linestyle="--")
    _axs[0].set(xlabel="PSIS-LOO pointwise ELPD", ylabel="WAIC pointwise ELPD", title="agreement per observation")
    _axs[1].scatter(np.asarray(_loo.pareto_k).ravel(), _waic.waic_i.values, s=10)
    _axs[1].axvline(0.7, color="C1", linestyle="--")
    _axs[1].set(xlabel="Pareto k", ylabel="WAIC pointwise ELPD", title="influential points have high k")

    _table = pd.DataFrame(
        {"ELPD": [_waic.elpd_waic, float(_loo.elpd)], "SE": [_waic.se, float(_loo.se)], "penalty": [_waic.p_waic, float(_loo.p)]},
        index=["utils.waic", "az.loo"],
    ).round(2)
    mo.vstack([_table, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving and showing figures

    `savefig` saves the current figure to `images/` at 300 dpi, and
    `display_image` shows an image from `images/` in the notebook. Lecture 13
    saves two figures that Lecture 14 shows again. The demo saves a figure,
    shows it, and deletes the file.
    """)
    return


@app.function
def savefig(filename):
    """Save a figure to the `./images` directory"""
    image_path = HERE / "images"
    if not image_path.exists():
        print(f"creating image directory: {image_path}")
        os.makedirs(image_path)

    figure_path = image_path / filename
    print(f"saving figure to {figure_path}")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")


@app.function
def display_image(filename, width=600):
    """Display an image saved to the `./images` directory"""
    import marimo as mo

    image_file = HERE / "images" / filename
    if mo.running_in_notebook():  # marimo doesn't render IPython images
        return mo.image(image_file, width=width)

    from IPython.display import Image, display

    return display(Image(filename=str(image_file), width=width))


@app.cell
def _():
    _fig, _ax = plt.subplots(figsize=(4, 3))
    plot_pymc_distribution(pm.Normal, mu=0, sigma=1)
    _ax.set_title("saved, then displayed")
    try:
        savefig("utils-demo.png")
        plt.close(_fig)
        _image = display_image("utils-demo.png", width=300)
    finally:
        os.remove(HERE / "images" / "utils-demo.png")
    _image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian updating on a grid

    `simulate_2_parameter_bayesian_learning` computes a grid posterior for a
    2-parameter model from the first observations, then draws the posterior
    contours (left) and a few sampled trends against the data (right). The
    lectures call it in a loop over sample sizes; move the slider to watch the
    posterior concentrate around the true parameters (black cross).
    """)
    return


@app.function
def simulate_2_parameter_bayesian_learning(
    x_obs,
    y_obs,
    param_a_grid,
    param_b_grid,
    true_param_a,
    true_param_b,
    model_func,
    posterior_func,
    n_posterior_samples=3,
    param_labels=None,
    data_range_x=None,
    data_range_y=None,
):
    """General function for simulating Bayesian learning in a 2-parameter model
    using grid approximation.

    Parameters
    ----------
    x_obs : np.ndarray
        The observed x values
    y_obs : np.ndarray
        The observed y values
    param_a_grid: np.ndarray
        The range of values the first model parameter in the model can take.
        Note: should have same length as param_b_grid.
    param_b_grid: np.ndarray
        The range of values the second model parameter in the model can take.
        Note: should have same length as param_a_grid.
    true_param_a: float
        The true value of the first model parameter, used for visualizing ground
        truth
    true_param_b: float
        The true value of the second model parameter, used for visualizing ground
        truth
    model_func: Callable
        A function `f` of the form `f(x, param_a, param_b)`. Evaluates the model
        given at data points x, given the current state of parameters, `param_a`
        and `param_b`. Returns a scalar output for the `y` associated with input
        `x`.
    posterior_func: Callable
        A function `f` of the form `f(x_obs, y_obs, param_grid_a, param_grid_b)
        that returns the posterior probability given the observed data and the
        range of parameters defined by `param_grid_a` and `param_grid_b`.
    n_posterior_samples: int
        The number of model functions sampled from the 2D posterior
    param_labels: Optional[list[str, str]]
        For visualization, the names of `param_a` and `param_b`, respectively
    data_range_x: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the domain used for model
        evaluation
    data_range_y: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the range used for model
        evaluation.
    """
    param_labels = param_labels if param_labels is not None else ["param_a", "param_b"]
    data_range_x = (x_obs.min(), x_obs.max()) if data_range_x is None else data_range_x
    data_range_y = (y_obs.min(), y_obs.max()) if data_range_y is None else data_range_y

    # NOTE: assume square parameter grid
    resolution = len(param_a_grid)

    param_a_grid, param_b_grid = np.meshgrid(param_a_grid, param_b_grid)
    param_a_grid = param_a_grid.ravel()
    param_b_grid = param_b_grid.ravel()

    posterior = posterior_func(x_obs, y_obs, param_a_grid, param_b_grid)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Posterior over intercept and slope params
    plt.sca(axs[0])
    plt.contour(
        param_a_grid.reshape(resolution, resolution),
        param_b_grid.reshape(resolution, resolution),
        posterior.reshape(resolution, resolution),
        cmap="gray_r",
    )

    # Sample locations in parameter space according to posterior
    sample_idx = np.random.choice(
        np.arange(len(posterior)),
        p=posterior / posterior.sum(),
        size=n_posterior_samples,
    )

    param_a_list = []
    param_b_list = []
    for ii, idx in enumerate(sample_idx):
        param_a = param_a_grid[idx]
        param_b = param_b_grid[idx]
        param_a_list.append(param_a)
        param_b_list.append(param_b)

        # Add sampled parameters to posterior
        plt.scatter(param_a, param_b, s=60, c=f"C{ii}", alpha=0.75, zorder=20)

    # Add the true params to the plot for reference
    plt.scatter(
        true_param_a, true_param_b, color="k", marker="x", s=60, label="true parameters"
    )

    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])

    # Plot the current training data and model trends sampled from posterior
    plt.sca(axs[1])
    plt.scatter(x_obs, y_obs, s=60, c="k", alpha=0.5)

    # Plot the resulting model functions sampled from posterior
    xs = np.linspace(data_range_x[0], data_range_x[1], 100)
    for ii, (param_a, param_b) in enumerate(zip(param_a_list, param_b_list)):
        ys = model_func(xs, param_a, param_b)
        plt.plot(xs, ys, color=f"C{ii}", linewidth=4, alpha=0.5)

    groundtruth_ys = model_func(xs, true_param_a, true_param_b)
    plt.plot(
        xs, groundtruth_ys, color="k", linestyle="--", alpha=0.5, label="true trend"
    )

    plt.xlim([data_range_x[0], data_range_x[1]])
    plt.xlabel("x value")

    plt.ylim([data_range_y[0], data_range_y[1]])
    plt.ylabel("y value")

    plt.title(f"N={len(y_obs)}")
    plt.legend(loc="upper left")


@app.cell
def _(mo):
    n_observations = mo.ui.slider(
        steps=[0, 2, 4, 8, 16, 32, 64], value=4, label="observations", show_value=True
    )
    n_observations
    return (n_observations,)


@app.cell
def _(n_observations):
    # Linear regression with a standard-normal prior on intercept and slope, as in Lecture 03
    def _linear_model(x, intercept, slope):
        return intercept + slope * x

    def _linear_regression_posterior(x_obs, y_obs, intercept_grid, slope_grid):
        log_prior = stats.norm(0, 1).logpdf(intercept_grid) + stats.norm(0, 1).logpdf(slope_grid)
        log_likelihood = stats.norm(
            intercept_grid[:, None] + slope_grid[:, None] * x_obs, 1.0
        ).logpdf(y_obs).sum(axis=1)
        log_posterior = log_prior + log_likelihood
        return np.exp(log_posterior - log_posterior.max())

    _intercept, _slope = -1, 0.5
    _x = np.random.default_rng(123).normal(size=64)
    _y = _intercept + _slope * _x + np.random.default_rng(456).normal(size=64) * 0.25
    _grid = np.linspace(-2, 2, 100)
    _n = n_observations.value

    np.random.seed(3)  # the helper samples trends with np.random
    simulate_2_parameter_bayesian_learning(
        x_obs=_x[:_n],
        y_obs=_y[:_n],
        param_a_grid=_grid,
        param_b_grid=_grid,
        true_param_a=_intercept,
        true_param_b=_slope,
        model_func=_linear_model,
        posterior_func=_linear_regression_posterior,
        param_labels=["intercept", "slope"],
        data_range_x=(-3, 3),
        data_range_y=(-3, 3),
    )
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
