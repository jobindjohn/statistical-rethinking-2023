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


def load_data(dataset, delimiter=";"):
    fname = f"{dataset}.csv"
    data_path = HERE / "data"
    data_file = data_path / fname
    return pd.read_csv(data_file, sep=delimiter)


def crosstab(x: np.array, y: np.array, labels: list[str] = None):
    """Simple cross tabulation of two discrete vectors x and y"""
    ct = pd.crosstab(x, y)
    if labels:
        ct.index = labels
        ct.columns = labels
    return ct


def center(vals: np.ndarray) -> np.ndarray:
    return vals - np.nanmean(vals)


def standardize(vals: np.ndarray) -> np.ndarray:
    centered_vals = center(vals)
    return centered_vals / np.nanstd(centered_vals)


def convert_to_categorical(vals):
    return vals.astype("category").cat.codes.values


def logit(p: float) -> float:
    return np.log(p / (1 - p))


def invlogit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


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


def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)"""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none"}

    for k, v in defaults.items():
        val = scatter_kwargs.get(k, v)
        scatter_kwargs[k] = val

    plt.scatter(xs, ys, **scatter_kwargs)


def plot_line(xs, ys, **plot_kwargs):
    """Plot line with consistent style (e.g. bordered lines)"""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    # Copy settings for background
    background_plot_kwargs = {k: v for k, v in plot_kwargs.items()}
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    del background_plot_kwargs["label"]  # no legend label for background

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)


def plot_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
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


def plot_x_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
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


def create_variables_dataframe(*variables: List[np.ndarray]) -> pd.DataFrame:
    """Converts a list of numpy arrays to a dataframe; infers column names from
    variable names
    """
    column_names = [get_variable_name(v) for v in variables]
    return pd.DataFrame(np.vstack(variables).T, columns=column_names)


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


# ---
# Plotting helpers for drawing onto existing matplotlib axes. ArviZ 1 plots
# create their own figures, so overlays and subplot panels use these instead.

SAMPLE_DIMS = ("chain", "draw", "sample")


def _hdi_prob(hdi_prob):
    return az.rcParams["stats.ci_prob"] if hdi_prob is None else hdi_prob


def _flatten_samples(values) -> np.ndarray:
    """Flatten array-like / DataArray draws to a 1D array without NaNs"""
    values = np.asarray(values).ravel()
    return values[~np.isnan(values)] if values.dtype.kind == "f" else values


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
    values = _flatten_samples(values)

    if values2 is not None:
        values2 = _flatten_samples(values2)
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

    hdi = array_stats.hdi(y, _hdi_prob(hdi_prob), axis=0)

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
    hdi_prob = _hdi_prob(hdi_prob)
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


def savefig(filename):
    """Save a figure to the `./images` directory"""
    image_path = HERE / "images"
    if not image_path.exists():
        print(f"creating image directory: {image_path}")
        os.makedirs(image_path)

    figure_path = image_path / filename
    print(f"saving figure to {figure_path}")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")


def display_image(filename, width=600):
    """Display an image saved to the `./images` directory"""
    from IPython.display import Image, display

    return display(Image(filename=f"images/{filename}", width=width))


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
