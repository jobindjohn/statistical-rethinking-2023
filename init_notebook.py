"""Shared setup for the lecture notebooks.

marimo notebooks import this module in their setup block:

    with app.setup:
        import init_notebook
        from init_notebook import cached_sample

        init_notebook.setup()

Jupyter notebooks can still `%run init_notebook.py`. That runs the
`__main__` block at the bottom: the common imports land in the notebook
namespace, and the aliases and watermark are printed.
"""

import hashlib
import pickle
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor
from matplotlib import style
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pytensor.graph.traversal import ancestors

HERE = Path(__file__).parent
STYLE = HERE / "statistical-rethinking-2023.mplstyle"

# Cached posterior samples live with marimo's own cache (gitignored)
SAMPLE_CACHE = HERE / "__marimo__" / "cache" / "samples"

# Packages reported by `watermark_report`
WATERMARK_PACKAGES = (
    "pymc,nutpie,arviz,numpy,pandas,scipy,xarray,statsmodels,matplotlib,marimo"
)


def setup():
    """Apply the plotting style, interval settings and warning filters."""
    warnings.filterwarnings("ignore")
    style.use(STYLE)

    # Report 89% highest density intervals, following the lectures
    az.rcParams["stats.ci_kind"] = "hdi"
    az.rcParams["stats.ci_prob"] = 0.89


def watermark_report() -> str:
    """System information and package versions, for a notebook footer."""
    from watermark import watermark

    return watermark() + "\n" + watermark(packages=WATERMARK_PACKAGES)


def _update_hash(h, value):
    """Feed one graph input (data, constant or setting) into the hash."""
    if isinstance(value, np.ndarray) and value.dtype != object:
        h.update(f"{value.dtype}{value.shape}".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    else:
        h.update(repr(value).encode())


def model_fingerprint(model=None, **sample_kwargs) -> str:
    """Hash of a model's structure, data and the arguments to `pm.sample`.

    Two calls give the same hash only when the model graph, every constant and
    data value in it, the coords and the sampling arguments are all the same.
    The model's random generator state is left out: it is reseeded on every
    run and does not affect the posterior.
    """
    model = pm.modelcontext(model)
    outputs = [model.logp(), *model.deterministics]

    h = hashlib.sha256()
    h.update(pm.__version__.encode())
    h.update(pytensor.printing.debugprint(outputs, file="str").encode())
    for var in ancestors(outputs):
        if isinstance(var, SharedVariable):
            value = var.get_value()
            if not isinstance(value, np.random.Generator):
                _update_hash(h, value)
        elif isinstance(var, Constant):
            _update_hash(h, var.data)
    _update_hash(h, sorted(model.coords.items()))
    _update_hash(h, sorted(sample_kwargs.items()))
    return h.hexdigest()


def sample_cache_path(model=None, **sample_kwargs) -> Path:
    """Where `cached_sample` stores the samples for this model and arguments."""
    key = model_fingerprint(model, **sample_kwargs)
    return SAMPLE_CACHE / f"{key}.pickle"


def cached_sample(model=None, **sample_kwargs):
    """`pm.sample`, cached on disk.

    Use it like `pm.sample`, inside the model's `with` block. The first call
    samples and saves the result; later calls with the same model, data and
    arguments load it, even after the notebook is restarted. Any change to the
    model or its data samples again. Delete `__marimo__/cache/samples` to clear
    the cache.
    """
    model = pm.modelcontext(model)
    path = sample_cache_path(model, **sample_kwargs)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    with model:
        inference = pm.sample(**sample_kwargs)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(inference, f)
    tmp_path.replace(path)
    return inference


if __name__ == "__main__":
    # `%run init_notebook.py` from a Jupyter notebook
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    import pymc as pm
    import xarray as xr
    import arviz as az
    import utils as utils

    from scipy import stats as stats
    from matplotlib import pyplot as plt

    from colorama import Fore

    print(Fore.RED + f"Module aliases imported by init_notebook.py:\n{'-' * 44}")
    for alias, module in [
        ("np", "numpy"),
        ("pd", "pandas"),
        ("smf", "statsmodels.formula.api"),
        ("pm", "pymc"),
        ("xr", "xarray"),
        ("az", "arviz"),
        ("utils", "utils"),
    ]:
        print(Fore.GREEN + "import" + Fore.BLUE + f" {module} " + Fore.GREEN + "as" + Fore.BLUE + f" {alias}")
    for module, name, alias in [("scipy", "stats", "stats"), ("matplotlib", "pyplot", "plt")]:
        print(
            Fore.GREEN + "from" + Fore.BLUE + f" {module} "
            + Fore.GREEN + "import" + Fore.BLUE + f" {name} "
            + Fore.GREEN + "as" + Fore.BLUE + f" {alias}"
        )

    print(Fore.RED + f"Watermark:\n{'-' * 10}")
    print(Fore.BLUE + watermark_report())

    setup()
