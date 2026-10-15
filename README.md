# Statistical Rethinking 2023 in Python

Python/PyMC 6 implementations of Richard McElreath's outstanding lecture series [Statistical Rethinking 2023](https://www.youtube.com/playlist?list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus).

## Setup

The environment is managed with [pixi](https://pixi.sh) via `pixi.toml` / `pixi.lock`, with locked dependencies for Windows, Linux and macOS (Intel and Apple Silicon):

```bash
pixi install    # create the environment from pixi.lock
pixi run lab    # launch JupyterLab with the lecture notebooks
pixi run utils  # open utils.py as a marimo notebook with demos of each helper
```

In VS Code, select the `.pixi/envs/default` Python interpreter as the notebook kernel.

Models are sampled with [nutpie](https://github.com/pymc-devs/nutpie), which `pm.sample()` uses automatically when every variable can be sampled with NUTS. PyMC doesn't print divergence, R-hat or ESS warnings after nutpie runs, so use `utils.check_sampling(inference)` to check a fit. Intervals default to the 89% HDI (set in `init_notebook.py`).

The notebooks were last executed with these versions (see [CHANGELOG.md](CHANGELOG.md) for the migration from PyMC 5):

| Package | Version |
| --- | --- |
| Python | 3.14.8 |
| pymc | 6.3.2 |
| pytensor | 3.3.3 |
| arviz | 1.3.0 |
| nutpie | 0.16.11 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| pandas | 3.0.6 |
| xarray | 2026.9.0 |
| statsmodels | 0.15.0 |
| matplotlib | 3.11.2 |
| marimo | 0.25.1 |
