# CHANGELOG

## 2026-10-15 - Convert notebooks to PyMC6

The lecture notebooks and `utils.py` now run on the latest PyMC and ArviZ releases.

### Environment

- The project is managed with [pixi](https://pixi.sh). All dependencies come from conda-forge.
- Main versions: Python 3.14, PyMC 6.3, PyTensor 3.3, ArviZ 1.3, nutpie 0.16, numpy 2.5, pandas 3.0, marimo 0.25.
- Pixi tasks:
  - `pixi run lab` opens JupyterLab.
  - `pixi run utils` opens `utils.py` as a marimo notebook.
- Sampling uses [nutpie](https://github.com/pymc-devs/nutpie). `pm.sample()` picks it automatically when every variable can be sampled with NUTS. Otherwise PyMC uses its own samplers.
- Intervals default to the 89% HDI, McElreath's convention. This is set globally in `init_notebook.py`.

### `utils.py`

- `utils.py` is now a [marimo](https://marimo.io) notebook. Each helper has an explanation and a live demo.
  - It still imports as a normal module (`import utils`).
  - To edit it, add new helpers as `@app.function`. Names starting with `_` are private to a cell, so helpers can't use them.
- New helpers that draw on existing matplotlib axes, which ArviZ 1 plots can't do:
  - `plot_dist`: a density curve. It replaces `az.plot_dist`.
  - `plot_hdi`: an HDI band. It replaces the removed `az.plot_hdi`.
  - `plot_forest`: means with HDI error bars. It replaces `az.plot_forest(..., ax=ax)`.
- `waic` computes WAIC from the `log_likelihood` group. ArviZ 1 removed `az.waic`.
- `check_sampling` prints warnings for divergences, R-hat above 1.01, and ESS below 100 per chain.
  - PyMC skips these warnings after sampling with nutpie.
  - Lectures 08 and 14 call it where they discuss diagnostics.
- `display_image` also works in marimo.

### API changes in the notebooks

| Old | New |
| --- | --- |
| `pm.MutableData` | `pm.Data` |
| `testval=` | `initval=` |
| `transforms.univariate_ordered` | `transforms.ordered` |
| `pm.summary`, `pm.plot_forest` | `az.summary`, `az.plot_forest` |
| `az.plot_dist`, `az.plot_hdi` on existing axes | `utils.plot_dist`, `utils.plot_hdi` |
| `az.plot_forest(..., ax=ax)` | `utils.plot_forest(..., ax=ax)` |
| `az.plot_trace(compact=..., legend=...)` | `az.plot_trace_dist(compact=...)` |
| `az.plot_ess(kind='evolution')` | `az.plot_ess_evolution` |
| `az.compare(..., scale='deviance')`, `pm.compare` | `az.compare(...)` on the ELPD scale |
| `az.waic(...)` | `utils.waic(...)` |
| `az.hdi(..., hdi_prob=)` with results indexed `[:, 0]` / `[:, 1]` | `az.hdi(..., prob=)` with results selected by `ci_bound='lower'` / `'upper'` |
| `gp.Marginal.marginal_likelihood(noise=)` | `marginal_likelihood(sigma=)` |
| `np.product` | `np.prod` |
| `pd.factorize([...])` | `pd.factorize(np.array([...]))` (pandas 3 rejects lists) |

### Changes in specific lectures

- **Lecture 07 and Lecture 12:**
  - Model comparison is reported as ELPD, where higher is better, instead of deviance, where lower is better.
  - The comparison tables, plots, axis labels and text were updated to match.
- **Lecture 08:** the trace plot has 42 panels. It raises ArviZ's 40-panel limit with `az.rc_context({"plot.max_subplots": 50})`.
- **Lecture 09 and Lecture 11:** data containers were renamed to `gender_id`, `department_id` and `education_level_id`. PyMC 6 doesn't allow a variable to have the same name as a dimension.
- **Lecture 14:** PyMC 6's `LKJCorr` returns the Cholesky factor `L` of the correlation matrix. The correlation matrix is now computed as `L @ L.T`.
- **Lecture 16:**
  - The `gp.Marginal` calls use `sigma=`.
  - A prior group is converted with `.to_dataset()` before `az.summary`.
- **Lecture 18:** the commented-out `marginal_likelihood` calls use `sigma=`.
- **Lecture 19:**
  - `pm.CustomDist` takes an explicit `signature`.
  - DataTree nodes are converted with `.to_dataset()` before `.to_dataframe()`.
  - One model samples with `init="adapt_diag"` to avoid an intermittent "Bad initial energy" error.

### Results that differ from the previous outputs

- Intervals are 89% HDIs, so they are narrower than the previous 94% intervals.
- Divergence and R-hat results are close to the previous outputs.
  - Lecture 03 and the devil's funnel in Lecture 14 still have divergences. The funnel's divergences are intentional.
  - Some models have R-hat slightly above 1.01, as before.
