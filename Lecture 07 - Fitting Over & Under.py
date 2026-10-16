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
    # [Lecture 07 - Fitting Over & Under](https://youtu.be/1VgYIsANQck?si=dXg7iajUYQJMU-Ye)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Infinite causes, finite data

    - There are an infinite number of estimators that could explain a given sample.
    - Often there is a tradeoff between simplicity (parsimony) and accuracy

    ## Two simultaneous struggles
    1. Causation: using logic and causal assumptions to design estimators; comparing and contrast alternative models
    2. Finite Data: how to make estimators work
       - the existence of an estimator is not enough, and having estimator doesn't mean that it's practical or possible to estimate
       - we need to think about the engineering considerations around estimation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Problems of prediction
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Create Toy Brain Volumes Dataset
    """)
    return


@app.cell
def _():
    species = ["afarensis", "africanus", "habilis", "boisei", "rudolfensis", "ergaster", "sapiens"]
    brain_volume_cc = [438 , 452 , 612, 521, 752, 871, 1350]
    mass_kg = [37.0 , 35.5 , 34.5 , 41.5 , 55.5 , 61.0 , 53.5]

    BRAIN_MASS_TOY = pd.DataFrame({"species": species, "brain": brain_volume_cc, "mass": mass_kg})
    return (BRAIN_MASS_TOY,)


@app.cell
def _(BRAIN_MASS_TOY):
    def plot_brain_mass_data(ax=None, data=BRAIN_MASS_TOY):
        if ax is not None:
            plt.sca(ax)
        utils.plot_scatter(data.mass, data.brain, color='k')
        plt.xlabel('mass (kg)')
        plt.ylabel('brain volume (cc)')

    return (plot_brain_mass_data,)


@app.cell(hide_code=True)
def _(plot_brain_mass_data):
    _fig, _ax = plt.subplots(figsize=(4, 4))
    plot_brain_mass_data(_ax)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Considerations
    - what possible functions **describe** these points
      - **curve fitting & compression**
    - what possible functions **explain** these points
      - this is the goal of **causal inference**
    - what would happen if we changed one of the points mass?
      - **intervention**'
    - given a new mass, what's the **expected value** of corresponding volume
      - **prediction**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Leave-one-out Cross Validation (LOCCV)
    Process for determining the ability for a model function to generalize -- i.e. accurately predicting **out-of-sample** data points.

    1. Drop one data point
    2. Fit the model function's parameters with the data point missing
    3. Predict the value of the dropped point, record the error
    4. Repeat 1-3 for all data points
    5. The LOCCV score is the sum of errors on all the dropped points
    """)
    return


@app.cell
def _(plot_brain_mass_data):
    def fit_polynomial_cross_validated(order, data):
        n_data_points = len(data)
        data_idx = np.arange(n_data_points)
        params = []
        for ii in range(n_data_points):
            shift = n_data_points -1 - ii
            cross_val_idx = sorted(np.roll(data_idx, shift)[:n_data_points-1])

            cross_val_x = data.mass[cross_val_idx]
            cross_val_y = data.brain[cross_val_idx]
            cross_val_params = np.polyfit(cross_val_x, cross_val_y, order)
            params.append(cross_val_params)
        return np.array(params)


    def estimate_n_degree_model_crossvalidated(order, data, ylims=(300, 1400)):
        cross_val_params = fit_polynomial_cross_validated(order, data)
        insample_model = np.poly1d(np.polyfit(data.mass, data.brain, order))

        ca = plt.gca()
        plot_brain_mass_data(ax=ca, data=data)

        out_of_sample_error = 0
        insample_error = 0

        xs = np.linspace(data.mass.min(), data.mass.max(), 100)
        for holdout_idx, params in enumerate(cross_val_params):

            model = np.poly1d(params)
            ys = model(xs)
            utils.plot_line(xs, ys, color='gray', label=None, alpha=.25)

            holdout_x = data.loc[holdout_idx, "mass"]
            holdout_y = data.loc[holdout_idx, "brain"]

            holdout_prediction = model(holdout_x)
            insample_prediction = insample_model(holdout_x)

            # Here we use absolute error
            holdout_error = np.abs(holdout_prediction - holdout_y)
            insample_error_ = np.abs(insample_prediction - holdout_y)

            out_of_sample_error += holdout_error
            insample_error += insample_error_

        ys = insample_model(xs)
        utils.plot_line(xs, ys, color='C1', linewidth=6, label="Average Model")
        plt.ylim(ylims)
        plt.title(f"Polynomial Terms: {order}\nIn-sample: {insample_error:1,.0f}\nOut-of-sample: {out_of_sample_error:1,.0f}")

        return insample_error, out_of_sample_error

    return (estimate_n_degree_model_crossvalidated,)


@app.cell
def _(BRAIN_MASS_TOY, estimate_n_degree_model_crossvalidated):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _insample_error, _outsample_error = estimate_n_degree_model_crossvalidated(1, data=BRAIN_MASS_TOY)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparing Polynomial Models with LO0CV
    """)
    return


@app.cell
def _(estimate_n_degree_model_crossvalidated):
    def simulate_n_degree_polynomial_overfitting(data, ylims=(300, 1200)):

        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        insample_errors = []
        outsample_errors = []
        for ii in range(1, 6):
            row = (ii-1) // 2
            col = (ii-1) % 2
            plt.sca(axs[row, col])
            insample_error, outsample_error = estimate_n_degree_model_crossvalidated(ii, data=data, ylims=ylims)
            insample_errors.append(insample_error)
            outsample_errors.append(outsample_error)


        # rescale errors
        insample_errors = np.array(insample_errors)
        relative_insample_errors = insample_errors / insample_errors.sum()
        outsample_errors = np.array(outsample_errors)
        relative_outsample_errors = outsample_errors / outsample_errors.sum()

        plt.sca(axs[-1, -1])
        xs = np.arange(1, 6)

        utils.plot_line(xs, relative_insample_errors, color='C1', label='in-sample', alpha=1)
        utils.plot_scatter(xs, relative_insample_errors, color='C1', label=None, zorder=100, alpha=1, facecolors='w')

        utils.plot_line(xs, relative_outsample_errors, color='C0', label='out-of-sample', alpha=1)
        utils.plot_scatter(xs, relative_outsample_errors, color='C0', label=None, zorder=100, alpha=1, facecolors='w')
        plt.xlabel("polynomial terms")
        plt.ylabel("relative error")
        plt.legend()

        return outsample_errors, insample_errors

    return (simulate_n_degree_polynomial_overfitting,)


@app.cell
def _(BRAIN_MASS_TOY, simulate_n_degree_polynomial_overfitting):
    _outsample_error, _insample_error = simulate_n_degree_polynomial_overfitting(data=BRAIN_MASS_TOY)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As we increase the polynomial order
    - The in-sample error reduces
    - The out-of-sample error SKYROCKETS
    - (at least for simple models) There's a tradeoff between model simplicity and complexity that is mirrored by the amount of in-sample and out-of-sample error
    - There will be a "sweet spot" in model complexity that is just complex enough to fit the data, but not too complex to overfit to noise.
    - This is known as the bias-variance tradeoff

    **NOTE** this is all applicable to the goal of **prediction**, not causal inference.
    """)
    return


@app.cell
def _(simulate_n_degree_polynomial_overfitting):
    np.random.seed(1234)
    _n_samples = 15
    # Run simulation with more data points
    PRIMATE_DATA = utils.load_data("Primates301")[["brain", "body"]].dropna()

    # preprocess the data a little
    PRIMATE_DATA.loc[:, 'brain'] = PRIMATE_DATA.brain * 10
    PRIMATE_DATA.loc[:, 'mass'] = PRIMATE_DATA.body / 100
    PRIMATE_DATA = PRIMATE_DATA[PRIMATE_DATA.mass >= 30]
    PRIMATE_DATA = PRIMATE_DATA[PRIMATE_DATA.mass <= 65]
    PRIMATE_DATA = PRIMATE_DATA[PRIMATE_DATA.mass <= 900]

    PRIMATE_DATA = PRIMATE_DATA.sample(n=_n_samples).reset_index( drop=True)

    np.random.seed(1234)
    outsample_error, insample_error = simulate_n_degree_polynomial_overfitting(data=PRIMATE_DATA, ylims=(250, 900))
    plt.gcf()
    return insample_error, outsample_error


@app.function
def plot_out_of_sample_penalty(outsample_error, insample_error):
    penalty = outsample_error - insample_error
    n_terms = len(penalty)
    xs = np.arange(1, n_terms + 1)

    _, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axs[0])
    utils.plot_scatter(xs, insample_error, color='C1', label=None, zorder=100, alpha=1, facecolors='w')
    utils.plot_line(xs, insample_error, color='C1', label="in-sample", alpha=1)

    utils.plot_scatter(xs, outsample_error, color='C0', label=None, zorder=100, alpha=1, facecolors='w')
    utils.plot_line(xs, outsample_error, color='C0',   label="out-of-sample", alpha=1)

    for ii, (out, inn) in enumerate(zip(outsample_error, insample_error)):
        label = 'difference' if not ii else None
        plt.plot((ii + 1, ii + 1), (out, inn), color='k', lw=3, label=label)

    plt.ylabel("prediction error")
    plt.xlabel("polynomial terms")
    plt.legend()

    plt.sca(axs[1])
    utils.plot_scatter(xs, penalty, color='black', label=None, zorder=100, alpha=1, facecolors='w')
    utils.plot_line(xs, penalty, color='black', label='difference', alpha=1)
    plt.xlabel("polynomial terms")
    plt.xticks(xs)
    plt.ylabel("out-of-sample-penalty")
    plt.legend()


@app.cell(hide_code=True)
def _(insample_error, outsample_error):
    plot_out_of_sample_penalty(outsample_error, insample_error)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Regularization

    - There will be a tradeoff between model flexibility and accuracy. Having a model that's too flexible can lead to overeighting the contribution of noise.
    - Do no want your estimator to get too excited about rare outlier events. You want it to find **regular features**.
    - Bayesian modeling uses tighter priors (in terms of variance) in order to be skeptical of unlikely events, tampering down the estimator's dendency to "get excited" about outlier events, thus reducing the model's flexibility to model those outliers.
      - Good priors are often tigher than one thinks
    - The goal isn't signal compression (i.e. memorizing all features of the data), it is generalization in predictions.

    > Cross validation is not regularization: CV can be used to compare models, but not to reduce flexibility (though we could average models)

    ### Regularizing Priors
    - Use science: there's no replacement for domain knowledge
    - Can tune priors with using CV
    - Many tasks are a mix of inference and prediction, so weight the ability to predict and make actionable inferences
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we define the out-of-sample-penalty (OOSP) as the difference between the out-of-sample and in-sample error, we can see that as model complexity increases, so does the OOSP. We can use this metric for comparing models of different form and complexity by providing a signal for overfitting.

    > McElreath goes on to show that Bayesian cross-validation metrics WAIC and PSIS closely track the OOSP returned by brute-force LOOCV (via the proxy lppd). It would be nice to replicate that chart in Python/PyMC, however, that's a lot of extra coding and model estimation for one plot, so I'm going to skip that one for now. That said, I'll show in the section on Robust Regression an example of using WAIC and PSIS returned from the PyMC.

    ## Penalty Prediction & Model (Mis-) Selection

    For the simple example above, runing cross-validation to obtain in- and out-of-sample penality is no big deal. However, for more complex models that may take a long time to train, retraining multiple times can be prohibitive. Luckily there are some approximations to the CV procedure that allow us to obtain similar metrics directly from Bayesian models without having to explicitly run CV. These metrics include

    - Pareto-smoothed Importance Sampling (PSIS)
    - Waikake's Information Criterion (WAIC)

    > When directly addressing causal inference problems, **do not use CV Penalties for selecting causal models**, this can result in selecting a confounded model. **Confounds often aid with prediction in the absence of intervention** by milking all association signals. However, there are many associations that we do not want to include when addressing causal problems.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: Simulated Fungus Growth
    The following is a translation of R code 6.13 from McElreath's v2 textbook that's used to simulate fungus growth on plants.
    """)
    return


@app.cell(hide_code=True)
def _():
    utils.draw_causal_graph(
        edge_list=[
            ("H0", "H1"),
            ("T", "H1"),
            ("T", "F"),
            ("F", "H1")
        ],
        node_props={
            "T": {"label": "Treatment, T", "color": "red"},
            "F": {"label": "Fungus, F"},
            "H0": {"label": "Initial Height, H0"},
            "H1": {"label": "Final Height, H1"},
        },
        edge_props={
            ("T", "F"): {"color": "red"},
            ("F", "H1"): {"color": "red"},
            ("T", "H1"): {"color": "red"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulate the plant growth experiment
    """)
    return


@app.cell
def _():
    np.random.seed(12)

    _n_plants = 100

    # Simulate initial heights
    _H0 = stats.norm.rvs(10, 2, size=_n_plants)


    # Treatment aids plant growth indirectly by reducing fungus,
    # which has a negative effect on plant growth

    # Simulate treatment
    treatment = stats.bernoulli.rvs(p=0.5, size=_n_plants)

    # Simulate fungus growth conditioned on treatment
    _p_star = 0.5 - (treatment * 0.4)  # fungus is random when untreated, otherwise 10% chance
    fungus = stats.bernoulli.rvs(p=_p_star)

    # Simulate final heights based on fungus
    _BETA_F = -3  # fungus hurts growth
    _ALPHA = 5
    _SIGMA = 1

    _mu = _ALPHA + _BETA_F * fungus
    _H1 = _H0 + stats.norm.rvs(_mu, _SIGMA)

    plants = pd.DataFrame(
        {
            "H0": _H0,
            "treatment": treatment,
            "fungus": fungus,
            "H1": _H1,
            "growth": _H1 - _H0
        }
    )
    plants.describe().T
    return fungus, plants, treatment


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we are focusing on the total causal affect of Treatment, $T$ on final height $H1$

    #### Incorrect adjustment set (stratifying by $F$)

    $$
    \begin{align*}
    H_1 &\sim \mathcal{N}(\mu_i, \sigma) \\
    \mu_i &= H_0 \times g_i \\
    g_i &= \alpha + \beta_T T_i + \beta_F F_i
    \end{align*}
    $$

    - incorporates treatment $T$ _and_ fungus, $F$, which is a post-treatment variable (bad for causal inference)
    - $F$ would not have been found using backdoor criterion
    - However, provides more accurate predictions than not incorporating $F$
    """)
    return


@app.cell
def _(fungus, plants, treatment):
    with pm.Model() as biased_model:
        # Priors
        _sigma = pm.Exponential("sigma", 1)
        _alpha = pm.LogNormal("alpha", 0.2)
        _beta_T = pm.Normal("beta_T", 0, 0.5)
        _beta_F = pm.Normal("beta_F", 0, 0.5)

        _H0 = plants.H0.values
        _H1 = plants.H1.values

        _T = pm.Data("T", treatment.astype(float))
        _F = pm.Data("F", fungus.astype(float))

        # proportional amount of growth
        _g = _alpha + _beta_T * _T + _beta_F * _F
        _mu = _g * _H0

        # Likelihood
        pm.Normal("H1", _mu, _sigma, observed=_H1)

        biased_inference = cached_sample(target_accept=.95)
        pm.compute_log_likelihood(biased_inference)

    pm.model_to_graphviz(biased_model)
    return (biased_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Correct adjustment set (not stratifying by $F$)

    $$
    \begin{align*}
    H_1 &\sim \mathcal{N}(\mu_i, \sigma) \\
    \mu_i &= H_0 \times g_i \\
    g_i &= \alpha + \beta_T T_i
    \end{align*}
    $$

    - includes only treatment T
    - less accurate predictions, despite being the correct causal model
    """)
    return


@app.cell
def _(fungus, plants, treatment):
    with pm.Model() as unbiased_model:
        # Priors
        _sigma = pm.Exponential("sigma", 1.)
        _alpha = pm.LogNormal("alpha", .2)
        _beta_T = pm.Normal("beta_T", 0, .5)

        _H0 = plants.H0.values
        _H1 = plants.H1.values

        _T = pm.Data("T", treatment.astype(float))
        _F = pm.Data("F", fungus.astype(float))

        # proportional amount of growth
        _g = _alpha + _beta_T * _T
        _mu = _g * _H0

        # Likelihood
        pm.Normal("H1", _mu, _sigma, observed=_H1)

        unbiased_inference = cached_sample(target_accept=.95)
        pm.compute_log_likelihood(unbiased_inference)

    pm.model_to_graphviz(unbiased_model)
    return (unbiased_inference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Biased model provides better _predictions_ of the data
    We can see that by comparing LOO Cross validation scores, the biased model is ranked higher:
    - biased model weight = 1
    - ELPD is higher (indicating better predictions out of sample)
    """)
    return


@app.cell
def _(biased_inference, unbiased_inference):
    comparison = az.compare(
        {
            "H1 ~ H0 + T + F": biased_inference,
            "H1 ~ H0 + T": unbiased_inference
        }
    )
    comparison
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison):
    az.plot_compare(comparison)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### However, the unbiased model recovers the correct causal effect
    """)
    return


@app.cell
def _(biased_inference):
    az.summary(biased_inference)
    return


@app.cell
def _(unbiased_inference):
    az.summary(unbiased_inference)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Plotting the posteriors of each model provide different results. The biased model suggests that there is no, or even negative effect of the treatment on plant growth. We know this is not true because the simulation either has 0 effect for the untreated, or a positive effect (indirectly by reducting fungus) on plant growth.
    """)
    return


@app.cell(hide_code=True)
def _(biased_inference, unbiased_inference):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    utils.plot_dist(biased_inference.posterior["beta_T"], ax=_ax, color='black', label='biased')
    utils.plot_dist(unbiased_inference.posterior["beta_T"], ax=_ax, color='C0', label='correct')
    plt.axvline(0, linestyle='dashed', color='k')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Why does the wrong model win at prediction?
    - Fungus is a better _predictor_ of growth than treatment.
      - there is a clear (negative) linear relationship between fungus and growth (left plot below)
        - i.e. the marginals are better separated when separating by fungus
      - control / treatment provides a much less
        - i.e. the marginals are highly overalping when splitting by treatment
      - there are fewer fungus points in the treatment group (see right plot below). This provides less predictive signal in the training set for assessing the predictive association between treatment and growth.
    """)
    return


@app.cell(hide_code=True)
def _(plants):
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4))

    plt.sca(_axs[0])
    for _treat, _label in enumerate(["control", "treatment"]):
        _plot_data = plants[plants.treatment == _treat]
        _jitter = .5 * (np.random.rand(len(_plot_data)) - .5)
        _color = f'C{np.abs(_treat - 1)}'
        utils.plot_scatter(_plot_data.fungus + _jitter, _plot_data.growth, color=_color, label=_label)

    plt.xticks([0, 1], labels=['no fungus', 'yo fungus'])
    plt.legend()
    plt.ylabel("growth")

    plt.sca(_axs[1])
    for _fungus, _label in enumerate(['no fungus', 'fungus']):
        _plot_data = plants[plants.fungus == _fungus]
        _jitter = .5 * (np.random.rand(len(_plot_data)) - .5)
        _color = f'C{np.abs(_fungus - 1)}'
        utils.plot_scatter(_plot_data.treatment + _jitter, _plot_data.growth, color=_color, label=_label)

    plt.xticks([0, 1], labels=["control", "treatment"])
    plt.ylabel("growth")

    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Mis-selection
    - **NEVER use predictive tools (WAIC, PSIS, CV / OOSPS) to choose a causal estimate**
    - Most analyses are a mix of prediction and inference.
    - Accurate functional descriptions create simpler, more parsimonious models that are less prone to overfitting, so **use science where possible**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Outliers and Robust Regression
    - Outliers are more influential on the posterior than "normal" points
      - You can quantify this using CV -- this is related to Importance Sampling
    - Do not drop outliers, they are still signal
      - The model is wrong, not the data
      - e.g. Mixture models & robust regression
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: Divorce Rate
    - Maine & Idaho both unusual
    - Maine: High divorce rate for average marriage age
    - Idaho: Low divorce rate and less-than-average marriage age

    What is the influence of outlier points like Idaho and Maine?
    """)
    return


@app.cell
def _():
    divorce = utils.load_data("WaffleDivorce")[["Location", "Divorce", "MedianAgeMarriage"]]

    inliers_idx = divorce.query("Location not in ('Idaho', 'Maine')").index.tolist()
    outliers_idx = divorce.query("Location in ('Idaho', 'Maine')").index.tolist()


    std_age = utils.standardize(divorce.MedianAgeMarriage)
    std_divorce = utils.standardize(divorce.Divorce)
    return divorce, inliers_idx, outliers_idx, std_age, std_divorce


@app.cell(hide_code=True)
def _(divorce, inliers_idx, outliers_idx, std_age, std_divorce):
    _fig, _ax = plt.subplots()
    utils.plot_scatter(std_age[inliers_idx], std_divorce[inliers_idx], color='k', label=None)
    utils.plot_scatter(std_age[outliers_idx], std_divorce[outliers_idx], color='C0', label=None)

    for _oi in outliers_idx:
        plt.annotate(divorce.loc[_oi, "Location"], (std_age[_oi] + .1, std_divorce[_oi]), fontsize=14)

    plt.xlabel("Age at mariage (standardized)")
    plt.ylabel("Divorce rate (standardized)")
    plt.axis('square')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit Least Square Model
    """)
    return


@app.cell
def _(std_age, std_divorce):
    with pm.Model() as lstsq_model:
        # Priors
        _sigma = pm.Exponential("sigma", 1)
        _alpha = pm.Normal("alpha", 0, .5)
        _beta_A = pm.Normal("beta_A", 0, .5)

        # Likelihood
        _mu = _alpha + _beta_A * std_age.values
        pm.Normal("divorce", _mu, _sigma, observed=std_divorce.values)
        lstsq_inference = cached_sample()

        # required for LOOCV score
        pm.compute_log_likelihood(lstsq_inference)
    return (lstsq_inference,)


@app.cell
def _(divorce, outliers_idx):
    # Calculate Pointwise LOOCV scores
    def plot_loocv(inference, title=None):
        plt.subplots(figsize=(6,3))
        pareto_k = az.loo(inference, pointwise=True).pareto_k
        waic = -utils.waic(inference, pointwise=True).waic_i

        utils.plot_scatter(pareto_k, waic,  color='C0', label=None)
        for oi in outliers_idx:
            plt.annotate(divorce.loc[oi, "Location"], (pareto_k[oi] + .01, waic[oi]), fontsize=14)
        plt.xlabel("PSIS Pareto K")
        plt.ylabel("WAIC")
        plt.title(title)

    return (plot_loocv,)


@app.cell(hide_code=True)
def _(lstsq_inference, plot_loocv):
    plot_loocv(lstsq_inference, title="Gaussian Likelihood Posterior\nAffected by Outliers")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we can see that the two outliers have large effect on the the Normal Likelihood model posterior. This is because the outliers have very low probability under the Normal distribution, and thus are very "surprising" or "salient".
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mixing Gaussians -- the Student-T
    - Adding Gaussians of differing variances results in a distribution with fatter tails
    - Implicilty captures unobserved heterogeneity in the population
      - Multiple overlapping processes could have different variances.
    """)
    return


@app.cell(hide_code=True)
def _():
    xs = np.linspace(-6, 6, 100)
    _pdfs = []
    _n_gaussians = 20
    _fig, _ax = plt.subplots()
    for _variance in np.linspace(.5, 5, _n_gaussians):
        _label = "Individual\nGaussians" if _variance == .5 else None
        _pdf = stats.norm(0, _variance).pdf(xs)
        _pdfs.append(_pdf)
        utils.plot_line(xs, _pdf, color='k', label=_label, alpha=.25)

    _sum_of_pdfs = np.array(_pdfs).sum(axis=0)
    _sum_of_pdfs /= _sum_of_pdfs.max()
    _sum_of_pdfs *= (1 - _n_gaussians / 100)

    utils.plot_line(xs, _sum_of_pdfs, color='C0', label='Mixture of\nGaussian')
    plt.legend()
    _fig
    return (xs,)


@app.cell(hide_code=True)
def _(xs):
    _fig, _ax = plt.subplots(figsize=(6,3))

    utils.plot_line(xs, stats.norm.pdf(xs), color='k', label='Gaussian')
    utils.plot_line(xs, stats.t(2).pdf(xs), color='C0', label='Student-t')
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Robust Linear Regression Using the Student-t Likelihood
    """)
    return


@app.cell
def _(std_age, std_divorce):
    with pm.Model() as studentt_model:
        # Priors
        _sigma = pm.Exponential("sigma", 1)
        _alpha = pm.Normal("alpha", 0, .5)
        _beta_A = pm.Normal("beta_A", 0, .5)

        # Likelihood
        _mu = _alpha + _beta_A * std_age.values
        # Replacing Gaussian with Student T -- note, we just "choose" 2 as our degrees of freedom
        pm.StudentT("divorce", nu=2, mu=_mu, sigma=_sigma, observed=std_divorce.values)
        studentt_inference = cached_sample()
        pm.compute_log_likelihood(studentt_inference)
    return (studentt_inference,)


@app.cell(hide_code=True)
def _(lstsq_inference, studentt_inference):
    _fig, _ax = plt.subplots(figsize=(6,3))
    utils.plot_dist(lstsq_inference.posterior['beta_A'], color='k', label='Gaussian Model')
    utils.plot_dist(studentt_inference.posterior['beta_A'], color='C0', label='Student-t Model')
    plt.xlabel("$\\beta_A$ (effect of age on divorce)")
    plt.ylabel("density")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we can see how the outliers pull the posterior closer to zero in the vanilla Gaussian linear regression. The more robust student-t model is less affected by those outliers.

    #### Outliers have less effect on Student-t posterior
    We can see that using a likelihood more robust to outliers weighs those outliers less, as indicated by Maine and Idaho given less extreme posteior importance weights, particularly for PSIS.
    """)
    return


@app.cell(hide_code=True)
def _(plot_loocv, studentt_inference):
    plot_loocv(studentt_inference, title="Student-t Likelihood Posterior\nLess Affected by Outliers")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Review: Robust Regression
    - Mixture of Gaussians / Student-t implicilty captures unobserved heterogeneity in the population
    - This results in "thick tailed" distributions that are "less surprised" by outliers
    - We "just set" the degrees of freedom parameters in the Student-t. It turns out it is hard to accurately estimate degrees of freedom from the data because outlier values are rare.
    - Should we just use Robust Regression by default for under-theorized domains?
      - Probably a good idea
      - Example: The world is becoming more peacful since early 1900s.
        - Thick tailed-distributions would suggest that large-scale conflicts like those in early 1900s are not surprising, but are to be expected from processes like societal conflicts that can compound and "run-away" (which tend to be long-tailed).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
