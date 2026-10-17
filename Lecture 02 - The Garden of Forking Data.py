import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from pprint import pprint

    import marimo as mo
    import numpy as np
    from scipy import stats
    from scipy.special import factorial
    from matplotlib import pyplot as plt

    import utils
    import init_notebook

    init_notebook.setup()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # [Lecture 02 - The Garden of Forking Data](https://youtu.be/R1vcdhPBlXA?si=rL3BOz9hHxkPt79m)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Task: What proportion of earth's surface is covered with water?

    ## Workflow (Drawing the Owl)

    1. Define generative model of tossing the globe
    2. Define an estimand -- in this case, the proportion of globe covered in water?
    3. Design a statistical means to produce estimate
    4. Test (3) using (1)
    5. Analyze real data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1, 2. Define generative model of globe tossing
    - $p$: proportion of water -- this is what we'd like to estimate
    - $N$: number of tosses  -- we control this via experiment
    - $W$: number of Water observations
    - $L$: number of Land observations
    """)
    return


@app.cell(hide_code=True)
def globe_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("p", "W"),
            ("p", "L"),
            ("N", "L"),
            ("N", "W")
        ],
        graph_direction="LR",
        node_props={
            "p": {"color": "red"}
        },
        edge_props={
            ("p", "W"): {"label": "influence"}
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - This graph defines a causal model, of how $p, N$ affect the values of $W, L$. This is the same as saying it defines some function $f$ that maps $p, N$ onto the values of $W, L \rightarrow$  $W, L = f(p, N)$
    - Scientific knowledge defines what $f$ is or can be

    #### Bayesian data analysis
    "Very simple, very humble"
    - For each possible explanation of the sample
    - Count all the ways the sample could occur
    - **The explanations with the largest number of ways to produce the sample are more plausible**

    The unglamourous basis of applied probability:
    > Things that can happen more ways are more plausible.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Design a statistical way to produce estimate
    ### Garden of Forking Data
    Following the mantra above...

    - for each possible **proportion of water**, $p$
    - count all the ways the sample of tosses could have occurred
    - the $p$ that are associated with more ways to produce the sample are more plausible
    """)
    return


@app.function
def calculate_n_ways_possible(observations: str, n_water: int, resolution: int = 4):
    """
    Calculate the number of ways of observing water ('W') given the toss of a globe
    with `resolution` sides, `n_water` of which are water.

    Note: this method results in numerical precision issues (due to the product) when the
    resolution reaches 16 or so, depending on your system.
    """
    assert n_water <= resolution

    # Convert observation string to an array
    observations = np.array(list(observations.upper()))

    # Create n-sided globe with possible outcomes
    possible = np.array(list("L" * (resolution - n_water)) + list("W" * n_water))

    # Tally up ways to obtain each observation given the possible outcomes
    # Here we use brute-force, but we could also use the analytical solution below
    ways = []
    for obs in observations:
        ways.append((possible == obs).sum())

    p_water = n_water / resolution
    # perform product in log space for numerical precision
    n_ways = np.round(np.exp(np.sum(np.log(ways)))).astype(int)
    return n_ways, p_water


@app.function
def run_globe_tossing_simulation(observations, resolution, current_n_possible_ways=None):
    """Simulate the number of ways you can observe water ('W') for a globe of `resolution`
    sides, varying the proportion of the globe that is covered by water.
    """
    # For Bayesian updates
    current_n_possible_ways = current_n_possible_ways if current_n_possible_ways is not None else np.array([])

    print(f"Observations: '{observations}'")
    p_water = np.array([])
    for n_W in range(0, resolution + 1):
        n_L = resolution - n_W
        globe_sides = "W" * n_W + "L" * n_L
        n_possible_ways, p_water_ = calculate_n_ways_possible(observations, n_water=n_W, resolution=resolution)
        print(f"({n_W+1}) {globe_sides} p(W) = {p_water_:1.2}\t\t{n_possible_ways} Ways to Produce")

        p_water = np.append(p_water, p_water_)
        current_n_possible_ways = np.append(current_n_possible_ways, n_possible_ways)

    return current_n_possible_ways, p_water


@app.cell
def wlw_simulation():
    RESOLUTION = 4
    observations = "WLW"
    n_possible_ways, p_water = run_globe_tossing_simulation(observations, resolution=RESOLUTION)
    return RESOLUTION, n_possible_ways, p_water


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bayesian (online) Updating
    """)
    return


@app.cell
def online_update(RESOLUTION, n_possible_ways, p_water):
    _new_observation_possible_ways, _ = run_globe_tossing_simulation("W", resolution=RESOLUTION)

    # Online update
    _updated_n_possible_ways = n_possible_ways * _new_observation_possible_ways

    print("\nUpdated Possibilities given new observation:")
    for _ii in range(0, RESOLUTION + 1):
        print(f"({_ii+1}) p(W) = {p_water[_ii]:1.2}\t\t{int(_updated_n_possible_ways[_ii])} Ways to Produce")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The whole sample
    """)
    return


@app.cell
def whole_sample_simulation(RESOLUTION):
    whole_sample = "WLWWWLWLW"
    n_W = len(whole_sample.replace("L", ""))
    n_L = len(whole_sample) - n_W

    whole_sample_n_ways, whole_sample_p_water = run_globe_tossing_simulation(whole_sample, resolution=RESOLUTION)
    return n_L, n_W, whole_sample, whole_sample_n_ways, whole_sample_p_water


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Show that we get identical answers with the analytical solution
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Results suggest the analytical solution: ways to produce $W, L = (Rp)^W \times (R - Rp)^L$
    where $R$ is the number of sides of the globe, in this case 4
    """)
    return


@app.cell
def analytic_solution(
    RESOLUTION,
    n_L,
    n_W,
    whole_sample_n_ways,
    whole_sample_p_water,
):
    def calculate_analytic_n_ways_possible(p, n_W, n_L, resolution=RESOLUTION):
        """This scales much better than the brute-force method"""
        return (resolution * p) ** n_W * (resolution - resolution * p) ** n_L

    _analytic_n_possible_ways = np.array([calculate_analytic_n_ways_possible(p, n_W, n_L) for p in whole_sample_p_water])
    assert (_analytic_n_possible_ways == whole_sample_n_ways).all()
    return (calculate_analytic_n_ways_possible,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Probability
    - non-negative values that sum to 1
    - normalizes large sums by the total counts
    """)
    return


@app.cell
def ways_table(RESOLUTION, whole_sample_n_ways, whole_sample_p_water):
    _n_possible_probabilities = whole_sample_n_ways / whole_sample_n_ways.sum()

    print("Proportion\tWays\tProbability")
    for _p, _n_w, _n_p in zip(whole_sample_p_water, whole_sample_n_ways, _n_possible_probabilities):
        print(f"{_p:1.12}\t\t{_n_w:0.0f}\t{_n_p:1.2f}")

    _probs = np.linspace(0, 1, RESOLUTION+1)
    _fig, _ax = plt.subplots(figsize=(5, 5))
    plt.bar(x=_probs, height=_n_possible_probabilities, width= .9 / RESOLUTION, color='k')
    plt.xticks(_probs)
    plt.ylabel("probability")
    plt.xlabel("proportion water")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Try it: count the ways
    Type a sequence of tosses and change the number of sides on the globe. The table counts the ways each possible globe could produce the sequence; the bars are those counts normalized into probabilities.
    """)
    return


@app.cell(hide_code=True)
def garden_controls():
    garden_sequence = mo.ui.text(value="WLWWWLWLW", label="Toss sequence (W/L)")
    garden_sides = mo.ui.slider(2, 10, value=4, label="Globe sides", show_value=True)
    return garden_sequence, garden_sides


@app.cell(hide_code=True)
def garden_explorer(
    calculate_analytic_n_ways_possible,
    garden_sequence,
    garden_sides,
):
    _observations = "".join(_c for _c in garden_sequence.value.upper() if _c in "WL")
    _n_W = _observations.count("W")
    _n_L = len(_observations) - _n_W
    _R = garden_sides.value

    _p_water = np.linspace(0, 1, _R + 1)
    _ways = np.array([calculate_analytic_n_ways_possible(_p, _n_W, _n_L, _R) for _p in _p_water])
    _posterior = _ways / _ways.sum()

    _table = mo.ui.table(
        [
            {"globe": "W" * _k + "L" * (_R - _k), "p(W)": round(_p, 3), "ways": f"{_w:,.0f}", "probability": round(_pr, 3)}
            for _k, (_p, _w, _pr) in enumerate(zip(_p_water, _ways, _posterior))
        ],
        selection=None,
        pagination=False,
    )

    _fig, _ax = plt.subplots(figsize=(5, 4))
    plt.bar(x=_p_water, height=_posterior, width=.9 / _R, color='k')
    plt.xticks(_p_water, labels=[f"{_p:.2f}" for _p in _p_water], rotation=45)
    plt.ylabel("probability")
    plt.xlabel("proportion water")
    plt.title(f"{_n_W} W, {_n_L} L")

    mo.vstack([
        mo.hstack([garden_sequence, garden_sides], justify="start", gap=2),
        mo.hstack([_table, _fig], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Test (3) using (1)

    ### Test Before You Est(imate) 🐤
    1. Code generative simulation
    2. Code an estimator
    3. Test 2 with 1; you should get expected output

    **IF YOU TEST NOTHING YOU MISS EVERYTHING**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4.1 Generative Simulation
    """)
    return


@app.function
def simulate_globe_toss(p=0.7, N=9):
    """Simulate N globe tosses with a specific/known proportion
    p: float
        The proportion of water
    N: int
        Number of globe tosses
    """
    return np.random.choice(list("WL"),  size=N, p=np.array([p, 1-p]), replace=True)


@app.cell
def toss_simulation_demo():
    np.random.seed(1)
    print(simulate_globe_toss())
    return


@app.cell
def toss_simulation_all_water():
    pprint([simulate_globe_toss(p=1, N=11).tolist() for _ in range(10)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Test on Extreme settings
    """)
    return


@app.cell
def large_n_convergence():
    # Large N should converge to known p
    _known_p = 0.5

    _simulated_ps = []
    _sample_sizes = np.linspace(10, 100_000, 10)
    for _N in _sample_sizes:
        _simulated_p = np.sum(simulate_globe_toss(p=_known_p, N=int(_N)) == 'W') / _N
        _simulated_ps.append(_simulated_p)

    _fig, _ax = plt.subplots()
    plt.axhline(_known_p, label=f"Known p={_known_p}", color='k', linestyle='--')
    plt.legend()
    plt.plot(_sample_sizes, _simulated_ps)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4.2 Code the estimator
    """)
    return


@app.cell
def posterior_functions(RESOLUTION, calculate_analytic_n_ways_possible):
    def compute_posterior(observations, resolution=RESOLUTION, ax=None):
        n_W = len(observations.replace("L", ""))
        n_L = len(observations) - n_W

        p_water = np.linspace(0, 1, resolution+1)
        n_possible_ways = np.array([calculate_analytic_n_ways_possible(p, n_W, n_L, resolution) for p in p_water])

        posterior = n_possible_ways / n_possible_ways.sum()

        probs = np.linspace(0, 1, resolution+1)
        return posterior, probs

    def plot_posterior(observations, resolution=RESOLUTION, ax=None):
        posterior, probs = compute_posterior(observations, resolution=resolution)
        if ax is not None:
            plt.sca(ax)
        plt.bar(x=probs, height=posterior, width= .9 / resolution, color='k')
        plt.xticks(probs[::2], rotation=45)
        plt.ylabel("probability")
        plt.xlabel("proportion water")
        plt.title(f"Posterior Calculated\nfrom # Samples: {len(observations)}")

    return (plot_posterior,)


@app.cell(hide_code=True)
def whole_sample_posterior_plot(plot_posterior, whole_sample):
    _fig, _ax = plt.subplots()
    plot_posterior(whole_sample, resolution=4, ax=_ax)
    _fig
    return


@app.cell
def estimator_test(plot_posterior):
    np.random.seed(2)
    _p = .4
    _simulated_observations = "".join(simulate_globe_toss(p=_p, N=100))

    _fig, _ax = plt.subplots()
    plot_posterior(_simulated_observations, resolution=20, ax=_ax)
    plt.axvline(_p, color='C0', label='True Proportion')
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Try it: test the estimator
    Simulate tosses from a globe with a known proportion of water, and check that the posterior concentrates around it as the number of tosses grows. The number of tosses stops at 150: beyond that the raw counts of ways overflow floating-point numbers, which is one reason real software works with log-probabilities.
    """)
    return


@app.cell(hide_code=True)
def estimator_controls():
    estimator_true_p = mo.ui.slider(0, 1, step=0.05, value=0.4, label="True proportion of water", show_value=True)
    estimator_n_tosses = mo.ui.slider(steps=[1, 2, 5, 10, 20, 50, 100, 150], value=100, label="Number of tosses", show_value=True)
    estimator_resolution = mo.ui.slider(2, 100, value=20, label="Globe sides", show_value=True)
    estimator_seed = mo.ui.number(start=0, stop=10_000, value=2, label="Random seed")
    return (
        estimator_n_tosses,
        estimator_resolution,
        estimator_seed,
        estimator_true_p,
    )


@app.cell(hide_code=True)
def estimator_explorer(
    estimator_n_tosses,
    estimator_resolution,
    estimator_seed,
    estimator_true_p,
    plot_posterior,
):
    np.random.seed(estimator_seed.value)
    _observations = "".join(simulate_globe_toss(p=estimator_true_p.value, N=estimator_n_tosses.value))

    _fig, _ax = plt.subplots()
    plot_posterior(_observations, resolution=estimator_resolution.value, ax=_ax)
    plt.xticks(np.linspace(0, 1, 11), rotation=45)
    plt.axvline(estimator_true_p.value, color='C0', label='True Proportion')
    plt.legend()

    mo.vstack([
        mo.hstack([estimator_true_p, estimator_n_tosses], justify="start", gap=2),
        mo.hstack([estimator_resolution, estimator_seed], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Infinite Possibilities

    ### Moving from an N-sided globe to an infinitely-sided globe.
    As we increase resolution of globe
    - there are more bars/finer-grained resolution along the proportion axis
    - bars get shorter with more possibilities -- they must sum to 1
    """)
    return


@app.cell
def posterior_by_resolution(plot_posterior):
    np.random.seed(12)
    _p = .7
    _simulated_observations = "".join(simulate_globe_toss(p=_p, N=30))
    _fig, _axs = plt.subplots(1, 3, figsize=(10, 4))
    for _ii, _possibilities in enumerate([5, 10, 20]):
        plot_posterior(_simulated_observations, resolution=_possibilities, ax=_axs[_ii])
        plt.ylim([-.05, 1])
        _axs[_ii].set_title(f"{_possibilities} possibilities")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Beta Distribution

    Analytical function that gives us the pdf as the limit as number of possibilities $\rightarrow \infty$

    $$
    \Pr(p \mid W, L) = \frac{(W + L + 1)!}{W!L!} p^W(1-p)^L
    $$

    where $\frac{(W + L + 1)!}{W!L!}$ is a normalizing constant to make the distribution sum to 1
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tossing the Globe
    """)
    return


@app.function
def beta(W, L, p):
    return factorial(W + L + 1) / (factorial(W) * factorial(L)) * p ** W * (1-p) ** L


@app.function
def plot_beta_from_observations(observations: str, resolution: int = 50, **plot_kwargs):
    """Calculate the posterior for a string of observations"""
    n_W = len(observations.replace("L", ""))
    n_L = len(observations) - n_W
    proportions = np.linspace(0, 1, resolution)

    probs = beta(n_W, n_L, proportions)
    plt.plot(proportions, probs, **plot_kwargs)
    plt.yticks([])
    plt.title(observations)


@app.cell(hide_code=True)
def sequential_updating():
    # Tossing the globe
    _observations = "WLWWWLWLW"
    _fig, _axs = plt.subplots(3, 3, figsize=(8, 8))
    for _ii in range(9):
        plt.sca(_axs[_ii // 3][_ii % 3])
        # Plot previous
        if _ii > 0:
            plot_beta_from_observations(_observations[:_ii], color='k', linestyle='--')
        else:
            # First observation, no previous data
            plot_beta_from_observations('', color='k', linestyle='--')

        _color = 'C1' if _observations[_ii] == 'W' else 'C0'
        plot_beta_from_observations(_observations[:_ii+1], color=_color, linewidth=4, alpha=.5)

        if not _ii % 3:
            plt.ylabel("posterior probability")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Try it: toss the globe yourself
    Add tosses one at a time, by hand or at random from a globe with the chosen proportion of water, and watch the posterior update. The dashed line is the posterior before the last toss.
    """)
    return


@app.cell(hide_code=True)
def tosser_state():
    get_tosses, set_tosses = mo.state("")
    return get_tosses, set_tosses


@app.cell(hide_code=True)
def tosser_true_p():
    toss_true_p = mo.ui.slider(0, 1, step=0.05, value=0.7, label="True proportion of water (random tosses)", show_value=True)
    return (toss_true_p,)


@app.cell(hide_code=True)
def tosser_buttons(set_tosses, toss_true_p):
    toss_water_button = mo.ui.button(label="W", on_click=lambda _: set_tosses(lambda s: s + "W"))
    toss_land_button = mo.ui.button(label="L", on_click=lambda _: set_tosses(lambda s: s + "L"))
    toss_random_button = mo.ui.button(
        label="Random toss",
        on_click=lambda _: set_tosses(lambda s: s + str(simulate_globe_toss(p=toss_true_p.value, N=1)[0])),
    )
    toss_reset_button = mo.ui.button(label="Reset", on_click=lambda _: set_tosses(""))
    return (
        toss_land_button,
        toss_random_button,
        toss_reset_button,
        toss_water_button,
    )


@app.cell(hide_code=True)
def tosser_display(
    get_tosses,
    toss_land_button,
    toss_random_button,
    toss_reset_button,
    toss_true_p,
    toss_water_button,
):
    _observations = get_tosses()
    _n_W = _observations.count("W")
    _n_L = len(_observations) - _n_W
    _proportions = np.linspace(0, 1, 200)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    if _observations:
        _previous = _observations[:-1]
        _previous_n_W = _previous.count("W")
        _previous_n_L = len(_previous) - _previous_n_W
        plt.plot(_proportions, stats.beta(_previous_n_W + 1, _previous_n_L + 1).pdf(_proportions), color='k', linestyle='--', label='before last toss')
    _color = 'C1' if _observations.endswith('W') else 'C0' if _observations else 'gray'
    plt.plot(_proportions, stats.beta(_n_W + 1, _n_L + 1).pdf(_proportions), color=_color, linewidth=4, alpha=.5, label='posterior')
    plt.axvline(toss_true_p.value, color='gray', linestyle=':', label='true proportion')
    plt.xlabel("proportion water")
    plt.ylabel("density")
    plt.yticks([])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.title(f"{_n_W} W, {_n_L} L")

    _shown = _observations if len(_observations) <= 60 else "..." + _observations[-60:]
    mo.vstack([
        mo.hstack([toss_water_button, toss_land_button, toss_random_button, toss_reset_button], justify="start"),
        toss_true_p,
        mo.md(f"**Tosses:** `{_shown or '(none yet)'}`"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## On Bayesian Inference...
    - **There is no minimum sample size** -- fewer samples fall back to prior
    - **Posterior shape embodies the sample size** -- more data makes the posterior more precise
    - There are no point estimates -- **the estimate is the entire posterior distribution**
    - There is no true interval -- there are an infinite number of intervals one could draw, each is arbitrary and depends on what you're trying to communicate/summarize
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From Posterior to Prediction
    - To make predictions, we must average (i.e. integrate) over the entire posterior -- this averages over the uncertainty in the posterior
    - We could do this with integral calculus
    - OR, we could **just take samples from the posterior and average over those**

    **TURN A CALCULUS PROBLEM INTO A DATA SUMMARY PROBLEM**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling from Posterior Distribution
    """)
    return


@app.cell
def beta_posterior_samples():
    _a, _b = 6, 3
    beta_posterior = stats.beta(_a, _b)
    _beta_posterior_samples = beta_posterior.rvs(size=1000)

    _fig, _ax = plt.subplots()
    plt.hist(_beta_posterior_samples, bins=50, density=True, label='samples')
    _probs = np.linspace(0, 1, 100)
    plt.plot(_probs, beta(_a-1, _b-1, _probs), linewidth=3, color='k', linestyle='--', label='beta distribution')
    plt.xlabel("proportion water")
    plt.ylabel("density")
    plt.legend()
    _fig
    return (beta_posterior,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling from Posterior Predictive Distribution
    **Posterior Prediction**: a prediction for out-of-sample data based on the current posterior estimate
    """)
    return


@app.cell
def posterior_predictive_simulation(beta_posterior):
    # Execute the entire simulation
    _posterior_samples = beta_posterior.rvs(size=10000)

    # Here we use samples from the posterior to simulate sampling 10 observations from our generative model
    _posterior_predictive = [(simulate_globe_toss(p, 10) == 'W').sum() for p in _posterior_samples]
    _ppd_unique, _ppd_counts = np.unique(_posterior_predictive, return_counts=True)

    _specific_prob = 0.64
    _specific_predictive = [(simulate_globe_toss(_specific_prob, 10) == 'W').sum() for _ in _posterior_samples]
    _specific_unique, _specific_counts = np.unique(_specific_predictive, return_counts=True)

    _fig, _ax = plt.subplots()
    plt.bar(_specific_unique, _specific_counts, width=.5, color='k', label=f'simulation at p={_specific_prob:1.2}')
    plt.bar(_ppd_unique, _ppd_counts, width=.2, color='C1', label='posterior predictive')
    plt.xlabel("number of 'W' samples")
    plt.ylabel('count')
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling is Handsome & Handy
    Things we'll compute via sampling
    - Forecasts
    - Causal effects
    - Counterfactuals
    - Prior Predictions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Summary: Bayesian Data Analysis
    - For each possible explanation of data
    - Count all the ways that data could happen under that explanation
    - The explanations with more ways to produce data are more plausible

    ## Bayesian Modesty
    - If your generative model is correct, you can't do better: this will be optimal
    - gives no guarantees, only provides what you put into it
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Bonus: Misclassification
    In previous examples, we do not consider sampling error or noise in measurement. In other words the number of Water observations that we measure may not be the _true_ value. This means that the _true_ value for $W$ is unknown / unmeasured, but we instead measure $W^*$, which is caused by the true, unmeasured $W$ and the measurement process $M$. If we know our measurement error rate, we can attempt to model it
    """)
    return


@app.cell(hide_code=True)
def misclassification_dag():
    utils.draw_causal_graph(
        edge_list=[
            ("p", "W"),
            ("W", "W*"),
            ("M", "W*"),
            ("N", "W")
        ],
        node_props={
            "p": {"color": "red", "style": "dashed"},
            "W": {"style": "dashed", "label": "actual W"},
            "W*": {"style": "dashed", "label": "noisy W, W*"},
            "unobserved": {"style": "dashed"},
            "M": {"label": "measurement error, M"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Misclassification Simulation
    """)
    return


@app.function
def simulate_noisy_globe_toss(p=0.7, N=9, error_rate=0.1):
    # True sample
    sample = np.random.choice(list("WL"),  size=N, p=np.array([p, 1-p]), replace=True)

    # Error-induced sample: each toss is misclassified with probability
    # `error_rate`, recording a W as L and an L as W
    error_trials = np.random.rand(N) < error_rate
    sample[error_trials] = np.where(sample[error_trials] == 'W', 'L', 'W')
    return sample


@app.cell
def noisy_toss_demo():
    simulate_noisy_globe_toss()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Misclassification Estimator
    """)
    return


@app.function
def calculate_unnormalized_n_ways_possible_with_error(p, n_W, n_L, error_rate=0.1):
    n_W_error = (p * (1 - error_rate) + ((1 - p) * error_rate)) ** n_W
    n_L_error = ((1 - p) * (1 - error_rate) + (p * error_rate)) ** n_L
    return n_W_error * n_L_error


@app.cell
def misclassification_posterior():
    _a, _b = 6, 3
    _resolution = 100
    _proportions = np.linspace(0, 1, _resolution)
    _error_rate = 0.1
    _error_posterior = np.array(
        [
            calculate_unnormalized_n_ways_possible_with_error(p, _a, _b, _error_rate) for p in _proportions
        ]
    )
    _beta_posterior = beta(_a, _b, _proportions)

    # Infer normalization constant Z from distribution
    _error_posterior *= _resolution / _error_posterior.sum()
    _beta_posterior *= _resolution / _beta_posterior.sum()

    _fig, _ax = plt.subplots(figsize=(6, 6))
    plt.plot(_proportions, _beta_posterior, label='previous posterior', color='k', linewidth=4)
    plt.plot(_proportions, _error_posterior, label=f'misclassification posterior\n(error rate={_error_rate:1.2})', linewidth=4)
    plt.xlabel("proportion of water")
    plt.ylabel("posterior probability")
    plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Try it: misclassification
    Change the error rate and the observed counts. At an error rate of 0.5 the observations carry no information, and the misclassification posterior is flat.
    """)
    return


@app.cell(hide_code=True)
def misclassification_controls():
    misclassification_error_rate = mo.ui.slider(0, 0.5, step=0.01, value=0.1, label="Error rate", show_value=True)
    misclassification_n_W = mo.ui.slider(0, 50, value=6, label="Observed W", show_value=True)
    misclassification_n_L = mo.ui.slider(0, 50, value=3, label="Observed L", show_value=True)
    return (
        misclassification_error_rate,
        misclassification_n_L,
        misclassification_n_W,
    )


@app.cell(hide_code=True)
def misclassification_explorer(
    misclassification_error_rate,
    misclassification_n_L,
    misclassification_n_W,
):
    _n_W, _n_L = misclassification_n_W.value, misclassification_n_L.value
    _error_rate = misclassification_error_rate.value
    _resolution = 100
    _proportions = np.linspace(0, 1, _resolution)
    _error_posterior = np.array(
        [
            calculate_unnormalized_n_ways_possible_with_error(p, _n_W, _n_L, _error_rate) for p in _proportions
        ]
    )
    _beta_posterior = beta(_n_W, _n_L, _proportions)

    # Infer normalization constant Z from distribution
    _error_posterior *= _resolution / _error_posterior.sum()
    _beta_posterior *= _resolution / _beta_posterior.sum()

    _fig, _ax = plt.subplots(figsize=(6, 5))
    plt.plot(_proportions, _beta_posterior, label='ignoring misclassification', color='k', linewidth=4)
    plt.plot(_proportions, _error_posterior, label=f'misclassification posterior\n(error rate={_error_rate:1.2})', linewidth=4)
    plt.xlabel("proportion of water")
    plt.ylabel("posterior probability")
    plt.legend()

    mo.vstack([
        mo.hstack([misclassification_error_rate, misclassification_n_W, misclassification_n_L], justify="start", gap=2),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Measurement Matters
    - better to model measurement error, than to ignore it
    - same goes for missing data
    - what matters is _why_ samples differ, and that we are explicit about how we model it
    """)
    return


@app.cell(hide_code=True)
def environment():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
