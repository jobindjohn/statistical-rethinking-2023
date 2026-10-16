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
    4. Test (3) using (2)
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
def _():
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
    - This graph defines a causal model, of how $p, N$ effect the values of $W, L$. This is the same as saying it defines some function $f$ that maps $p, N$ onto the values of $W, L \rightarrow$  $W, L = f(p, N)$
    - Scientific knowledge defines what $f$ is or can be

    #### Bayesian data analysis
    "Very simple, very humble"
    - For each possible explanaint of the sample
    - Count all the ways the sampel could occur
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
    Calculate the number of ways to observing water ('W') given the toss of a globe
    with `resolution` number of sides and `n_water` faces.

    Note: this method results in numerical precision issues (due to the product) when the
    resolution of 16 or so, depending on your system.
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
def _():
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
def _(RESOLUTION, n_possible_ways, p_water):
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
def _(RESOLUTION):
    whole_sample = "WLWWWLWLW"
    n_W = len(whole_sample.replace("L", ""))
    n_L = len(whole_sample) - n_W

    whole_sample_n_ways, whole_sample_p_water = run_globe_tossing_simulation(whole_sample, resolution=RESOLUTION)
    return n_L, n_W, whole_sample, whole_sample_n_ways, whole_sample_p_water


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    show that we get identical answers with the analytical solution
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Results suggest the Analytical Soluiton $W,L = (Rp)^W \times (R - Rp)^L$
    where $R$ is the number of possible globes, in this case 4
    """)
    return


@app.cell
def _(RESOLUTION, n_L, n_W, whole_sample_n_ways, whole_sample_p_water):
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
def _(RESOLUTION, whole_sample_n_ways, whole_sample_p_water):
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
        The propotion of water
    N: int
        Number of globe tosses
    """
    return np.random.choice(list("WL"),  size=N, p=np.array([p, 1-p]), replace=True)


@app.cell
def _():
    np.random.seed(1)
    print(simulate_globe_toss())
    return


@app.cell
def _():
    pprint([simulate_globe_toss(p=1, N=11).tolist() for _ in range(10)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Test on Extreme settings
    """)
    return


@app.cell
def _():
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
def _(RESOLUTION, calculate_analytic_n_ways_possible):
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
def _(plot_posterior, whole_sample):
    _fig, _ax = plt.subplots()
    plot_posterior(whole_sample, resolution=4, ax=_ax)
    _fig
    return


@app.cell
def _(plot_posterior):
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
    ## Infinite Possibilities

    ### Moving from an N-sided globe to an infinitely-sided globe.
    As we increase resolution of globe
    - there are more bars/finer-grained resolution along the proportion axis
    - bars get shorter with more possibilities -- they must sum to 1
    """)
    return


@app.cell
def _(plot_posterior):
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
    p = \frac{(W + L + 1)!}{W!L!} p^W(1-p)^L
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
    """Calcualte the posterior for a string of observations"""
    n_W = len(observations.replace("L", ""))
    n_L = len(observations) - n_W
    proportions = np.linspace(0, 1, resolution)

    probs = beta(n_W, n_L, proportions)
    plt.plot(proportions, probs, **plot_kwargs)
    plt.yticks([])
    plt.title(observations)


@app.cell(hide_code=True)
def _():
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
    ## On Bayesian Inference...
    - **There is no minimun sample size** -- fewer samples fall back to prior
    - **Posterior shape embodies the sample size** -- more data makes the posterior more precise
    - There is no point estimates -- **the estimate is the entire posterior distribution**
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
def _():
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
def _(beta_posterior):
    # Execute the entire simulation
    _posterior_samples = beta_posterior.rvs(size=10000)

    # Here we use samples for the posterior to simulate sampling 10 observations from our generative model
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
    ### Sampling is Handsom & Handy
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
    - The explanations with more ways to produce data are more plausable

    ## Bayesian Modesty
    - If your generative model is correct, you can't do better: this will be an optimal
    - givens no gaurantees, only provides what you put into it
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Bonus: Misclassification
    In previous examples, we  do not consider sampling error or noise in measurement. In other words the number of Water observations that we measure may not be the _true_ number value. This means that the _true_ value for $W$ is unknown / unmeasured, but we instead measure $W^*$ that is a cause of the true, unmeasured $W$ and the measurement process M. If we know our measurement error rate, we can attempt to model it
    """)
    return


@app.cell(hide_code=True)
def _():
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
    ## Missclassification Simulation
    """)
    return


@app.function
def simulate_noisy_globe_toss(p=0.7, N=9, error_rate=0.1):
    # True sample
    sample = np.random.choice(list("WL"),  size=N, p=np.array([p, 1-p]), replace=True)

    # Error-induced sample
    error_trials = np.random.rand(N) < error_rate
    errors_effect_sample_trials = (sample == 'W') & error_trials
    sample[errors_effect_sample_trials] = 'L'
    return sample


@app.cell
def _():
    simulate_noisy_globe_toss()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Missclassification Estimator
    """)
    return


@app.function
def calculate_unnormalized_n_ways_possible_with_error(p, n_W, n_L, error_rate=0.1):
    n_W_error = (p * (1 - error_rate) + ((1 - p) * error_rate)) ** n_W
    n_L_error = ((1 - p) * (1 - error_rate) + (p * error_rate)) ** n_L
    return n_W_error * n_L_error


@app.cell
def _():
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
    ## Measurement Matters
    - beter to model measurement error, than to ignore it
    - same goes for mssing data
    - what matters is _why_ samples differ, and that we are explicit about how model it
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Environment": mo.plain_text(init_notebook.watermark_report())})
    return


if __name__ == "__main__":
    app.run()
