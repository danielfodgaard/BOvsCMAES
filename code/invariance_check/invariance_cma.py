"""H1 for CMA-ES (pycma), d = 5. Run with: python code/invariance_check/invariance_cma.py

A1: termination off, 2000 evaluations; compare the mean and sigma of every generation with the reference.
A2: default termination; record the stopping generation and reason.
Outputs results/invariance/cma.csv and first_divergence_cma.png.
"""
import cma
import numpy as np

from invariance_util import A_VALS, B_VALS, BOX, RESULTS, SEEDS, compare, make_problem, plot_first_divergence, write_csv

CMA_BUDGET = 2000
SIGMA0 = 2.5
NO_TERMINATION = {"tolfun": 0, "tolfunhist": 0}


def run_cma(f, seed, d=5, opts=None, use_stop=False):
    """Run pycma on f for CMA_BUDGET evaluations, or until es.stop() if use_stop.

    Returns (means, sigmas, stop): the mean and step size after every generation as arrays of
    shape (generations, d) and (generations,), and the final es.stop() dict.
    """
    x0 = np.random.default_rng(seed).uniform(*BOX, d)
    options = {
        "bounds": BOX,
        "seed": seed + 1,  # pycma treats seed=0 as "pick one at random"
        "verbose": -9,
        "popsize": 8,  # pycma's default for d = 5, spelled out: 2000 / 8 = 250 generations
    }
    options.update(opts or {})
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, options)

    means, sigmas = [], []
    while es.countevals < CMA_BUDGET:
        if use_stop and es.stop():
            break
        X = es.ask()
        es.tell(X, [f(x) for x in X])
        # es.mean lives in an unbounded internal space; xfavorite is that mean folded into the box
        means.append(es.result.xfavorite.copy())
        sigmas.append(es.sigma)
    return np.array(means), np.array(sigmas), dict(es.stop())


def experiment_cma(d=5):
    """A1 and A2 for every seed and (a, b). Returns one row per case.

    Columns: first_div (earliest generation where mean or sigma differs bit-for-bit from the reference,
    None if never), steps (generations run), max_diff_mean, max_diff_sigma, stop_gen and stop_reason (A2).
    """
    rows = []
    for seed in SEEDS:
        f = make_problem(d, seed)
        ref_means, ref_sigmas, _ = run_cma(f, seed, d, opts=NO_TERMINATION)
        for a in A_VALS:
            for b in B_VALS:
                g = lambda x: a * f(x) + b  # what the optimiser sees
                means, sigmas, _ = run_cma(g, seed, d, opts=NO_TERMINATION)  # A1
                _, stopped_sigmas, stop = run_cma(g, seed, d, use_stop=True)  # A2
                first_mean, max_mean = compare(means, ref_means)
                first_sigma, max_sigma = compare(sigmas, ref_sigmas)
                firsts = [i for i in (first_mean, first_sigma) if i is not None]
                rows.append({"seed": seed, "a": a, "b": b,
                             "first_div": min(firsts, default=None), "steps": len(means),
                             "max_diff_mean": max_mean, "max_diff_sigma": max_sigma,
                             "stop_gen": len(stopped_sigmas), "stop_reason": ", ".join(stop)})
    return rows


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = experiment_cma()
    write_csv(rows, RESULTS / "cma.csv")
    plot_first_divergence({"CMA-ES, d = 5": rows}, RESULTS / "first_divergence_cma.png")
