"""H1 for CMA-ES (pycma), d = 5. Run with: python code/invariance_check/invariance_cma.py

A1: termination off, 2000 evaluations; compare the mean and sigma of every generation with the reference.
A2: default termination; record the stopping generation and reason.
Outputs results/invariance/cma.csv, cma_gaps.csv (gap in best untransformed f, with the seed null band),
cma_curves.csv (best-so-far untransformed f after every evaluation, for the figure), first_divergence_cma.png, best_so_far_cma.png and best_so_far_cma_budget.png (fixed budget only, the report's figure).
"""
import cma
import numpy as np

from invariance_util import A_VALS, B_VALS, BOX, RESULTS, SEEDS, best_so_far, start_point, compare, log_gap, make_problem, plot_best_so_far, plot_first_divergence, summarise_gaps, write_csv

CMA_BUDGET = 2000
SIGMA0 = 2.5
NO_TERMINATION = {"tolfun": 0, "tolfunhist": 0}  # redundant in fixed-budget runs: run_cma only calls es.stop() with
# use_stop=True, so there every termination criterion is off anyway; kept so the options say what the run does


def run_cma(f, seed, d=5, opts=None, use_stop=False, x0=None):
    """Run pycma on f for CMA_BUDGET evaluations, or until es.stop() if use_stop.

    x0 defaults to start_point(d, seed), a seeded uniform draw from the box independent of x*; opts may override any option, e.g. per-axis "bounds" or "CMA_stds".

    Returns (means, sigmas, stop, X): the mean and step size after every generation as arrays of
    shape (generations, d) and (generations,), the final es.stop() dict, and every evaluated point
    in evaluation order as an array of shape (evaluations, d).
    """
    x0 = start_point(d, seed) if x0 is None else x0
    options = {
        "bounds": BOX,
        "seed": seed + 1,  # pycma treats seed=0 as "pick one at random"
        "verbose": -9,
        "popsize": 8,  # pycma's default for d = 5, spelled out: 2000 / 8 = 250 generations
    }
    options.update(opts or {})
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, options)

    means, sigmas, points = [], [], []
    while es.countevals < CMA_BUDGET:
        if use_stop and es.stop():
            break
        X = es.ask()
        es.tell(X, [f(x) for x in X])
        # es.mean lives in an unbounded internal space; xfavorite is that mean folded into the box
        means.append(es.result.xfavorite.copy())
        sigmas.append(es.sigma)
        points.extend(X)
    return np.array(means), np.array(sigmas), dict(es.stop()), np.array(points)


def experiment_cma(d=5):
    """A1 and A2 for every seed and (a, b). Returns (rows, curve_rows): one row per case, and one row per
    case, arm (budget = A1, stop = A2) and evaluation with the best-so-far untransformed f.

    Columns: first_div (earliest generation where mean or sigma differs bit-for-bit from the reference,
    None if never), steps (generations run), max_diff_mean, max_diff_sigma, stop_gen and stop_reason (A2),
    best_f / best_f_stop (best untransformed f at the end of A1 / at the A2 stop) and gap / gap_stop
    (their log10 difference to the reference run of the same seed).
    """
    rows, curve_rows = [], []
    for seed in SEEDS:
        f = make_problem(d, seed)
        ref_means, ref_sigmas, _, ref_X = run_cma(f, seed, d, opts=NO_TERMINATION)
        ref_best, ref_best_stop = best_so_far(f, ref_X)[-1], best_so_far(f, run_cma(f, seed, d, use_stop=True)[3])[-1]
        for a in A_VALS:
            for b in B_VALS:
                g = lambda x: a * f(x) + b  # what the optimiser sees
                means, sigmas, _, X = run_cma(g, seed, d, opts=NO_TERMINATION)  # A1
                _, stopped_sigmas, stop, X_stop = run_cma(g, seed, d, use_stop=True)  # A2
                curve, curve_stop = best_so_far(f, X), best_so_far(f, X_stop)  # scored on the untransformed f, like every run
                best, best_stop = curve[-1], curve_stop[-1]
                for arm, cv in (("budget", curve), ("stop", curve_stop)):
                    curve_rows += [{"seed": seed, "a": a, "b": b, "arm": arm, "eval": i + 1, "best_f": v} for i, v in enumerate(cv)]
                first_mean, max_mean = compare(means, ref_means)
                first_sigma, max_sigma = compare(sigmas, ref_sigmas)
                firsts = [i for i in (first_mean, first_sigma) if i is not None]
                rows.append({"seed": seed, "a": a, "b": b,
                             "first_div": min(firsts, default=None), "steps": len(means),
                             "max_diff_mean": max_mean, "max_diff_sigma": max_sigma,
                             "stop_gen": len(stopped_sigmas), "stop_reason": ", ".join(stop),
                             "best_f": best, "gap": log_gap(best, ref_best),
                             "best_f_stop": best_stop, "gap_stop": log_gap(best_stop, ref_best_stop)})
    return rows, curve_rows


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, curve_rows = experiment_cma()
    write_csv(rows, RESULTS / "cma.csv")
    write_csv(curve_rows, RESULTS / "cma_curves.csv")
    write_csv(summarise_gaps(rows, ("a", "b")) + summarise_gaps(rows, ("a", "b"), "gap_stop"), RESULTS / "cma_gaps.csv")
    plot_first_divergence({"CMA-ES, d = 5": rows}, RESULTS / "first_divergence_cma.png")
    plot_best_so_far(curve_rows, RESULTS / "best_so_far_cma.png", col="a", line="b", row="arm", ref={"a": 1, "b": 0})
    # the report's H1 figure: the fixed-budget search only (the default-stopping row is E1's story)
    plot_best_so_far([r for r in curve_rows if r["arm"] == "budget"], RESULTS / "best_so_far_cma_budget.png", col="a", line="b", ref={"a": 1, "b": 0})
