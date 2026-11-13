"""E2: rescaling the axes of x, CMA-ES (pycma) and BO (BoTorch), d = 5. Run with: python code/e2_rescaling/rescaling.py

Five problems per seed, all with the same x* draw, CMA-ES start point and Sobol design (mapped along where the box moves):
  sphere      f(z) = |z - z*|^2 on [-5, 5]^d: the reference.
  stretched   the sphere in new units x = u z with u_i = 1 / s_i, s_i = 10^((i-1)/(d-1)) in [1, 10]: f(x) = sum s_i^2 (x_i - x*_i)^2
              with x* = u z* and box u_i [-5, 5]. The same problem, so an invariant method must behave like it does on the sphere
              (Hansen's diagonal invariance, given the initial step sizes per axis).
  uniform_0.01, uniform_100
              the sphere in new units x = k z with one factor k on every axis (k = 0.01: box [-0.05, 0.05]^d; k = 100: box
              [-500, 500]^d), x* and the start point mapped along. Hansen's scale invariance, given sigma0 scaled by k. Needed
              because the stretched problem tests the per-axis case only.
  ellipsoid   the project's ellipsoid, same weights s_i^2 but box [-5, 5]^d and x* in [-2, 2]^d kept (H1/E1's problem):
              the sphere stretched without stretching the box, so relative to f the box is 10x wider on the stiffest axis.
Arms. CMA-ES: default (pycma defaults, sigma0 = 2.5; with bounds pycma caps each coordinate's std at a third of that
coordinate's box width, option maxstd_boundrange = 1/3, which on the stretched box is a per-axis multiplier of 1 to 0.13, so
the box itself tells pycma the scale; the cap can only shrink sigma, so on uniform_100, where sigma0 = 2.5 is 100x too small,
it does nothing), CMA_stds (sigma0 multiplied per axis by the units factor u_i; Hansen's "transform the initial state too"),
CMA_stds_nobounds (the same without bounds: the clean test of Hansen's claim, since pycma's bound handler is not
scale-free; on uniform_0.01 its margin 0.05 max(|lb|, 1) is the whole half-width) and nocap (the cap switched off with
maxstd = inf: sigma0 = 2.5 on every axis, the user's one number kept after the change of units). BO: Normalize (inputs normalised to the box,
as H1/E1 and the BoTorch tutorials) vs raw (no input transform, BoTorch's literal default: the GP and its lengthscale prior
see x in its own units).
Every setting is run on all five problems with a fixed budget (CMA-ES 2000 evaluations, termination off; BO 10 Sobol + 80).
Measures: gap = log10 best f - log10 best f on the sphere, same method, arm and seed; evaluations to reach each target in
TARGETS (the cost of the defaults); for the rescaled problems, where the run first departs from the sphere run (first_div):
for CMA-ES the first generation that ranks its population differently, which is scale-free and ignores rounding that never
changes a ranking (an absolute tolerance on the mean cannot see a difference once the mean is within it of the optimum);
for BO the first proposal that differs from the sphere run's, mapped back to sphere units (x / u), by more than TOL.

Outputs in results/e2_rescaling/: runs.csv (one row per run, with the last GP's lengthscales and noise for BO; the "setting" column is the arm),
curves.csv (best-so-far f per evaluation; CMA-ES once per generation), gaps.csv, summary.csv (per arm and problem: median
gap, Wilcoxon p, seeds worse, and the paired extra evaluations to each target with their p; see summarise_runs),
kernel_check.csv (why raw-input BO fails at u = 100, see kernel_check), best_so_far_cma.png, best_so_far_bo.png.
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit deprecation raised inside botorch's import

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions import InputDataWarning
from botorch.utils.sampling import draw_sobol_samples
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=InputDataWarning)  # the raw arm deliberately feeds the GP un-normalised inputs
warnings.filterwarnings("ignore", message="ValueWarning")  # pycma's warning, deliberately triggered by the nocap arm

sys.path.insert(0, str(Path(__file__).parents[1] / "invariance_check"))  # reuse H1's runners and helpers
from invariance_bo import BO_INIT, TOL, run_bo
from invariance_cma import CMA_BUDGET, NO_TERMINATION, run_cma
from invariance_util import BOX, best_so_far, compare, log_gap, make_problem, plot_best_so_far, start_point, summarise_gaps, write_csv

D = 5
SEEDS = {"cma": range(50), "bo": range(10)}  # CMA-ES: 50, since the seed spread spans two orders of magnitude and 10 paired seeds cannot resolve one
BO_BUDGET, POPSIZE = 90, 8  # as E1; CMA-ES budget is run_cma's 2000
S = 10 ** (np.arange(D) / (D - 1))  # axis i is stretched by s_i; s^2 = 1..100 are the ellipsoid's weights
TARGETS = [1, 1e-2, 1e-4, 1e-8]
# x = u z with u = 1 / INV[name]; f is computed as sphere(INV * x), not sphere(x / u): the two differ in the last bit,
# which BO amplifies into different trajectories
INV = {"stretched": S, "uniform_0.01": np.full(D, 100.0), "uniform_100": np.full(D, 0.01)}
UNITS = {name: 1 / inv for name, inv in INV.items()}
PROBLEMS = ["sphere", *UNITS, "ellipsoid"]
ARMS = {"cma": ["default", "CMA_stds", "CMA_stds_nobounds", "nocap"], "bo": ["Normalize", "raw"]}
CMA_OPTS = {"default": lambda scale: {}, "CMA_stds": lambda scale: {"CMA_stds": scale},
            "CMA_stds_nobounds": lambda scale: {"CMA_stds": scale, "bounds": None}, "nocap": lambda scale: {"maxstd": np.inf}}
RESULTS = Path(__file__).parents[2] / "results" / "e2_rescaling"


def problem(name, seed):
    """Returns (f, box as a 2 x d array, CMA-ES start point, units factor u per axis, which CMA_stds multiplies sigma0 by).
    z0 is run_cma's own default draw."""
    sphere, z0 = make_problem(D, seed, alpha=1), start_point(D, seed)
    if name == "sphere":
        return sphere, np.outer(BOX, np.ones(D)), z0, np.ones(D)
    if name in UNITS:
        u = UNITS[name]
        return (lambda x: sphere(INV[name] * np.asarray(x))), np.outer(BOX, u), z0 * u, u  # np.asarray: BO passes torch tensors
    return make_problem(D, seed), np.outer(BOX, np.ones(D)), z0, 1 / S  # the ellipsoid in the sphere's box: same weights as stretched


def run_case(method, arm, name, seed):
    """One run. Top level so a worker process can run it. Returns a dict with X (evaluated points) plus the trace."""
    torch.set_num_threads(1)  # tiny problems: many single-threaded runs in parallel beat one multi-threaded run
    f, box, x0, scale = problem(name, seed)
    if method == "cma":
        opts = {**NO_TERMINATION, "bounds": box.tolist(), **CMA_OPTS[arm](scale)}
        means, sigmas, _, X = run_cma(f, seed, D, opts=opts, x0=x0)
        return {"X": X, "means": means, "sigmas": sigmas}
    return run_bo(f, seed, LogExpectedImprovement, D, BO_BUDGET, bounds=box, normalize=arm == "Normalize")


def evals_to(curve):
    return {f"evals_to_{t:g}": next((i + 1 for i, v in enumerate(curve) if v <= t), None) for t in TARGETS}


def experiment():
    """Every method, arm, problem and seed in parallel. Returns (rows, curve_rows)."""
    cases = [(m, arm, name, seed) for m, arms in ARMS.items() for arm in arms for name in PROBLEMS for seed in SEEDS[m]]
    with ProcessPoolExecutor() as pool:
        logs = dict(zip(cases, pool.map(run_case, *zip(*cases))))

    rows, curve_rows = [], []
    for (method, arm, name, seed), log in logs.items():
        f, ref = problem(name, seed)[0], logs[method, arm, "sphere", seed]
        curve = best_so_far(f, log["X"])
        ref_best = best_so_far(problem("sphere", seed)[0], ref["X"])[-1]
        if name in UNITS:
            u = UNITS[name]
            if method == "cma":  # with bounds pycma's boundary margin 0.05 max(|lb|, 1) is not scale-free: first_div shows where it bites
                first_div, max_diff = first_rank_flip(f, log["X"], problem("sphere", seed)[0], ref["X"]), None
            else:  # self-check of the mapping: BO's Sobol design is the sphere's in new units
                assert np.allclose(log["X"][:BO_INIT] / u, ref["X"][:BO_INIT])
                first_div, max_diff = compare(np.array(log["proposal"]) / u, ref["proposal"], TOL)
        else:
            first_div, max_diff = None, None
        rows.append({"method": method, "setting": arm, "problem": name, "seed": seed, "best_f": curve[-1],
                     "gap": log_gap(curve[-1], ref_best), **evals_to(curve), "first_div": first_div, "max_diff_mapped": max_diff,
                     **({"noise_final": log["noise"][-1], **{f"ls{j + 1}_final": v for j, v in enumerate(log["lengthscale"][-1])}}
                        if method == "bo" else {"noise_final": None, **{f"ls{j + 1}_final": None for j in range(D)}})})
        step = POPSIZE if method == "cma" else 1
        curve_rows += [{"method": method, "setting": arm, "problem": name, "seed": seed, "eval": i + 1, "best_f": curve[i]}
                       for i in range(step - 1, len(curve), step)]
    return rows, curve_rows


def kernel_check():
    """Why raw-input BO fails at u = 100: the RBF kernel between its 10 Sobol design points at BoTorch's initial
    lengthscale, the mode of its LogNormal(sqrt(2) + log(d)/2, sqrt(3)) prior. One row per BO seed: the smallest distance
    between two design points and the largest off-diagonal kernel value; 0.0 means the GP sees no correlation between
    any two points, so the marginal likelihood is flat in the lengthscale and the fit never moves (runs.csv: every
    raw u = 100 run ends with lengthscale 0.458). The design is run_bo's own (draw_sobol_samples, same seed and box)."""
    ls0 = float(np.exp(np.sqrt(2) + np.log(D) / 2 - 3))
    rows = []
    for seed in SEEDS["bo"]:
        box = torch.tensor(problem("uniform_100", seed)[1], dtype=torch.double)
        dist = pdist(draw_sobol_samples(box, n=BO_INIT, q=1, seed=seed).squeeze(1).numpy())
        rows.append({"seed": seed, "initial_lengthscale": ls0, "min_distance": float(dist.min()),
                     "max_kernel_offdiag": float(np.exp(-dist.min() ** 2 / (2 * ls0 ** 2)))})
    return rows


def first_rank_flip(f, X, f_ref, X_ref, popsize=POPSIZE):
    """First CMA-ES generation in which the run ranks its population differently from the reference run (None if never).
    While every ranking agrees, the two runs make the same updates and differ only by rounding."""
    n = min(len(X), len(X_ref)) // popsize * popsize
    rank = lambda f, X: np.argsort(np.reshape([f(x) for x in X[:n]], (-1, popsize)), axis=1, kind="stable")
    differing = np.flatnonzero((rank(f, X) != rank(f_ref, X_ref)).any(axis=1))
    return int(differing[0]) if len(differing) else None


def summarise_runs(rows, ref, budgets):
    """One row per method, setting and non-reference problem: median gap with its paired Wilcoxon p and the seeds worse,
    and per target the median paired difference in evaluations to reach it (run minus reference, same seed) with its
    Wilcoxon p. A run that never reaches a target counts as budget + 1 evaluations, so no pair is dropped and the
    difference is a lower bound where that happens (censored_* counts the pairs with a censored run). Printed too."""
    key = lambda r: (r["method"], r["setting"], r["problem"])
    pval = lambda d: float(wilcoxon(d).pvalue) if any(d) else float("nan")
    reached = lambda r, t, budget: r[f"evals_to_{t:g}"] if r[f"evals_to_{t:g}"] is not None else budget + 1
    refs = {(r["method"], r["setting"], r["seed"]): r for r in rows if r["problem"] == ref}
    out = []
    for k in dict.fromkeys(key(r) for r in rows if r["problem"] != ref):
        rs = [r for r in rows if key(r) == k]
        d = [r["gap"] for r in rs]
        row = {"method": k[0], "setting": k[1], "problem": k[2], "seeds": len(rs), "gap_median": float(np.median(d)),
               "gap_p": pval(d), "seeds_worse": sum(x > 0 for x in d)}
        for t in TARGETS:
            pairs = [(r, refs[r["method"], r["setting"], r["seed"]]) for r in rs]
            e = [reached(r, t, budgets[k[0]]) - reached(q, t, budgets[k[0]]) for r, q in pairs]
            row |= {f"extra_evals_to_{t:g}": float(np.median(e)), f"p_evals_{t:g}": pval(e),
                    f"censored_{t:g}": sum(r[f"evals_to_{t:g}"] is None or q[f"evals_to_{t:g}"] is None for r, q in pairs)}
        out.append(row)
    print(f"{'method':6} {'setting':17} {'problem':12} {'gap':>6} {'p':>6} {'worse':>6}  extra evals to target (p, censored pairs)")
    for s in out:
        print(f"{s['method']:6} {s['setting']:17} {s['problem']:12} {s['gap_median']:>+6.2f} {s['gap_p']:>6.3f} {s['seeds_worse']:>3}/{s['seeds']:<2}  "
              + "  ".join(f"{t:g}: {s[f'extra_evals_to_{t:g}']:+.0f} ({s[f'p_evals_{t:g}']:.3f}, {s[f'censored_{t:g}']})" for t in TARGETS))
    return out


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, curve_rows = experiment()
    write_csv(rows, RESULTS / "runs.csv")
    write_csv(curve_rows, RESULTS / "curves.csv")
    gaps = summarise_gaps(rows, ("method", "setting", "problem"), ref={"problem": "sphere"})
    write_csv(gaps, RESULTS / "gaps.csv")
    label = lambda k, v: str(v)
    for method in ARMS:
        plot_best_so_far([r for r in curve_rows if r["method"] == method], RESULTS / f"best_so_far_{method}.png",
                         col="setting", line="problem", label=label)

    write_csv(summarise_runs(rows, "sphere", {"cma": CMA_BUDGET, "bo": BO_BUDGET}), RESULTS / "summary.csv")
    write_csv(kernel_check(), RESULTS / "kernel_check.csv")

