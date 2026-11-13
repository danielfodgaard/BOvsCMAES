"""E3: rotating x, CMA-ES (pycma) and BO (BoTorch), d = 5. Run with: python code/e3_rotation/rotation.py

Two problems per seed, with the same x* and Sobol design as E1/E2's ellipsoid:
  ellipsoid   f(x) = sum w_i (x_i - x*_i)^2, w_i = 100^((i-1)/(d-1)), x* in [-2, 2]^d, box [-5, 5]^d: the reference.
  rotated     the same function in rotated coordinates y = R x, R a seeded random rotation (a new one per seed):
              f_R(y) = f(R^T y), optimum y* = R x* (norm <= 2 sqrt(d) < 5, so inside the box), CMA-ES start R x0.
              The box is kept at [-5, 5]^d, since a rotated box is not a box; BO's Sobol design is therefore the same points.
CMA-ES start x0: uniform in the ball of radius 5 (start_in_ball), not in the box as in E1/E2. The ball lies inside the box
and is closed under rotation, so R x0 is a valid start and every rotated run is a paired copy of its reference run. (E1's draw
from the box would leave it after rotation in 16/50 seeds and be clipped into it, which would unpair those runs.)
Settings. CMA-ES: bounds (pycma defaults with the box, as H1/E1; pycma's boundary handling is coordinate-wise, so it cannot
be rotation-invariant), bounds_nocap (the box kept but pycma's per-coordinate std cap maxstd_boundrange switched off with
maxstd = inf, to separate the cap from the bound transform) and no_bounds (bounds off, the clean test of Hansen's claim: with
the start point rotated along, the run should be the reference run rotated). BO: ARD (BoTorch's default kernel, one lengthscale per axis, which is
expected to notice the rotation) and isotropic (the same RBF kernel and prior with a single lengthscale, the control for
that prediction; the design and the acquisition optimiser's box are still axis-aligned).
Fixed budgets: CMA-ES 2000 evaluations, termination off; BO 10 Sobol + 140, with the gap after 90 (E1/E2's budget) kept
as gap_90. CMA-ES runs 50 seeds (0.2 s each), because with a fixed budget the seed spread of the
final f is two orders of magnitude and 10 seeds cannot resolve that; BO 10.
Measures: gap = log10 best f - log10 best f on the ellipsoid, same method, setting and seed, against the spread over seeds;
evaluations to reach each target in TARGETS. No step-by-step trajectory comparison as in E1/E2: for a fixed seed the Gaussian
samples are not rotated along with the problem, so CMA-ES's rotation invariance holds in distribution, not bit for bit.

Outputs in results/e3_rotation/: runs.csv (one row per run; BO rows carry the last GP's lengthscales and noise; the
isotropic kernel has one lengthscale, in ls1_final), curves.csv, gaps.csv, summary.csv (as E2's: gap, its p, and paired
extra evaluations to each target with their p, never-reached runs counted as budget + 1), summary_90.csv (BO's gap after 90 evaluations, see summary_90),
best_so_far_cma.png and best_so_far_bo.png.
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from math import log, sqrt
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit deprecation raised inside botorch's import

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel
from gpytorch.priors import LogNormalPrior
from scipy.stats import special_ortho_group, wilcoxon

sys.path.insert(0, str(Path(__file__).parents[1] / "invariance_check"))  # reuse H1's runners and helpers
sys.path.insert(0, str(Path(__file__).parents[1] / "e2_rescaling"))  # and E2's targets and summary
from invariance_bo import BO_INIT, run_bo
from invariance_cma import CMA_BUDGET, NO_TERMINATION, run_cma
from invariance_util import BOX, best_so_far, log_gap, make_problem, plot_best_so_far, summarise_gaps, write_csv
from rescaling import POPSIZE, evals_to, summarise_runs

D = 5
BO_BUDGET, BO_EARLY = 150, 90  # 150 so the rotated run can catch up if it is only slower; the gap at E1/E2's 90 is kept as gap_90
SEEDS = {"cma": range(50), "bo": range(10)}
PROBLEMS = ["ellipsoid", "rotated"]
ARMS = {"cma": ["bounds", "bounds_nocap", "no_bounds"], "bo": ["ARD", "isotropic"]}
CMA_OPTS = {"bounds": {}, "bounds_nocap": {"maxstd": np.inf}, "no_bounds": {"bounds": None}}
RESULTS = Path(__file__).parents[2] / "results" / "e3_rotation"


def isotropic_kernel():
    """BoTorch's default RBF kernel and lengthscale prior (Hvarfner et al. 2024) with one lengthscale instead of one per axis."""
    prior = LogNormalPrior(loc=sqrt(2) + log(D) / 2, scale=sqrt(3))
    return RBFKernel(lengthscale_prior=prior, lengthscale_constraint=GreaterThan(2.5e-2, transform=None, initial_value=prior.mode))


BO_KERNELS = {"ARD": None, "isotropic": isotropic_kernel}


def start_in_ball(seed, radius=BOX[1]):
    """CMA-ES start, uniform in the ball of the given radius, from the same seed stream as start_point ([seed, 2])."""
    rng = np.random.default_rng([seed, 2])
    v = rng.standard_normal(D)
    return radius * rng.uniform() ** (1 / D) * v / np.linalg.norm(v)


def problem(name, seed):
    """Returns (f, CMA-ES start point, rotation R). The rotation has its own seed stream ([seed, 1])."""
    f, x0 = make_problem(D, seed), start_in_ball(seed)
    if name == "ellipsoid":
        return f, x0, np.eye(D)
    R = special_ortho_group.rvs(D, random_state=np.random.default_rng([seed, 1]))
    return (lambda y: f(R.T @ np.asarray(y, dtype=float).ravel())), R @ x0, R


def run_case(method, arm, name, seed):
    """One run. Top level so a worker process can run it. Returns a dict with X (evaluated points) plus the trace."""
    torch.set_num_threads(1)  # tiny problems: many single-threaded runs in parallel beat one multi-threaded run
    f, x0, _ = problem(name, seed)
    if method == "cma":
        assert np.all(np.abs(x0) < BOX[1]), x0  # the ball start stays inside the box after rotation, so nothing is clipped
        means, sigmas, _, X = run_cma(f, seed, D, opts={**NO_TERMINATION, **CMA_OPTS[arm]}, x0=x0)
        return {"X": X, "means": means, "sigmas": sigmas}
    return run_bo(f, seed, LogExpectedImprovement, D, BO_BUDGET, kernel=BO_KERNELS[arm])


def summary_90(rows):
    """BO's gap on the rotated problem after BO_EARLY evaluations (E1/E2's budget), per kernel: median, seeds worse, Wilcoxon p."""
    out = []
    for arm in ARMS["bo"]:
        d = [r["gap_90"] for r in rows if (r["method"], r["setting"], r["problem"]) == ("bo", arm, "rotated")]
        out.append({"setting": arm, "evaluations": BO_EARLY, "gap_median": float(np.median(d)), "seeds_worse": sum(x > 0 for x in d),
                    "seeds": len(d), "p": float(wilcoxon(d).pvalue)})
    return out


def experiment():
    """Every method, setting, problem and seed in parallel. Returns (rows, curve_rows)."""
    cases = [(m, arm, name, seed) for m, arms in ARMS.items() for arm in arms for name in PROBLEMS for seed in SEEDS[m]]
    with ProcessPoolExecutor() as pool:
        logs = dict(zip(cases, pool.map(run_case, *zip(*cases))))

    rows, curve_rows = [], []
    for (method, arm, name, seed), run in logs.items():
        f, ref = problem(name, seed)[0], logs[method, arm, "ellipsoid", seed]
        curve = best_so_far(f, run["X"])
        ref_best = best_so_far(problem("ellipsoid", seed)[0], ref["X"])[-1]
        if name == "rotated" and method == "bo":  # self-check: the box is the same, so the Sobol design is the same points
            assert np.array_equal(run["X"][:BO_INIT], ref["X"][:BO_INIT])
        ls = list(run["lengthscale"][-1]) if method == "bo" else []
        rows.append({"method": method, "setting": arm, "problem": name, "seed": seed, "best_f": curve[-1],
                     "gap": log_gap(curve[-1], ref_best), **evals_to(curve),
                     "gap_90": log_gap(curve[BO_EARLY - 1], best_so_far(problem("ellipsoid", seed)[0], ref["X"][:BO_EARLY])[-1]) if method == "bo" else None,
                     "noise_final": run["noise"][-1] if method == "bo" else None,
                     **{f"ls{j + 1}_final": ls[j] if j < len(ls) else None for j in range(D)}})
        step = POPSIZE if method == "cma" else 1
        curve_rows += [{"method": method, "setting": arm, "problem": name, "seed": seed, "eval": i + 1, "best_f": curve[i]}
                       for i in range(step - 1, len(curve), step)]
    return rows, curve_rows


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, curve_rows = experiment()
    write_csv(rows, RESULTS / "runs.csv")
    write_csv(curve_rows, RESULTS / "curves.csv")
    gaps = summarise_gaps(rows, ("method", "setting", "problem"), ref={"problem": "ellipsoid"})
    write_csv(gaps, RESULTS / "gaps.csv")
    for method in ARMS:
        plot_best_so_far([r for r in curve_rows if r["method"] == method], RESULTS / f"best_so_far_{method}.png",
                         col="setting", line="problem", label=lambda k, v: str(v))
    write_csv(summarise_runs(rows, "ellipsoid", {"cma": CMA_BUDGET, "bo": BO_BUDGET}), RESULTS / "summary.csv")
    s90 = summary_90(rows)
    write_csv(s90, RESULTS / "summary_90.csv")
    for s in s90:
        print(f"bo {s['setting']}: gap at {BO_EARLY} evaluations, median {s['gap_median']:+.2f}, worse in {s['seeds_worse']}/{s['seeds']}, Wilcoxon p = {s['p']:.3f}")
