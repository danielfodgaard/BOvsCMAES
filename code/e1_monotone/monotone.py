"""E1: monotone transforms g(f), CMA-ES (pycma) and BO (BoTorch), d = 5. Run with: python code/e1_monotone/monotone.py

Every method runs on g(f) for each g in TRANSFORMS with the same seed, start point and initial design as on f.
CMA-ES uses H1's runner: A1 has f-based termination off and a fixed budget, and is compared bit for bit with
the run on f; A2 uses pycma's default termination, whose thresholds are absolute differences in g.
BO uses H1's runner with LogEI: the default arm fits the GP to g(f), the rank arm to the ranks of g(f) (Shirakawa),
the low-noise arm is the default with BoTorch's noise floor lowered from 1e-4 to NOISE_FLOOR (on f, the default floor
is a noise std of about 6 in f units, 80x the best f reached).
Always scored on the untransformed f: gap = log10 best f on g - log10 best f on f, same seed and arm.

Outputs in results/e1_monotone/: cma.csv and bo.csv (one row per run), gaps.csv (median and max |gap| per
transform with the seed null band), bo_evals.csv (every BO evaluation: point, f, best-so-far f and the fitted
lengthscales, noise and incumbent_gap of the GP that proposed it; evals 1-10 are the initial design), cma_curves.csv, and the
figures gap.png, best_so_far_bo.png, best_so_far_cma.png, gp_fit.png and best_so_far_{bo,cma}_selected.png
(log f, f^4 and 1000f only, the report's figures).
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit deprecation raised inside botorch's import

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from scipy.stats import gaussian_kde, norm, rankdata

sys.path.insert(0, str(Path(__file__).parents[1] / "invariance_check"))  # reuse H1's runners and helpers
from invariance_bo import BO_INIT, TOL, run_bo
from invariance_cma import NO_TERMINATION, run_cma
from invariance_util import best_so_far, compare, log_gap, make_problem, plot_best_so_far, summarise_gaps, write_csv

D, SEEDS = 5, range(10)
BO_BUDGET = 90  # 10 Sobol + 80 iterations (results/e1_monotone_run1/ used 100)
POPSIZE = 8  # set in run_cma; CMA curves are saved once per generation to keep the CSV small
C_SAT = 100  # about the 1 % quantile of f over the box, so g > 0.5 on 99 % of it
NOISE_FLOOR = 1e-8  # noise variance bound for the low-noise arm, standardised units: noise std 1e-4 of the spread
TRANSFORMS = {  # g applied to f >= 0; every one is strictly increasing
    "log f": lambda v: np.log(max(v, 1e-300)),  # the usual real-world transform; floor only guards f == 0 exactly
    "f^0.25": lambda v: v**0.25,
    "f^0.5": lambda v: v**0.5,
    "f^1": lambda v: v,
    "f^2": lambda v: v**2,
    "f^4": lambda v: v**4,
    f"f/(f+{C_SAT})": lambda v: v / (v + C_SAT),
    "1000f": lambda v: 1000 * v,  # affine control (H1): same problem in exact arithmetic, so its gap is the rounding-only null
}
REF = "f^1"
OFFSETS = {  # R's controls for the warp arm only: affine, so default BO (H1) and ranks cannot tell them from f, but Box-Cox can
    "f+10": lambda v: v + 10,
    "f+1000": lambda v: v + 1000,
    "f-10": lambda v: v - 10,  # negative values once f < 10, which forces the Yeo-Johnson branch
}
SHIFTED_LOGS = {  # follow-up to R: log f with the singular point moved off the optimum; run on the default arm only
    f"log(f+{c:g})": (lambda v, c=c: np.log(v + c)) for c in (0.1, 10, 1000)
}


def lebesgue(X01, y):
    """Shirakawa's (2016) estimate of Akimoto's Lebesgue-measure objective, -vol{x : f(x) at least as good}^(2/d).

    The volume is a mean over the evaluated points at least as good, each weighted by 1 / (kernel density estimate of
    the sampling density at it, Scott's rule). It uses the values only through their order, so it is exactly invariant.
    """
    p = gaussian_kde(X01.T)(X01.T)
    return -(((y[None, :] >= y[:, None]) / p).mean(axis=1) ** (2 / X01.shape[1]))


FIT_TO = {  # follow-up to E1/R: invariant GP targets other than plain ranks (run_bo's fit_to hook)
    "bo_nscore": lambda X01, y: norm.ppf((rankdata(y) - 0.5) / len(y)),  # normal scores: ranks with the tails stretched
    "bo_lebesgue": lebesgue,
}
BIG_BOX = [[-10.0] * D, [10.0] * D]  # R follow-up's boundary-stall test: the same x*, twice the box
BO_ARM_KWARGS = {  # run_bo's keyword arguments per arm; E1's three arms first, the rest are R's and R follow-up's
    "bo_default": {},
    "bo_rank": {"rank": True},
    "bo_lownoise": {"noise_floor": NOISE_FLOOR},
    "bo_warp": {"warp": True},
    **{arm: {"fit_to": fit_to} for arm, fit_to in FIT_TO.items()},
    "bo_postinc": {"incumbent": "posterior"},
    "bo_bigbox": {"bounds": BIG_BOX},
}
BO_ARMS = ["bo_default", "bo_rank", "bo_lownoise"]
RESULTS = Path(__file__).parents[2] / "results" / "e1_monotone"


def experiment_cma():
    """A1 and A2 for every seed and transform. Returns (rows, curve_rows).

    rows: first_div (earliest generation where mean or sigma differs bit for bit from the run on f, None if
    never), max_diff_mean/sigma, best_f and gap after the full budget (A1), stop_evals, stop_reason,
    best_f_stop and gap_stop (A2).
    """
    rows, curve_rows = [], []
    for seed in SEEDS:
        f = make_problem(D, seed)
        runs = {name: (run_cma(lambda x: g(f(x)), seed, D, opts=NO_TERMINATION), run_cma(lambda x: g(f(x)), seed, D, use_stop=True))
                for name, g in TRANSFORMS.items()}
        (ref_means, ref_sigmas, _, ref_X), (*_, ref_X_stop) = runs[REF]
        ref_best, ref_best_stop = best_so_far(f, ref_X)[-1], best_so_far(f, ref_X_stop)[-1]
        for name, ((means, sigmas, _, X), (_, _, stop, X_stop)) in runs.items():
            curve, curve_stop = best_so_far(f, X), best_so_far(f, X_stop)  # scored on the untransformed f, like every run
            first_mean, max_mean = compare(means, ref_means)
            first_sigma, max_sigma = compare(sigmas, ref_sigmas)
            rows.append({"method": "cma", "seed": seed, "transform": name,
                         "first_div": min((i for i in (first_mean, first_sigma) if i is not None), default=None),
                         "max_diff_mean": max_mean, "max_diff_sigma": max_sigma,
                         "best_f": curve[-1], "gap": log_gap(curve[-1], ref_best),
                         "stop_evals": len(X_stop), "stop_reason": ", ".join(stop) or "budget",
                         "best_f_stop": curve_stop[-1], "gap_stop": log_gap(curve_stop[-1], ref_best_stop)})
            for arm, cv in (("budget", curve), ("stop", curve_stop)):
                curve_rows += [{"seed": seed, "transform": name, "arm": arm, "eval": i + 1, "best_f": cv[i]}
                               for i in range(POPSIZE - 1, len(cv), POPSIZE)]
    return rows, curve_rows


def run_case(method, seed, name):
    """One BO run on g(f). Top level so a worker process can run it."""
    torch.set_num_threads(1)  # tiny problems: many single-threaded runs in parallel beat one multi-threaded run
    f, g = make_problem(D, seed), {**TRANSFORMS, **OFFSETS, **SHIFTED_LOGS}[name]
    return run_bo(lambda x: g(f(x)), seed, LogExpectedImprovement, D, BO_BUDGET, **BO_ARM_KWARGS[method])


def experiment_bo(arms=BO_ARMS, names=tuple(TRANSFORMS), seeds=SEEDS):
    """Every arm (any key of BO_ARM_KWARGS), seed and transform; names must include REF. Returns (rows, eval_rows).

    rows: best_f and gap at the end of the budget, first_div (earliest iteration where the proposal differs
    from the run on f by more than TOL, None if never) and max_diff_proposal, acquisition retries, and the
    fitted noise and lengthscales of the last GP. eval_rows: one per evaluation, see the module docstring.
    """
    cases = [(method, seed, name) for method in arms for seed in seeds for name in names]
    with ProcessPoolExecutor() as pool:
        logs = dict(zip(cases, pool.map(run_case, *zip(*cases))))

    rows, eval_rows = [], []
    for (method, seed, name), log in logs.items():
        f, ref = make_problem(D, seed), logs[method, seed, REF]
        fx = [f(x) for x in log["X"]]
        curve = np.minimum.accumulate(fx)
        first_div, max_diff = compare(log["proposal"], ref["proposal"], TOL)
        rows.append({"method": method, "seed": seed, "transform": name,
                     "best_f": curve[-1], "gap": log_gap(curve[-1], best_so_far(f, ref["X"])[-1]),
                     "first_div": first_div, "max_diff_proposal": max_diff, "retries": sum(log["retries"]),
                     "noise_final": log["noise"][-1], **{f"ls{j + 1}_final": v for j, v in enumerate(log["lengthscale"][-1])},
                     **({"lam_final": log["lam"][-1], "family_final": log["family"][-1],
                         "boxcox_fits": log["family"].count("boxcox")} if "lam" in log else {})})
        for i, (x, v, best) in enumerate(zip(log["X"], fx, curve)):
            it = i - BO_INIT  # index of the GP fit that proposed this point; the initial design has none
            gp = [log["noise"][it], *log["lengthscale"][it], log["incumbent_gap"][it]] if it >= 0 else [""] * (D + 2)
            warp = {"lam": log["lam"][it] if it >= 0 else "", "family": log["family"][it] if it >= 0 else ""} if "lam" in log else {}
            eval_rows.append({"method": method, "seed": seed, "transform": name, "eval": i + 1, "f": v, "best_f": best,
                              **{f"x{j + 1}": c for j, c in enumerate(x)},
                              **dict(zip(["noise"] + [f"ls{j + 1}" for j in range(D)] + ["incumbent_gap"], gp)), **warp})
    return rows, eval_rows


ARMS = {  # (method, gap key): legend label, fixed categorical slot, marker as the second encoding
    ("bo_default", "gap"): ("BO, default", "#2a78d6", "o"),
    ("bo_rank", "gap"): ("BO, ranks", "#eb6834", "s"),
    ("bo_lownoise", "gap"): ("BO, low noise floor", "#8e5bd6", "v"),
    ("cma", "gap"): ("CMA-ES, fixed budget", "#1baf7a", "^"),
    ("cma", "gap_stop"): ("CMA-ES, default stopping", "#eda100", "D"),
}


def label(key, value):
    """Readable name for an arm, CMA-ES setting or transform, for titles and legends."""
    if (value, "gap") in ARMS:
        return ARMS[value, "gap"][0]
    if value in ("budget", "stop"):
        return ARMS["cma", "gap" if value == "budget" else "gap_stop"][0]
    if value == "f^1":
        return "f"
    return f"$f^{{{value[2:]}}}$" if str(value).startswith("f^") else str(value)


def _style(ax, names, ylabel, title):
    ax.set_xticks(range(len(names)), [label("transform", n) for n in names], rotation=30)
    ax.set(ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def _strip(ax, names, arms, values):
    """Per arm, one small dot per seed and a large marker at the median, offset sideways within each transform."""
    for k, arm in enumerate(arms):
        label, color, marker = ARMS[arm]
        for i, name in enumerate(names):
            v = values(arm, name)
            x = i + 0.8 * (k - (len(arms) - 1) / 2) / len(arms)
            ax.scatter(np.full(len(v), x), v, s=12, color=color, marker=marker, alpha=0.35, linewidth=0)
            ax.scatter(x, np.median(v), s=64, color=color, marker=marker, edgecolor="white", linewidth=1.5,
                       label=label if i == 0 else None, zorder=3)


def plot_gap(rows, path):
    """Left: paired gap per transform and arm. Right: evaluations CMA-ES with default stopping uses before it stops.
    No absolute-level panel: BO's 90 and CMA-ES's 2000 evaluations are different regimes, so only paired gaps compare."""
    by = lambda method, name: [r for r in rows if r["method"] == method and r["transform"] == name]
    names = list(TRANSFORMS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})

    others = [n for n in names if n != REF]
    axes[0].axhline(0, color="#8a8984", linewidth=1)
    _strip(axes[0], others, list(ARMS), lambda arm, name: [r[arm[1]] for r in by(arm[0], name)])
    _style(axes[0], others, "log10 best f on g − log10 best f on f", "Paired gap (0 = invariant; > 0 = worse on g)")

    _strip(axes[1], names, [("cma", "gap_stop")], lambda arm, name: [r["stop_evals"] for r in by("cma", name)])
    _style(axes[1], names, "evaluations used before stopping", "CMA-ES, default stopping")

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_gp_fit(bo_rows, path):
    """Fitted lengthscales of the last GP per dimension (median over seeds) and its noise, per transform and arm.

    Lengthscales are in normalised-input units; the ellipsoid's axis weights grow with the dimension index,
    so on f a well-fitted GP has lengthscales shrinking from dimension 1 to 5.
    """
    names = list(TRANSFORMS)
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # dimension 1 (weight 1) light -> 5 (weight 100) dark
    fig, axes = plt.subplots(1, len(BO_ARMS) + 1, figsize=(5.3 * (len(BO_ARMS) + 1), 4.4), sharex=True)
    for ax, method in zip(axes, BO_ARMS):
        for j, color in enumerate(ramp):
            med = [np.median([r[f"ls{j + 1}_final"] for r in bo_rows if r["method"] == method and r["transform"] == n]) for n in names]
            ax.plot(range(len(names)), med, color=color, marker="o", markersize=6, linewidth=2, label=f"dim {j + 1}")
        ax.set_yscale("log")
        _style(ax, names, "lengthscale (normalised inputs)", f"{ARMS[method, 'gap'][0]}: last GP's lengthscales, median")
    _strip(axes[-1], names, [(m, "gap") for m in BO_ARMS],
           lambda arm, name: [r["noise_final"] for r in bo_rows if r["method"] == arm[0] and r["transform"] == name])
    axes[-1].set_yscale("log")
    _style(axes[-1], names, "noise variance (standardised outputs)", "Last GP's fitted noise")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    cma_rows, cma_curves = experiment_cma()
    write_csv(cma_rows, RESULTS / "cma.csv")
    write_csv(cma_curves, RESULTS / "cma_curves.csv")
    bo_rows, bo_evals = experiment_bo()
    write_csv(bo_rows, RESULTS / "bo.csv")
    write_csv(bo_evals, RESULTS / "bo_evals.csv")

    ref = {"transform": REF}
    gaps = [summarise_gaps(cma_rows, ("method", "transform"), k, ref) for k in ("gap", "gap_stop")]
    write_csv(sum(gaps, summarise_gaps(bo_rows, ("method", "transform"), ref=ref)), RESULTS / "gaps.csv")
    plot_gap(cma_rows + bo_rows, RESULTS / "gap.png")
    plot_gp_fit(bo_rows, RESULTS / "gp_fit.png")
    plot_best_so_far(bo_evals, RESULTS / "best_so_far_bo.png", col="transform", line="method", row="method", ref=ref, label=label)
    plot_best_so_far(cma_curves, RESULTS / "best_so_far_cma.png", col="transform", line="arm", row="arm", ref=ref, label=label)
    # the report's selection: one compressing g, one expanding g and the affine control
    shown = ("log f", "f^4", "1000f", REF)
    plot_best_so_far([r for r in bo_evals if r["transform"] in shown], RESULTS / "best_so_far_bo_selected.png", col="transform", line="method", row="method", ref=ref, label=label)
    plot_best_so_far([r for r in cma_curves if r["transform"] in shown], RESULTS / "best_so_far_cma_selected.png", col="transform", line="arm", row="arm", ref=ref, label=label)
