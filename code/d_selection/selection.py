"""D: can BO choose a good representation of f from its own data? BO (BoTorch), d = 5. Run with:
python code/d_selection/selection.py (after code/e1_monotone/monotone.py and code/r_robustness/robustness.py, whose
default, rank and warp runs it compares against). Unit tests of the criterion: python code/d_selection/test_selection.py.

Question. E1 showed that default BO's result depends on how f is written down (g(f)), and R that learning a power
transform by Box-Cox's criterion (Gaussian-looking values) maps every f^p to a near-log representation, about 2 orders
worse than f itself. D asks whether a different criterion, applied to the same data at no extra evaluations, picks a
representation that undoes g, i.e. turns g(f) back into f, and so recovers default BO's result on f.

Procedure. The black box returns v = g(f(x)) for g in TRANSFORMS (E1's problem, seeds, Sobol designs and budget:
10 Sobol + 80 LogEI iterations). Before every GP fit:
  1. Each candidate h in CANDIDATES that is defined on the observed values is applied to them, and BoTorch's default GP
     (ARD RBF, Standardize, inputs normalised to the box) is fitted to -h(v) (BoTorch maximises).
  2. Each fitted GP is scored by its leave-one-out (LOO) predictive log-likelihood of the observed values, in the units
     of v: sum_i log N(z_i; mu_-i, s2_-i) - n log sd_h + sum_i log |h'(v_i)|. The LOO mean and variance are the closed
     forms of Rasmussen & Williams (2006, eq. 5.12), with the hyperparameters of the fit to all n points (as Jones et al.
     1998 do); z are the standardised targets and sd_h the standard deviation Standardize divided by. The last two terms
     convert densities of z into densities of v (change of variables), so candidates are compared on the same scale.
     Without them a compressing h would win by making every value look easy to predict.
  3. The candidate with the highest score is used for this iteration: run_bo fits the GP to its values (same seed, so
     the same fit as in step 1) and LogEI proposes the next point. Ties go to the earlier candidate (identity first).
Variants: "every" re-selects before every fit; "once" selects after the 10-point design and keeps that h (Jones et al.'s
procedure), unless it stops being defined on the data (a power or log once a value is <= 0), in which case it re-selects
among the defined candidates and records a fallback.

Candidates (on the minimised values v): v^lam for lam in {1, 1/16, 1/8, 1/4, 1/2, 2, 4}, log v, and exp(v - max v).
The grid is chosen so that f and f^4 offer the same representations of f: f^(1/4), f^(1/2), f, f^2, f^4 and log f are
reachable from both (with lam only in {1/4, ..., 4}, f^4's data would offer f^(1/2) but not f^(1/8) of f^4 = f^(1/2),
which would make the two runs' choices incomparable). Powers other than 1 and log need v > 0 and are skipped in iterations where
some v <= 0 (log f once f < 1). exp(v - max v) equals exp(v) up to a factor, which Standardize removes, so it is the exact
inverse of log f; unlike the powers it is not scale-free (on 1000 f it is a different candidate than on f), so the 1000 f
control is exact only while exp is not chosen. A candidate whose GP fit fails (BoTorch gives up after 5 attempts, or the
covariance is not positive definite) counts as undefined for that iteration and is recorded.
What is recorded is the chosen h and the representation of f it amounts to, h(g(f)) up to an affine map (representation()):
f itself is reachable from f and 1000 f (v^1), f^4 (v^(1/4)) and log f (exp), not from f / (f + 100).
Note: on default BO's data from f, the criterion already prefers compressing h (v^(1/2), log) over
f once n >= 30, so D is framed by its level against the baselines, not by whether it finds a "correct" h.

Measures (untransformed f after 90 evaluations): level = log10 best f; gap = level on g - level on f, same variant and
seed; paired comparison with E1's default arm on the same g and on f, E1's rank arm, and R's warp arm on the same g
(sign count and Wilcoxon p over the 10 seeds; with about 40 tests and n = 10, sign counts lead and p values are not
corrected for multiplicity); share of fits whose representation is f itself. The comparison with the warp arm mixes
criterion and candidate family (exp, the inverse of log f, has no Box-Cox counterpart); f^4 is the clean comparison.

Outputs in results/d_selection/: runs.csv (one row per run), choices.csv (one row per GP fit: chosen h, its
representation of f, failed candidate fits, and every candidate's score, None if undefined), evals.csv (every evaluation: point, f, best-so-far f, chosen h of the fit that
proposed it), summary.csv (per variant and g), level.png (level per g for default, ranks, warp and both D variants),
choices.png (share of fits choosing each candidate per g), and level_report.png and choices_report.png (compact
versions for the report).
"""
import csv
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit deprecation raised inside botorch's import

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.utils.errors import NotPSDError
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parents[1] / "e1_monotone"))
sys.path.insert(0, str(Path(__file__).parents[1] / "invariance_check"))
import monotone as e1
from invariance_bo import BO_INIT, run_bo
from invariance_util import best_so_far, log_gap, make_problem, write_csv

TRANSFORMS = ["f^1", "f^4", "log f", "f/(f+100)", "1000f"]  # E1's names; f^1 is each variant's reference
VARIANTS = ["every", "once"]
POWERS = [1, 1 / 16, 1 / 8, 0.25, 0.5, 2, 4]  # identity first, so it wins ties
P = {"f^1": 1, "1000f": 1, "f^4": 4}  # g = a f^p
SEEDS = e1.SEEDS
RESULTS = Path(__file__).parents[2] / "results" / "d_selection"
R_RESULTS = Path(__file__).parents[2] / "results" / "r_robustness"


def candidates(v):
    """{name: (h(v), log |h'(v)|)} for every candidate defined on the observed values v (1-D array, minimised)."""
    out = {}
    for lam in POWERS:
        if lam == 1:
            out["v^1"] = (v.copy(), np.zeros_like(v))
        elif v.min() > 0:
            out[f"v^{lam:g}"] = (v**lam, np.log(lam) + (lam - 1) * np.log(v))
    if v.min() > 0:
        out["log"] = (np.log(v), -np.log(v))
    out["exp"] = (np.exp(v - v.max()), v - v.max())
    return {k: (h, j) for k, (h, j) in out.items() if np.isfinite(h).all() and np.isfinite(j).all() and h.std() > 0}


def representation(g, h):
    """The representation of f that candidate h amounts to on data g(f), up to an affine map; "f^1" is f itself."""
    lam = None if h in ("log", "exp") else float(h[2:])
    if g in P:
        return f"f^{lam * P[g]:g}" if lam is not None else ("log f" if h == "log" else f"exp({g})")
    if g == "log f":
        return "f^1" if h == "exp" else ("log log f" if h == "log" else ("log f" if lam == 1 else f"(log f)^{lam:g}"))
    return f"log({g})" if h == "log" else (f"exp({g})" if h == "exp" else (g if lam == 1 else f"({g})^{lam:g}"))


def fit_gp(X01, targets, seed):
    """BoTorch's default GP on inputs already in the unit cube, fitted with run_bo's seeding."""
    X = torch.from_numpy(X01)
    torch.manual_seed(seed)
    gp = SingleTaskGP(X, torch.from_numpy(targets).unsqueeze(1), input_transform=Normalize(X.shape[1], bounds=torch.tensor([[0.0] * X.shape[1], [1.0] * X.shape[1]], dtype=torch.double)))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def loo_terms(gp, X01):
    """Closed-form LOO predictive mean and variance of the standardised targets z, and z itself (R&W eq. 5.12)."""
    with torch.no_grad():
        X = torch.from_numpy(X01)
        K = gp.covar_module(X).to_dense() + gp.likelihood.noise * torch.eye(len(X), dtype=X.dtype)
        z, m = gp.train_targets, gp.mean_module.constant
        K_inv = torch.cholesky_inverse(torch.linalg.cholesky(K))
        var = 1 / torch.diagonal(K_inv)
        mu = z - (K_inv @ (z - m)) * var
    return mu.numpy(), var.numpy(), z.numpy()


def loo_score(gp, X01, log_jac):
    """LOO predictive log-likelihood of the observed values v, in the units of v (see the module docstring, step 2)."""
    mu, var, z = loo_terms(gp, X01)
    sd = float(gp.outcome_transform.stdvs.squeeze())
    return float(np.sum(-0.5 * np.log(2 * math.pi * var) - 0.5 * (z - mu) ** 2 / var) - len(z) * math.log(sd) + np.sum(log_jac))


class Selector:
    """run_bo's fit_to hook: chooses h before each fit and returns the GP's targets -h(v). Records every choice."""

    def __init__(self, variant, seed, g):
        self.variant, self.seed, self.g, self.kept = variant, seed, g, None
        self.log = []  # one dict per fit: n, chosen, fallback, score per candidate

    def __call__(self, X01, y):
        v = -y
        cands = candidates(v)
        fallback = self.variant == "once" and self.kept is not None and self.kept not in cands
        failed = []
        if self.variant == "once" and self.kept in cands:
            chosen, scores = self.kept, {}
        else:
            scores = {}
            for name, (h, j) in cands.items():
                try:
                    scores[name] = loo_score(fit_gp(X01, -h, self.seed), X01, j)
                except (ModelFittingError, NotPSDError, torch.linalg.LinAlgError):
                    failed.append(name)
            chosen = max(scores, key=scores.get)  # dicts keep insertion order and max keeps the first maximum: identity wins ties
            if self.variant == "once":
                self.kept = chosen
        self.log.append({"n": len(v), "chosen": chosen, "representation": representation(self.g, chosen), "fallback": fallback,
                         "failed": " ".join(failed), **{f"score_{k}": scores.get(k) for k in ALL_NAMES}})
        return -cands[chosen][0]


ALL_NAMES = ["v^1", *(f"v^{lam:g}" for lam in POWERS if lam != 1), "log", "exp"]


def run_case(variant, seed, name):
    """One BO run on g(f) with transform selection. Top level so a worker process can run it."""
    torch.set_num_threads(1)
    f, g = make_problem(e1.D, seed), e1.TRANSFORMS[name]
    selector = Selector(variant, seed, name)
    log = run_bo(lambda x: g(f(x)), seed, LogExpectedImprovement, e1.D, e1.BO_BUDGET, fit_to=selector)
    log["choices"] = selector.log
    return log


def experiment():
    """Every variant, seed and transform in parallel. Returns (rows, choice_rows, eval_rows)."""
    cases = [(variant, seed, name) for variant in VARIANTS for seed in SEEDS for name in TRANSFORMS]
    with ProcessPoolExecutor() as pool:
        logs = dict(zip(cases, pool.map(run_case, *zip(*cases))))
    rows, choice_rows, eval_rows = [], [], []
    for (variant, seed, name), log in logs.items():
        f = make_problem(e1.D, seed)
        curve = best_so_far(f, log["X"])
        ref_best = best_so_far(f, logs[variant, seed, "f^1"]["X"])[-1]
        chosen = [c["chosen"] for c in log["choices"]]
        reps = [c["representation"] for c in log["choices"]]
        mode = lambda xs: max(dict.fromkeys(xs), key=xs.count)  # ties go to the first one chosen; set() order is not reproducible
        rows.append({"variant": variant, "seed": seed, "transform": name, "best_f": curve[-1], "level": float(np.log10(curve[-1])),
                     "gap": log_gap(curve[-1], ref_best),
                     "share_rep_f": None if name == "f/(f+100)" else reps.count("f^1") / len(reps),
                     "first_choice": chosen[0], "final_choice": chosen[-1], "mode_choice": mode(chosen), "mode_representation": mode(reps),
                     "fallbacks": sum(c["fallback"] for c in log["choices"]),
                     "failed_fits": sum(len(c["failed"].split()) for c in log["choices"]),
                     "noise_final": log["noise"][-1], **{f"ls{j + 1}_final": v for j, v in enumerate(log["lengthscale"][-1])}})
        choice_rows += [{"variant": variant, "seed": seed, "transform": name, "fit": i + 1, **c} for i, c in enumerate(log["choices"])]
        for i, (x, best) in enumerate(zip(log["X"], curve)):
            eval_rows.append({"variant": variant, "seed": seed, "transform": name, "eval": i + 1, "f": f(x), "best_f": best,
                              **{f"x{j + 1}": c for j, c in enumerate(x)},
                              "chosen": chosen[i - BO_INIT] if i >= BO_INIT else ""})
    return rows, choice_rows, eval_rows


def read_baselines():
    """{(arm, transform, seed): best f} for E1's default and rank arms and R's warp arm."""
    out = {}
    for path in (e1.RESULTS / "bo.csv", R_RESULTS / "bo.csv"):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["method"] in ("bo_default", "bo_rank", "bo_warp"):
                    out[r["method"], r["transform"], int(r["seed"])] = float(r["best_f"])
    return out


def summarise(rows, base):
    """Per variant and g: median level and gap, choice of h, and paired comparisons with the baselines."""
    out = []
    for variant in VARIANTS:
        for name in TRANSFORMS:
            rs = [r for r in rows if (r["variant"], r["transform"]) == (variant, name)]
            row = {"variant": variant, "transform": name, "level_median": float(np.median([r["level"] for r in rs])),
                   "gap_median": float(np.median([r["gap"] for r in rs])),
                   "share_rep_f_mean": None if name == "f/(f+100)" else float(np.mean([r["share_rep_f"] for r in rs])),
                   "fallbacks": sum(r["fallbacks"] for r in rs), "failed_fits": sum(r["failed_fits"] for r in rs)}
            for label, arm, g in (("default_same_g", "bo_default", name), ("default_on_f", "bo_default", "f^1"),
                                  ("rank", "bo_rank", "f^1"), ("warp_same_g", "bo_warp", name)):
                d = [r["level"] - np.log10(base[arm, g, r["seed"]]) for r in rs]
                row |= {f"vs_{label}_median": float(np.median(d)), f"vs_{label}_better": sum(x < 0 for x in d),
                        f"vs_{label}_p": float(wilcoxon(d).pvalue) if any(d) else float("nan")}
            out.append(row)
    return out


LEVEL_ARMS = {  # label, colour, marker; the first three are read from E1 and R
    "bo_default": ("BO, default (E1)", "#2a78d6", "o"), "bo_rank": ("BO, ranks (E1)", "#eb6834", "s"),
    "bo_warp": ("BO, learned Box-Cox transformation (E4)", "#d6457a", "X"),
    "every": ("BO, chosen by LOO every iteration (E5)", "#00a3a3", "<"), "once": ("BO, chosen by LOO once (E5)", "#a37f00", ">"),
}


def plot_level(rows, base, path, compact=False):
    """Level per g and arm: small markers are seeds, large markers medians. compact: narrower, legend below (report)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8) if compact else (8.5, 3.6))
    for k, (arm, (label, color, marker)) in enumerate(LEVEL_ARMS.items()):
        for i, name in enumerate(TRANSFORMS):
            v = ([np.log10(base[arm, "f^1" if arm == "bo_rank" else name, s]) for s in SEEDS] if arm.startswith("bo_")
                 else [r["level"] for r in rows if (r["variant"], r["transform"]) == (arm, name)])
            x = i + 0.8 * (k - 2) / len(LEVEL_ARMS)
            ax.scatter(np.full(len(v), x), v, s=12, color=color, marker=marker, alpha=0.35, linewidth=0)
            ax.scatter(x, np.median(v), s=64, color=color, marker=marker, edgecolor="white", linewidth=1.5, label=label if i == 0 else None, zorder=3)
    ax.set_xticks(range(len(TRANSFORMS)), [e1.label("transform", n) for n in TRANSFORMS])
    ax.set(ylabel="log10 best untransformed f after 90 evaluations", title="Level per transform (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if compact:
        ax.set(ylabel="log10 best f after 90 evaluations")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    else:
        ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")  # outside: inside it covered seeds
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_choices(choice_rows, path, compact=False):
    """Share of fits choosing each candidate, per g: one panel per variant, stacked bars; outlined: h(g(f)) is f itself.
    compact: the every-iteration panel only, legend below (report)."""
    variants = ["every"] if compact else VARIANTS
    colors = dict(zip(ALL_NAMES, ["#2a78d6", "#d9e8fb", "#b3d1f6", "#86b6ef", "#5598e7", "#1c5cab", "#104281", "#eb6834", "#1baf7a"]))
    assert len(colors) == len(ALL_NAMES)
    fig, axes = plt.subplots(1, len(variants), figsize=(5.6, 4.8) if compact else (5.2 * len(variants), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, variant in zip(axes, variants):
        bottom = np.zeros(len(TRANSFORMS))
        for cand in ALL_NAMES:
            share = np.array([np.mean([c["chosen"] == cand for c in choice_rows if (c["variant"], c["transform"]) == (variant, name)])
                              for name in TRANSFORMS])
            edges = ["black" if representation(name, cand) == "f^1" else "none" for name in TRANSFORMS]
            ax.bar(range(len(TRANSFORMS)), share, bottom=bottom, color=colors[cand], edgecolor=edges, linewidth=1.2, label=cand)
            bottom += share
        ax.set_xticks(range(len(TRANSFORMS)), [e1.label("transform", n) for n in TRANSFORMS], rotation=30)
        ax.set(title=f"h chosen {variant if variant == 'once' else 'every iteration'}", ylim=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of GP fits (all seeds)")
    if compact:
        axes[-1].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), title="h (black edge: h(g) = f)")
    else:
        axes[-1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", title="h (black edge: h(g) = f)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, choice_rows, eval_rows = experiment()
    write_csv(rows, RESULTS / "runs.csv")
    write_csv(choice_rows, RESULTS / "choices.csv")
    write_csv(eval_rows, RESULTS / "evals.csv")
    base = read_baselines()
    summary = summarise(rows, base)
    write_csv(summary, RESULTS / "summary.csv")
    plot_level(rows, base, RESULTS / "level.png")
    plot_choices(choice_rows, RESULTS / "choices.png")
    plot_level(rows, base, RESULTS / "level_report.png", compact=True)  # the report's side-by-side versions
    plot_choices(choice_rows, RESULTS / "choices_report.png", compact=True)
    n = len(SEEDS)
    for s in summary:
        rep_f = "-" if s["share_rep_f_mean"] is None else f"{s['share_rep_f_mean']:.2f}"
        print(f"{s['variant']:5s} {s['transform']:10s} level {s['level_median']:+.2f}  gap {s['gap_median']:+.2f}  share f itself {rep_f}  "
              f"vs default same g {s['vs_default_same_g_median']:+.2f} ({s['vs_default_same_g_better']}/{n}, p = {s['vs_default_same_g_p']:.3f})  "
              f"vs default on f {s['vs_default_on_f_median']:+.2f} ({s['vs_default_on_f_better']}/{n})  "
              f"vs ranks {s['vs_rank_median']:+.2f} ({s['vs_rank_better']}/{n})  vs warp {s['vs_warp_same_g_median']:+.2f} ({s['vs_warp_same_g_better']}/{n})")
