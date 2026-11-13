"""H1 for Bayesian optimisation (BoTorch), d = 2. Run with: python code/invariance_check/invariance_bo.py

Two acquisition arms, EI and LogEI, 10 Sobol points plus 30 BO iterations. Every iteration logs four
stages in the order a difference can cascade: standardised data -> fitted hyperparameters ->
acquisition value at its argmax -> proposed point. The 90 cases (arm, seed, a, b) run in parallel;
the reference for each (arm, seed) is its own a = 1, b = 0 case.

Outputs in results/invariance/: bo.csv (one row per case), bo_diffs.csv (|difference| to the reference
per case, stage and iteration), bo_gaps.csv (gap in best untransformed f, with the seed null band),
bo_curves.csv (best-so-far untransformed f after every evaluation, for the figure), first_divergence_bo.png,
divergence_growth_bo.png, best_so_far_bo.png and best_so_far_bo_logei.png (LogEI only, the report's figure).
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit deprecation raised inside botorch's import

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.exceptions import NumericsWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from scipy.stats import boxcox, rankdata, yeojohnson

from invariance_util import A_VALS, B_VALS, BOX, RESULTS, SEEDS, best_so_far, compare, diffs, log_gap, make_problem, plot_best_so_far, plot_first_divergence, summarise_gaps, write_csv

warnings.filterwarnings("ignore", category=NumericsWarning)  # BoTorch warns on every EI construction; the EI arm is deliberate

BO_INIT, BO_BUDGET = 10, 40
ACQUISITIONS = {"EI": ExpectedImprovement, "LogEI": LogExpectedImprovement}
STAGES = ["data", "hyper", "acq", "proposal"]
TOL = 1e-6  # transformed and reference data differ by rounding, which the GP fit amplifies; below TOL counts as agreement, raw diffs are saved anyway


def run_bo(f, seed, acq_class, d=2, budget=BO_BUDGET, rank=False, noise_floor=None, bounds=None, normalize=True, kernel=None, warp=False, fit_to=None, incumbent="observed"):
    """BO on -f (BoTorch maximises) with SingleTaskGP defaults, i.e. standardised outcomes.

    Returns a dict with one list per stage in STAGES, each holding one entry per iteration:
    data (standardised Y), hyper (raw GP parameters), acq (value at the argmax), proposal (the point);
    plus retries, how often BoTorch restarted the acquisition optimiser after scipy reported failure,
    lengthscale and noise (fitted GP values, normalised-input and standardised-output units),
    incumbent_gap (best value the GP was fitted to minus its best posterior mean at the evaluated points, in standardised
    units: > 0 when the fitted noise lets the GP's mean sit below the observed best; computed after the acquisition is
    optimised, so it does not change the run),
    and X, every evaluated point in order (initial design first).
    rank=True fits the GP to the ranks of the observed values instead of the values (Shirakawa's fix),
    which makes the run invariant to any strictly increasing transform of f.
    noise_floor lowers BoTorch's bound on the fitted noise variance (default 1e-4, in standardised-output units,
    i.e. 1 % of the spread of the observed values as noise std); the LogNormal(-4, 1) prior on it is kept.
    bounds (2 x d, default the BOX on every axis) is the search box, used for the Sobol design, the acquisition
    optimiser and the Normalize input transform; normalize=False drops that transform (BoTorch's literal default),
    so the GP and its lengthscale prior see the raw inputs.
    kernel, if given, is a zero-argument callable returning a fresh covariance module per fit (default: BoTorch's ARD RBF).
    warp=True fits the GP to a power transform of the observed (minimised) values, refitted by maximum likelihood every
    iteration, with HEBO's family rule: Box-Cox if all values are positive, else Yeo-Johnson (only the output transform is
    HEBO's; the GP, noise prior and acquisition stay BoTorch's defaults). The fitted lambda is logged as lam
    and the family as family ("boxcox" or "yeojohnson").
    fit_to, if given, maps (X scaled to the unit cube, n x d array; the maximised values -f, length n) to the n values the
    GP is fitted to (higher is better), recomputed every iteration: the hook for other invariant targets (see E1's FIT_TO).
    incumbent is the best_f the acquisition improves on: "observed" (BoTorch's usual choice, the best value the GP was fitted
    to) or "posterior" (the best posterior mean at the evaluated points; R follow-up's test of the boundary stall).
    """
    assert not (fit_to and (rank or warp)), "fit_to replaces the fitted values, so it cannot be combined with rank or warp"
    def likelihood():  # fresh per fit, like BoTorch's default; None keeps the default
        if noise_floor is None:
            return None
        prior = LogNormalPrior(loc=-4.0, scale=1.0)  # BoTorch's default prior, so only the bound changes
        return GaussianLikelihood(noise_prior=prior, noise_constraint=GreaterThan(noise_floor, transform=None, initial_value=prior.mode))
    bounds = torch.tensor([[BOX[0]] * d, [BOX[1]] * d] if bounds is None else bounds, dtype=torch.double)
    X = draw_sobol_samples(bounds, n=BO_INIT, q=1, seed=seed).squeeze(1)  # same initial design for every (a, b)
    Y = torch.tensor([[-f(x)] for x in X], dtype=torch.double)
    log = {stage: [] for stage in STAGES + ["retries", "lengthscale", "noise", "incumbent_gap"] + ["lam", "family"] * warp}
    while len(X) < budget:
        # ranks are recomputed from all observations every iteration; best gets the highest rank
        Y_fit = torch.from_numpy(rankdata(Y.numpy(), axis=0)) if rank else Y
        if warp:
            y = -Y.numpy().ravel()
            y = y / y.std()  # as HEBO; Box-Cox's lambda is unchanged by it (scale-equivariant), Yeo-Johnson's is not
            family = "boxcox" if y.min() > 0 else "yeojohnson"
            w, lam = (boxcox if family == "boxcox" else yeojohnson)(y)
            assert np.isfinite(w).all() and w.std() > 0, (family, lam)
            Y_fit = torch.from_numpy(-w).unsqueeze(1)
            log["lam"].append(float(lam))
            log["family"].append(family)
        if fit_to:
            Y_fit = torch.from_numpy(fit_to(((X - bounds[0]) / (bounds[1] - bounds[0])).numpy(), Y.numpy().ravel())).unsqueeze(1)
        torch.manual_seed(seed)  # re-seed before every fit and every acquisition optimisation
        gp = SingleTaskGP(X, Y_fit, likelihood=likelihood(), covar_module=kernel() if kernel else None,
                          input_transform=Normalize(d, bounds=bounds) if normalize else None)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        torch.manual_seed(seed)
        if incumbent == "observed":
            best_f = Y_fit.max()
        else:
            assert incumbent == "posterior", incumbent
            with torch.no_grad():
                best_f = gp.posterior(X).mean.max()  # in the units of Y_fit: Standardize is undone by the posterior
        acq = acq_class(gp, best_f=best_f)
        with warnings.catch_warnings(record=True) as caught:  # scipy "ABNORMAL" retries are data, not noise
            warnings.simplefilter("always")
            x_next, acq_value = optimize_acqf(acq, bounds, q=1, num_restarts=10, raw_samples=256)

        log["data"].append(gp.train_targets.detach().numpy().copy())
        log["hyper"].append(torch.cat([p.detach().ravel() for p in gp.parameters()]).numpy())
        log["acq"].append(acq_value.item())
        log["proposal"].append(x_next.squeeze(0).numpy().copy())
        log["retries"].append(sum("Trying again" in str(w.message) for w in caught))
        log["lengthscale"].append(gp.covar_module.lengthscale.detach().numpy().ravel().copy())
        log["noise"].append(gp.likelihood.noise.item())
        with torch.no_grad():
            log["incumbent_gap"].append(float((Y_fit.max() - gp.posterior(X).mean.max()) / gp.outcome_transform.stdvs.squeeze()))
        X = torch.cat([X, x_next])
        Y = torch.cat([Y, torch.tensor([[-f(x_next)]], dtype=torch.double)])
    log["X"] = X.numpy()
    return log


def run_case(acq_name, seed, a, b, d=2):
    """One BO run on a * f + b. Top level so a worker process can run it."""
    torch.set_num_threads(1)  # the problems are tiny: many single-threaded runs in parallel beat one multi-threaded run
    f = make_problem(d, seed)
    return run_bo(lambda x: a * f(x) + b, seed, ACQUISITIONS[acq_name], d)


def rescaled_acq(acq_values, acq_name, a):
    """Undo the dependence of the acquisition value on a: EI(a f) = a EI(f), LogEI(a f) = LogEI(f) + ln a."""
    acq_values = np.array(acq_values)
    return acq_values / a if acq_name == "EI" else acq_values - np.log(a)


def experiment_bo():
    """Both arms, every seed and (a, b). Returns (rows, diff_rows, curve_rows).

    rows: one per case with first_div (earliest iteration where any stage differs from the reference by
    more than TOL, None if never), first_stage (earliest stage in cascade order differing at that
    iteration), steps, retries, max_diff_<stage>, best_f (best untransformed f at the end of the budget)
    and gap (its log10 difference to the reference run of the same seed).
    diff_rows: one per case, stage and iteration with the raw |difference| to the reference.
    curve_rows: one per case and evaluation with the best-so-far untransformed f.
    """
    cases = [(acq, seed, a, b) for acq in ACQUISITIONS for seed in SEEDS for a in A_VALS for b in B_VALS]
    with ProcessPoolExecutor() as pool:
        logs = dict(zip(cases, pool.map(run_case, *zip(*cases))))
    for (acq_name, _, a, _), log in logs.items():
        log["acq"] = rescaled_acq(log["acq"], acq_name, a)

    rows, diff_rows, curve_rows = [], [], []
    for (acq_name, seed, a, b), log in logs.items():
        ref = logs[acq_name, seed, 1, 0]
        f = make_problem(log["X"].shape[1], seed)  # scored on the untransformed f, like every run
        curve = best_so_far(f, log["X"])
        best, ref_best = curve[-1], best_so_far(f, ref["X"])[-1]
        curve_rows += [{"acq": acq_name, "seed": seed, "a": a, "b": b, "eval": i + 1, "best_f": v} for i, v in enumerate(curve)]
        first = {stage: compare(log[stage], ref[stage], TOL) for stage in STAGES}
        first_it = min((it for it, _ in first.values() if it is not None), default=None)
        first_stage = None if first_it is None else next(stage for stage in STAGES if first[stage][0] == first_it)
        rows.append({"acq": acq_name, "seed": seed, "a": a, "b": b,
                     "first_div": first_it, "first_stage": first_stage,
                     "steps": len(log["acq"]), "retries": sum(log["retries"]),
                     "best_f": best, "gap": log_gap(best, ref_best),
                     **{f"max_diff_{stage}": max_diff for stage, (_, max_diff) in first.items()}})
        for stage in STAGES:
            for it, diff in enumerate(diffs(log[stage], ref[stage])):
                diff_rows.append({"acq": acq_name, "seed": seed, "a": a, "b": b, "stage": stage, "it": it, "diff": diff})
    return rows, diff_rows, curve_rows


def plot_divergence_growth(diff_rows, path, stage="proposal"):
    """log10 |difference to the reference| per iteration, median over seeds: a row per arm, a column per a, a line per b."""
    colors = dict(zip(B_VALS, ["#2a78d6", "#eb6834", "#1baf7a"]))
    fig, axes = plt.subplots(len(ACQUISITIONS), len(A_VALS), figsize=(4 * len(A_VALS), 3 * len(ACQUISITIONS)),
                             sharex=True, sharey=True, squeeze=False)
    for row, acq in zip(axes, ACQUISITIONS):
        for ax, a in zip(row, A_VALS):
            for b in B_VALS:
                per_seed = [[r["diff"] for r in diff_rows
                             if (r["acq"], r["seed"], r["a"], r["b"], r["stage"]) == (acq, seed, a, b, stage)] for seed in SEEDS]
                median = np.median(per_seed, axis=0)
                ax.plot(np.log10(np.maximum(median, 1e-17)), color=colors[b], linewidth=1.5, label=f"b = {b:g}")
            ax.set_title(f"{acq}, a = {a:g}")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("iteration")
    fig.supylabel(f"log10 |{stage} - reference|, median over seeds (0 drawn at -17)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, diff_rows, curve_rows = experiment_bo()
    write_csv(rows, RESULTS / "bo.csv")
    write_csv(diff_rows, RESULTS / "bo_diffs.csv")
    write_csv(curve_rows, RESULTS / "bo_curves.csv")
    write_csv(summarise_gaps(rows, ("acq", "a", "b")), RESULTS / "bo_gaps.csv")
    plot_first_divergence({f"BO {acq}, d = 2": [r for r in rows if r["acq"] == acq] for acq in ACQUISITIONS},
                          RESULTS / "first_divergence_bo.png")
    plot_divergence_growth(diff_rows, RESULTS / "divergence_growth_bo.png")
    plot_best_so_far(curve_rows, RESULTS / "best_so_far_bo.png", col="a", line="b", row="acq", ref={"a": 1, "b": 0})
    # the report's H1 figure: LogEI only, the acquisition every later experiment uses
    plot_best_so_far([r for r in curve_rows if r["acq"] == "LogEI"], RESULTS / "best_so_far_bo_logei.png", col="a", line="b", ref={"a": 1, "b": 0})
