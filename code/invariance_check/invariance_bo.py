"""H1 for Bayesian optimisation (BoTorch), d = 2. Run with: python code/invariance_check/invariance_bo.py

Two acquisition arms, EI and LogEI, 10 Sobol points plus 30 BO iterations. Every iteration logs four
stages in the order a difference can cascade: standardised data -> fitted hyperparameters ->
acquisition value at its argmax -> proposed point. The 90 cases (arm, seed, a, b) run in parallel;
the reference for each (arm, seed) is its own a = 1, b = 0 case.

Outputs in results/invariance/: bo.csv (one row per case), bo_diffs.csv (|difference| to the reference
per case, stage and iteration), first_divergence_bo.png and divergence_growth_bo.png.
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
from gpytorch.mlls import ExactMarginalLogLikelihood

from invariance_util import A_VALS, B_VALS, BOX, RESULTS, SEEDS, compare, diffs, make_problem, plot_first_divergence, write_csv

warnings.filterwarnings("ignore", category=NumericsWarning)  # BoTorch warns on every EI construction; the EI arm is deliberate

BO_INIT, BO_BUDGET = 10, 40
ACQUISITIONS = {"EI": ExpectedImprovement, "LogEI": LogExpectedImprovement}
STAGES = ["data", "hyper", "acq", "proposal"]
TOL = 1e-6  # the GP fitter itself is only reproducible to ~1e-9; below TOL counts as agreement, raw diffs are saved anyway


def run_bo(f, seed, acq_class, d=2):
    """BO on -f (BoTorch maximises) with SingleTaskGP defaults, i.e. standardised outcomes.

    Returns a dict with one list per stage in STAGES, each holding one entry per iteration:
    data (standardised Y), hyper (raw GP parameters), acq (value at the argmax), proposal (the point);
    plus retries, how often BoTorch restarted the acquisition optimiser after scipy reported failure.
    """
    bounds = torch.tensor([[BOX[0]] * d, [BOX[1]] * d], dtype=torch.double)
    X = draw_sobol_samples(bounds, n=BO_INIT, q=1, seed=seed).squeeze(1)  # same initial design for every (a, b)
    Y = torch.tensor([[-f(x)] for x in X], dtype=torch.double)
    log = {stage: [] for stage in STAGES + ["retries"]}
    while len(X) < BO_BUDGET:
        torch.manual_seed(seed)  # re-seed before every fit and every acquisition optimisation
        gp = SingleTaskGP(X, Y, input_transform=Normalize(d, bounds=bounds))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        torch.manual_seed(seed)
        acq = acq_class(gp, best_f=Y.max())
        with warnings.catch_warnings(record=True) as caught:  # scipy "ABNORMAL" retries are data, not noise
            warnings.simplefilter("always")
            x_next, acq_value = optimize_acqf(acq, bounds, q=1, num_restarts=10, raw_samples=256)

        log["data"].append(gp.train_targets.detach().numpy().copy())
        log["hyper"].append(torch.cat([p.detach().ravel() for p in gp.parameters()]).numpy())
        log["acq"].append(acq_value.item())
        log["proposal"].append(x_next.squeeze(0).numpy().copy())
        log["retries"].append(sum("Trying again" in str(w.message) for w in caught))
        X = torch.cat([X, x_next])
        Y = torch.cat([Y, torch.tensor([[-f(x_next)]], dtype=torch.double)])
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
    """Both arms, every seed and (a, b). Returns (rows, diff_rows).

    rows: one per case with first_div (earliest iteration where any stage differs from the reference by
    more than TOL, None if never), first_stage (earliest stage in cascade order differing at that
    iteration), steps, retries and max_diff_<stage>.
    diff_rows: one per case, stage and iteration with the raw |difference| to the reference.
    """
    cases = [(acq, seed, a, b) for acq in ACQUISITIONS for seed in SEEDS for a in A_VALS for b in B_VALS]
    with ProcessPoolExecutor() as pool:
        logs = dict(zip(cases, pool.map(run_case, *zip(*cases))))
    for (acq_name, _, a, _), log in logs.items():
        log["acq"] = rescaled_acq(log["acq"], acq_name, a)

    rows, diff_rows = [], []
    for (acq_name, seed, a, b), log in logs.items():
        ref = logs[acq_name, seed, 1, 0]
        first = {stage: compare(log[stage], ref[stage], TOL) for stage in STAGES}
        first_it = min((it for it, _ in first.values() if it is not None), default=None)
        first_stage = None if first_it is None else next(stage for stage in STAGES if first[stage][0] == first_it)
        rows.append({"acq": acq_name, "seed": seed, "a": a, "b": b,
                     "first_div": first_it, "first_stage": first_stage,
                     "steps": len(log["acq"]), "retries": sum(log["retries"]),
                     **{f"max_diff_{stage}": max_diff for stage, (_, max_diff) in first.items()}})
        for stage in STAGES:
            for it, diff in enumerate(diffs(log[stage], ref[stage])):
                diff_rows.append({"acq": acq_name, "seed": seed, "a": a, "b": b, "stage": stage, "it": it, "diff": diff})
    return rows, diff_rows


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
    rows, diff_rows = experiment_bo()
    write_csv(rows, RESULTS / "bo.csv")
    write_csv(diff_rows, RESULTS / "bo_diffs.csv")
    plot_first_divergence({f"BO {acq}, d = 2": [r for r in rows if r["acq"] == acq] for acq in ACQUISITIONS},
                          RESULTS / "first_divergence_bo.png")
    plot_divergence_growth(diff_rows, RESULTS / "divergence_growth_bo.png")
