"""Shared helpers used by every experiment (first written for H1): problems, start points, comparisons, CSVs and plots.

H1: an optimiser that only ranks or standardises objective values should behave identically
on f and on a*f + b. Each method is run on the nine (a, b) transforms below and compared,
step by step, with the untransformed reference run (a = 1, b = 0) under the same seed.
This module holds the problem, the transform grid, the comparison and the plot; the
optimiser-specific runners and drivers are in invariance_cma.py and invariance_bo.py.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

A_VALS, B_VALS = [1e-3, 1, 1e3], [0, 1e3, 1e8]  # the optimiser sees a * f(x) + b
BOX = [-5, 5]
SEEDS = range(5)
RESULTS = Path(__file__).parents[2] / "results" / "invariance"


def optimum(d, seed):
    """x*, uniform in [-2, 2]^d: the first draw of the seed's own stream."""
    return np.random.default_rng(seed).uniform(-2, 2, d)


def start_point(d, seed):
    """CMA-ES start point, uniform in the box, from its own stream.

    results/e1_monotone_run1/ used an earlier start drawn from x*'s own stream (default_rng(seed)), which made it exactly 2.5 * x*.
    """
    return np.random.default_rng([seed, 2]).uniform(*BOX, d)  # [seed, 1] is E3's rotation stream


def make_problem(d, seed, alpha=100):
    """Separable ellipsoid sum_i w_i (x_i - x*_i)^2 with condition number alpha and f* = 0 at x* = optimum(d, seed)."""
    x_opt = optimum(d, seed)
    w = alpha ** (np.arange(d) / (d - 1))

    def f(x):
        z = np.asarray(x, dtype=float).ravel() - x_opt
        return float(np.sum(w * z**2))

    return f


def diffs(trace, ref):
    """max |trace - ref| at every step of the common prefix.

    A trace is a sequence with one entry per generation/iteration; entries may be scalars or vectors,
    including vectors whose length grows with the step. Only the common prefix is compared, so a run
    that stopped early is judged on the steps it did produce.
    """
    return np.array([np.abs(np.asarray(t) - np.asarray(r)).max() for t, r in zip(trace, ref)])


def compare(trace, ref, tol=0.0):
    """First step at which trace differs from ref by more than tol (None if never), and max |trace - ref|."""
    diff_per_step = diffs(trace, ref)
    differing = np.flatnonzero(diff_per_step > tol)
    first = int(differing[0]) if len(differing) else None
    return first, float(diff_per_step.max())


def best_so_far(f, X):
    """Best-so-far untransformed f after each evaluation, for the evaluated points X in order."""
    return np.minimum.accumulate([f(x) for x in X])


def log_gap(best, ref_best):
    """log10 best f on a*f + b minus log10 best f on f, for the same seed. Floored so an exact 0 does not give -inf."""
    tiny = np.finfo(float).tiny
    return float(np.log10(max(best, tiny)) - np.log10(max(ref_best, tiny)))


def summarise_gaps(rows, group, key="gap", ref=None):
    """One row per group (e.g. ("a", "b") or ("acq", "a", "b")): median and max |gap| over seeds.

    ref_range is the null band: the max - min over seeds of log10 best f on the reference (ref, default
    a = 1, b = 0) within the same group on the other keys. A gap inside it is indistinguishable from seed noise.
    """
    ref = ref or {"a": 1, "b": 0}
    out = []
    for g in sorted({tuple(r[k] for k in group) for r in rows}):
        rs = [r for r in rows if tuple(r[k] for k in group) == g]
        same_arm = lambda r: all(r[k] == v for k, v in zip(group, g) if k not in ref)
        refs = [np.log10(r[key.replace("gap", "best_f")]) for r in rows if all(r[k] == v for k, v in ref.items()) and same_arm(r)]
        out.append({**dict(zip(group, g)), "arm": key, "gap_median": float(np.median([r[key] for r in rs])),
                    "gap_max_abs": float(np.max(np.abs([r[key] for r in rs]))), "ref_range": float(np.ptp(refs))})
    return out


def plot_best_so_far(curve_rows, path, col, line, row=None, ref=None, label=None):
    """Median over seeds of best-so-far untransformed f against evaluations, log y.

    One subplot per value of col (and of row, if given), one line per value of line, in the order the values first
    appear in curve_rows. Rows need seed, eval and best_f plus those keys. ref, e.g. {"a": 1, "b": 0}, selects the
    reference rows, drawn dashed in every subplot of the same row for comparison; a column holding only the reference
    is left out. label(key, value) formats titles and legend entries. Each row gets its own legend.
    A median at an evaluation uses the seeds still running there, so stopped runs just end early.
    """
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd6", "#c9a400", "#d6457a", "#5b8e7d", "#7a6a53"]
    label = label or (lambda k, v: f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}")
    is_ref = lambda r: ref is not None and all(r[k] == v for k, v in ref.items())

    def median_curve(rows):
        by_eval = {}
        for r in rows:
            by_eval.setdefault(r["eval"], []).append(r["best_f"])
        evals = sorted(by_eval)
        return evals, [np.median(by_eval[e]) for e in evals]

    row_vals = list(dict.fromkeys(r[row] for r in curve_rows)) if row else [None]
    col_vals = [v for v in dict.fromkeys(r[col] for r in curve_rows)
                if not (ref and col in ref and len(ref) == 1 and v == ref[col])]  # its panel would show the reference alone
    line_vals = list(dict.fromkeys(r[line] for r in curve_rows))
    assert len(line_vals) <= len(colors), f"{len(line_vals)} lines but {len(colors)} colours: zip would drop lines silently"
    fig, axes = plt.subplots(len(row_vals), len(col_vals), figsize=(3.6 * len(col_vals) + 0.6, 2.9 * len(row_vals)),
                             sharex=True, sharey=True, squeeze=False)
    for ax_row, rv in zip(axes, row_vals):
        in_row = [r for r in curve_rows if row is None or r[row] == rv]
        ref_curve = median_curve([r for r in in_row if is_ref(r)]) if ref else None
        for ax, cv in zip(ax_row, col_vals):
            for color, lv in zip(colors, line_vals):
                rows = [r for r in in_row if r[col] == cv and r[line] == lv and not is_ref(r)]
                if rows:
                    ax.plot(*median_curve(rows), color=color, linewidth=1.5, label=label(line, lv))
            if ref_curve:
                ax.plot(*ref_curve, color="black", linestyle="--", linewidth=1.2, label="reference")
            ax.set_yscale("log")
            ax.set_title(label(col, cv) if row is None else f"{label(row, rv)}, {label(col, cv)}")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        entries = {l: h for ax in ax_row for h, l in zip(*ax.get_legend_handles_labels())}  # every line in the row, not just panel 1's
        ax_row[0].legend(entries.values(), entries.keys(), frameon=False, fontsize=8, loc="best")
        ax_row[0].set_ylabel("best-so-far f\n(median over seeds)")  # an axis label, so tight_layout keeps it inside the figure
    for ax in axes[-1]:
        ax.set_xlabel("evaluations")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_csv(rows, path):
    """Write a list of dicts as a CSV with a header row. Columns are the union of the keys in first-appearance order;
    a row without a column leaves it empty (e.g. the warp arm's lambda columns next to default BO's rows)."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def plot_first_divergence(rows_by_method, path):
    """Scatter the first differing step against log10(b/a): one panel per method, one point per seed.

    Rows need the keys a, b, seed, first_div and steps (the run length). Runs that never diverged are
    drawn as hollow markers at the budget. Rows with b = 0 are left out since log10(0) is undefined.
    """
    blue = "#2a78d6"
    fig, axes = plt.subplots(1, len(rows_by_method), figsize=(5 * len(rows_by_method), 4), squeeze=False)
    for ax, (method, rows) in zip(axes[0], rows_by_method.items()):
        rows = [r for r in rows if r["b"] > 0]
        x = np.array([np.log10(r["b"] / r["a"]) + 0.06 * (r["seed"] - np.mean(SEEDS)) for r in rows])  # jitter per seed
        y = np.array([r["steps"] if r["first_div"] is None else r["first_div"] for r in rows], dtype=float)
        never = np.array([r["first_div"] is None for r in rows])

        ax.scatter(x[~never], y[~never], s=28, color=blue, alpha=0.75, edgecolor="white", linewidth=0.5)
        if never.any():
            ax.scatter(x[never], y[never], s=28, facecolor="none", edgecolor=blue, label="never diverged (drawn at budget)")
            ax.legend(loc="lower left", frameon=False)
        ax.set(title=method, xlabel="log10(b / a)", ylabel="first differing step", ylim=(-0.5, max(y) + 0.5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # steps are whole numbers; -0.5 keeps markers at 0 whole
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
