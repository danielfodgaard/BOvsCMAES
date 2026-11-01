"""Shared pieces of the H1 invariance experiment.

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


def make_problem(d, seed, alpha=100):
    """Separable ellipsoid sum_i w_i (x_i - x*_i)^2 with condition number alpha and f* = 0 at x* in [-2, 2]^d."""
    x_opt = np.random.default_rng(seed).uniform(-2, 2, d)
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


def write_csv(rows, path):
    """Write a list of dicts with identical keys as a CSV with a header row."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0])
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
        ax.set(title=method, xlabel="log10(b / a)", ylabel="first differing step", ylim=(0, None))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
