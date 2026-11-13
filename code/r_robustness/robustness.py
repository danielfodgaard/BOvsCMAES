"""R: does learned output warping recover what the rank fix gives? BO (BoTorch), d = 5. Run with: python code/r_robustness/robustness.py

One more BO arm on E1's transforms, same seeds, problems and Sobol designs: bo_warp fits the GP to a power transform of
the observed g(f), refitted by maximum likelihood every iteration (HEBO's family rule: Box-Cox if all values are positive,
else Yeo-Johnson; see run_bo). This is BoTorch's default BO plus HEBO's output transform, not HEBO: the GP, noise prior
and LogEI are unchanged. Box-Cox contains the inverse of every f^p (lambda -> lambda / p gives the same warped values up
to an affine map, which Standardize removes), so the arm is invariant on a * f^p in exact arithmetic. It is not invariant
on log f or the saturating g, and, unlike default BO (H1), not on f + b: Box-Cox has its singular point at 0, where f*
sits. The warp arm therefore also runs on the offsets in monotone.OFFSETS (f - 10 goes negative and forces Yeo-Johnson).
The default and rank arms are NOT rerun: they are read from results/e1_monotone/ (run code/e1_monotone/monotone.py first).

Two measures, both on the untransformed f after 90 evaluations:
  gap    log10 best f on g - log10 best f on f, same seed and arm (invariance, as E1);
  level  log10 best f itself (what the user gets; the gap hides how good the arm's own reference is).

Outputs in results/r_robustness/: bo.csv (one row per warp run, with the last fit's lambda and family), bo_evals.csv,
gaps.csv (all three arms: median gap and its range, median level, seeds where the arm beats default BO and rank BO on
the same g with the paired median difference; on the offsets, default and rank BO are their runs on f), gap.png,
best_so_far_bo.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "e1_monotone"))
sys.path.insert(0, str(Path(__file__).parents[1] / "invariance_check"))
import monotone as e1
from invariance_util import plot_best_so_far, write_csv

RESULTS = Path(__file__).parents[2] / "results" / "r_robustness"
E1 = e1.RESULTS
SHOWN = ["bo_default", "bo_rank", "bo_warp"]
e1.ARMS["bo_warp", "gap"] = ("BO, Box-Cox warp", "#d6457a", "X")  # neither E1's low-noise nor its default-stopping glyph
NAMES = [*e1.TRANSFORMS, *e1.OFFSETS]


def read_e1(name, methods=("bo_default", "bo_rank")):
    """E1's rows for the arms R compares against, with the numeric fields R uses converted from text."""
    with open(E1 / name, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["method"] in methods]
    for r in rows:
        r.update({k: float(r[k]) for k in ("best_f", "gap") if k in r}, seed=int(r["seed"]), **({"eval": int(r["eval"])} if "eval" in r else {}))
    return rows


def summarise(rows):
    """Per arm and transform: median gap, how many seeds are worse than on f, median level, and seeds beating default BO."""
    best = {(r["method"], r["transform"], r["seed"]): r["best_f"] for r in rows}
    # default (to rounding, H1) and rank BO (exactly) on f + b are their runs on f, so the offsets are compared with those
    other = lambda method, r: best.get((method, r["transform"], r["seed"]), best[method, e1.REF, r["seed"]])
    vs = lambda method, rs: [float(np.log10(r["best_f"]) - np.log10(other(method, r))) for r in rs]
    out = []
    for method in SHOWN:
        for name in NAMES:
            rs = [r for r in rows if (r["method"], r["transform"]) == (method, name)]
            if not rs and name in e1.OFFSETS:
                continue
            assert len(rs) == len(e1.SEEDS), (method, name, len(rs))
            gaps = [r["gap"] for r in rs]
            out.append({"method": method, "transform": name, "gap_median": float(np.median(gaps)),
                        "gap_min": min(gaps), "gap_max": max(gaps), "seeds_worse_than_on_f": sum(g > 0 for g in gaps),
                        "level_median": float(np.median([np.log10(r["best_f"]) for r in rs])),
                        **{k: v for m in ("default", "rank") for d in [vs(f"bo_{m}", rs)]
                           for k, v in ((f"seeds_beating_{m}", sum(x < 0 for x in d)), (f"median_vs_{m}", float(np.median(d))))}})
    return out


def plot(rows, path):
    """Left: paired gap per transform (invariance). Middle: log10 best f per transform, f included (what the user gets).
    Right: the warp arm under f + b, against the medians of default and rank BO on f (which do not see b)."""
    by = lambda method, name: [r for r in rows if r["method"] == method and r["transform"] == name]
    names, arms = list(e1.TRANSFORMS), [(m, "gap") for m in SHOWN]
    others = [n for n in names if n != e1.REF]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.4, 1.6, 0.9]})
    axes[0].axhline(0, color="#8a8984", linewidth=1)
    e1._strip(axes[0], others, arms, lambda arm, name: [r["gap"] for r in by(arm[0], name)])
    e1._style(axes[0], others, "log10 best f on g − log10 best f on f", "Paired gap (0 = invariant; > 0 = worse on g)")
    e1._strip(axes[1], names, arms, lambda arm, name: [np.log10(r["best_f"]) for r in by(arm[0], name)])
    e1._style(axes[1], names, "log10 best untransformed f after 90 evaluations", "Level (lower is better)")
    shifted = [e1.REF, *e1.OFFSETS]
    e1._strip(axes[2], shifted, [("bo_warp", "gap")], lambda arm, name: [np.log10(r["best_f"]) for r in by(arm[0], name)])
    for method in ("bo_default", "bo_rank"):
        label, color, _ = e1.ARMS[method, "gap"]
        axes[2].axhline(np.median([np.log10(r["best_f"]) for r in by(method, e1.REF)]), color=color, linestyle="--", linewidth=1.5, label=f"{label}, median on f")
    e1._style(axes[2], shifted, "log10 best untransformed f after 90 evaluations", "Warp arm under f + b")
    for ax in (axes[0], axes[2]):
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    warp_rows, warp_evals = e1.experiment_bo(["bo_warp"], NAMES)
    write_csv(warp_rows, RESULTS / "bo.csv")
    write_csv(warp_evals, RESULTS / "bo_evals.csv")
    rows = read_e1("bo.csv") + warp_rows
    summary = summarise(rows)
    write_csv(summary, RESULTS / "gaps.csv")
    plot(rows, RESULTS / "gap.png")
    plot_best_so_far(read_e1("bo_evals.csv") + [r for r in warp_evals if r["transform"] in e1.TRANSFORMS], RESULTS / "best_so_far_bo.png", col="transform", line="method",
                     row="method", ref={"transform": e1.REF}, label=e1.label)
    for s in summary:
        print(f"{s['method']:11s} {s['transform']:10s} gap {s['gap_median']:+.2f} [{s['gap_min']:+.2f}, {s['gap_max']:+.2f}]  "
              f"level {s['level_median']:+.2f}  beats default {s['seeds_beating_default']}/{len(e1.SEEDS)} ({s['median_vs_default']:+.2f})  "
              f"beats rank {s['seeds_beating_rank']}/{len(e1.SEEDS)} ({s['median_vs_rank']:+.2f})")
