"""R follow-up: questions E1 and R left open. BO (BoTorch), d = 5. Run with: python code/r_robustness/followup.py
(after code/e1_monotone/monotone.py and code/r_robustness/robustness.py, whose results it reads).

1. Is the 2-order cost of invariance a cost of ranks? Two more exactly invariant arms, run on f (and on f^4 as the
   bit-for-bit invariance check): bo_nscore (GP fitted to the normal scores of the values) and bo_lebesgue (Shirakawa's
   estimate of the Lebesgue-measure objective, which on this ellipsoid is f itself when exact).
2. Is the stall under log f caused by the singular optimum? Default BO on log(f + c), c in monotone.SHIFTED_LOGS: the
   optimum is a regular point for every c > 0, so if the singularity were the cause, any c should restore the level on f.
3. Why do stalled runs end up sampling the faces of the box? Two arms on f and log f:
   bo_postinc  LogEI improves on the best posterior mean at the evaluated points instead of the best observed value.
               Candidate (a): with fitted noise, the GP's mean at the incumbent is below the observed best, so nothing
               near it is expected to improve and the acquisition's maximum moves to where the variance is, the faces.
   bo_bigbox   the box doubled to [-10, 10]^5 with the same x* (so the optimum is farther from every face), with default
               BO on f in the same box as its reference: a larger box costs evaluations by itself (see E2). It also
               makes the Sobol design 32x sparser and the curvature 4x larger in normalised units, so the gap alone cannot
               separate "the stall level moves with the face" from "the box is harder"; section3.csv therefore records, per
               run, the level a run parked on the faces of its incumbent would have if every other coordinate were at x*
               (sum over the face axes of w_i (half-width - |x*_i|)^2), to compare with the observed level.
   run_bo logs incumbent_gap for every BO run (best fitted value minus best posterior mean, standardised units), the premise of
   candidate (a); stall.csv gives its median over evaluations 61-90 per arm, so bo_postinc can be read conditional on it.
4. How often do the bimodal rows stall? Default BO on log(f + 1000) and the warp arm on f + 1000 and
   f - 10 split 10 seeds into a group as good as default BO on f and a group parked on a face. Rerun with 30 seeds; per
   seed record whether the run stalled and where x* lies, to see whether the geometry predicts it. Stalled (primary): at
   least 90 % of the proposals in evaluations 61-90 lie on a face. Secondary: the final incumbent lies on a face (it
   undercounts: an incumbent found before the stall need not be on a face). Stalled fractions come with 95 % Clopper-Pearson
   intervals.

Same Sobol designs and budget (10 + 80, LogEI) as E1; seeds 0-9, except section 4 (0-29). A point is "on a face" if some
coordinate is within 0.2 % of the box's half-width of the bound (|x_i| > 4.99 in the usual box).

Outputs in results/r_followup/:
  bo.csv, bo_evals.csv   one row per run / per evaluation, sections 1-3
  summary.csv            sections 1-2: per arm and transform, median level log10 best f, the paired difference to E1's
                         default and rank arms on f with sign count and Wilcoxon p; for bo_lebesgue also the log-log slope
                         of the final estimate against f (1 if it is affine in f)
  stall.csv              section 3 and the face diagnostic for every BO run of E1, R and this script: per arm and
                         transform, median level, share of proposals on a face in evaluations 11-30 and 61-90, median
                         evaluation of the last improvement
  section3_summary.csv   section 3: per arm and g, paired difference to default BO on the same g, seeds better, Wilcoxon p
  section3.csv           section 3, one row per run: level, gap, paired difference to default BO on the same g and seed,
                         face axes of the final incumbent and the level predicted for a run parked there
  bimodal.csv            section 4, one row per run: level, gap, share of evaluations 61-90 on a face, both stall flags,
                         face axes of the incumbent, x*_1 and the distance of x* to the nearest face along axis 1
  bimodal_summary.csv    section 4: per arm and g, stalled runs out of 30 with a 95 % Clopper-Pearson interval, and the
                         median level of stalled and other runs
  bimodal_bo.csv, bimodal_evals.csv   section 4's raw runs (incl. the 30-seed reference runs on f)
  best_so_far_bo.png     sections 1-3; bimodal.png section 4 (level against the distance of x*_1 to the face)
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import robustness  # also puts E1 and the shared helpers on sys.path
import monotone as e1
from invariance_util import BOX, optimum, plot_best_so_far, write_csv

W = 100 ** (np.arange(e1.D) / (e1.D - 1))  # the ellipsoid's axis weights, as in make_problem

RESULTS = Path(__file__).parents[2] / "results" / "r_followup"
CHECK = "f^4"  # invariant arms must propose bit-identical points on it
FACE = 0.998  # fraction of the half-width beyond which a coordinate counts as on a face (4.99 in [-5, 5])
BIMODAL = {"bo_default": [e1.REF, "log(f+1000)"], "bo_warp": [e1.REF, "f+1000", "f-10"]}  # section 4, with each arm's reference
BIMODAL_SEEDS = range(30)
e1.ARMS["bo_postinc", "gap"] = ("BO, posterior-mean incumbent", "#5b8e7d", "P")
e1.ARMS["bo_bigbox", "gap"] = ("BO, box [-10, 10]^5", "#7a6a53", "h")
e1.ARMS["bo_nscore", "gap"] = ("BO, normal scores", "#a05195", "p")
e1.ARMS["bo_lebesgue", "gap"] = ("BO, Lebesgue estimate", "#3d7ea6", "*")


def half_width(method):
    return e1.BIG_BOX[1][0] if method == "bo_bigbox" else BOX[1]


def on_face(x, method):
    """Indices (1-based) of the coordinates of x that lie on a face of the method's box."""
    return [j + 1 for j, v in enumerate(x) if abs(v) > FACE * half_width(method)]


def point(r):
    return [float(r[f"x{j + 1}"]) for j in range(e1.D)]


def slope(evals, top=None):
    """Median over seeds of the log-log slope of -(final Lebesgue estimate) against f, over the best `top` points (all if None)."""
    out = []
    for seed in e1.SEEDS:
        rs = [r for r in evals if (r["method"], r["transform"], r["seed"]) == ("bo_lebesgue", e1.REF, seed)]
        X01 = (np.array([point(r) for r in rs]) + 5) / 10
        f = np.array([r["f"] for r in rs])
        keep = np.argsort(f)[:top]
        out.append(np.polyfit(np.log(f[keep]), np.log(-e1.lebesgue(X01, -f)[keep]), 1)[0])
    return float(np.median(out))


def summarise(rows, evals):
    """Sections 1-2: level of each arm and transform, paired against E1's default and rank arms on f."""
    base = {(r["method"], r["seed"]): np.log10(r["best_f"]) for r in robustness.read_e1("bo.csv") if r["transform"] == e1.REF}
    out = []
    for method, name in dict.fromkeys((r["method"], r["transform"]) for r in rows):
        if method in ("bo_postinc", "bo_bigbox"):
            continue  # section 3, summarised in stall.csv
        rs = [r for r in rows if (r["method"], r["transform"]) == (method, name)]
        level = [np.log10(r["best_f"]) for r in rs]
        row = {"method": method, "transform": name, "level_median": float(np.median(level)),
               "gap_median": float(np.median([r["gap"] for r in rs])), "max_diff_proposal": max(r["max_diff_proposal"] for r in rs)}
        for other in ("default", "rank"):
            d = [v - base[f"bo_{other}", r["seed"]] for v, r in zip(level, rs)]
            row |= {f"median_vs_{other}_on_f": float(np.median(d)), f"seeds_beating_{other}": sum(x < 0 for x in d),
                    f"p_vs_{other}": float(wilcoxon(d).pvalue) if any(d) else float("nan")}
        row |= {"slope_all": slope(evals), "slope_best30": slope(evals, 30)} if (method, name) == ("bo_lebesgue", e1.REF) else {"slope_all": None, "slope_best30": None}
        out.append(row)
    return out


def stall_table(evals):
    """Per arm and transform: median level, share of proposals on a face early and late, median last improvement."""
    out = []
    for method, name in dict.fromkeys((r["method"], r["transform"]) for r in evals):
        rs = [r for r in evals if (r["method"], r["transform"]) == (method, name)]
        share = lambda lo, hi: float(np.mean([bool(on_face(point(r), method)) for r in rs if lo <= r["eval"] <= hi]))
        last, levels = [], []
        for seed in dict.fromkeys(r["seed"] for r in rs):
            best = [r["best_f"] for r in sorted((r for r in rs if r["seed"] == seed), key=lambda r: r["eval"])]
            last.append(1 + max(i for i in range(len(best)) if i == 0 or best[i] < best[i - 1]))
            levels.append(np.log10(best[-1]))
        gaps = [float(r["incumbent_gap"]) for r in rs if 61 <= r["eval"] <= 90 and r.get("incumbent_gap") not in (None, "")]
        out.append({"method": method, "transform": name, "seeds": len(levels), "level_median": float(np.median(levels)),
                    "face_share_11_30": share(11, 30), "face_share_61_90": share(61, 90), "last_improvement_median": float(np.median(last)),
                    "incumbent_gap_median_61_90": float(np.median(gaps)) if gaps else None})
    return out


def bimodal_table(rows, evals):
    """Section 4, one row per run: whether the final incumbent is on a face, where x* is along axis 1, and for the warp
    arm in how many of its 80 fits it used Box-Cox (all values positive) rather than Yeo-Johnson."""
    out = []
    for r in rows:
        rs = [e for e in evals if (e["method"], e["transform"], e["seed"]) == (r["method"], r["transform"], r["seed"])]
        incumbent = min(rs, key=lambda e: (e["f"], e["eval"]))
        late = float(np.mean([bool(on_face(point(e), r["method"])) for e in rs if 61 <= e["eval"] <= 90]))
        x1 = optimum(e1.D, r["seed"])[0]
        out.append({"method": r["method"], "transform": r["transform"], "seed": r["seed"], "best_f": r["best_f"],
                    "level": float(np.log10(r["best_f"])), "gap": r["gap"], "face_share_61_90": late, "stalled": late >= 0.9,
                    "incumbent_on_face": bool(on_face(point(incumbent), r["method"])),
                    "face_axes": " ".join(map(str, on_face(point(incumbent), r["method"]))),
                    "x_opt_1": x1, "dist_to_face_1": BOX[1] - abs(x1), "boxcox_fits": r.get("boxcox_fits")})
    return out


def section3_table(rows, evals):
    """Section 3, one row per run: paired difference to default BO on the same g and seed, and the parked-level prediction."""
    base = {(r["transform"], r["seed"]): r["best_f"] for r in rows if r["method"] == "bo_default"}
    out = []
    for r in rows:
        if r["method"] not in ("bo_postinc", "bo_bigbox"):
            continue
        rs = [e for e in evals if (e["method"], e["transform"], e["seed"]) == (r["method"], r["transform"], r["seed"])]
        incumbent = point(min(rs, key=lambda e: (e["f"], e["eval"])))
        axes = on_face(incumbent, r["method"])
        x_opt, hw = optimum(e1.D, r["seed"]), half_width(r["method"])
        parked = sum(W[j - 1] * (np.sign(incumbent[j - 1]) * hw - x_opt[j - 1]) ** 2 for j in axes)  # on whichever face it is
        out.append({"method": r["method"], "transform": r["transform"], "seed": r["seed"], "level": float(np.log10(r["best_f"])),
                    "gap": r["gap"], "vs_default_same_g": float(np.log10(r["best_f"]) - np.log10(base[r["transform"], r["seed"]])),
                    "face_axes": " ".join(map(str, axes)), "parked_level_prediction": float(np.log10(parked)) if axes else None})
    return out


def plot_bimodal(table, path):
    """One panel per bimodal row: level against the distance of x*_1 to the nearest face of axis 1, filled if stalled.
    Dashed: log10 of that distance squared, the f of a point with x_1 on that face and every other coordinate at x*."""
    rows = [(m, n) for m, names in BIMODAL.items() for n in names if n != e1.REF]
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.6), sharey=True, squeeze=False)
    for ax, (method, name) in zip(axes[0], rows):
        rs = [t for t in table if (t["method"], t["transform"]) == (method, name)]
        for stalled, style in ((True, {"color": "#eb6834"}), (False, {"facecolor": "none", "edgecolor": "#2a78d6"})):
            sel = [t for t in rs if t["stalled"] == stalled]
            ax.scatter([t["dist_to_face_1"] for t in sel], [t["level"] for t in sel], s=30, **style,
                       label=f"stalled: >= 90 % of evals 61-90 on a face ({len(sel)}/{len(rs)})" if stalled else f"not stalled ({len(sel)}/{len(rs)})")
        d = np.linspace(min(t["dist_to_face_1"] for t in rs), max(t["dist_to_face_1"] for t in rs), 50)
        ax.plot(d, np.log10(d**2), color="#8a8984", linestyle="--", linewidth=1, label="parked on the axis-1 face")
        ax.set(title=f"{e1.label('method', method)}, {name}", xlabel="distance of $x^*_1$ to the nearest face of axis 1")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="best")
    axes[0, 0].set_ylabel("log10 best f after 90 evaluations")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def section3_summary(section3):
    """Section 3 per arm and g: median paired difference to default BO on the same g and seed, seeds better, Wilcoxon p."""
    out = []
    for method in ("bo_postinc", "bo_bigbox"):
        for name in (e1.REF, "log f"):
            d = [t["vs_default_same_g"] for t in section3 if (t["method"], t["transform"]) == (method, name)]
            out.append({"method": method, "transform": name, "median_vs_default_same_g": float(np.median(d)),
                        "seeds_better": sum(x < 0 for x in d), "seeds": len(d), "p": float(wilcoxon(d).pvalue) if any(d) else float("nan")})
    return out


def bimodal_summary(bimodal):
    """Section 4 per row: stall rate with its Clopper-Pearson 95 % interval, incumbents on a face, median level of the
    stalled and the other runs, and the Mann-Whitney p for the distance of x*_1 to the axis-1 face, stalled against not."""
    out = []
    for method, names in BIMODAL.items():
        for name in names[1:]:
            rs = [t for t in bimodal if (t["method"], t["transform"]) == (method, name)]
            st, ok = [t for t in rs if t["stalled"]], [t for t in rs if not t["stalled"]]
            ci = binomtest(len(st), len(rs)).proportion_ci()  # Clopper-Pearson by default
            med = lambda ts: float(np.median([t["level"] for t in ts])) if ts else None
            out.append({"method": method, "transform": name, "stalled": len(st), "runs": len(rs), "ci_low": float(ci.low),
                        "ci_high": float(ci.high), "incumbent_on_face": sum(bool(t["incumbent_on_face"]) for t in rs),
                        "level_median_stalled": med(st), "level_median_not_stalled": med(ok),
                        "mannwhitney_p_dist_to_face_1": float(mannwhitneyu([t["dist_to_face_1"] for t in st], [t["dist_to_face_1"] for t in ok]).pvalue)
                        if st and ok else float("nan")})
    return out


def merged(*parts):
    rows, evals = (sum(p, []) for p in zip(*parts))
    return rows, evals


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, evals = merged(e1.experiment_bo(list(e1.FIT_TO), [e1.REF, CHECK]),                          # section 1
                         e1.experiment_bo(["bo_default"], [e1.REF, "log f", *e1.SHIFTED_LOGS]),         # section 2
                         e1.experiment_bo(["bo_postinc", "bo_bigbox"], [e1.REF, "log f"]))              # section 3
    write_csv(rows, RESULTS / "bo.csv")
    write_csv(evals, RESULTS / "bo_evals.csv")
    summary = summarise(rows, evals)
    write_csv(summary, RESULTS / "summary.csv")
    plot_best_so_far(evals, RESULTS / "best_so_far_bo.png", col="method", line="transform", label=e1.label)

    # face diagnostic over every BO run with 90 evaluations on seeds 0-9 (E1, R, this script)
    def read(path):
        with open(path, newline="") as fh:
            rs = list(csv.DictReader(fh))
        for r in rs:
            r.update(seed=int(r["seed"]), eval=int(r["eval"]), f=float(r["f"]), best_f=float(r["best_f"]))
        return rs
    all_evals = (read(e1.RESULTS / "bo_evals.csv") + read(robustness.RESULTS / "bo_evals.csv")
                 + [r for r in evals if not (r["method"] == "bo_default" and r["transform"] in (e1.REF, "log f"))])  # rerun here
    stall = stall_table(all_evals)
    write_csv(stall, RESULTS / "stall.csv")
    section3 = section3_table(rows, evals)
    write_csv(section3, RESULTS / "section3.csv")

    bim_rows, bim_evals = merged(*(e1.experiment_bo([m], names, BIMODAL_SEEDS) for m, names in BIMODAL.items()))  # section 4
    write_csv(bim_rows, RESULTS / "bimodal_bo.csv")
    write_csv(bim_evals, RESULTS / "bimodal_evals.csv")
    bimodal = bimodal_table([r for r in bim_rows if r["transform"] != e1.REF], bim_evals)
    write_csv(bimodal, RESULTS / "bimodal.csv")
    plot_bimodal(bimodal, RESULTS / "bimodal.png")

    n = len(e1.SEEDS)
    for s in summary:
        print(f"{s['method']:11s} {s['transform']:12s} level {s['level_median']:+.2f}  gap {s['gap_median']:+.2f}  "
              f"vs default on f {s['median_vs_default_on_f']:+.2f} (better in {s['seeds_beating_default']}/{n}, p = {s['p_vs_default']:.3f})  "
              f"vs ranks {s['median_vs_rank_on_f']:+.2f} (better in {s['seeds_beating_rank']}/{n}, p = {s['p_vs_rank']:.3f})"
              + (f"  slope {s['slope_all']:.2f} (best 30: {s['slope_best30']:.2f})" if s["slope_all"] is not None else ""))
    for s in stall:
        print(f"{s['method']:11s} {s['transform']:12s} level {s['level_median']:+.2f}  on a face: evals 11-30 {s['face_share_11_30']:.2f}, "
              f"61-90 {s['face_share_61_90']:.2f}  last improvement {s['last_improvement_median']:.0f}  incumbent gap {s['incumbent_gap_median_61_90']}")
    s3 = section3_summary(section3)
    write_csv(s3, RESULTS / "section3_summary.csv")
    for s in s3:
        print(f"{s['method']:11s} {s['transform']:6s} vs default BO on the same g: median {s['median_vs_default_same_g']:+.2f}, "
              f"better in {s['seeds_better']}/{s['seeds']}, Wilcoxon p = {s['p']:.3f}")
    bs = bimodal_summary(bimodal)
    write_csv(bs, RESULTS / "bimodal_summary.csv")
    fmt = lambda v: "-" if v is None else f"{v:+.2f}"
    for s in bs:
        print(f"{s['method']:11s} {s['transform']:12s} stalled {s['stalled']}/{s['runs']} (95 % CI {s['ci_low']:.2f}-{s['ci_high']:.2f}; "
              f"incumbent on a face {s['incumbent_on_face']}/{s['runs']}), level stalled {fmt(s['level_median_stalled'])}, "
              f"not stalled {fmt(s['level_median_not_stalled'])}; distance of x*_1 to the face, stalled vs not: "
              f"Mann-Whitney p = {s['mannwhitney_p_dist_to_face_1']:.3f}")

    # self-checks, after the files are written so a failure does not lose the runs
    bad = [r for r in rows if r["method"] in e1.FIT_TO and r["max_diff_proposal"] != 0]
    assert not bad, f"invariant arms not bit-identical on {CHECK}: {[(r['method'], r['seed']) for r in bad]}"
    old = {(r["transform"], r["seed"]): r["best_f"] for r in robustness.read_e1("bo.csv", ("bo_default",))}
    assert all(np.isclose(r["best_f"], old[r["transform"], r["seed"]], rtol=1e-9) for r in rows if (r["method"], r["transform"]) in (("bo_default", e1.REF), ("bo_default", "log f"))), "default arm does not reproduce E1"
    first10 = {(r["method"], r["transform"], r["seed"]): r["best_f"] for r in rows}
    with open(robustness.RESULTS / "bo.csv", newline="") as fh:
        first10 |= {(r["method"], r["transform"], int(r["seed"])): float(r["best_f"]) for r in csv.DictReader(fh)}
    again = [r for r in bim_rows if r["seed"] in e1.SEEDS and (r["method"], r["transform"], r["seed"]) in first10]
    assert len(again) == sum(map(len, BIMODAL.values())) * len(e1.SEEDS) and all(np.isclose(r["best_f"], first10[r["method"], r["transform"], r["seed"]], rtol=1e-9) for r in again), \
        "section 4 does not reproduce seeds 0-9"
