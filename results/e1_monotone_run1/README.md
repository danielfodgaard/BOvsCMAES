# E1, first run (2026-11-06, commit 5727ffe)

Frozen copy of `results/e1_monotone/` before the rerun with `log f`, the `bo_lownoise` arm and BO_BUDGET = 90.
Seven transforms, BO budget 100, arms `bo_default` and `bo_rank`. Kept because these are the results that motivated
the low-noise arm: in `bo_evals.csv` (`bo_default`, `f^1`), the std of f over the 100 evaluated points is 615 (median
seed), so BoTorch's noise bound of 1e-4 (standardised variance) is a noise std of 0.01 × 615 ≈ 6 in f units, against a
median best f of 0.077. `bo.csv` shows `noise_final` sitting exactly on the bound for f, f^2, f^4 and 1000f.

Other changes for the rerun that this run motivated:
- `log f` added. The prediction that compressing g (p < 1, saturating) would help BO failed here (`gaps.csv`: +2.1 for
  f^0.25, +1.3 for f/(f+100)); `gp_fit.png` shows their fitted noise rising to about 1e-2, consistent with the cusp of
  f^p at the optimum being read as noise. log f is the compressing transform without a cusp, so it is the remaining test.
- Middle panel of `gap.png` (absolute best f at 100 evaluations) dropped. It read CMA-ES at evaluation 100 of its
  2000 (`best_f_at_bo_budget` in `cma.csv`) and invited a cross-method comparison the design does not support.
- BO budget cut from 100 to 90. `best_so_far_bo.png` and `bo_evals.csv`: default BO on f made its last improvement
  around evaluation 65 (median), rank-BO around 27, so the last 10 evaluations carry little.
- Y-axis label of `best_so_far_*.png` overlapped the tick labels; layout fixed in `plot_best_so_far`.
