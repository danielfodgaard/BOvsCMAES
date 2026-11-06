## H1 (affine transforms). 

For $a > 0$, both methods are invariant to $a·f + b$ in exact arithmetic: 

CMA-ES through ranks, BO through output standardisation. 

In the implementation we predict three leaks:
- (i) by design: 
  - pycma’s $f$-difference termination (tolfun, tolfunhist) changes stopping time with $a$; 
  - plain EI’s scale with $a$ interacts with absolute optimiser tolerances, while LogEI removes this;
- (ii) floating point: 
  - large b/a collapses distinct values near the optimum into ties, breaking both methods;
- (iii) amplification:
  - For BO with LogEI under clean conditions (moderate b/a), the transformed run differs from the reference only by rounding in the last bits after standardisation. 
    - These differences can grow through the steps that follow 
    (hyperparameter fitting and acquisition optimisation), 
    for example when two acquisition maxima are nearly tied. 
    So proposals may start identical to machine precision and diverge after some iteration. 
  - CMA-ES has no such step, since it only sorts the values. 

## E1 (monotone transforms).

For strictly increasing $g$, CMA-ES sees only ranks and so cannot tell $g(f)$ from $f$;
BO fits a GP to the values, so it can.
Tested with $g = f^p$ ($p \in \{1/4, 1/2, 1, 2, 4\}$), $f/(f+100)$ and the affine control
$1000f$, scored as the paired gap $\log_{10}$ best untransformed $f$ on $g$ $-$ same on $f$.

Predictions:
- (i) CMA-ES, fixed budget: bit-identical across $g$, gap exactly 0.
- (ii) CMA-ES, default stopping: not invariant, since `tolfun` compares absolute
  differences in $g$; gap $> 0$ where $g' \to 0$ at the optimum, $< 0$ where
  $g' \to \infty$, about $\pm\log_{10} g'$ for the affine and saturating cases.
- (iii) BO, default: not invariant beyond the control's rounding band; expanding $g$
  ($p > 1$) hurts most, compressing $g$ ($p < 1$, saturating) may help by reducing
  the dynamic range.
- (iv) BO, ranks: exactly invariant, at a cost on $f$ itself, since ranks flatten
  the values near the optimum.

### Changes after the first run (2026-11-06, `results/e1_monotone_run1/`)

The first run confirmed (i), (ii) and (iv) and falsified the second half of (iii):
compressing $g$ hurt too, consistent with the cusp of $f^p$ at the optimum being
fitted as noise. Changes for the rerun, each with its prediction:
- $\log f$ added, the compressing transform without a cusp: the one $g$ that may
  still help default BO.
- Third BO arm with the noise floor lowered from $10^{-4}$ to $10^{-8}$ (on $f$ the
  default floor is a noise std of about 6 in $f$ units against a best $f$ of 0.08):
  should improve the tail on $f$ and leave the ordering of transforms in (iii)
  unchanged.
- BO budget 100 $\to$ 90 (default BO on $f$ stopped improving around evaluation 65)
  and the absolute-level panel dropped (it compared BO at 100 evaluations with
  CMA-ES at 100 of 2000, which the design does not support): no change in results
  expected.
