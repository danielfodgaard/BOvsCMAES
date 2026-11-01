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