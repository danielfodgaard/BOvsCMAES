# Results

## H1 (affine transforms a·f + b) — `results/invariance/`, 5 seeds, a ∈ {10⁻³, 1, 10³}, b ∈ {0, 10³, 10⁸}

CMA-ES is bit-identical to the run on f for every a when b = 0, and with b > 0 diverges only once
a·f + b stops resolving differences near the optimum in floating point (generation ~60 for b = 10⁸,
~110–180 for b = 10³), after which it stops on `tolflatfitness`; so scaling leaks nothing and shifts
leak only through rounding. BO (EI and LogEI) leaves the reference within 0–4 iterations for every
(a, b) ≠ (1, 0), since standardised data agree only to ~10⁻¹⁶ and the GP fit and acquisition optimiser
amplify that to proposals up to the box width apart; E1's affine control (1000·f, 10 seeds) shows this
is a trajectory difference, not a performance one, with a paired gap in best f inside ±0.35 log10,
within the seed spread.
