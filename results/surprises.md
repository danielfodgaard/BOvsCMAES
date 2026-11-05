# Surprises and things worth remembering

A running log of things that were surprising, non-obvious, or took a while to understand during the
project. One entry per item: date, which experiment it belongs to, what the insight is, and why it
matters for the report. Newest at the bottom. Keep each entry short enough to read in one go.

---

## 2026-11-03 · H1 · Why H1 is a *control* and not just a first experiment

In E1 you will transform f nonlinearly, for example f⁴, and measure how much worse each method does.
But f⁴ also changes the scale of the values enormously. If a method were already sensitive to a plain
rescaling a·f + b, then a bad E1 result could be caused by the scale change rather than by the
nonlinearity. H1 exists to rule that out. It has to show that a·f + b leaves the result unchanged, so
that whatever E1 shows can be blamed on nonlinearity alone.

*Why it matters:* H1 must measure the same quantity E1 uses (best untransformed f at the end of the
budget), not only whether trajectories stay step-for-step identical. BO's trajectory diverges within a
few iterations under any transform because rounding gets amplified, but that is not the same as doing
worse.

## 2026-11-03 · H1 · Why best-so-far *untransformed* f has to be logged

"Unchanged" can mean two things. Trajectory-identical: the run on a·f + b visits exactly the same
points, step by step, as the run on f. Performance-identical: after the budget, the best f found is the
same, up to the normal spread between seeds. The H1 code measures only the first (CMA-ES mean and σ;
BO standardised data, hyperparameters, acquisition value, proposal) and reports the first step where
anything differs from the reference run.

For CMA-ES that is enough: bit-identical for every a when b = 0, and identical until generation 60 to
160 otherwise, so performance follows. For BO it is not. Every transformed BO run leaves the reference
within four iterations, and later proposals can differ by half the box width. On its own that table
says "BO is not affine-invariant in practice", which would break the control: E1 could then not tell
"f⁴ hurts BO because it is nonlinear" from "any change to the values changes what BO does".

The likely truth is milder: standardisation works (standardised data differ by 1e-16), the GP fit and
the acquisition optimiser amplify those rounding differences into a visibly different path, but both
paths find the optimum about equally well. The experiment cannot say so, because neither runner records
the untransformed f of the evaluated points. The quantity E1 uses (best untransformed f at the end of
the budget) was never measured in H1.

*Why it matters:* log best-so-far untransformed f per step in both runners, then compute the same gap
E1 uses (best f on a·f + b minus best f on f, median over seeds) next to the spread across seeds on the
untransformed problem. If the gap sits inside that spread, H1 is a working control, and the
trajectory tables become the explanation of *why* the paths differ rather than the headline. Note that
with 5 seeds the null band is wide; the outline asks for 10.
