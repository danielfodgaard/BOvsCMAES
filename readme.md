# Which invariances survive implementation? CMA-ES vs. Bayesian optimisation

In theory, CMA-ES does not care how a problem is written down. It behaves the same if you replace the objective f by
log f or f⁴, or if you rescale or rotate the search space (as long as the starting point is transformed too).
Bayesian optimisation (BO) has none of them built in, only what preprocessing of the inputs and outputs gives it. This project tests whether that holds in
practice, in the libraries people actually use: **pycma** and **BoTorch**.

The idea is simple: run each method on a problem and on a transformed copy of the same problem, with the same seed.
The transformation does not change the problem, so any difference in the result is caused by the method.

The report is [`report/report.pdf`](report/report.pdf) (10 pages), built from `report/report.tex`.

## Setup

You need Python 3.13. The exact library versions used for the results are pinned in `requirements.txt`, and
everything runs on the CPU.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

No data is needed. Every test problem is a synthetic function (an ellipsoid or a sphere) that the scripts generate
from the seed.

## Experiments

The report numbers the experiments 0–5. The code and the result folders use short code names.

| Report | Code name | What it tests | Script | Results in | Runtime* |
| --- | --- | --- | --- | --- | --- |
| Experiment 0 | H1 | Affine transformations a·f + b (a control) | `code/invariance_check/invariance_cma.py` and `invariance_bo.py` | `results/invariance/` | 15 s + 80 s |
| Experiment 1 | E1 | Monotone transformations g(f), such as log f and f⁴ | `code/e1_monotone/monotone.py` | `results/e1_monotone/` | 20 min |
| Experiment 2 | E2 | Changing the units of x | `code/e2_rescaling/rescaling.py` | `results/e2_rescaling/` | 10 min |
| Experiment 3 | E3 | Rotating x | `code/e3_rotation/rotation.py` | `results/e3_rotation/` | 7 min |
| Experiment 4 | R | BO with a learned output transformation (Box-Cox, as in HEBO) | `code/r_robustness/robustness.py` | `results/r_robustness/` | 7 min |
| Partly used | R follow-up | 30-seed runs for experiment 4's shifted problems, plus extra analysis not in the report | `code/r_robustness/followup.py` | `results/r_followup/` | 30–35 min |
| Experiment 5 | D | BO choosing an output transformation by leave-one-out cross-validation | `code/d_selection/selection.py` | `results/d_selection/` | 33 min |

\*On a 16-thread laptop. Runs are parallelised over seeds.

Each script starts with a docstring that explains its design, settings, seeds and the files it writes. The code
shared by all experiments (the CMA-ES and BO runners, the test problems, and the plotting and CSV helpers) is in
`code/invariance_check/`.

## How to reproduce the results

Run the scripts from the repository root, in the order below, because later scripts read earlier results. Each script
overwrites its own results folder. The whole run takes about two hours.

```bash
.venv/bin/python code/invariance_check/invariance_cma.py   # experiment 0, CMA-ES
.venv/bin/python code/invariance_check/invariance_bo.py    # experiment 0, BO
.venv/bin/python code/e1_monotone/monotone.py              # experiment 1
.venv/bin/python code/e2_rescaling/rescaling.py            # experiment 2
.venv/bin/python code/e3_rotation/rotation.py              # experiment 3
.venv/bin/python code/r_robustness/robustness.py           # experiment 4
.venv/bin/python code/r_robustness/followup.py             # follow-up (30-seed runs for experiment 4)
.venv/bin/python code/d_selection/selection.py             # experiment 5
.venv/bin/python code/d_selection/test_selection.py        # unit tests for experiment 5 (about 10 s)
```

**What to expect when you rerun**

- CMA-ES runs are deterministic. They have been reproduced bit for bit on a second machine.
- BO runs are deterministic on the same machine. On a different CPU the numerical libraries (MKL, OpenBLAS) round
  slightly differently, so single BO runs can end differently, but the overall results hold. In an independent rerun
  on another machine the BO medians moved by 0.1–0.3 orders of magnitude and the conclusions stayed the same.
- The saved results were produced on an AMD Ryzen 7 PRO 7840U under Linux.
- Run the whole chain on one machine. `followup.py` checks that its BO runs reproduce the saved experiment 1 and 4
  runs exactly, so if you run it on another machine against the shipped CSVs it will stop with an `AssertionError`.
  Rerunning experiments 1 and 4 first on the same machine avoids this.

**Rebuilding the report**

Copy the figures the report uses from `results/` into `report/figures/`, then build the PDF. On Overleaf, upload the
`report/` folder and set `report.tex` as the main file instead.

```bash
cp results/invariance/best_so_far_cma_budget.png    report/figures/h1_cma.png
cp results/invariance/best_so_far_bo_logei.png      report/figures/h1_bo.png
cp results/e1_monotone/best_so_far_cma_selected.png report/figures/e1_cma.png
cp results/e1_monotone/best_so_far_bo_selected.png  report/figures/e1_bo.png
cp results/e2_rescaling/best_so_far_cma.png         report/figures/e2_cma.png
cp results/e2_rescaling/best_so_far_bo.png          report/figures/e2_bo.png
cp results/e3_rotation/best_so_far_cma.png          report/figures/e3_cma.png
cp results/e3_rotation/best_so_far_bo.png           report/figures/e3_bo.png
cp results/d_selection/level_report.png             report/figures/d_level.png
cp results/d_selection/choices_report.png           report/figures/d_choices.png
cd report && pdflatex report && bibtex report && pdflatex report && pdflatex report
```

## More results than in the report

The report only has room for the most informative results. The `results/` folder also contains:

- the rest of the follow-up (`r_followup/`): two more ways of making BO invariant (normal scores and Shirakawa's
  Lebesgue estimate), BO on log(f + c), an analysis of how often BO's proposals sit on the edge of the box, BO with a
  posterior-mean incumbent, a box twice as wide, and stall rates over 30 seeds;
- experiment 1's other transformations (f^1/4, f^1/2, f², f/(f+100)) and the fitted GP noise and lengthscales
  (`e1_monotone/`);
- experiment 0 with EI as well as LogEI, CMA-ES with its default stopping rule, and the step at which each run first
  diverges (`invariance/`);
- the number of extra evaluations each rescaled or rotated run needs to reach fixed targets (`e2_rescaling/summary.csv`,
  `e3_rotation/summary.csv`), and experiment 3's BO results after 90 evaluations (`e3_rotation/summary_90.csv`);
- which transformation experiment 5 picks when it chooses only once (`d_selection/choices.png`).

## Built-in checks

- **Invariance is measured, not assumed.** The CSVs record the first step at which a run departs from its reference
  run (`first_div`, empty if never) and the largest difference (`max_diff_*`). For CMA-ES in experiments 0 and 1 this
  is checked bit for bit.
- **Key assumptions are asserted in the code**, for example that the rotated start point stays inside the box, that
  the invariant BO variants propose identical points on f and f⁴, and that the follow-up reproduces the earlier runs.
- **Experiment 5's selection criterion has unit tests** (`code/d_selection/test_selection.py`).
- **The reason raw-input BO fails in experiment 2** is saved in `e2_rescaling/kernel_check.csv`.

## Statistics

- Every comparison is paired: the same seed is run on the original and the transformed problem, and both are scored
  on the original f.
- The gap is log10(best f on the transformed problem) − log10(best f on the original). A positive gap means the
  transformation hurt.
- The report shows medians over seeds. The scripts also save sign counts, paired Wilcoxon tests (not corrected for
  multiple comparisons) and confidence intervals, in `e2_rescaling/summary.csv`, `e3_rotation/summary.csv` and
  `summary_90.csv`, `d_selection/summary.csv`, and `r_followup/section3_summary.csv` and `bimodal_summary.csv`.
- Seeds: BO uses 10 (5 in experiment 0, 30 for experiment 4's shifted problems). CMA-ES uses 50 in experiments 2 and
  3, 10 in experiment 1 and 5 in experiment 0.

## Repository layout

```
code/
  invariance_check/   shared runners and helpers, and experiment 0
  e1_monotone/  e2_rescaling/  e3_rotation/  r_robustness/  d_selection/
results/
  <experiment>/       the CSVs and figures each script writes (columns explained in the script's docstring)
  e1_monotone_run1/   an earlier first run of experiment 1, kept because it motivated the low-noise BO variant;
                      it was made by monotone.py as of commit 5727ffe (BO budget 100), not by the current code
report/
  report.tex          report source (report.pdf is the built report)
  refs.bib            bibliography
  figures/            copies of the result figures used in the report
authors.md            name and student number
requirements.txt      pinned library versions
```
