# ---------- 0. imports ---------
import time
from pathlib import Path

import cma
import numpy as np
import scipy.stats
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

# ---------- 1. config ----------
D = 5
BOUNDS = [-5, 5]
CMA_BUDGET = 1000
BO_BUDGET = 60
SEEDS = [0, 1, 2]
kappa = 10e4
RESULTS = Path(__file__).parent.parent / "results" / "pilot"

# ---------- 2. test functions ----------
def sphere(z):
    sum = 0
    for i in range(len(z)):
        sum += z[i]**2

    return sum

def ellipsoid(z):
    sum = 0
    for i in range(len(z)):
        sum += (kappa**(i/(len(z)-1))) * (z[i]**2)
    return sum

def make_problem(name, seed):
    rng = np.random.default_rng(seed)
    base_fn, rotate = PROBLEMS[name]


    #random uniform in [-3, 3]^D        (fixed by seed)
    x_opt = rng.uniform(-3, 3, D)

    #random rotation (or identity)        (fixed by seed)
    if rotate:
        R = scipy.stats.special_ortho_group(D, rng).rvs()
    #no rotation
    else:
        R = np.eye(D)

    # f* = 0
    def f(x):
        x = np.array(x).flatten()
        return base_fn(R @ (x - x_opt))

    return f

# True means rotation/ vice versa for false
PROBLEMS = {"sphere":       (sphere,    False),
            "ellipsoid":    (ellipsoid, False),
            "ellipsoid_rot":(ellipsoid, True)}

# ---------- 3. logging wrapper ----------
class Logged:
    #wraps f; every call appends (x, f(x), time) to a list
    #-> afterwards: history of all evaluations, in order
    def __init__(self, f):
        self.f = f
        self.history = []
        self.time_start = time.perf_counter()

    def __call__(self, x):
        x = np.array(x).flatten()
        value = float(self.f(x))
        t = time.perf_counter() - self.time_start

        tup = (x, value, t)
        self.history.append(tup)

        return value


# ---------- 4. one runner per method ----------
def run_cma(f, seed):
    x0 = np.random.default_rng(seed).uniform(BOUNDS[0], BOUNDS[1], D)
    # cma treats seed=0 as "pick a random seed", so shift by one
    es = cma.CMAEvolutionStrategy(x0, 2.5, {"bounds": BOUNDS, "seed": seed + 1, "verbose": -9})
    while len(f.history) < CMA_BUDGET:
        X = es.ask()
        es.tell(X, [f(x) for x in X])


def run_bo(f, seed):
    torch.manual_seed(seed)  # optimize_acqf draws random start points
    bounds = torch.tensor([[BOUNDS[0]] * D, [BOUNDS[1]] * D], dtype=torch.double)
    X = draw_sobol_samples(bounds, n=10, q=1, seed=seed).squeeze(1)
    # BoTorch maximizes, so the GP models -f
    Y = torch.tensor([[-f(x.numpy())] for x in X], dtype=torch.double)

    while len(X) < BO_BUDGET:
        gp = SingleTaskGP(X, Y, input_transform=Normalize(D, bounds=bounds),
                          outcome_transform=Standardize(1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acq = LogExpectedImprovement(gp, best_f=Y.max())
        x_next, _ = optimize_acqf(acq, bounds, q=1, num_restarts=10, raw_samples=256)

        X = torch.cat([X, x_next])
        Y = torch.cat([Y, torch.tensor([[-f(x_next.numpy())]], dtype=torch.double)])


# ---------- 5. main loop ----------
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for problem in PROBLEMS:
        for method in [run_cma, run_bo]:
            for seed in SEEDS:
                f = Logged(make_problem(problem, seed))
                method(f, seed)

                # one row per evaluation: eval number, f(x), time, x_1..x_D
                rows = [[i, value, t, *x] for i, (x, value, t) in enumerate(f.history, 1)]
                name = f"{problem}_{method.__name__[4:]}_{seed}"
                header = "eval,f,time," + ",".join(f"x{j}" for j in range(1, D + 1))
                np.savetxt(RESULTS / f"{name}.csv", rows, delimiter=",", header=header, comments="")
                print(f"{name}: best f = {min(v for _, v, _ in f.history):.3g}")


if __name__ == "__main__":
    main()
