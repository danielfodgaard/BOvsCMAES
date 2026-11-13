"""Unit tests of D's selection criterion. Run with: python code/d_selection/test_selection.py (about 10 s).

1. The closed-form LOO mean and variance equal an explicit leave-one-out prediction with the same hyperparameters.
2. Change of variables: on f^4 data the candidate v^(1/4) recovers f, and its score equals v^1's score on f data minus
   sum log g'(f_i), g(f) = f^4. So the criterion compares f^4's candidates exactly as densities of the same values.
3. Scale: multiplying the data by 1000 lowers every power's and log's score by n log 1000 (to 1e-2; the GP fits themselves agree to about 1e-3)
   and leaves the choice unchanged.
4. Candidates that are undefined on the data (powers other than 1 and log, once a value is <= 0) are left out.
5. representation(): f and f^4 offer the same representations of f (the grid is symmetric), and h(g(f)) = f exactly for
   v^1 on f and 1000 f, v^(1/4) on f^4 and exp on log f, and for nothing on f / (f + 100).
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import selection as d
from invariance_util import make_problem

SEED, N = 3, 15
X01 = np.random.default_rng(0).uniform(0, 1, (N, d.e1.D))
f = np.array([make_problem(d.e1.D, SEED)(10 * x - 5) for x in X01])


def scores(v):
    return {k: d.loo_score(d.fit_gp(X01, -h, SEED), X01, j) for k, (h, j) in d.candidates(v).items()}


def test_closed_form_loo():
    gp = d.fit_gp(X01, -f, SEED)
    mu, var, z = d.loo_terms(gp, X01)
    with torch.no_grad():
        Y = torch.from_numpy(-f)
        assert np.allclose(z, ((Y - Y.mean()) / Y.std()).numpy()), "train_targets are not the standardised targets"
        X, m, noise = torch.from_numpy(X01), gp.mean_module.constant, gp.likelihood.noise
        for i in range(N):
            o = [k for k in range(N) if k != i]
            K = gp.covar_module(X[o]).to_dense() + noise * torch.eye(N - 1, dtype=X.dtype)
            k = gp.covar_module(X[o], X[i:i + 1]).to_dense().squeeze(1)
            w = torch.linalg.solve(K, k)
            mu_i = m + w @ (torch.from_numpy(z[o]) - m)
            var_i = gp.covar_module(X[i:i + 1]).to_dense().squeeze() + noise - k @ w
            assert math.isclose(mu[i], float(mu_i), rel_tol=1e-6, abs_tol=1e-8), (i, mu[i], float(mu_i))
            assert math.isclose(var[i], float(var_i), rel_tol=1e-6), (i, var[i], float(var_i))


def test_change_of_variables():
    s_f, s_f4 = scores(f), scores(f**4)
    expected = s_f["v^1"] - np.sum(np.log(4 * f**3))
    assert math.isclose(s_f4["v^0.25"], expected, rel_tol=1e-5, abs_tol=1e-4), (s_f4["v^0.25"], expected)


def test_scale():
    s1, s1000 = scores(f), scores(1000 * f)
    for k in s1:
        if k != "exp":
            assert math.isclose(s1000[k] - s1[k], -N * math.log(1000), abs_tol=1e-2), (k, s1000[k] - s1[k])  # GP fits agree to ~1e-3
    best = lambda s: max((k for k in s if k != "exp"), key=s.get)
    assert best(s1) == best(s1000)


def test_undefined_candidates():
    v = np.log(f / f.min()) - 0.5  # some values negative, as log f once f < 1
    assert (v <= 0).any() and set(d.candidates(v)) == {"v^1", "exp"}
    assert set(d.candidates(f)) == set(d.ALL_NAMES)


def test_representation():
    on = lambda g: {d.representation(g, h) for h in d.ALL_NAMES}
    shared = {"f^0.25", "f^0.5", "f^1", "f^2", "f^4", "log f"}
    assert shared <= on("f^1") and shared <= on("f^4"), (on("f^1"), on("f^4"))
    hits = {g: [h for h in d.ALL_NAMES if d.representation(g, h) == "f^1"] for g in d.TRANSFORMS}
    assert hits == {"f^1": ["v^1"], "f^4": ["v^0.25"], "log f": ["exp"], "f/(f+100)": [], "1000f": ["v^1"]}, hits


if __name__ == "__main__":
    torch.set_num_threads(1)
    for test in (test_closed_form_loo, test_change_of_variables, test_scale, test_undefined_candidates, test_representation):
        test()
        print(f"{test.__name__}: ok")
    print("scores on f:", {k: round(v, 2) for k, v in scores(f).items()})
