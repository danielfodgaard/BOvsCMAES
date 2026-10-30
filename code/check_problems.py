import numpy as np
from first_experiment import make_problem, sphere, ellipsoid, Logged, PROBLEMS, D, SEEDS, kappa

e = np.zeros(D)
e[-1] = 1
assert np.isclose(ellipsoid(e), kappa), "last slot should be kappa"

f = make_problem("ellipsoid_rot", 0)
assert np.isclose(f(np.zeros((1, D))), f(np.zeros(D))), "(1, D) and (D,) inputs should agree"

for name in PROBLEMS:
    for seed in SEEDS:
        # same seed -> same first draw, so this matches the x_opt inside make_problem
        x_opt = np.random.default_rng(seed).uniform(-3, 3, D)
        f = make_problem(name, seed)
        assert np.isclose(f(x_opt), 0), f"{name}/{seed}: f(x_opt) should be 0"

        x = np.random.default_rng(100 + seed).uniform(-5, 5, D)
        assert f(x) == make_problem(name, seed)(x), f"{name}/{seed}: same seed should give same f"

logged = Logged(sphere)
values = [logged([1.0, 2.0]), logged(np.zeros((1, 2))), logged([3.0, 0.0])]
assert len(logged.history) == 3
assert all(type(v) is float for v in values)
assert values == [5.0, 0.0, 9.0]
times = [t for _, _, t in logged.history]
assert times == sorted(times), "times should increase"

print("all checks passed")
