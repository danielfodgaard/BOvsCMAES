import matplotlib.pyplot as plt
import numpy as np
from first_experiment import PROBLEMS, RESULTS

fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(5 * len(PROBLEMS), 4), sharey=True)
for ax, problem in zip(axes, PROBLEMS):
    for method, color in [("cma", "tab:blue"), ("bo", "tab:orange")]:
        for i, path in enumerate(sorted(RESULTS.glob(f"{problem}_{method}_*.csv"))):
            f = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
            best_so_far = np.minimum.accumulate(f)
            ax.plot(np.arange(1, len(f) + 1), np.log10(best_so_far), color=color,
                    label=method.upper() if i == 0 else None)
    ax.set_title(problem)
    ax.set_xlabel("evaluation")
    ax.legend()
axes[0].set_ylabel("log10(best f so far)")

fig.tight_layout()
fig.savefig(RESULTS / "pilot.png", dpi=150)
