"""Run the baseline and write all observed losses and the figure."""
from dataclasses import asdict
import json
import os
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline import Config, ROOT, fit, make_problem


def main():
    c = Config.load()
    teacher, train, test, initial = make_problem(c)
    final, losses, t1, t2 = fit(c, train, test, initial)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    steps = np.arange(len(losses))
    np.savetxt(out / "losses.csv", np.column_stack((steps, losses)),
               fmt=["%d", "%.17g", "%.17g"], delimiter=",", comments="",
               header="updates,train_mse,test_mse")
    np.savez(out / "weights.npz", teacher=teacher, initial=initial, final=final)
    summary = dict(config=asdict(c), t1=t1, t2=t2, observations=len(losses),
                   delay=None if None in (t1, t2) else t2-t1,
                   python=platform.python_version(), platform=platform.platform(),
                   numpy=np.__version__, matplotlib=matplotlib.__version__,
                   openblas_num_threads=os.getenv("OPENBLAS_NUM_THREADS"))
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(steps, losses[:, 0], label="Training", color="#166087")
    ax.plot(steps, losses[:, 1], label="Test", color="#ad492a")
    ax.axhline(c.epsilon, color="0.4", ls="--", label=f"MSE = {c.epsilon:g}")
    for step, col in [(t1, 0), (t2, 1)]:
        if step is not None:
            ax.plot(step, losses[step, col], "o", color="black", ms=4)
            ax.annotate(f"t{col+1} = {step:,}", (step, losses[step, col]),
                        xytext=(0, 12), textcoords="offset points", ha="center")
    ax.set(xscale="symlog", yscale="log", xlabel="Gradient updates (log scale beyond 2)",
           ylabel="Mean squared error", title="Grokking in ridge regression")
    ax.legend()
    fig.savefig(out / "baseline.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
