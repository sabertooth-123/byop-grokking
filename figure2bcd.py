"""Run one of the remaining Section 5.1 sweeps from its first trial."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

from provenance import write_run
from baseline import Config, ROOT, make_problem
from setup_reference import REFERENCE, REVISION, verify_checkout
from weight_decay import PREFIX, TRIALS, crossings, plt, save_csv

GRIDS = {
    "sample_size": ("n_train", (10, 20, 40, 80, 160), "figure2b.png", "Training sample size n"),
    "feature_dimension": ("dimension", (1000, 1500, 2000, 2500, 3000), "figure2c.png", "Feature dimension m"),
    "initialization": ("student_variance", tuple(10**k for k in range(7)), "figure2d.png", "Initialization variance ν²"),
}
UNOBSERVED = "unobserved"


def reference_module():
    verify_checkout(REFERENCE)
    sys.path.insert(0, str(REFERENCE))
    import ridge_regression
    return ridge_regression


def configs(panel, base):
    field, grid, _, _ = GRIDS[panel]
    for i, value in enumerate(grid):
        for trial in range(TRIALS):
            offset = 1000 * (i * TRIALS + trial)
            yield value, trial+1, replace(base, **{field: value}, teacher_seed=base.teacher_seed+offset,
                                          train_seed=base.train_seed+offset, test_seed=base.test_seed+offset)


def smallest_positive(x):
    # The two Gram matrices share their nonzero spectrum.
    gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
    spectrum = np.linalg.eigvalsh(gram)
    return spectrum[spectrum > 1e-12].min()


def bounds(ref, panel, c, smallest):
    n, m, v = c.n_train, c.dimension, c.student_variance
    args = dict(eta=c.step_size, epsilon=c.epsilon, weight_decay=c.weight_decay)
    upper_args = dict(n=n, m=m, eta=c.step_size, epsilon=c.epsilon, lambda_min_pos=smallest)
    scale = math.sqrt(v) if panel == "initialization" else v
    out = dict(t1_upper=ref.theoretical_upper_bound_t1(**upper_args, nu_square=scale))
    if panel == "sample_size":
        lower = ref.theoretical_lower_bound_sample_size_t2([n], m=m, nu_square=v, **args)
    elif panel == "feature_dimension":
        lower = ref.theoretical_lower_bound_width_t2([m], n=n, nu_square=v, **args)
    else:
        lower = ref.theoretical_lower_bound_initialization_t2([v], n=n, m=m, **args)
        out["t1_upper_variance"] = ref.theoretical_upper_bound_t1(**upper_args, nu_square=v)
    out["t2_lower"] = lower[0]
    return out


def summarize(panel, rows):
    stats = []
    metrics = ["t1", "t2", "delay", "t1_upper", "t2_lower"]
    if panel == "initialization":
        metrics.append("t1_upper_variance")
    for value in GRIDS[panel][1]:
        group = [r for r in rows if r["value"] == value]
        if sorted(r["trial"] for r in group) != list(range(1, TRIALS+1)):
            raise ValueError(f"Incomplete or duplicate trials for {panel}={value}")
        entry = dict(value=value, trials=len(group))
        for key in metrics:
            data = [r[key] for r in group if r[key] != UNOBSERVED]
            entry[key+"_observed"] = len(data)
            if len(data) < len(group):
                mean = sd = float("nan")  # np.mean/var over a reference row containing NaN
            elif min(data) == max(data):
                mean, sd = data[0], 0.
            else:
                mean, sd = np.mean(data), np.std(data, ddof=0)
            entry[key+"_mean"], entry[key+"_sd"] = mean, sd
        stats.append(entry)
    return stats


def plot(panel, stats, out):
    _, values, filename, label = GRIDS[panel]
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    for key, fmt, color, legend in [
        ("t1", "x-", "#236193", r"$t_1$ empirical"), ("t1_upper", "x--", "#236193", r"$t_1$ upper bound"),
        ("t2", "o-", "#ac3425", r"$t_2$ empirical"), ("t2_lower", "o--", "#ac3425", r"$t_2$ lower bound")
    ]:
        errors = None if key == "t2_lower" else [s[key+"_sd"] for s in stats]
        ax.errorbar(values, [s[key+"_mean"] for s in stats], yerr=errors, fmt=fmt,
                    color=color, capsize=4, label=legend)
    if panel == "initialization":
        ax.set_xscale("log", base=10)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    else:
        ax.set_yscale("log", base=2)
        if panel == "sample_size":
            ax.set_xscale("log", base=2)
        ax.set_xticks(values, [str(v) for v in values])
    ax.set(xlabel=label, ylabel="Gradient updates")
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(out / filename, dpi=180)
    plt.close(fig)


def run(panel):
    start = perf_counter()
    ref = reference_module()
    base = Config.load()
    out = ROOT / "results" / panel
    out.mkdir(exist_ok=True)
    files = ("figure2bcd.py", "baseline.py", "weight_decay.py", "setup_reference.py")
    metadata = dict(panel=panel, base_config=asdict(base), field=GRIDS[panel][0], grid=GRIDS[panel][1],
                    trials_per_value=TRIALS, data_seed_stride=1000, ddof=0, prefix=PREFIX,
                    eigenvalue_cutoff=1e-12, reference_commit=REVISION,
                    project_head=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                    python=platform.python_version(), numpy=np.__version__, matplotlib=plt.matplotlib.__version__,
                    platform=platform.platform(), openblas_num_threads=os.getenv("OPENBLAS_NUM_THREADS"))
    rows = []
    for value, trial, c in configs(panel, base):
        trial_start = perf_counter()
        _, train, test, initial = make_problem(c)
        t1, t2, skipped = crossings(c, train, test, initial)
        smallest = smallest_positive(train[0])
        rows.append(dict(value=value, trial=trial, teacher_seed=c.teacher_seed, train_seed=c.train_seed,
                         test_seed=c.test_seed, student_seed=c.student_seed, student_std=math.sqrt(c.student_variance),
                         t1=UNOBSERVED if t1 is None else t1, t2=UNOBSERVED if t2 is None else t2,
                         delay=UNOBSERVED if None in (t1, t2) else t2-t1, lambda_min_positive=smallest,
                         **bounds(ref, panel, c, smallest), skipped_updates=skipped, seconds=perf_counter()-trial_start))
        if trial == TRIALS:
            print(f"{panel}={value} completed ({len(rows)} trials)", flush=True)
    stats = summarize(panel, rows)
    save_csv(out / "trials.csv", rows)
    save_csv(out / "summary.csv", stats)
    plot(panel, stats, out)
    metadata["wall_seconds"] = perf_counter()-start
    write_run(out/"run.json", metadata,
              "Restart seed progression per panel; add 1000 per trial to teacher, train and test seeds; fixed student seed.",
              "Independent Gaussian rows with covariance I/m; noiseless unit teacher; override the configured panel field.")
    print(f"{panel} finished in {metadata['wall_seconds']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=GRIDS)
    run(parser.parse_args().panel)
