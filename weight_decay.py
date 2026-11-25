"""Milestone 2A. Reproduce the regularization branch of the Figure 2 driver."""
import argparse
import csv
from dataclasses import asdict, replace
import json
import math
import os
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from provenance import write_run
from baseline import Config, ROOT, fit, make_problem
from setup_reference import REVISION

DECAYS = tuple(1e-6 * 2**k for k in range(1, 6))
TRIALS = 10
PREFIX = 2048
METRICS = ("t1", "t2", "delay", "t1_upper", "t2_lower")


def trial_configs(base):
    for group, decay in enumerate(DECAYS):
        for trial in range(TRIALS):
            offset = 1000 * (group * TRIALS + trial)
            yield trial + 1, replace(base, weight_decay=decay, teacher_seed=base.teacher_seed + offset,
                                     train_seed=base.train_seed + offset, test_seed=base.test_seed + offset)


def theory(c, features):
    eigs = np.linalg.eigvalsh(features.T @ features)
    smallest = eigs[eigs > 1e-12].min()
    upper = c.n_train * math.log(14 * c.dimension * c.student_variance / c.epsilon)
    lower = math.log((c.dimension - c.n_train) * c.student_variance / (8 * c.dimension * c.epsilon))
    return smallest, upper / (2 * c.step_size * smallest), lower / (2.02 * c.step_size * c.weight_decay)


def advance_above_threshold(c, train, test, weights, remaining):
    """Advance through discrete GD steps whose held-out RMSE lower bound exceeds sqrt(epsilon)."""
    x, y = train
    xt, yt = test
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    curvature = s**2 / len(y) + c.weight_decay
    row_decay = 1 - c.step_size * curvature
    null_decay = 1 - c.step_size * c.weight_decay
    if not (0 < null_decay < 1 and np.all((0 <= row_decay) & (row_decay < 1))):
        return 0, weights

    projection = vt @ weights
    null = weights - vt.T @ projection
    optimum = (s * (u.T @ y)) / (len(y) * curvature)
    row_error = projection - optimum
    # At k more updates the held-out residual is a + null_decay**k b
    # plus Xt V diag(row_decay**k) row_error. Bound the last term by its norm.
    a = (xt @ (vt.T @ optimum) - yt) / math.sqrt(len(yt))
    b = (xt @ null) / math.sqrt(len(yt))
    aa, ab, bb = a @ a, a @ b, b @ b
    row_bound = np.linalg.norm(xt) * np.linalg.norm(row_error) / math.sqrt(len(yt))
    skip = remaining
    for start in range(1, remaining + 1, 32768):
        k = np.arange(start, min(start + 32768, remaining + 1))
        z = null_decay**k
        main_norm = np.sqrt(np.maximum(0., aa + z * (2 * ab + z * bb)))
        lower = main_norm - row_bound * row_decay.max()**k
        possible = np.flatnonzero(lower <= math.sqrt(c.epsilon) + 1e-10)
        if possible.size:
            skip = int(k[possible[0]]) - 1
            break
    # Leave several literal updates around each candidate, including the step cap.
    skip = max(0, skip - 8)
    if skip == 0:
        return 0, weights
    advanced = null_decay**skip * null + vt.T @ (optimum + row_decay**skip * row_error)
    return skip, advanced


def crossings(c, train, test, initial, literal=False):
    prefix = c.max_steps if literal else min(PREFIX, c.max_steps)
    weights, _, t1, t2 = fit(replace(c, max_steps=prefix), train, test, initial)
    if prefix == c.max_steps or (t1 is not None and t2 is not None):
        return t1, t2, 0
    skipped = 0
    if t1 is not None:
        skipped, weights = advance_above_threshold(c, train, test, weights, c.max_steps - prefix)
    offset = prefix + skipped
    _, _, later_train, later_test = fit(replace(c, max_steps=c.max_steps-offset), train, test, weights)
    if t1 is None and later_train is not None:
        t1 = offset + later_train
    if t2 is None and later_test is not None:
        t2 = offset + later_test
    return t1, t2, skipped


def summarize(rows):
    summary = []
    for decay in DECAYS:
        group = [r for r in rows if r["weight_decay"] == decay]
        if len(group) != TRIALS:
            raise ValueError(f"Expected {TRIALS} trials for {decay}")
        entry = dict(weight_decay=decay, trials=len(group))
        for key in METRICS:
            values = np.asarray([r[key] for r in group], dtype=float)
            # The t2 lower bound is the same scalar for every trial at a given decay.
            if np.all(values == values[0]):
                entry[key + "_mean"], entry[key + "_sd"] = values[0], 0.
            else:
                entry[key + "_mean"], entry[key + "_sd"] = values.mean(), values.std(ddof=0)
        summary.append(entry)
    return summary


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.5), layout="constrained")
    for key, fmt, color, label in [
        ("t1", "x-", "#236193", r"$t_1$ empirical"), ("t1_upper", "x--", "#236193", r"$t_1$ upper bound"),
        ("t2", "o-", "#ac3425", r"$t_2$ empirical"), ("t2_lower", "o--", "#ac3425", r"$t_2$ lower bound")
    ]:
        errors = None if key == "t2_lower" else [row[key + "_sd"] for row in summary]
        ax.errorbar(DECAYS, [row[key + "_mean"] for row in summary], yerr=errors,
                    fmt=fmt, color=color, capsize=4, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(DECAYS, [f"{v*1e6:g}" for v in DECAYS])
    ax.set(xlabel=r"Weight decay $\lambda$ ($\times 10^{-6}$)", ylabel="Gradient updates",
           title="Effect of weight decay on grokking")
    ax.grid(alpha=.3)
    ax.legend(loc="center left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--literal", action="store_true", help="Evaluate every update with baseline.fit")
    args = parser.parse_args()
    start = perf_counter()
    base = Config.load()
    name = "weight_decay_literal" if args.literal else "weight_decay"
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial, c in trial_configs(base):
        trial_start = perf_counter()
        _, train, test, initial = make_problem(c)
        t1, t2, skipped = crossings(c, train, test, initial, args.literal)
        eig, upper, lower = theory(c, train[0])
        rows.append(dict(weight_decay=c.weight_decay, trial=trial, teacher_seed=c.teacher_seed,
                         train_seed=c.train_seed, test_seed=c.test_seed, student_seed=c.student_seed,
                         t1=t1, t2=t2, delay=None if None in (t1,t2) else t2-t1, lambda_min_positive=eig,
                         t1_upper=upper, t2_lower=lower, skipped_updates=skipped, seconds=perf_counter()-trial_start))
        print(f"Trial {len(rows):2}/50: lambda={c.weight_decay:g}, t1={t1}, t2={t2}", flush=True)
    summary = summarize(rows)
    save_csv(out / "trials.csv", rows)
    save_csv(out / "summary.csv", summary)
    plot(summary, out / "figure2a.png")
    metadata = dict(base_config=asdict(base), weight_decays=DECAYS, trials_per_decay=TRIALS,
                    data_seed_stride=1000, ddof=0, gram_eigenvalue_cutoff=1e-12,
                    solver="literal" if args.literal else "certified interval skip with literal GD at crossings",
                    direct_prefix=base.max_steps if args.literal else PREFIX, reference_commit=REVISION,
                    python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                    platform=platform.platform(), processor=platform.processor(),
                    openblas_num_threads=os.getenv("OPENBLAS_NUM_THREADS"), wall_seconds=perf_counter()-start)
    write_run(out/"run.json", metadata,
              "Teacher, train and test seeds advance by 1000 per trial across settings; fixed student seed.",
              "Independent Gaussian rows with covariance I/m; noiseless unit teacher; fixed Gaussian test split.")
    print(f"Elapsed: {metadata['wall_seconds']:.2f} s")


if __name__ == "__main__":
    main()
