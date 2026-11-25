"""Milestone 3. Threshold sensitivity and paired finite-test versus population crossings."""
from dataclasses import asdict, replace
import hashlib
import json
import os
import platform
from time import perf_counter

import numpy as np

from provenance import write_run
from baseline import Config, ROOT, make_problem
from weight_decay import crossings, plt, save_csv

EPSILONS = (.005, .01, .02, .05)
TEST_OFFSETS = {100: 3, 500: 4, 2000: 2, 10000: 5}
TRIALS = 20
BOOTSTRAP_SEED = 3102026
RESAMPLES = 20000
OUT = ROOT / "results" / "robustness"
MISSING = "unobserved"


def trial_config(base, trial):
    offset = 1000 * (trial-1)
    return replace(base, teacher_seed=base.teacher_seed+offset, train_seed=base.train_seed+offset,
                   test_seed=base.test_seed+offset, student_seed=base.student_seed+offset)


def finite_set(c, teacher, size, seed):
    x = np.random.default_rng(seed).normal(0., 1/np.sqrt(c.dimension), (size, c.dimension))
    return x, x @ teacher


def measure(c, train, evaluation, initial):
    t1, t2, skipped = crossings(c, train, evaluation, initial)
    return dict(t1=MISSING if t1 is None else t1, t2=MISSING if t2 is None else t2,
                delay=MISSING if t1 is None or t2 is None else t2-t1, skipped_updates=skipped)


def differences(t2, population):
    if MISSING in (t2, population):
        return MISSING, MISSING
    return t2-population, abs(t2-population)


def run_trial(c, trial):
    start = perf_counter()
    teacher, train, baseline_test, initial = make_problem(c)
    seeds = dict(trial=trial, teacher_seed=c.teacher_seed, train_seed=c.train_seed, student_seed=c.student_seed)
    threshold_rows = []
    for epsilon in EPSILONS:
        result = measure(replace(c, epsilon=epsilon), train, baseline_test, initial)
        threshold_rows.append(dict(**seeds, epsilon=epsilon, test_seed=c.test_seed, test_size=c.n_test, **result))
        if epsilon == c.epsilon:
            shared = result
    a_seconds = perf_counter()-start
    start = perf_counter()
    # Mean squared residual on the identity design is exactly ||w - teacher||²/m.
    population = measure(c, train, (np.eye(c.dimension), teacher), initial)
    common = dict(**seeds, epsilon=c.epsilon, population_t2=population["t2"])
    test_rows = [dict(**common, evaluation="population", test_size="NA", test_seed="NA", **population,
                      t2_difference="NA", abs_t2_difference="NA")]
    for size, offset in TEST_OFFSETS.items():
        seed = 1000*(trial-1) + offset
        if size == c.n_test and seed == c.test_seed:
            result = shared
        else:
            result = measure(c, train, finite_set(c, teacher, size, seed), initial)
        if result["t1"] != population["t1"]:
            raise ArithmeticError("Different training crossings for the same trajectory")
        delta, absolute = differences(result["t2"], population["t2"])
        test_rows.append(dict(**common, evaluation="finite", test_size=size, test_seed=seed,
                              **result, t2_difference=delta, abs_t2_difference=absolute))
    return threshold_rows, test_rows, a_seconds, perf_counter()-start


def aggregate(rows, group_key, settings, paired=False):
    """Percentile bootstrap of complete trial vectors, using the same draws in each group."""
    trial_ids = sorted({r["trial"] for r in rows})
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(trial_ids), size=(RESAMPLES, len(trial_ids)))
    output = []
    for setting in settings:
        group = sorted((r for r in rows if r[group_key] == setting), key=lambda r: r["trial"])
        if [r["trial"] for r in group] != trial_ids:
            raise ValueError(f"Missing or duplicate trial row at {group_key}={setting}")
        metrics = ["t1", "t2", "delay"]
        if paired and setting != "NA":
            metrics += ["t2_difference", "abs_t2_difference"]
        for metric in metrics:
            values = [r[metric] for r in group]
            observed = len(values) - values.count(MISSING)
            entry = {group_key: setting, "metric": metric, "trials": len(values),
                     "observed": observed, "unobserved": len(values)-observed}
            stats = dict.fromkeys(("mean", "median", "sd", "mean_ci_low", "mean_ci_high",
                                   "median_ci_low", "median_ci_high"), float("nan"))
            if observed == len(values):
                x = np.asarray(values, dtype=float)
                samples = x[draws]
                stats.update(mean=x.mean(), median=np.median(x), sd=x.std(ddof=1))
                for key, estimates in [("mean", samples.mean(axis=1)), ("median", np.median(samples, axis=1))]:
                    stats[key+"_ci_low"], stats[key+"_ci_high"] = np.quantile(estimates, [.025, .975])
            output.append(dict(**entry, **stats))
    return output


def mean_intervals(ax, rows, metric, positions, label, color):
    selected = [r for r in rows if r["metric"] == metric]
    y = np.array([r["mean"] for r in selected])
    lo = np.array([r["mean_ci_low"] for r in selected])
    hi = np.array([r["mean_ci_high"] for r in selected])
    ax.errorbar(positions, y, yerr=[y-lo, hi-y], marker="o", color=color, capsize=4, label=label)
    ax.grid(axis="y", alpha=.25)


def plots(threshold, finite):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout="constrained")
    for ax, metric, label in zip(axes, ["t1", "t2", "delay"], ["Training crossing", "Test crossing", "Grokking delay"]):
        mean_intervals(ax, threshold, metric, range(len(EPSILONS)), label, "#21618c")
        ax.set(xticks=range(len(EPSILONS)), xticklabels=[str(e) for e in EPSILONS],
               xlabel="MSE threshold ε", title=label)
    axes[0].set_ylabel("Gradient updates")
    fig.suptitle("Threshold sensitivity · trial means and 95% confidence intervals")
    fig.savefig(OUT / "threshold_sensitivity.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    sizes = list(TEST_OFFSETS)
    rows = [r for r in finite if r["test_size"] != "NA"]
    for ax, metric, title in zip(axes, ["t2_difference", "abs_t2_difference"],
                                 ["Finite − population crossing", "Absolute crossing-time difference"]):
        mean_intervals(ax, rows, metric, range(len(sizes)), title, "#97492c")
        ax.set(xticks=range(len(sizes)), xticklabels=sizes, xlabel="Held-out examples", title=title)
    axes[0].axhline(0, color="0.4", ls="--", lw=1)
    axes[0].set_ylabel("Gradient updates")
    fig.suptitle("Paired test-set error · trial means and 95% confidence intervals")
    fig.savefig(OUT / "finite_vs_population.png", dpi=180)
    plt.close(fig)


def main():
    start = perf_counter()
    base = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    a_rows, b_rows = [], []
    a_time = b_time = 0.
    for trial in range(1, TRIALS+1):
        a, b, ta, tb = run_trial(trial_config(base, trial), trial)
        a_rows.extend(a)
        b_rows.extend(b)
        a_time += ta
        b_time += tb
        print(f"Finished trial {trial}/{TRIALS}", flush=True)
    threshold = aggregate(a_rows, "epsilon", EPSILONS)
    finite = aggregate(b_rows, "test_size", ["NA", *TEST_OFFSETS], paired=True)
    for name, rows in [("threshold_trials", a_rows), ("threshold_summary", threshold),
                       ("population_trials", b_rows), ("population_summary", finite)]:
        save_csv(OUT / (name + ".csv"), rows)
    plots(threshold, finite)
    files = ["robustness.py", "weight_decay.py", "baseline.py", "baseline.json"]
    meta = dict(base_config=asdict(base), trials=TRIALS, epsilons=EPSILONS, test_seed_offsets=TEST_OFFSETS,
                seed_stride=1000, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=RESAMPLES,
                sd_ddof=1, confidence_level=.95, python=platform.python_version(), numpy=np.__version__,
                matplotlib=plt.matplotlib.__version__, platform=platform.platform(),
                openblas_num_threads=os.getenv("OPENBLAS_NUM_THREADS"),
                sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                threshold_seconds=a_time, population_seconds=b_time, total_seconds=perf_counter()-start)
    write_run(OUT/"run.json", meta,
              "Advance teacher, train and student seeds by 1000; independent finite test seeds use recorded offsets.",
              "Independent Gaussian rows with covariance I/m; paired finite splits and exact isotropic population scoring.")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
