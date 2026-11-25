"""Replay full-size trial 1 by direct GD and compare with the recorded robustness results."""
import csv
import json
from time import perf_counter

import numpy as np

from baseline import Config, gradient, make_problem, population_mse
from robustness import EPSILONS, MISSING, OUT, TEST_OFFSETS, finite_set, trial_config


def verify():
    start = perf_counter()
    c = trial_config(Config.load(), 1)
    teacher, train, test, w = make_problem(c)
    x, y = train
    small_x, small_y = finite_set(c, teacher, 100, TEST_OFFSETS[100])
    train_times = dict.fromkeys(EPSILONS)
    finite_times = dict.fromkeys(EPSILONS)
    population_time = small_time = None
    for step in range(c.max_steps+1):
        residual = x @ w-y
        train_loss = np.mean(residual**2)
        finite_loss = np.mean((test[0] @ w-test[1])**2)
        for e in EPSILONS:
            if train_times[e] is None and train_loss <= e:
                train_times[e] = step
            if finite_times[e] is None and finite_loss <= e:
                finite_times[e] = step
        if population_time is None and population_mse(w, teacher) <= c.epsilon:
            population_time = step
        if small_time is None and np.mean((small_x @ w-small_y)**2) <= c.epsilon:
            small_time = step
        if None not in (*train_times.values(), *finite_times.values(), population_time, small_time):
            break
        if step < c.max_steps:
            w -= c.step_size * gradient(x, residual, w, c.weight_decay)

    def trial_one(name):
        with (OUT/name).open(newline="") as stream:
            return [r for r in csv.DictReader(stream) if int(r["trial"]) == 1]

    def check(row, t1, t2):
        for key, val in [("t1", t1), ("t2", t2), ("delay", None if None in (t1,t2) else t2-t1)]:
            np.testing.assert_equal(row[key], MISSING if val is None else str(val))

    a = trial_one("threshold_trials.csv")
    np.testing.assert_array_equal([float(r["epsilon"]) for r in a], EPSILONS)
    for r in a:
        e = float(r["epsilon"])
        check(r, train_times[e], finite_times[e])
    b = {r["test_size"]: r for r in trial_one("population_trials.csv")}
    for group, t2 in [("NA", population_time), ("100", small_time), ("2000", finite_times[c.epsilon])]:
        check(b[group], train_times[c.epsilon], t2)
    expected_delta = MISSING if None in (small_time, population_time) else str(small_time-population_time)
    np.testing.assert_equal(b["100"]["t2_difference"], expected_delta)
    receipt = dict(trial=1, literal_updates=step, threshold_train=train_times, threshold_test=finite_times,
                   population_t2=population_time, test_100_t2=small_time, seconds=perf_counter()-start)
    (OUT/"verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    verify()
