"""Compare saved Figure 2A bounds and one complete trial with the pinned official functions."""
import csv
import json
import sys
from time import perf_counter

import numpy as np

from provenance import write_validation
from baseline import Config, ROOT, make_problem
from setup_reference import REFERENCE, REVISION, verify_checkout
from weight_decay import trial_configs


def check_neighbors(c, train, test, initial, row):
    """Independently evaluate steps on either side of each reported crossing from step zero."""
    x, y = train
    eigenvalues, u = np.linalg.eigh(x @ x.T)
    basis = x.T @ u
    # Row-space coordinates in the (unnormalized) basis X.T U.
    start = (u.T @ (x @ initial)) / eigenvalues
    fixed = (u.T @ y) / (eigenvalues + len(y) * c.weight_decay)
    null = initial - basis @ start
    q = 1 - c.step_size * (eigenvalues / len(y) + c.weight_decay)
    q_null = 1 - c.step_size * c.weight_decay
    margins = []
    for key, (features, labels) in [("t1", train), ("t2", test)]:
        t = int(row[key])
        for step in (t-1, t):
            weights = q_null**step * null + basis @ (fixed + q**step * (start-fixed))
            mse = np.mean((features @ weights - labels)**2)
            np.testing.assert_equal(mse <= c.epsilon, step == t)
            margins.append(abs(mse-c.epsilon))
    return min(margins)


def main():
    start = perf_counter()
    verify_checkout(REFERENCE)
    sys.path.insert(0, str(REFERENCE))
    import ridge_regression as official

    out = ROOT / "results" / "weight_decay"
    meta = json.loads((out / "run.json").read_text())
    with (out / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    configs = list(trial_configs(Config(**meta["base_config"])))
    if len(rows) != len(configs):
        raise ValueError("Expected a raw result for each trial")
    bound_errors = []
    margins = []
    for row, (trial, c) in zip(rows, configs):
        np.testing.assert_equal((float(row["weight_decay"]), int(row["trial"])), (c.weight_decay, trial))
        for key in ("teacher_seed", "train_seed", "test_seed", "student_seed"):
            np.testing.assert_equal(int(row[key]), getattr(c, key))
        _, train, test, initial = make_problem(c)
        margins.append(check_neighbors(c, train, test, initial, row))
        eigenvalue = official.lambda_min_positive_gram(train[0])
        upper = official.theoretical_upper_bound_t1(c.n_train, c.dimension, c.step_size, c.epsilon,
                                                   c.student_variance, eigenvalue)
        lower, = official.theoretical_lower_bound_weight_decay_t2(
            [c.weight_decay], c.n_train, c.dimension, c.step_size, c.epsilon, c.student_variance)
        expected = np.array([upper, lower])
        actual = np.array([float(row["t1_upper"]), float(row["t2_lower"])])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        bound_errors.append(abs(actual-expected))
        if trial == 1 and c.weight_decay == configs[-1][1].weight_decay:
            literal_start = perf_counter()
            _, hist = official.ridge_gd_with_tracking(*train, *test, weight_decay=c.weight_decay,
                                                      epsilon=c.epsilon, max_steps=c.max_steps,
                                                      step_size=c.step_size, verbose=False)
            t1, t2 = hist["t1_train_epsilon"], hist["t2_test_epsilon"]
            np.testing.assert_equal([int(row[k]) for k in ("t1", "t2", "delay")], [t1, t2, t2-t1])
            literal = dict(weight_decay=c.weight_decay, trial=trial, teacher_seed=c.teacher_seed, t1=t1,
                           t2=t2, delay=t2-t1, seconds=perf_counter()-literal_start)
    upper_error, lower_error = np.max(bound_errors, axis=0)
    report = dict(reference_commit=REVISION, bounds_checked=len(rows), max_abs_upper_error=upper_error,
                  max_abs_lower_error=lower_error, crossing_neighbors_checked=4*len(rows),
                  smallest_threshold_margin=min(margins), literal_reference_trial=literal, seconds=perf_counter()-start)
    write_validation(out/"verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
