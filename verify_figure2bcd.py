"""Check saved trials against the released Gram calculation, Eq. 8, and full GD."""
import argparse
import csv
import hashlib
import json
import math
from time import perf_counter

import numpy as np

from baseline import Config, ROOT, make_problem
from figure2bcd import GRIDS, UNOBSERVED, configs, reference_module
from setup_reference import REVISION
from verify_weight_decay import check_neighbors


def verify(panel):
    start = perf_counter()
    ref = reference_module()
    out = ROOT / "results" / panel
    meta = json.loads((out / "run.json").read_text())
    with (out / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = configs(panel, Config(**meta["base_config"]))
    representative = {"sample_size": 20, "feature_dimension": 1500, "initialization": 1000}[panel]
    errors, margins, literal_checks = [], [], []
    for row, (value, trial, c) in zip(rows, expected, strict=True):
        for key in ("teacher_seed", "train_seed", "test_seed", "student_seed"):
            np.testing.assert_equal(int(row[key]), getattr(c, key))
        np.testing.assert_equal((int(row["value"]), int(row["trial"])), (value, trial))
        _, train, test, initial = make_problem(c)
        # Compare both crossing neighbors from step zero, independently of interval skipping.
        observed = UNOBSERVED not in (row["t1"], row["t2"])
        if observed:
            margins.append(check_neighbors(c, train, test, initial, row))
            np.testing.assert_equal(int(row["delay"]), int(row["t2"])-int(row["t1"]))
        else:
            np.testing.assert_equal(row["delay"], UNOBSERVED)

        eig = ref.lambda_min_positive_gram(train[0])  # original m by m matrix
        n, m, v = c.n_train, c.dimension, c.student_variance
        arg = math.sqrt(v) if panel == "initialization" else v
        upper = n * math.log(14*m*arg/c.epsilon) / (2*c.step_size*eig)
        lower = math.log((m-n)*v/(8*m*c.epsilon)) / (2.02*c.step_size*c.weight_decay)
        pairs = [(eig, row["lambda_min_positive"]), (upper, row["t1_upper"]), (lower, row["t2_lower"])]
        if panel == "initialization":
            pairs.append((n*math.log(14*m*v/c.epsilon)/(2*c.step_size*eig), row["t1_upper_variance"]))
        reference, saved = np.asarray(pairs, dtype=float).T
        np.testing.assert_allclose(saved, reference, rtol=1e-12, atol=1e-12)
        errors.append(abs(saved-reference))

        # A missing crossing requires a literal run through the cap as an independent check.
        if (value == representative and trial == 1) or not observed:
            before = perf_counter()
            tracker = ref.ridge_gd_with_tracking
            kwargs = {}
            if panel == "initialization":
                tracker = ref.ridge_gd_with_tracking_scaling
                kwargs["ini_scale"] = math.sqrt(v)
            _, hist = tracker(*train, *test, weight_decay=c.weight_decay, epsilon=c.epsilon,
                              max_steps=c.max_steps, step_size=c.step_size, verbose=False, **kwargs)
            t1, t2 = hist["t1_train_epsilon"], hist["t2_test_epsilon"]
            for key, crossing in (("t1", t1), ("t2", t2)):
                np.testing.assert_equal(row[key], UNOBSERVED if crossing is None else str(crossing))
            literal_checks.append(dict(value=value, trial=trial, t1=t1, t2=t2, seconds=perf_counter()-before))
    names = ["lambda_min_positive", "t1_upper", "t2_lower", "t1_upper_variance"]
    report = dict(reference_commit=REVISION, verifier_sha256=hashlib.sha256((ROOT/"verify_figure2bcd.py").read_bytes()).hexdigest(),
                  trials_checked=len(rows), neighbor_checks=4*len(margins),
                  minimum_observed_margin=min(margins, default=None),
                  max_abs_errors=dict(zip(names, np.max(errors, axis=0))),
                  literal_checks=literal_checks, seconds=perf_counter()-start)
    (out / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=GRIDS)
    verify(parser.parse_args().panel)
