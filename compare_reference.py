"""Repeat the saved run with the authors' tracker and compare every observation."""
import hashlib
import json
import sys

import numpy as np
from provenance import write_validation
from baseline import Config, ROOT, make_problem
from setup_reference import REFERENCE, REVISION, verify_checkout

SCRIPT_BLOB = "1fc000535b8aa649c4a0213a73c5397f9966dcbf"


def validate_initialization(c):
    if c.student_seed != 100 or c.student_variance != 1.0:
        raise ValueError("Official ridge_gd_with_tracking requires student_seed=100 and "
                         f"student_variance=1.0. Saved run uses student_seed={c.student_seed}, "
                         f"student_variance={c.student_variance}.")


def main():
    out = ROOT / "results"
    run = json.loads((out / "run.json").read_text())
    c = Config(**run["config"])
    validate_initialization(c)
    verify_checkout(REFERENCE)
    source = (REFERENCE / "ridge_regression.py").read_bytes().replace(b"\r\n", b"\n")
    blob = hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
    if blob != SCRIPT_BLOB:
        raise ValueError(f"Expected the official ridge_regression.py from {REVISION}")
    sys.path.insert(0, str(REFERENCE))
    from ridge_regression import generate_linear_data, ridge_gd_with_tracking

    independent_teacher, train, test, _ = make_problem(c)
    teacher = np.random.default_rng(c.teacher_seed).normal(0, 1, c.dimension)
    teacher /= np.linalg.norm(teacher)
    np.testing.assert_array_equal(teacher, independent_teacher)
    ref_train = generate_linear_data(c.n_train, c.dimension, teacher, c.train_seed)
    ref_test = generate_linear_data(c.n_test, c.dimension, teacher, c.test_seed)
    for ours, theirs in zip((*train, *test), (*ref_train, *ref_test)):
        np.testing.assert_array_equal(ours, theirs)
    weights, history = ridge_gd_with_tracking(
        *ref_train, *ref_test, weight_decay=c.weight_decay, epsilon=c.epsilon,
        max_steps=c.max_steps, step_size=c.step_size, verbose=False
    )
    curves = np.column_stack((history["train_losses"], history["test_losses"]))
    stored = np.loadtxt(out / "losses.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(stored[:, 0], np.arange(len(curves)))
    np.testing.assert_allclose(stored[:, 1:], curves, rtol=1e-12, atol=1e-14)
    with np.load(out / "weights.npz") as saved:
        np.testing.assert_array_equal(saved["teacher"], teacher)
        np.testing.assert_array_equal(saved["initial"], np.random.default_rng(100).normal(size=c.dimension))
        weight_error = np.max(np.abs(saved["final"] - weights))
        np.testing.assert_allclose(saved["final"], weights, rtol=1e-12, atol=1e-14)
    crossings = [history["t1_train_epsilon"], history["t2_test_epsilon"]]
    np.testing.assert_array_equal(crossings, [run["t1"], run["t2"]])
    diffs = np.max(np.abs(stored[:, 1:] - curves), axis=0)
    report = dict(reference_commit=REVISION, script_git_blob=blob,
                  observations=len(curves), reference_t1=crossings[0], reference_t2=crossings[1],
                  max_abs_train_mse_difference=diffs[0], max_abs_test_mse_difference=diffs[1],
                  max_abs_final_weight_difference=weight_error, rtol=1e-12, atol=1e-14)
    write_validation(out/"comparison.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
