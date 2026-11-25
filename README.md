# Milestone 1 · Grokking in ridge regression

With the Section 5.1 defaults and the official first-trial seeds, training MSE reached 0.01 at update **232**, when test MSE was still **0.85045**. Test MSE first reached 0.01 at update **22,769**, a delay of **22,537** updates. These numbers come from a single deterministic run. [Loss curves](results/baseline.png) · [Per-update data](results/losses.csv)

| Completed updates | Train MSE | Test MSE |
| ---: | ---: | ---: |
| 0 | 0.784190 | 0.972880 |
| 231 | 0.0100544 | 0.850648 |
| 232 | 0.00990283 | 0.850450 |
| 22,768 | 1.06019e-7 | 0.0100011 |
| 22,769 | 1.06019e-7 | 0.00999926 |

## Experimental protocol

The student predicts the inner product of its weights and the input, using identity features and zero intercept. The experiment uses 100 training examples, 2,000 held-out examples and 1,000 features. Each input coordinate is independently Gaussian with variance 1/1,000. Labels are noiseless inner products with a shared unit-length teacher. [Paper, Section 5.1](https://arxiv.org/html/2601.19791v4#S5.SS1) · [Official implementation](https://github.com/raymondmyxu/to-grok-grokking/blob/4b01ed1d878a37a36b7872e6cbd2beabb8b23cce/ridge_regression.py)

The teacher is a standard Gaussian vector divided by its Euclidean norm. Separate NumPy default_rng streams use seeds 0, 1, 2 and 100 for the teacher, training inputs, test inputs and student respectively. The initial student's coordinates are independent standard normals. All arrays use float64. [Configuration](baseline.json)

For training matrix X, labels y, weights w and residual r = Xw − y, the objective is ‖r‖²/(2n) + λ‖w‖²/2. Each full-batch update subtracts η(Xᵀr/n + λw) from w, with η = 1 and λ = 0.0001. The tracker computes the unregularized mean squared residual on each fixed split at initialization and after every update. Each crossing is the first observation at or below 0.01. The loop ends once both crossings have occurred and at least one update has run, or at 3,000,000 updates. [Independent implementation](baseline.py)

## Comparison and scope

The verifier regenerates the splits with the official generator and executes the official ridge tracker using the configuration in `results/run.json`. That tracker fixes the student seed at 100 and variance at 1.0. The verifier rejects a saved run with any other initialization before loading the reference. It checks both splits, both thresholds, every per-update loss and the final weights. Across 22,770 observations, the maximum absolute differences in training MSE, test MSE and final weight coordinates were all zero on this machine. [Reference comparison](results/comparison.json)

The eight tests include two complete, independently initialized runs of the default configuration. They check exact repeatability of the trajectories and weights, both threshold times, the gradient by finite differences, the seeded generator, a solvable ridge recurrence, step-zero equality, expiry at the update cap, and the reference initialization guard. For the noiseless teacher and input covariance I/m, the exact population MSE is ‖w − θ*‖²/m. A test checks it against an exact second-moment calculation using signed coordinate vectors. The experiment still uses held-out MSE for its test curve and stopping rule. [Tests](test_baseline.py)

Section 5.1 Figure 2 summarizes crossings over hyperparameter sweeps. This curve uses its default parameter tuple and the starting seeds from the sweep driver. The official driver runs ten trials per setting and advances the data and teacher seeds. This milestone covers one initial trajectory. Its test loss is a finite-sample estimate of population MSE. The reference tracker can take one extra unrecorded update if the step cap is exhausted. The independent tracker returns the weights at its last observation. Both runs stopped at the test crossing, well before the cap.

## Reproduction

From the project root, use Git and Python 3.12.3. In PowerShell, create an environment, install the recorded versions, and prepare the reference checkout once per clone.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_reference.py
```

Setup clones the authors' GitHub repository into the ignored `reference/` directory at `4b01ed1d878a37a36b7872e6cbd2beabb8b23cce`. Later invocations verify the existing revision and tracked files offline. If setup reports a conflicting checkout, move it aside to preserve local work and rerun setup. [Setup script](setup_reference.py)

Run the complete validation from the same directory. The unit tests and baseline run independently of the reference checkout.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -m unittest -v test_baseline
.\.venv\Scripts\python.exe run_baseline.py
.\.venv\Scripts\python.exe compare_reference.py
```

Expected results are eight passing tests, crossings at 232 and 22,769, and 22,770 observations. The comparison requires exact threshold agreement and checks losses and final weights with relative tolerance 1e-12 and absolute tolerance 1e-14. The measured maximum differences are printed and saved in `results/comparison.json`.

The runner writes the per-update CSV, figure, teacher and student weights, and a snapshot of the configuration and software environment. A single BLAS thread was used for both implementations. Last-bit differences are possible with different numerical libraries or hardware. [Run metadata](results/run.json) · [Weight arrays](results/weights.npz)
