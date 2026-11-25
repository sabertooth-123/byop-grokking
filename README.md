# Grokking in ridge regression

## Milestone 1 · Baseline

With the Section 5.1 defaults and the official first-trial seeds, training MSE reached 0.01 at update **232**, when test MSE was still **0.85045**. Test MSE first reached 0.01 at update **22,769**, a delay of **22,537** updates. These numbers come from a single deterministic run. [Loss curves](results/baseline.png) · [Per-update data](results/losses.csv)

| Completed updates | Train MSE | Test MSE |
| ---: | ---: | ---: |
| 0 | 0.784190 | 0.972880 |
| 231 | 0.0100544 | 0.850648 |
| 232 | 0.00990283 | 0.850450 |
| 22,768 | 1.06019e-7 | 0.0100011 |
| 22,769 | 1.06019e-7 | 0.00999926 |

### Experimental protocol

The student predicts the inner product of its weights and the input, using identity features and zero intercept. The experiment uses 100 training examples, 2,000 held-out examples and 1,000 features. Each input coordinate is independently Gaussian with variance 1/1,000. Labels are noiseless inner products with a shared unit-length teacher. [Paper, Section 5.1](https://arxiv.org/html/2601.19791v4#S5.SS1) · [Official implementation](https://github.com/raymondmyxu/to-grok-grokking/blob/4b01ed1d878a37a36b7872e6cbd2beabb8b23cce/ridge_regression.py)

The teacher is a standard Gaussian vector divided by its Euclidean norm. Separate NumPy default_rng streams use seeds 0, 1, 2 and 100 for the teacher, training inputs, test inputs and student respectively. The initial student's coordinates are independent standard normals. All arrays use float64. [Configuration](baseline.json)

For training matrix X, labels y, weights w and residual r = Xw − y, the objective is ‖r‖²/(2n) + λ‖w‖²/2. Each full-batch update subtracts η(Xᵀr/n + λw) from w, with η = 1 and λ = 0.0001. The tracker computes the unregularized mean squared residual on each fixed split at initialization and after every update. Each crossing is the first observation at or below 0.01. The loop ends once both crossings have occurred and at least one update has run, or at 3,000,000 updates. [Independent implementation](baseline.py)

### Comparison and scope

The verifier regenerates the splits with the official generator and executes the official ridge tracker using the configuration in `results/run.json`. That tracker fixes the student seed at 100 and variance at 1.0. The verifier rejects a saved run with any other initialization before loading the reference. It checks both splits, both thresholds, every per-update loss and the final weights. Across 22,770 observations, the maximum absolute differences in training MSE, test MSE and final weight coordinates were all zero on this machine. [Reference comparison](results/comparison.json)

The eight tests include two complete, independently initialized runs of the default configuration. They check exact repeatability of the trajectories and weights, both threshold times, the gradient by finite differences, the seeded generator, a solvable ridge recurrence, step-zero equality, expiry at the update cap, and the reference initialization guard. For the noiseless teacher and input covariance I/m, the exact population MSE is ‖w − θ*‖²/m. A test checks it against an exact second-moment calculation using signed coordinate vectors. The experiment still uses held-out MSE for its test curve and stopping rule. [Tests](test_baseline.py)

Section 5.1 Figure 2 summarizes crossings over hyperparameter sweeps. This curve uses its default parameter tuple and the starting seeds from the sweep driver. The official driver runs ten trials per setting and advances the data and teacher seeds. This milestone covers one initial trajectory. Its test loss is a finite-sample estimate of population MSE. The reference tracker can take one extra unrecorded update if the step cap is exhausted. The independent tracker returns the weights at its last observation. Both runs stopped at the test crossing, well before the cap.

### Reproduction

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

## Milestone 2A · Effect of weight decay

The upper-left panel of Figure 2 uses five weight decays in ascending order and ten trials at each value. Numbering the runs from 0 to 49, run *j* uses seeds 1000*j*, 1000*j* + 1, and 1000*j* + 2 for the teacher, training inputs, and test inputs. The student starts from the same seed, 100, every time. Each run uses the Milestone 1 data generator, objective, learning rate, threshold and update cap. Error bars show one population standard deviation across the ten trials (NumPy `ddof=0`). Delay is computed within each trial before aggregation. [Paper, Figure 2 and Eq. 8](https://arxiv.org/html/2601.19791v4#S5.F2) · [Pinned experiment driver](https://github.com/raymondmyxu/to-grok-grokking/blob/4b01ed1d878a37a36b7872e6cbd2beabb8b23cce/ridge_regression.py)

The dashed training-time bound uses the smallest eigenvalue of XᵀX above 10⁻¹², computed separately for each training matrix. The test-time bound depends only on the configuration, so each weight decay has a single theoretical value. The plot uses base-2 logarithmic axes. [Plot](results/weight_decay/figure2a.png) · [Per-trial measurements and bounds](results/weight_decay/trials.csv) · [Group statistics](results/weight_decay/summary.csv)

| λ (×10⁻⁶) | Training crossing, mean ± SD | Test crossing, mean ± SD | Delay, mean ± SD |
| ---: | ---: | ---: | ---: |
| 2 | 258.2 ± 12.9 | 1,156,242 ± 9,245 | 1,155,984 ± 9,242 |
| 4 | 257.1 ± 12.7 | 576,646 ± 5,667 | 576,389 ± 5,670 |
| 8 | 260.0 ± 13.9 | 288,638 ± 3,366 | 288,378 ± 3,374 |
| 16 | 261.8 ± 9.5 | 144,078 ± 846 | 143,817 ± 845 |
| 32 | 261.1 ± 16.5 | 71,970 ± 888 | 71,709 ± 893 |

All 50 trials reached both thresholds. Raising λ by a factor of 16 shortened mean test-crossing time by a factor of 16.07. Mean training-crossing time stayed between 257.1 and 261.8 updates. These trends and the ordering of all four curves agree with the published panel. The bounds have visible slack in both figures. Here, mean test-crossing times are about 1.92–1.93 times their lower bounds, and mean training upper bounds are about 5.6–5.8 times the observed means. The pinned release provides the rendered panel but omits its raw trial arrays, which limits direct numerical comparison to data regenerated from the official functions. [Published panel](https://arxiv.org/html/2601.19791v4/plots/ridge_regularization.png)

The sweep, including data generation, bounds and plotting, took **150.83 seconds** on a Core i7-13650HX using one OpenBLAS thread. Across all trials the solver advanced through 22,272,895 interior updates via the discrete recurrence. Times are wall-clock measurements on this machine.

### Computation and validation

The default runner uses the validated literal GD loop for the first 2,048 updates. Once training has crossed, it can advance through a long interval of discrete GD using the singular vectors of the training matrix. In that basis each row-space error coordinate contracts by 1 − η(λ + sᵢ²/n) per update, and the null-space component contracts by 1 − ηλ. The held-out residual is the sum of a fixed vector, the decaying null-space prediction, and the remaining row-space error.

For every candidate skipped step, the runner computes the norm of the first two terms and subtracts an upper bound on the third. It skips only while this lower bound on **held-out RMSE** exceeds √ε by more than 10⁻¹⁰. It leaves eight extra steps before the first possible crossing and resumes the literal GD loop from the reconstructed weights. This preserves the discrete update count without assuming monotone test loss. The derivation is exact in real arithmetic. Floating-point matrix powers and repeated GD can differ in their last digits. [Runner and bounds](weight_decay.py)

With the Milestone 1 environment and pinned checkout prepared, run the following from the project root.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -m unittest -v test_baseline test_weight_decay
.\.venv\Scripts\python.exe weight_decay.py
.\.venv\Scripts\python.exe verify_weight_decay.py
```

The verifier compares all per-trial theoretical bounds with the official functions. An independent eigendecomposition of XXᵀ checks the empirical MSE immediately before and at both reported crossings for every trial. The verifier also runs the complete official GD tracker on the first trial at the largest decay. To evaluate all 50 trials with literal GD, run `python weight_decay.py --literal`. That writes to `results/weight_decay_literal/` and requires substantially more time and memory for the loss histories. [Run configuration and timing](results/weight_decay/run.json) · [Verification](results/weight_decay/verification.json)

All 12 tests passed. Both bounds matched the official functions exactly for all 50 trials. All 200 crossing-neighbor checks agreed, with the closest MSE approximately 7.88 × 10⁻¹¹ from the threshold. The full official trial returned t₁ = 263, t₂ = 72,884 and delay = 72,621, matching the saved trial exactly. Verification took 79.17 seconds, including 73.19 seconds for that literal trial.
