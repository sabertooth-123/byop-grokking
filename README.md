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

## Milestone 2B · Sample size, feature dimension and initialization

Each panel starts a fresh sequence of ten trials per setting. For zero-based trial index *j* across that panel, the teacher, training and test seeds are 1000*j*, 1000*j* + 1 and 1000*j* + 2. The student seed is 100 throughout. The sweeps use λ = 0.0001, η = 1, ε = 0.01, 2,000 held-out examples and a cap of 3,000,000 updates. The default training size, feature dimension and student variance are 100, 1,000 and 1 respectively. The sample-size panel uses base-2 log axes. Dimension uses a linear horizontal axis and base-2 vertical axis. Initialization uses a base-10 horizontal axis and linear vertical axis. [Paper, Section 5.1](https://arxiv.org/html/2601.19791v4#S5.SS1) · [Official driver](https://github.com/raymondmyxu/to-grok-grokking/blob/4b01ed1d878a37a36b7872e6cbd2beabb8b23cce/ridge_regression.py)

| Panel | Grid | Plot | Individual trials | Means, SDs and counts |
| --- | --- | --- | --- | --- |
| Training sample size | 10, 20, 40, 80, 160 | [Figure 2B](results/sample_size/figure2b.png) | [CSV](results/sample_size/trials.csv) | [CSV](results/sample_size/summary.csv) |
| Feature dimension | 1000, 1500, 2000, 2500, 3000 | [Figure 2C](results/feature_dimension/figure2c.png) | [CSV](results/feature_dimension/trials.csv) | [CSV](results/feature_dimension/summary.csv) |
| Initialization variance | 1, 10, 100, 1000, 10000, 100000, 1000000 | [Figure 2D](results/initialization/figure2d.png) | [CSV](results/initialization/trials.csv) | [CSV](results/initialization/summary.csv) |

The runner reuses the Milestone 1 generator and Milestone 2A discrete GD solver. Each trial records its two first threshold crossings, their paired difference, and its theoretical bounds. A crossing missing at the cap is recorded as `unobserved`, as is a delay with either crossing missing. Each summary gives the observed count. A partial group yields NaN for the affected mean and SD, matching the released use of ordinary NumPy mean and variance on arrays containing NaN. Error bars are one SD across the ten trials, with divisor ten. [Runner](figure2bcd.py)

**Initialization convention.** For variance *v* = ν², the released driver samples each initial weight with SD √*v*. It also passes √*v* as the `nu_square` argument of the **training** upper-bound function. Equation 8 uses *v* in that logarithm. Figure 2D and the `t1_upper` columns use the released call. The separate `t1_upper_variance` columns evaluate the same function on the same training matrix with *v*. The test lower bound takes *v* in both cases. These two upper bounds share the same measured trajectories. [Released upper-bound call](https://github.com/raymondmyxu/to-grok-grokking/blob/4b01ed1d878a37a36b7872e6cbd2beabb8b23cce/ridge_regression.py#L677-L687)

### Reproduction and numerical checks

Use the environment and pinned reference checkout from Milestone 1. Run these commands in PowerShell from the project root.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe setup_reference.py
.\.venv\Scripts\python.exe -m unittest -v test_baseline test_weight_decay test_figure2bcd
foreach ($panel in 'sample_size', 'feature_dimension', 'initialization') {
    .\.venv\Scripts\python.exe figure2bcd.py $panel
    if ($LASTEXITCODE -ne 0) { throw "Sweep failed for $panel" }
    .\.venv\Scripts\python.exe verify_figure2bcd.py $panel
    if ($LASTEXITCODE -ne 0) { throw "Validation failed for $panel" }
}
```

Each panel directory contains a `run.json` with the base configuration, overrides, seed stride, runtime, package versions, project HEAD, SHA-256 hashes of the runner and its local dependencies, and the official commit. The hashes identify the executed working-tree sources, including the new runner. `verification.json` records the numerical checks and their runtime. The production path calls the pinned reference bound functions with the smallest positive eigenvalue above 10⁻¹² from XXᵀ. This matrix has the same positive eigenvalues as XᵀX and at most 160 rows in these sweeps.

The verifier independently evaluates Equation 8 with the released m-by-m Gram eigensolver for every trial. It also evaluates held-out and training MSE from step zero using the independent discrete solution at the two adjacent updates around each crossing. One full-size trial per panel runs through the official literal GD tracker, using its scaled-initialization variant for Figure 2D. [Verification code](verify_figure2bcd.py) · [Numerical tests](test_figure2bcd.py)

### Comparison with Figure 2

- **Sample size.** Increasing *n* from 10 to 160 raised mean t₁ from 21.7 to 446.1 updates. Mean t₂ fell modestly, from 23,648.3 to 22,689.6. Mean delay consequently shrank from 23,626.6 to 22,243.5. This reproduces the direction of both empirical curves in the upper-right panel. [Sample-size summary](results/sample_size/summary.csv)
- **Feature dimension.** Tripling *m* from 1,000 to 3,000 at *n* = 100 moved mean t₁ from 254.9 to 236.9 updates and mean t₂ from 23,124.4 to 22,940.7. Mean t₂ varied by about 1.3% across the whole grid, consistent with the nearly flat empirical curve in the lower-left panel. [Feature-dimension summary](results/feature_dimension/summary.csv)
- **Initialization.** Going from variance 1 to 10⁶ raised mean t₁ from 254.9 to 1,421.3 and mean t₂ from 23,124.4 to 92,104.9. Mean t₂ rose by about 11,500 updates per decade of variance, consistent with the paper's logarithmic dependence. At 10⁶, the released mean t₁ upper bound was 2,193.0 ± 43.3 updates. Passing the variance as in Equation 8 gave 2,912.4 ± 57.5. At variance 1 both calculations gave 1,491.2. [Initialization summary](results/initialization/summary.csv)

**Published training bounds.** The three arXiv v4 panel PNGs matched the pinned repository PNGs byte for byte (SHA-256). The training upper bound in the published sample-size panel reads approximately 41 and 1,520 updates at the two endpoints, whereas the released calculation gives 83.4 and 3,073.0. The feature-dimension image also puts the bound lower: at *m* = 1,000 its marker reads about 620 updates, versus 1,491.2 from the released code. The image estimates use the blue dashed marker centers and the labeled log-axis grid. The released arrays underlying those images are unavailable, so the cause of the offset remains unresolved. The initialization argument mismatch cannot explain these two panels because their variance is 1. [Published sample-size panel](https://arxiv.org/html/2601.19791v4/plots/ridge_sample_size.png) · [Published dimension panel](https://arxiv.org/html/2601.19791v4/plots/ridge_width.png) · [Published initialization panel](https://arxiv.org/html/2601.19791v4/plots/ridge_initialization.png)

### Validation results and wall time

All 170 trials crossed both thresholds. All 17 unit tests passed in 48.50 s. All 680 independent checks of MSE before and at the reported crossings passed. The largest absolute difference in training upper bounds using the two Gram calculations was 2.32 × 10⁻¹¹ updates. All test lower bounds agreed exactly. Direct GD matched both integer crossing times for each full-size representative below.

| Panel | Trials | Sweep | Verification | Official literal trial 1, (t₁, t₂) |
| --- | ---: | ---: | ---: | --- |
| [Sample size](results/sample_size/verification.json) | 50 | 54.50 s | 37.93 s | *n* = 20, (44, 23,238) |
| [Feature dimension](results/feature_dimension/verification.json) | 50 | 231.39 s | 149.50 s | *m* = 1,500, (275, 22,849) |
| [Initialization](results/initialization/verification.json) | 70 | 103.19 s | 58.48 s | ν² = 1,000, (771, 57,431) |

Each process used one OpenBLAS thread. Sample-size validation finished before the dimension and initialization sweeps started. Those latter two jobs ran concurrently, so their elapsed times overlap. Test crossings describe MSE on a fixed finite held-out set per trial. The SDs describe variation across the released data and teacher seeds with fixed student seed.
