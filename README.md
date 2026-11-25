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

## Milestone 3 · Robustness of the crossing-time measurement

These additional experiments keep the Section 5.1 default ridge problem, step size, penalty, initialization distribution and 3,000,000-update budget. The endpoints are the first integer updates with unregularized MSE ≤ ε, including update zero. Each condition replays the same deterministic trajectory for a given trial using the validated Milestone 2A solver. [Evaluation code](robustness.py) · [Run configuration and timings](results/robustness/run.json)

**Independent trials and pairing.** There are 20 independent trial seeds, compared with ten per setting in Figure 2. For zero-based trial *j*, the seeds are 1000*j*, 1000*j* + 1 and 1000*j* + 100 for the teacher, training data and student. Varying the student seed samples initialization variability across trials. Trial 1 reproduces the original baseline seeds. Conditions share all three seeds within each trial. The default test set uses 1000*j* + 2.

- **A · Threshold sensitivity.** Apply ε ∈ {0.005, 0.01, 0.02, 0.05} to the same training and 2,000-example held-out sets within each trial. Record t₁, t₂ and t₂ − t₁. [Trial records](results/robustness/threshold_trials.csv) · [Statistics](results/robustness/threshold_summary.csv) · [Plot](results/robustness/threshold_sensitivity.png)
- **B · Finite test versus population.** Fix ε = 0.01. For Gaussian inputs with covariance I/m, the exact squared prediction error is ‖w − θ*‖²/m. The solver measures it by scoring the identity matrix against θ*, which yields this MSE at every weight vector. Independently sampled Gaussian test sets of sizes 100, 500, 2,000 and 10,000 use seed offsets 3, 4, 2 and 5 respectively. Each stays fixed across updates. The 2,000-example observation is reused from A. The population row has `test_size=NA` and no test seed. Each finite row contains the signed and absolute differences from its own population crossing. Because t₁ is shared, the signed difference in t₂ equals the difference in measured grokking delay. [Paired trial records](results/robustness/population_trials.csv) · [Statistics](results/robustness/population_summary.csv) · [Plot](results/robustness/finite_vs_population.png)

**Uncertainty.** The tables give arithmetic means, medians and sample SDs (divisor 19). They also give pointwise 95% percentile bootstrap intervals for both means and medians. Resampling uses 20,000 draws of 20 complete trial IDs with replacement, with NumPy seed 3102026 and identical draws across conditions. Paired differences are formed before resampling. Plot intervals are for the mean. These intervals quantify sampling uncertainty across the specified trial population. They provide neither simultaneous coverage of the grid nor prediction intervals for an individual run.

An endpoint missing at the cap is recorded as `unobserved`. Dependent delays and differences inherit that status. Summary rows give observed and unobserved counts. If any observation for a metric is censored, its unrestricted moments and intervals are reported as NaN. This avoids substituting the cap or conditioning silently on successful trials.

**Why test size matters.** At a fixed update, conditional on the training trajectory, each independent Gaussian test residual has mean zero and variance equal to the population MSE. For positive population MSE, the ratio of finite to population MSE has distribution χ²(N)/N, where N is the test-set size. Its relative standard deviation falls from 14.1% at N = 100 to 1.41% at N = 10,000. A finite set that underestimates loss near ε can cross early, and one that overestimates it can cross late. Reusing the set over time correlates successive estimates. The fixed-update distribution therefore explains a source of timing variability without specifying the distribution of the first crossing itself.

**Reproduction.** From the environment described under Milestone 1, run these commands in PowerShell. The first command invoking Python runs the original numerical tests along with the new evaluation tests. The last command replays trial 1 without acceleration, measuring population loss directly from the weights.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -m unittest -v test_baseline test_weight_decay test_figure2bcd test_robustness
.\.venv\Scripts\python.exe robustness.py
.\.venv\Scripts\python.exe verify_robustness.py
```

### Robustness results

All 20 trials crossed both thresholds in every condition. Tightening ε from 0.05 to 0.005 increased mean training-crossing time by 153 updates and mean test-crossing time by 12,413 updates. Delayed generalization was observed throughout this threshold range, but its measured duration depended strongly on ε.

| ε | Mean t₁ | Mean t₂ | Mean delay | Median delay | Delay SD | 95% CI for mean delay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 306.6 | 26,860.3 | 26,553.7 | 26,504.0 | 348.0 | [26,408.5, 26,703.8] |
| 0.010 | 257.8 | 22,875.6 | 22,617.8 | 22,565.5 | 326.7 | [22,479.9, 22,756.8] |
| 0.020 | 211.3 | 19,168.9 | 18,957.6 | 18,927.0 | 318.5 | [18,822.9, 19,092.4] |
| 0.050 | 153.9 | 14,447.8 | 14,293.9 | 14,285.5 | 315.7 | [14,160.7, 14,426.9] |

At ε = 0.01, the exact population mean t₂ was 22,931.4 updates (95% CI [22,822.3, 23,032.2]). The exact population mean delay was 22,673.6. The largest finite test set had a smaller mean absolute timing error than the smallest one. Positive signed differences indicate later finite-test crossings.

| Test examples | Mean finite t₂ | Mean paired difference | SD of paired differences | Mean absolute difference (95% CI) |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 22,850.4 | −81.1 | 781.1 | 624.9 [430.8, 820.2] |
| 500 | 23,073.5 | +142.1 | 228.0 | 205.1 [136.1, 280.0] |
| 2,000 | 22,875.6 | −55.8 | 236.4 | 182.6 [118.4, 250.7] |
| 10,000 | 22,962.4 | +31.0 | 80.3 | 71.2 [52.6, 91.5] |

The signed differences varied in direction across sizes. For N = 500, the pointwise interval for the mean signed difference was [+46.3, +241.4] updates. The other three signed intervals included zero. There is one independent test draw of each size per trajectory, with 20 trajectories overall. This supports a descriptive comparison of timing error under the baseline Gaussian model. The data provide limited precision for small biases and comparisons between adjacent sizes.

The evaluation took 490.8 s with one OpenBLAS thread, including 190.4 s for A (including shared trajectory setup) and 295.9 s for B (reusing the shared 2,000-example result). The balance covers aggregation and output generation. The 22 tests passed in 104.6 s, with some overlap with the evaluation run.

A separate 26.0 s replay of 26,690 literal updates matched every training and test crossing for the four thresholds in trial 1. At ε = 0.01 the weight-norm calculation crossed at 23,159 updates, versus 22,627 for 100 test examples and 22,769 for 2,000. All three matched the accelerated measurements exactly. [Literal verification](results/robustness/verification.json)

## Milestone 4 · Feature correlation

Across 20 paired seeds, stronger equicorrelation delayed training convergence to MSE 0.01 and advanced the population crossing. All 80 runs crossed both thresholds. Times are integer GD updates per trial. Intervals below are pointwise percentile bootstrap intervals for **mean within-trial differences** (20,000 draws of 20 whole trial IDs, seed 4102026). SDs use divisor 19. [Delay and paired differences](results/feature_correlation/delay.png) · [Training and population MSE histories](results/feature_correlation/losses.png)

| ρ | Mean training t₁ | Mean population t₂ | Mean delay ± SD | Mean delay change from 0 (95% CI) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 257.8 | 22,931.4 | 22,673.6 ± 245.4 | 0 by construction |
| 0.30 | 331.5 | 21,041.0 | 20,709.6 ± 248.9 | −1,964.1 [−1,985.8, −1,932.4] |
| 0.60 | 482.9 | 18,097.9 | 17,615.0 ± 248.5 | −5,058.6 [−5,083.2, −5,026.9] |
| 0.85 | 855.5 | 13,075.3 | 12,219.8 ± 251.7 | −10,453.8 [−10,491.3, −10,415.0] |

**Design.** Each trial reuses its Milestone 3 teacher, student and Gaussian training draws across all four settings. Input dimension is 1,000, training size 100, step size 1 and ridge penalty 0.0001. For a baseline row z with covariance I/m, scale its projection onto u = 1/√m times the all-ones vector by √(1+(m−1)ρ) and its orthogonal projection by √(1−ρ). The resulting Gaussian rows have diagonal variance 1/m and off-diagonal covariance ρ/m. Labels use the transformed inputs and the fixed noiseless teacher. Population MSE for error d = w − θ* is (1−ρ)‖d‖²/m + ρ(uᵀd)². Raw records include both initial MSEs, seeds, spectrum and alignment. Missing crossings at 3,000,000 updates use `unobserved`; affected aggregate metrics and intervals are NaN with observation counts retained. [Trial data](results/feature_correlation/trials.csv) · [Summary](results/feature_correlation/summary.csv)

**Discrete dynamics.** The runner uses the baseline generator and evaluates the exact ridge GD recurrence in the training matrix's singular basis. Each row-space coordinate's displacement from its ridge fixed point scales by (1−η(s²/n+λ)) per update. The training-null component scales by (1−ηλ). The runner evaluates both MSEs at every integer step in blocks through the first crossings, including step zero. It saves additional common checkpoints through update 30,000 for comparisons across settings. This uses the same discrete objective and initialization as the baseline. [Runner](feature_correlation.py) · [Checkpoint losses and directional components](results/feature_correlation/losses.csv)

**Spectrum and interpretation.** The covariance trace stays at one. As ρ increases from 0 to 0.85, the shared-direction eigenvalue increases from 0.001 to 0.85015 and the other 999 eigenvalues decrease from 0.001 to 0.00015. The participation ratio, (tr Σ)²/tr(Σ²), falls from 1,000 to 1.38. The mean smallest positive eigenvalue of XᵀX/n falls from 0.00477 to 0.000717, consistent with slower decay of some training residuals.

The population metric itself changes with ρ. At update 5,000, the mean squared Euclidean weight error was 329.48 at ρ = 0 and 329.51 at ρ = 0.85. Yet mean population MSE was 0.32948 and 0.04978, respectively. At ρ = 0.85 the shared direction contributed only 0.00035 of that loss. Once this high-variance direction was fit, the remaining errors received less weight in the population metric. This accounts for much of the earlier population crossing without requiring faster overall recovery of the teacher weights. Initial population losses also fluctuated more across seeds as variance concentrated in one direction (SD 0.050 at ρ = 0 and 0.704 at ρ = 0.85).

**Teacher alignment.** For positive ρ the unique dominant axis is u. At ρ = 0 all axes have equal variance. Squared teacher cosine with u ranged from 0.000028 to 0.002831 across these randomly oriented teachers. The across-seed Pearson correlations of alignment with the paired delay change were −0.16, +0.04 and +0.25 for the three positive settings. All three bootstrap intervals included zero (at ρ = 0.85, [−0.10, 0.65]). This exploratory comparison has limited precision and very little coverage of strongly aligned teachers. It does not establish a teacher-alignment effect. [Alignment estimates for training, population and delay](results/feature_correlation/alignment.csv)

**Reproduction and validation.** From the existing project environment, run the following in PowerShell. The sweep took 6.61 s and the literal comparison took 5.08 s with one OpenBLAS thread. All 20 isotropic crossing pairs matched the saved Milestone 3 population controls. For seed 1 at all four correlations, literal GD through 30,000 updates matched both first crossings exactly. Maximum absolute MSE disagreement across logged checkpoints was below 4 × 10⁻¹⁴. Independent tests check the exact covariance square root, marginal and sampled covariances, dense quadratic population loss, signed GD factors and cap handling. [Config and runtime](results/feature_correlation/run.json) · [Verification receipt](results/feature_correlation/verification.json)

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
python -m unittest -v test_feature_correlation
python feature_correlation.py
python verify_feature_correlation.py
```

The paired observations support an effect of positive equicorrelation on the measured delay for this threshold, scale, ridge penalty and covariance family. They do not establish a universal direction for other spectra, thresholds or teacher distributions.
