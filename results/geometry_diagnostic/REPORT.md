# Milestone 4B · Does feature correlation speed up teacher recovery?

For the original random teachers, changing the evaluation distribution explains essentially the entire advance in the population crossing. Training still takes longer. The delay also shortens under a common metric, primarily because its starting event (the training crossing) happens later. [Crossing times](crossings.png) · [Normalized trajectories](normalized_losses.png)

### Same weights, two evaluations

We replayed the 20 paired trials from commit `0620b2b` at ρ = 0, 0.3, 0.6 and 0.85. Each trajectory uses its original correlated training matrix, teacher and initialization. Both scores are evaluated on the same discrete GD iterates. For weight error d and unit shared axis u, the training-distribution risk is (1−ρ)‖d‖²/m + ρ(uᵀd)². The common risk is ‖d‖²/m, which measures parameter recovery on one scale across settings. We record the first integer update with absolute loss ≤ 0.01, including step zero, with a cap of 3,000,000. All 80 original and 48 controlled trajectories crossed under both metrics. The runner records a missing endpoint as `unobserved` and propagates it to dependent contrasts.

| Training ρ | Mean t₁ | Mean t₂, own distribution | Mean t₂, common isotropic | Paired change in common t₂ from ρ = 0 (95% CI) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 257.8 | 22,931.4 | 22,931.4 | 0 |
| 0.30 | 331.5 | 21,041.0 | 22,931.95 | +0.55 [−2.65, +4.05] |
| 0.60 | 482.85 | 18,097.85 | 22,932.00 | +0.60 [−2.65, +4.15] |
| 0.85 | 855.5 | 13,075.3 | 22,932.35 | +0.95 [−2.35, +4.50] |

At ρ = 0.85 the original delay fell by 10,453.8 updates relative to zero correlation. Using common risk for the endpoint reduces that difference to −596.75 updates [−616.90, −576.35]. The difference between these contrasts, 9,857.05 updates, comes from changing the evaluation metric on the same trajectory. It accounts for about 94% of the original delay change at this threshold. The remaining change follows from t₁ moving 597.7 updates later while common t₂ moved 0.95 updates later.

**Loss normalization.** At each saved checkpoint, divide each loss by its own step-zero value within that trial and metric, then average ratios across trials. The absolute crossing threshold is applied before normalization. At update 100, normalized isotropic risk was 0.900 at ρ = 0 and 0.955 at ρ = 0.85, so an early difference survives common scoring. At update 5,000 the two isotropic ratios were 0.33227 and 0.33231. The own-distribution ratios were still well separated at 0.33227 and 0.16654. This distinguishes an early transient from a persistent advantage in recovering weights. [Checkpoint data](checkpoints.csv)

### Controlled alignment

We selected the first six seeds in advance and ran both unit-norm teachers at all four settings. The parallel teacher is +u. For the orthogonal teacher, project that trial's original teacher onto u⊥ and renormalize. Within a pair, only the teacher and the labels it induces change. The student and inputs are identical. At ρ = 0, u is a nominated axis in a degenerate spectrum. For positive ρ the two unit teachers have different target variances under the training distribution, a consequence of the alignment manipulation.

At ρ = 0.85, common t₂ averaged 22,563.5 for parallel teachers and 23,012.7 for orthogonal teachers. The paired parallel-minus-orthogonal difference was −449.2 updates [−496.7, −403.0], compared with −12.7 [−67.3, +40.2] at ρ = 0. Within the parallel condition, most of the common-metric advance was already present at ρ = 0.3 (−423.2 updates relative to zero, versus −434.7 at 0.85). Within the orthogonal condition, the corresponding changes were +1.0 and +1.8. The controlled runs support an alignment-specific effect on recovery at this threshold. Six seeds, two extreme orientations, one teacher norm and one spectrum family limit the inference. [Raw trials](trials.csv) · [Means, SDs and contrasts](summary.csv)

### Reproduction and checks

From the project root in the existing environment:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python geometry_diagnostic.py
python verify_geometry_diagnostic.py
```

All intervals are pointwise percentile bootstrap intervals for means, using 20,000 resamples of trial IDs with replacement (seed 4202026). Differences are computed within trial before resampling. All 80 original training times, own-distribution crossings and delays matched the saved Milestone 4 records exactly. Six full-size literal GD replays across four seeds and all three teacher types matched all 18 crossing counts exactly. Raw and normalized checkpoint discrepancies through update 30,000 were below 4 × 10⁻¹⁴. The diagnostic took 9.32 s and the literal verification took 6.09 s with one OpenBLAS thread. [Configuration](run.json) · [Literal replay results](verification.json)
