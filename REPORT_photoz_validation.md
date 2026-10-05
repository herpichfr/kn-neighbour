# Photo-z environment validation (Phase 2)

*Run 2026-10-05 on top of commit f9880d5 plus the Phase 2 changes. Command: `PYTHONPATH=src python -m knn.cli validate-photoz config/mkw4.yaml`. Machine-readable results: `data/derived/photoz_validation.json`.*

## Question

Photo-z-only galaxies (614 in 0.007 < z < 0.037) need an environment. How well does an environment measured from a photo-z recover the environment the same galaxy has at its spec-z?

## Setup

- **Validation sample:** 1,114 galaxies with a spec-z in 0.007–0.037 and a photo-z. Each one's "true" environment is measured at its spec-z position; its photo-z environment is measured as if only the photo-z were known. The galaxy itself is excluded from its own neighbours in both cases.
- **Tracers:** 5,264 spec-z galaxies in 0.002–0.042. The wider tracer window keeps galaxies near the edges of 0.007–0.037 from seeing a truncated volume.
- **Estimators (k = 10):**
  - `knn3d`: distance to the 10th neighbour in 3D comoving space.
  - `sigma2d`: projected density Σ₁₀ inside a redshift slice. The slice is ±1,500 km/s for spec-z and ±1.5σ(1+z) for photo-z.
- **Photo-z modes:**
  - `point`: the galaxy sits at `zml`.
  - `gauss`: 200 draws from N(zml − bias(1+zml), σ(1+zml)).
  - `pdf`: 200 draws from the catalogue's 3-Gaussian mixture.
- **Comparison:** 910 galaxies have a valid value in all three modes (point mode loses the galaxies whose `zml` lies outside the window).

## Photo-z quality in the window

- Point estimate: bias = +0.0046 and σ_NMAD = 0.0066 in (zml − z)/(1 + z). Over z < 0.1 the scatter is about twice as large (σ_NMAD ≈ 0.013, bias −0.004).
- Calibration of the redshift distributions, shown as a histogram of where each spec-z falls in its own distribution (10 bins; a calibrated distribution gives about 111 per bin):

| Distribution | Histogram |
|---|---|
| Catalogue mixture | 321, 242, 177, 106, 82, 54, 57, 41, 20, 14 |
| Gaussian (`gauss` mode) | 196, 89, 79, 80, 110, 123, 138, 125, 81, 93 |

The catalogue mixtures are biased high: the spec-z mostly sits in the low end of each distribution. The Gaussian is roughly flat but has heavier tails than σ_NMAD implies. Removing components wider than 0.5 changes nothing for this sample.

## Results

Metrics compare the photo-z log(1+δ) with the true log(1+δ):

- **ρ:** Spearman rank correlation.
- **Offset and NMAD:** median difference and robust scatter, in dex.
- **Quartile:** fraction of galaxies placed in the correct quartile of the true values. Chance is 0.25.
- **Coverage:** fraction of true values inside the 16–84 percentile range. It is 0.68 if the ranges are right.

| Estimator | Mode | ρ | Offset | NMAD | Quartile | Coverage |
|---|---|---|---|---|---|---|
| knn3d | point | 0.07 | −0.83 | 0.86 | 0.27 | — |
| knn3d | gauss | 0.09 | −0.82 | 0.61 | 0.25 | 0.32 |
| knn3d | pdf | 0.05 | −0.90 | 0.62 | 0.25 | 0.31 |
| sigma2d | point | 0.64 | −0.23 | 0.29 | 0.47 | — |
| **sigma2d** | **gauss** | **0.78** | **−0.09** | **0.24** | **0.57** | **0.58** |
| sigma2d | pdf | 0.76 | −0.19 | 0.26 | 0.50 | 0.42 |

By photo-z `odds` tercile (low / mid / high, about 303 galaxies each), sigma2d with gauss gives ρ = 0.71 / 0.73 / 0.88 and quartile accuracy 0.51 / 0.55 / 0.65. For knn3d, ρ stays between −0.07 and 0.13 in every tercile and mode.

**Independent cross-check:** a separate implementation of sigma2d gave ρ = 0.60 on 1,064 galaxies, using a single bias-corrected photo-z centre and no draws. That is consistent with the point and gauss rows above.

## Conclusions

1. **3D k-th neighbour distances cannot be used for photo-z galaxies.** Recovery is at chance level in every mode and every odds tercile. σ(1+z) ≈ 0.0066 is about 28 Mpc along the line of sight, much larger than the 10th-neighbour distance (median 2.5 Mpc). The `knn3d` values on photo-z rows of the enriched catalogue should be hidden or flagged.
2. **Projected density with Gaussian draws is the default for photo-z galaxies** (`sigma2d`, `photoz.mode: gauss`). It recovers the true ranking well (ρ = 0.78) and puts 57% of galaxies in the right quartile.
3. **The catalogue's own redshift distributions are not the better choice.** They are miscalibrated for these nearby galaxies.
4. **Photo-z environments stay noisier than spec-z ones.** The scatter is 0.24 dex, the offset −0.09 dex, and the uncertainty ranges are too narrow (58% coverage against 68%). Analyses should carry the `_lo`/`_hi` columns, cut on `p_in_window`, and check every result separately for spec-z and photo-z galaxies.

## Caveats

- **In-sample parameters:** the Gaussian's bias and σ are measured on the same 1,114 galaxies used for validation. Only two parameters are fitted, so the effect should be small, but it is not zero.
- **Provisional normalisation:** log(1+δ) is normalised by the median tracer density, not by a completeness-corrected mean.
- **Brighter validation sample:** galaxies with both redshifts are brighter than typical photo-z-only galaxies, so the recovery for the real photo-z sample is probably somewhat worse than shown here.
