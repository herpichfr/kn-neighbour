# Refactor plan: an interactive environment explorer for galaxy evolution

*Drafted 2026-10-05. Status: proposal, open questions at the end.*

## 0. Notes on the current code

Reviewed with the author on 2026-10-05:

1. **Spec-z only, by design.** At `src/calculate_kn_neighbour.py:52-54`, only galaxies with a spec-z in the window are kept, chosen for their precision. `MKW4+v7_k10_neighbours.csv` therefore has 4,328 rows, all spec-z, and the 617 photo-z-only galaxies in the window are excluded. The new pipeline makes this an explicit choice of tracer set (§1).
2. **The angle × 3D-separation metric was a verification test.** `sorted(sep2d.value * sep3d.value)[11]` multiplies an angle in degrees by a 3D separation. The new pipeline uses comoving 3D distance; the legacy metric is kept only to reproduce the existing table in a regression test.
3. **Neighbour index.** The sorted list holds the galaxy itself at index 0, so index `i` is the i-th neighbour: `[11]` is the 11th neighbour and the 10th would be `[10]`. The new code takes `k` as a parameter and excludes the galaxy itself explicitly, so the existing table corresponds to `k = 11`.
4. **Redshift as distance.** `distance=redshift, unit=Mpc` works out to Hubble-flow geometry scaled by c/H₀, which is adequate at z < 0.04. The new pipeline uses comoving distances from an astropy cosmology so that wider windows work.
5. **S-PLUS vs Legacy magnitudes.** The S-PLUS and Legacy filter systems differ, so their magnitudes are not expected to match exactly. For the mass step (§3): `r_auto` agrees with Legacy `mag_r` to a median 0.08 mag, while `r_PStotal` is a median 0.88 mag fainter with 0.7 mag scatter. Which S-PLUS magnitude feeds the mass-to-light relation is decided in Phase 3.

There's also a speed problem: the code compares every galaxy to every other one, and the parallel pool re-sends the whole `SkyCoord` array with each task. A `scipy.spatial.cKDTree` on Cartesian comoving coordinates does the same job in milliseconds for 10⁴ galaxies. That speed is what makes live recomputation in a web app possible.

## 1. Spectroscopic vs photometric redshifts (the central decision)

From the current data:

| | Value |
|---|---|
| Spec-z in the window 0.007–0.037 | 4,328 |
| Photo-z only in the window | 617 (this parquet looks pre-selected; 2,846 galaxies are photo-z only overall) |
| Photo-z scatter σ_NMAD (3,682 galaxies with both) | **0.013**, bias −0.004 |
| Photo-z scatter in distance | ≈ 56 Mpc, about half the whole window. Cluster velocity smearing for spec-z is ≈ 7 Mpc. |

So photo-z galaxies **cannot** go into a 3D nearest-neighbour calculation at face value. The recommendation is to split *tracers* from the *population being studied*:

- **Tracers (they define the density field):** spec-z only, cut to a luminosity- or mass-limited sample so the tracer density doesn't change with z just because survey depth changes. If spec-z coverage is uneven across the sky, add a completeness map.
- **Population (galaxies whose environment is measured):** both spec-z and photo-z galaxies. For each photo-z galaxy, its density is averaged over its redshift probability distribution. This distribution already exists as a 3-Gaussian mixture in `pdf_weights_*`, `pdf_means_*` and `pdf_stds_*`. Repeated random draws from it give a spread on the density, not just one number.
- **2D alternative for wide areas:** projected density Σ_k inside a redshift slice whose width depends on redshift type. That would be about ±1,500 km/s for spec-z and about ±1.5σ(1+z) for photo-z. This is the standard approach for photo-z surveys.
- **Built-in check:** take the 3,682 galaxies that have both redshifts, compute their density using only the photo-z, and compare it with the spec-z result. That measures how well photo-z environments recover the truth. It goes in a `REPORT_photoz_validation.md`.

Every row carries `z_best`, `z_type`, `sigma_z` and a quality flag (`odds`), and the interface can always split results by `z_type`.

## 2. Environment ("concentration") parameters

These are plug-ins with a common interface, so the interface can list them and new ones are easy to add. Each one outputs a value, an uncertainty and an overdensity log(1+δ):

- **k-th nearest neighbour in 3D:** r_k and ρ_k = k / (4/3 π r_k³), with k adjustable.
- **Projected Σ_k** inside a velocity slice (Σ₅, Σ₁₀).
- **Fixed-aperture counts:** neighbours within R Mpc and ±Δv.
- **Mass-weighted density:** the stellar mass of the k nearest neighbours per area or volume, instead of their number. This is "mass as a concentration parameter".
- **Luminosity-weighted density:** the same with luminosity.
- **Cluster-centric position:** R/R₂₀₀ and the phase-space position (R/R₂₀₀, Δv/σ), which separates recently infalling galaxies from the virialised core. `radius_Mpc` and `v_offset` already exist.
- **Substructure / group membership** from `mkw4_substructures.csv` and `id_group_final`.
- **Later:** distance to the filament spine (minimum spanning tree first, DisPerSE if needed), which is the natural parameter for a filament study, and Voronoi density.

## 3. Stellar masses from magnitudes

- **Default method: a colour-based mass-to-light relation.** Taylor et al. (2011): log M/L_i = −0.68 + 0.70(g−i), using S-PLUS or Legacy g and i. As a cross-check, Cluver et al. (2014) uses WISE W1 with the W1−W2 colour (9,297 galaxies have W1). WISE magnitudes must be on the Vega system for that relation.
- **Required corrections:** cosmological distance modulus from `z_best`, Galactic extinction (check whether the catalogue already applies it), and K-corrections (small at z < 0.04, but cheap to include). The magnitude mismatch from §0.4 must be resolved first.
- **Photo-z caveat:** at z ≈ 0.02, σ_z = 0.013 means about 0.5 dex of uncertainty in luminosity and therefore in mass. Photo-z masses need error bars carried from the redshift draws, and mass-weighted density should use spec-z tracers only.
- **Validation:** most of the spec-z comes from DESI DR1 (4,148 objects), whose value-added catalogues include stellar masses. Matching against them gives a direct check.
- **Optional later step:** full SED fitting on all 12 S-PLUS bands plus WISE, for example with CIGALE.

## 4. Galaxy-evolution indicators

- **Colours and quenching:** g−r and u−r; a red-sequence / blue-cloud split, preferably with a slope that depends on mass; WISE W2−W3 as a cleaner star-forming vs passive indicator.
- **Spectral classes:** `class_spec` and `subclass_final`.
- **Morphology** (459 galaxies in `MKW4_properties_concat_all_final.csv`): Sérsic index, CAS, Gini–M20, and size–mass.
- **Hα from J0660:** the filter covers Hα only up to z ≈ 0.019, so MKW4 (z ≈ 0.02) sits at its edge. Mark this as unreliable for the cluster itself.
- **Fixed-mass comparisons:** the analysis view always offers "indicator vs environment at fixed stellar mass", for example the quenched fraction against log(1+δ) in mass bins. Mass and environment are correlated, and any evolution signal is unreadable without separating them.

## 5. Architecture

```
kn-neighbour/
  pyproject.toml
  config/mkw4.yaml          # catalogue paths, column mapping, cosmology, z-window, cuts
  src/knn/
    io.py                   # load/cross-match catalogues via the column map
    redshift.py             # z_best, z_type, sigma_z, sampling from the photo-z distribution
    geometry.py             # astropy cosmology → comoving XYZ, KD-tree
    environment/            # one module per estimator, common interface
    mass.py                 # mass-to-light relations, K-corr, extinction, error propagation
    indicators.py           # colours, quenched flags, morphology joins
    build.py                # writes enriched parquet + provenance (config hash, git sha)
    cli.py                  # `knn build config/mkw4.yaml`
  app/                      # web interface
  tests/                    # regression + synthetic-catalogue tests
```

**Data flow:** `knn build` writes one enriched parquet with all galaxy columns plus the defaults for environment, mass and indicators. The app reads that file. With about 10⁴ galaxies, a KD-tree query takes milliseconds, so the app can **recompute an estimator live** when k, Δv, the tracer set or the sample cuts change. Only the expensive parts (random redshift draws, SED fits) are computed in advance.

**Web stack:** **Plotly Dash** is recommended. It's pure Python, so the science code is reused directly. Its WebGL 3D scatter handles 10⁵ points, it supports linked selection, and it runs locally or on an institute server. Panel/HoloViews is the alternative if its linked-brushing model is preferred. Streamlit is simpler, but it reruns the whole script on every change and is weak at linked selections.

**Interface layout:**
- **Sample panel:** z-window, spec/photo/both, odds and σ_z cuts, magnitude or mass limit, membership, substructure, field.
- **Environment panel:** estimator, its parameters, and the tracer set.
- **Main view:** a 3D map (RA, Dec, comoving distance, or Cartesian Mpc) with colour and size mapped to any column, plus a 2D sky map with a density map underneath.
- **Analysis panel:** indicator vs environment (running median, fractions with error bars) in mass bins, plus histograms. Everything is linked, so a selection in 3D highlights in every plot.
- **Export:** the selection as CSV, figures as PDF/PNG, and the full settings as YAML so any figure can be reproduced.

## 6. Phases

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | KD-tree engine, legacy regression test | The legacy metric reproduces `separation_k10` in the existing table, and the new estimator passes on a synthetic Poisson catalogue against the analytic expectation |
| 1 | Package layout, YAML config, `knn build` | One command produces the enriched parquet from `MKW4+v7.parquet` |
| 2 | Redshift module + photo-z validation | Report on photo-z vs spec-z environment recovery |
| 3 | Masses + indicators | Mass comparison against DESI written up; magnitude mismatch resolved |
| 4 | Dash MVP | 3D map, sample cuts and estimator switch with live recomputation |
| 5 | Analysis panels + export | Fixed-mass trend plots, YAML snapshot reproduces a figure |
| 6 | Optional | Filament distances, uncertainty bands from redshift draws, SED masses, deployment |

## Decisions (2026-10-05)

The author accepted the plan. Defaults adopted where a question was open:

- **Photo-z handling:** tracer/population split (§1), with spec-z tracers.
- **Masses:** colour-based mass-to-light relation first; SED fitting later.
- **Deployment:** local Dash app first; a shared server later if needed.
- **3D view:** raw Hubble-flow coordinates by default, with line-of-sight compression as an option.

Still open: whether a larger photo-z catalogue exists. The design does not assume a catalogue size.
