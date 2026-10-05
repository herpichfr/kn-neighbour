# Stellar masses, indicators and environment weighting (Phase 3)

*Run 2026-10-05, `knn build config/mkw4.yaml` on commit 7a8e06c plus the Phase 3 changes. Provenance: `data/derived/MKW4_enriched.parquet.provenance.json`.*

## Method

- **Photometry.** Default set `legacy` (Legacy Surveys mag_g, mag_i, mag_r, W1–W3, AB). Values of 99, NaN and inf are treated as missing. Galactic extinction comes from the SFD map (median E(B−V) = 0.021) and K-corrections (g−i, g−r, u−r) from Chilingarian et al. The catalogue magnitudes are assumed not to be extinction-corrected.
- **Distance.** Luminosity distance from the config cosmology (H0 = 70, Om0 = 0.3). For photo-z galaxies, the median and 16th/84th percentiles (`_lo`/`_hi`) are taken over the same 200 Gaussian redshift draws as the environment. This captures the distance uncertainty only.
- **Mass.** The default `logmstar` uses Taylor et al. 2011: log M*/L_i = −0.68 + 0.70 (g−i), with M_i,sun = 4.58. As a cross-check, `logmstar_w1` uses Cluver et al. 2014 (Eq. 2): log M*/L_W1 = −1.96 (W1−W2) − 0.03, in Vega magnitudes with M_sun = 3.24. The WISE AB-to-Vega offsets are 2.699 (W1) and 3.339 (W2).
- **Mass-weighted environment.** Entries with `weight: mass` sum the stellar mass of the k nearest tracers per volume (knn3d) or per area (sigma2d). Tracers without a mass are excluded, leaving 3,973 of 5,264. log(1+δ) is normalised by the tracer median, as in the count version. The columns end in `*_k10_mass`. knn3d is set to NaN for photo-z rows (`photoz_mode: none`; see REPORT_photoz_validation.md).
- **Indicators.**
  - Colours: g−r, g−i, u−r (S-PLUS u) and W2−W3.
  - Red sequence: a linear fit of g−r against log M* on spec-z galaxies with log M* in [9.5, 12]. The fit is seeded with the upper half of each 0.4-dex mass bin, then iterated with asymmetric sigma clipping (−2/+3 times the NMAD).
  - `rs_offset` is the colour minus the sequence; `quenched` means offset > −2σ_RS.
- **Joins.** Nearest position within 1.5 arcsec; ambiguous matches are flagged.

## Sources

- **Taylor et al. 2011** (arXiv:1108.0635, Eq. 7). Their printed Eq. 8 omits the +0.4 M_i,sun term, so Eq. 7 with M_i,sun is used.
- **Cluver et al. 2014** (arXiv:1401.0837, Eq. 2 and Sec. 4.2).
- **Extinction coefficients:** Schlafly & Finkbeiner 2011 (Table 6) and the Legacy Surveys DR9 catalogue documentation.
- **K-corrections:** Chilingarian et al. 2010, 2012 (kcor.sai.msu.ru).
- **WISE Vega/AB offsets:** WISE Explanatory Supplement, Sec. 4.4h.
- **Cluster:** Sarkar et al. 2021 (arXiv:2012.05160) for z_cl = 0.02 and R200 = 0.884 Mpc.

## Results

- **Magnitude set.**
  - Legacy g−i agrees with S-PLUS `auto` to a median −0.005 mag (NMAD 0.13, n = 955).
  - S-PLUS PStotal and aper_6 miss total light: their M_i is 1.6 and 1.0 mag fainter than Legacy.
  - Legacy covers 3,393 of 4,328 spec-z galaxies, against 1,098 for S-PLUS.
- **Masses** (log M*).
  - Spec-z (3,393 galaxies), 5th/50th/95th percentiles: 7.02 / 8.47 / 10.37.
  - Photo-z (500 galaxies): 6.59 / 7.52 / 8.51, with a median uncertainty (hi − lo) of 0.43 dex.
- **Independent check.** A separate Taylor calculation without extinction or K-correction gives masses a median 0.036 dex higher than the pipeline (5th–95th percentile: 0.018–0.070 dex), which is the expected size of those corrections.
- **Taylor vs W1.** The median Taylor − W1 is −0.33 dex (NMAD 0.23, n = 2,500), and the offset varies with W1−W2 colour.
- **Red sequence.**
  - Fit: g−r = 0.738 + 0.056 (log M* − 10), σ = 0.035; 363 of 691 galaxies are in the fit.
  - Quenched fraction for spec-z: 0.55 at log M* ≥ 9.5 and 0.20 overall (extrapolated below 9.5). It rises from 0.05 at log M* < 8 to 0.81 above 10.5.
- **Joins.**
  - Morphology: 443 matched, 2 ambiguous.
  - Substructure: 160 of 168 matched.
  - Groups: 1,001 and 1,162 matched, at exact positions. The two Nr200 files number their groups differently, so the ids are kept in separate columns rather than merged.
- **Phase space.**
  - The cluster's RA, Dec and σ_v are unknown (null), so R comes from the catalogue's radius_Mpc.
  - The velocity offset is computed with z_cl = 0.02 and matches the catalogue's v_offset to 0.15 km/s.
  - The catalogue's columns imply a centre at (180.990, +1.886) deg.

## Caveats

- The absolute mass scale is not validated; Taylor and W1 masses differ by 0.3 dex.
- Red-sequence results depend strongly on the lower mass edge, and offsets below log M* = 9.5 are extrapolations. The mass comes from g−i while the sequence is in g−r, so the two are partly degenerate.
- The photo-z mass uncertainty covers distance only. Photometric errors and the 0.1 dex scatter of the mass-to-light relation are not propagated.
- S-PLUS and DECam filters use SDSS extinction and K-correction coefficients, and WISE magnitudes get no K-correction.
- W3 has no S/N cut, so `sf_w2w3` is unreliable.
- The colour plausibility limits are the authors' own choice, not taken from a paper.
- The SFD map directory is an absolute path outside the repository (`extinction.sfd_dir` in config/mkw4.yaml).
