# E-058 amendment A1 — user accuracy targets (registered before any E-058 analysis)

Registered 2026-10-24 ~20:10 local, while S3 acquisitions were running and before
S4 or any E-058 velocity/flow result existed. The user set the phantom success
targets: **throat flow within ±8 %, peak throat velocity within 8 %, with ≥ 90 %
coverage.** This amendment fixes how they are computed. It adds an evaluation;
it changes no S0–S4 gate, data, reference or the PROTOCOL.md decision rule.

## Shared definitions

- Arms: E055 300k (`baseline_coarse`), E058 `baseline_fine`, E058
  `short_echo_fine`. Each is compared with CFD averaged at its own actual ADC
  times (E048 references for baseline timing, E058 S1 references for the
  candidate).
- Per-bin support (the coverage definition): in bin b, a voxel is supported when
  all four REF/X/Y/Z magnitudes are ≥ 5 % of that encoding's bin-b maximum and
  > 0. This is the per-bin rule E055 already reported as `bin_coverage`. The
  fixed all-bin/all-encoding mask is reported as a sensitivity.
- Velocity: the unchanged `velocities(images, 5 m/s)` helper; axial component.
- Throat region: source z in [−10, +10) mm (`REGIONS["throat"]`), eligible
  CFD lumen (fraction > 0), fraction-weighted. Throat section: the plane
  nearest source z = 0 (`PLANES["throat"]`), area = plane fraction × 3.125 mm².
- Peak bin b*: the bin of maximum full-section CFD throat flow (baseline timing),
  fixed for all arms.

## Targets (all three must hold for PASS)

- **T1 throat flow:** cycle-integrated throat-section flow, Koma versus CFD on the
  identical retained area per bin, |error| ≤ 8 %. Reported alongside: Koma
  retained-area flow versus full-section CFD flow (includes coverage loss),
  peak-bin flow error and worst per-bin error.
- **T2 peak throat velocity:** at b*, maximum axial velocity over supported
  throat-region voxels versus the CFD maximum over the whole eligible throat
  region, |error| ≤ 8 %. Sensitivities: the 99th percentile and same-support CFD.
- **T3 coverage:** throat-section area coverage ≥ 90 % in every bin and
  throat-region volume coverage ≥ 90 % at b*. Whole-lumen per-bin coverage is
  reported, not gated.

No fitting, filling, regularized/hybrid voxels, threshold tuning or smoothing.
Noiseless, single spin sample, same-source internal consistency; a pass would
not be independent physical validation or an SNR result. 1M stays on hold.
