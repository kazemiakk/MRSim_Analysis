# E-060 S4 analysis — registered before any E-060 signal exists

Applies PROTOCOL.md "S4" with the E-058 amendment-A1 definitions. Script:
`analyze_e060.py`. References: `build_references_e060.py` (S4a). Neither may be
edited after it has run; a correction is a new versioned output with the
earlier one preserved.

## Data and conventions

- Velocity per axis: `angle(E_axis · conj(E_REF)) · VENC_axis / π`, with VENC
  (1.5, 1.5, 3.0) m/s for (R, P, S). This is the frozen helper's sign
  convention, applied per axis. S is the vessel axis (axial component).
- Arrays are (bin, S, P, R). For reuse, they are transposed to (bin, R, P, S)
  and the components reordered to (S, P, R). That makes the vessel axis last
  and the axial component index 0, so E-058's frozen `evaluate()`,
  `per_bin_support()` and `coverage()` run unchanged.
- CFD reference: the S4a actual-ADC average on the E-060 grid.
  - Section area = plane area fraction × 1.5625 mm².
  - Throat section = partition 15 (z = 0).
  - Throat region = |z| < 10 mm (z ∈ [−10, 10) mm, as in E-058).
  - Peak bin = argmax of the CFD full-section throat flow.

## Targets (A1, unchanged)

- **T1** — cycle same-area throat-section flow error |e| ≤ 8 %, with section
  coverage ≥ 90 % in every bin.
- **T2** — peak-bin maximum axial throat-region velocity versus the CFD region
  maximum, |e| ≤ 8 %.
- **T3** — throat-section coverage ≥ 90 % in all bins, and throat-region
  coverage ≥ 90 % at the peak bin.
- **Support** — per-bin, all four encodings ≥ 5 % of that bin's maximum
  magnitude.

Sensitivities (reported, never substituted):

- Fixed all-bin mask.
- T2 p99.
- T2 interior-only (CFD fraction = 1), added after E-058's diagnostic.
- Same-area flow error at the upstream sections, partitions 3–7
  (z −30 to −20 mm).

## Comparison with the measured scan (qualitative; the CFD carries ~60 % more flow)

- **Method.** E-059's frozen `analyse()` (`measured_s1_v2.py`) is applied
  unchanged to the Koma images:
  - Inputs: the central 24 partitions (3–26), which map one-to-one onto scan
    slices 0–23 with the throat at slice 12; velocity in cm/s; magnitude |REF|.
  - Segmentation: thresholds 20/25/30 %, with 25 % primary.
- **Reported** for Koma next to the scan:
  - Throat slice and diameter.
  - Throat-versus-upstream mean and peak flow.
  - Flow-conservation CV.
  - Peak forward velocity.
- **Flow-normalized waveforms** are compared in figures only.

## Hypothesis readout

E-059's hypothesis — that protocol geometry explains the scanner's throat-flow
conservation — is called **supported in simulation** only if both hold:

- Koma's scan-method throat-versus-upstream mean flow is within ±8 % (the
  scan's is +2.8 %).
- T1 is within ±8 %.

Otherwise it is **not supported**. In either case the result is internal
consistency plus a qualitative comparison with the scan, not physical
validation.
