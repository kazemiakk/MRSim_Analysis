# E-061 / D-064 — reference precision, axial VENC and phase unwrapping

Registered 2026-10-25 by user choice ("lets do 1, 2, and 3"), before any E-061
output exists.

Root: `D:/MRSim/outputs/venc_unwrap_e061`. E-060 outputs are read-only inputs.
No 1M run.

## Question

E-060 met the user's targets but showed two problems:

- **Axial aliasing.** At 3.0 m/s axial VENC, 87 voxel-bin samples wrap in
  bins 5–12. This causes the systolic throat-flow dip, for example −17.8 % at
  bin 9.
- **Reference imprecision.** The reference's own section-flow quadrature
  varies with CV 4.7 % along the slab.

E-061 measures how much of the remaining error each fix removes.

## Arms

All arms use E-060's phantom, seam, numerics and A1 target definitions,
through the frozen `evaluate()` reached via `analyze_e060.py` helpers.

1. **REF3 — reference precision.**
   - The S4a v2 reference, rebuilt with 4×4×4 volume sub-samples and 4×4
     S-normal section sub-samples per voxel. Previously 2.
   - Same frozen functions, seam and gates (ADC-weight identity; throat and
     upstream flow within 5 % of E048).
   - Reported: along-slab section-flow CV, and the change in every target
     metric against `references_v2`.
2. **VENC4 — axial VENC 4.0 m/s.**
   - Only the S (Z) encoding is re-acquired.
   - Gates before any signal:
     - Builder regression and all S0 audits, as in E-060 v2.
     - REF, X and Y must be event-equivalent to E-060 `sequences_v2`.
     - The ADC timing and label arrays must be exactly equal for all four
       encodings. This compares the arrays, not the HDF5 file bytes.
     - The Z first moment must equal −1/(2γ·4.0).
     - The E-060 seam markers must be valid on the new Z, and the Z step
       self-difference at 16 µs versus 8 µs must be ≤ 1e-4.
   - Acquisition: E-060 REF/X/Y results plus the new Z, on GPU1.
3. **UNWRAP — declared temporal phase unwrapping**, a standard 4D-flow
   post-processing step that uses no CFD information.
   - For each voxel and velocity component, the phase difference
     angle(E_axis · conj(E_REF)) is unwrapped along the 30 cardiac bins with
     `numpy.unwrap` (period 2π), starting from bin 1.
   - Velocity = unwrapped phase × VENC / π.
   - Applied to the E-060 data, and to VENC4 as a check that it changes
     nothing there.

## Reporting

Descriptive; no new pass/fail rule. For each arm, reported against
`references_v2` and REF3:

- T1, T2 and T3 (A1 definitions; per-bin support primary, fixed mask as a
  sensitivity).
- Regional axial RMSE and bias, upstream/throat/downstream/slab.
- Per-partition cycle flow error.
- Count of wrap-like samples (CFD > 1.5 m/s while Koma < 0).

Evidence class: same-source internal consistency, noiseless, one 300k seed.
Noise is not modelled, so the √(4/3) noise cost of the higher VENC is not
represented.
