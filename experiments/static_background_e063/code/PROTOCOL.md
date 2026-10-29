# E-063 / D-066 — static background (stationary spins) for the phantom and aorta

Registered 2026-10-26 by user decision ("No static tissue"; "Yes, run it") before any signal.
Root: `D:/MRSim/outputs/static_background_e063`.

## Why

Real 4D-flow images contain stationary material. Its phase is zero (uniform grey) and only air is noise. The real
phantom scan (E-059) shows two stationary objects (water bottles) beneath the tube. The E-060 and E-062 Koma phantoms
contain only flowing blood, so everything outside the vessel was noise.

## Method: exact linear superposition

MR signal is the sum over independent spins, and the perfect-spoiling model acts per spin. The signal of a phantom
containing blood and stationary material therefore equals the existing blood-only signal plus the signal of a
stationary-only phantom simulated with the identical sequence and numerics. The flow runs are not repeated.

**Phantom (E-060 sequences, E-060 numerics, 3 T).**

- **Geometry:**
  - Stationary spins fill the two bottles segmented from the time-mean scan magnitude: non-vessel components above
    25 % of the 99.5th percentile, present in at least half the slices.
  - They are placed relative to the vessel axis with the scan's pixel offsets: scan rows → Koma y (P), columns →
    Koma x (R), 1.25 mm pixels.
  - Extruded along the vessel axis for |S| ≤ 36 mm and clipped to one voxel inside the field of view.
- **Spin grid:** 2 × 2 in-plane per voxel (0.625 mm), at partition centres along S. ρ = 1, T1 = 1.0 s, T2 = 0.1 s.
- **Simulation:** REF and X are simulated in full, one per GPU, with the E-060 runner numerics: Float64,
  MigrationPerfectSpoiling, PrecessionMaxStepRule 16 µs / 0.5 µs, the same verified zero-amplitude seam markers,
  and B0 = 3 T.
- **Gate G1 (static identity):** stationary spins carry no velocity phase, so every encoding must equal REF.
  - Full X versus full REF: relative L2 ≤ 1e-6.
  - 30-profile prefixes of Y, Z (VENC 3) and E-061 Z (VENC 4) versus the REF prefix: ≤ 1e-6 each.
  - If G1 passes, the REF stationary image is used for every encoding. If it fails, stop.
- **Calibration (one scalar, stated):** the stationary image is scaled so that its median bottle magnitude relative
  to the median time-mean lumen magnitude of the flow REF image equals the same ratio in the real scan.

**Aorta (E-062 20-bin sequences, frozen koma_aorta simulator, 1.5 T).**

- **Geometry:** synthetic soft tissue fills the field of view, with a 2 mm margin, everywhere farther than 2 mm from
  any CFD node. The vessel wall is therefore a ≤ 1 mm dark rim.
- **Spin grid and tissue:** 2 × 2 × 2 per voxel. T1 = 0.9 s, T2 = 0.045 s (muscle-like). Proton density 0.8 × blood
  per volume, applied as the stated scale factor `0.8 × blood spin density / tissue spin density`.
- **Simulation:** REF/X/Y/Z, run on both GPUs and then combined, as in E-062.
- **Gate:** the stationary encodings agree with REF (relative L2 ≤ 1e-4, f32 simulation).
- **Not anatomy:** this tissue is a uniform block, not body anatomy.

**Combination and display (S2).**

- Combined complex image = flow image + scaled stationary image, per encoding and bin.
- Display noise as before: phantom SNR 84 matched to the scan; aorta SNR 12, with the flow phase shown as for
  VENC 0.3 m/s (labelled).
- Figures: scanner-style native slices (magnitude, through-plane and two in-plane phase images).
  - Phantom: shown beside the real scan.
  - Aorta: axial slices and an oblique candy-cane slab.

## Evidence class

Presentation and realism only.

- The bottle intensity is calibrated to the scan; the aortic tissue is synthetic.
- No registered E-060, E-061 or E-062 metric changes. Stationary spins sit outside the lumen, so the vessel velocities
  are unchanged except in partial-volume edge voxels, which are reported descriptively.
- Cost estimate: about 2 × 1 GPU-hour (phantom) plus under 1 GPU-hour (aorta). Stop if a warm prefix projects more
  than 12 GPU-hours.
