# E-064 / D-067 — aorta acquired in the candy-cane orientation

Registered 2026-10-26 by user decision ("Aligned re-simulation") before any signal.
Root: `D:/MRSim/outputs/aorta_aligned_e064`.

## Why

The E-062 aorta was acquired with the scan volume aligned to the CFD axes. The aorta lies obliquely in that volume,
so a candy-cane reformat (E-063 `candycane_v2`) cuts across the volume's corners. Real candy-cane 4D-flow scans plan
the volume along the aorta.

## Design

**Frame.** Taken from E-063 `candycane_v2`:

- e2 (vertical) = descending-aorta axis, pointing toward the arch.
- e1 (horizontal) = in-plane direction, with the ascending aorta on the left.
- n = e1 × e2 (normal), sign chosen so that det[e2, e1, n] = +1 (a proper rotation, no mirror).
- Centre c = midpoint of the CFD node extents in that frame. In the E-062 MRI frame, c = (6.67, −15.81, −0.96) mm.

**New scanner axes.**

- Readout = e2: FOV 288 mm, 72 samples.
- Phase = e1: FOV 160 mm, 40 lines.
- Partition / slab = n: FOV 96 mm, 24 partitions.
- Voxels are 4 × 4 × 4 mm (E-062: 3.89 × 4 × 4 mm).
- Aorta node extents are 263 × 129 × 52 mm, leaving at least 12 mm half-margin on each axis.

**Phantom.** The E-062 1M-spin FlowPath phantom (`aorta_motion_1m_8bin.phantom`, SHA d2324133…) is rigidly rotated:

- base' = R(r0 − c); displacement paths d' = R d.
- Resets, time curve and period (0.72616 s) are unchanged.
- Gates: proper rotation (det +1); spin count and period unchanged; the inverse rotation reproduces the original
  positions (≤ 1e-6 m, Float32); every position stays inside the new FOV with at least 4 mm half-margin.

**Sequences.** The E-062 builder (copied verbatim) is used with the new FOV and matrix. Everything else is unchanged:
20 bins, 11 lines per bin per beat, TR 3.30 ms, 0.16 ms beat pad, RR 0.72616 s, VENC 2.5 m/s, flip 10°.

- **G0a/G0b:** the regression gates rebuild the production 8-bin files byte-identically.
- **G0c:** 19,200 unique labels per encoding; DEF RR = 0.72616; TR = 3.3 ms.
- **G0d:** Python sequence_timing; heartbeat periodicity < 1e-9 s; REF/X/Y/Z timing weights identical.

**Simulation.** The frozen koma_aorta `simulate_pc.jl` (read-only), with the E-062 numerics. Two parts (REF+X on
GPU0, Y+Z on GPU1), then a provenance-checked combine. Simulator QC must pass.

**Stationary tissue.** As in E-063, but filling the new FOV with a 2 mm margin and excluded within 2 mm of any rotated
CFD node.

- T1 0.9 s, T2 0.045 s, PD 0.8 × blood; 2 × 2 × 2 spins per voxel.
- Simulated with the same simulator and added by exact superposition.
- Gate: stationary encodings agree with REF (relative L2 ≤ 1e-2 and phase p99 ≤ 0.01 rad, f32 per E-063 A1).

**Display.**

- Candy-cane: magnitude-weighted 20 mm slab centred on the plane through the descending axis and the ascending
  centroid.
- Axial slice through the ascending and descending aorta.
- Display noise SNR 12; flow phase shown as for VENC 0.3 m/s (labelled); Tukey k-space filter.

## Evidence class

Presentation and realism. No registered metric changes. The CFD reference is not re-sampled in this frame.

**Cost estimate:** about 1 h on two GPUs for the flow runs, plus about 10 min for the tissue runs. Stop if a gate
fails or the projected cost exceeds 12 GPU-hours.
