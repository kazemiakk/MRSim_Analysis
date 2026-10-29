# E-060 / D-063 — Koma in the measured scanner geometry (registration)

Registered 2026-10-24 by the user's choice ("Yes, 30 phases"; CFD compared to the
scan only qualitatively for now). Root: `D:/MRSim/outputs/scanner_geometry_e060`.
New code/results only here; all earlier artifacts and both frozen E053 copies are
read-only. No 1M run. E-058 continues unchanged.

## Question and evidence class

E-059 found that the real scanner conserves throat flow (throat vs upstream mean
flow +2.8 %, peak −0.2 %) while the readout-along-vessel simulation loses ~20 %
(E055 T1 −20.6 %). Hypothesis: the difference is protocol geometry — the scanner
measures flow along the phase-encoded slab axis, where position and velocity are
encoded close in time, and resolves the vessel cross-section at 1.25 × 1.25 mm.
Does Koma, run in the scanner's geometry and protocol, meet the user's targets
(throat flow ±8 %, peak throat velocity 8 %, ≥90 % coverage) against its own CFD?

Evidence class: same-source internal consistency (Koma vs the driving CFD), plus a
qualitative, flow-normalized comparison with the independent measured scan. The
dp150 CFD carries ~60 % more flow than the scan, so no quantitative
measured-versus-simulation accuracy claim is made.

## Fixed design (from the scan's DICOM headers and Siemens protocol block)

- Vessel axis along the slab/partition (Koma z) axis; in-plane readout x and
  phase y across the vessel. Throat at the slab centre, as in the scan (throat at
  slice 12 of 24, slab from 30 mm upstream to 27.5 mm downstream).
- 64 × 64 in-plane at 1.25 mm (FOV 80 × 80 mm); 30 partitions at 2.5 mm (75 mm
  partition FOV, 25 % oversampling as scanned), reconstructing the central 24;
  slab thickness 60 mm.
- VENC per axis: 1.5 m/s (x), 1.5 m/s (y), 3.0 m/s (z, along the vessel). One
  shared REF; all encodings share the velocity-lobe timing set by the largest first
  moment; each axis's lobe amplitude gives exactly π/VENC_axis.
- 30 cardiac bins, RR 1.012 s, exact-RR scheduling, nominal TR 23.44 ms,
  TE 2.74 ms target, 350 Hz/pixel, flip 8°, 10 ramp shots, RF spoiling 117°,
  perfect transverse spoiling; Gmax 40 mT/m, Smax 180 T/m/s, B1 10 µT, as in
  all earlier protocols.
- RF slab profile is NOT in the scan's protocol data. Declared assumption:
  primary windowed sinc TBP 4 (Hanning a = 0.46); the old TBP 1 / a = 0 profile is
  evaluated as a sensitivity. Before any signal, an independent finite-RF Bloch
  profile (static spins along z) must show excitation |Mxy| ≤ 2 % of the slab
  centre beyond ±37.5 mm (the partition FOV edge, where fold-in would start).
- If TE 2.74 ms is infeasible at VENC 1.5 m/s with these hardware limits, stop
  and ask; never relabel a different TE.

## Builder extension (D-063)

A copy of the hash-checked E056/E058 `PC.jl` with two changes only: `venc` may be
a per-axis 3-tuple, and `slab_thickness` may differ from the partition FOV. With
scalar `venc` and no `slab_thickness`, the copy must reproduce the original real
baseline sequences event-for-event (regression gate). Per-axis differential M0
≤ 2e-12 T·s/m and first-moment error ≤ 2e-15 T·s²/m against −1/(2γ·VENC_axis);
labels, RF history, k-space grid and hardware/timing audits as in E-058 S0.

## Later stages (registered here, run in order)

S1 phantom: the E055 coupled 300k phantom and cycle map, rigidly rotated
(x,y,z)_new = (y, z, x − x_throat) — a proper rotation, so the map is unchanged.
S2 numerics: RF 0.5 µs; free-precession step re-verified on a prefix (self-
difference ≤ 1e-4 between h* and h*/2). S3 acquisition on GPU1, coupled phantom.
S4 CFD reference at the actual ADC times on the new grid (frozen E048 functions);
analysis with the E-058 A1 target definitions adapted to the throat slice, plus
the throat-to-upstream flow ratio that can be compared with the scan's +2.8 %.
Noiseless; single spin sample; separate REF/X/Y/Z; stroboscopic 30 bins versus
the scanner's retrospectively gated, interpolated 30 phases.
