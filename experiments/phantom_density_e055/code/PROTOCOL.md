# E-055: corrected 300k stenotic-phantom confirmation

Authorized staged plan, 2026-10-24. Prerequisite E-054 must pass all 24
predefined model/mode gates and its independent saved-signal/path validation.
The current E-054 report SHA256 is
`c88a592791b0536f190c3507f18b0b54fadeb00866fcc19cfe5e0f7c4a8eb5f0`.
Register E-055/D-058 in the external D: journal while E-053's original
every-file migration snapshot remains frozen. Queue canonical amendments.

Evidence class: same-source internal consistency and a controlled nested
particle-count confirmation, not independent physical validation or convergence.

Use the preserved 1M MRI-coordinate trajectories, whose SHA256 is
`f904e81e3626baf26007f3a20858ef7d9e15816b96978f64b4d0553fd505978c`.
Reproduce the original 100k identity sample with seed 20260831 and assert its
index digest. Add 200k uniformly selected unused identities with seed 20261024.
Sort the final 300k without replacement. Do not retrack or substitute CFD.
Verify the original 100k paths/flags/properties within the new source export.
Prepend t=0, preserve all stored Float32 trajectory values, and keep the same
literal Float32 motion endpoints subsequently promoted to Float64.

Preserve the physical sequence and all acquired bins/labels: 3T, VENC 5m/s,
400x64x32, 15 phases, original REF/X/Y/Z and RF/ADC clocks. Keep perfect
spoiling, sorted event lookup, Float64 simulation, Inf gradient step and 50us
RF step. Store ComplexF64 raw signal and the same primary ComplexF32 sort/FFT.
Use one coupled 300001-spin phantom (including its zero-density sentinel),
never independent mapped spin batches. Rebuild the SAME upstream nearest-
neighbor cycle-map algorithm for the new population and restore genuine final
tracking reset flags. Report the map's approximation, multiplicity and distances.

Equal per-spin rho=1 is retained exactly from E050. Raw signal scales with N;
any amplitude comparison must divide by physical particle count (a declared
constant normalization, not a fitted gain). Phase and relative support gates
are invariant to that global scale. Do not claim an absolute SNR improvement
from an unnormalized amplitude increase. No preferential throat oversampling.

Before full acquisition: verify frozen source, helpers, manifest, exact timing,
E054 validation, free D: storage and an idle GPU1. Run 3/15-profile REF prefixes
with the COMPLETE coupled 300k population. Fail closed on nonfinite output,
precision/count mismatch, allocation, source or timing failure. Keep all failed
attempts and checkpoint completed encodings atomically.

Primary reference remains E048 centered-FFT actual-ADC averaged CFD. Compare
against the completed E050 Float64 100k acquisition, NOT the old Float32 arm.
Use an unchanged 5% all-bin/all-encoding fixed signal gate; also report inherited
10/15/20% sensitivities. Report both arms' own retained volume and errors on
identical paired support, including whole, throat and other original regions.
Compute true-plane partial flows on identical paired areas; do not compare
different own-area flow amounts as a density improvement. Report paired
velocity RMSE/bias, regional p95/p99 speeds and peak/low-flow profile errors.
Audit sort/FFT directly from the raw signals and do not fit a shift, gain,
smoothing kernel, time offset, mask or velocity correction.

1M DECISION GATE: no automatic launch from image appearance or coverage alone.
Require improved own coverage AND improved paired throat/local velocity error,
with no worsened relevant same-area flow comparison. Repeated-seed variability
has NOT yet been established by the deterministic E054 controls; if a single
300k comparison cannot establish robust improvement, explicitly hold 1M for
additional seed confirmation. An error plateau or failed numerical gate stops
scale-up. Do not relabel incomplete coverage as full-lumen flow accuracy.

Limitations remain: 10ms CFD with 0.5ms trajectory integration, synthetic final
80ms closure, separate scanner-derived/geometry-adapted/VENC-adjusted encodings,
nearest-neighbor ensemble-state remapping, and encoded-image versus CFD-mean
observable differences. No TKE, wall, pressure or clinical-accuracy promotion.
