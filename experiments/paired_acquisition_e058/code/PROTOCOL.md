# E-058 / D-061 — paired real-CFD acquisition test with converged numerics

Registered 2026-10-24 after E-057 passed, by explicit user choice ("300k, two
arms"). Root: `D:/MRSim/outputs/paired_acquisition_e058`. New code and results
live only here. E048/E050/E055/E056/E057 artifacts and both frozen E053 project
copies are read-only. No 1M run; 1M stays on hold whatever this shows.

## Question and evidence class

On the matched real stenotic-phantom CFD, with simulator numerics converged, does
the E-057-supported shorter-echo acquisition (700 Hz/pixel, actual TE 2.1 ms)
reduce local throat velocity error versus the baseline protocol (350 Hz/pixel,
TE 2.74 ms) without losing coverage or worsening same-area flow? Secondary: how
much do E055's numbers move when only the numerics are converged?

Evidence class: same-source internal consistency (CFD-driven tracks versus the
same CFD), noiseless. Not independent physical validation, not an SNR study, not
scanner-exact. A 700 Hz/pixel readout carries about √2 more thermal noise per
voxel on a real scanner; that penalty is outside this noiseless experiment.

## Arms (identical phantom, map, scanner, sorter, reconstruction)

Coupled 300001-spin E055 phantom (`phantom_density_e055/inputs_v1`, `map_v1`),
never split into batches. Float64, perfect spoiling, `max_block_length` 512,
physical GPU1, frozen E046 helpers and pinned manifest, E055 phantom loader and
timing contract copied verbatim.

| Arm | Sequences | RF step | Free-precession step |
|---|---|---|---|
| `baseline_fine` | original real REF/X/Y/Z (`koma_dp150/pc_3d_scanner_matched_venc5`) | 0.5 µs | h* |
| `short_echo_fine` | new 700 Hz/pixel, TE 2.1 ms real-protocol REF/X/Y/Z | 0.5 µs | h* |
| `baseline_coarse` | existing E055 300k result, not rerun | 50 µs | Inf |

Free-precession refinement uses the E-057 `PrecessionMaxStepRule`, copied with
its hash checked against the certified E-057 file.

## S0 — sequences (CPU, no signal)

(a) Rebuild the baseline with E056's copied builder helpers (only the
Scanner.limits API adaptation) at the real 400×64×32 matrix, 500×80×80 mm FOV and
the original generator's exact arguments. Certify event equivalence to each
original `.seq`: same block count; per block equal duration, RF amplitude/phase/
delay/waveform, gradient amplitudes/timings, ADC N/dwell/delay/phase and labels
(times ≤ 1e-9 s, amplitudes ≤ 1e-9 relative). Failure stops E-058: the candidate
builder would not be certified.

(b) Build the candidate with only `BWpp=700` and `target_TE=0.0021` changed.
Require on readback: `check_timing`/`check_hw_limits` pass (40 mT/m,
180 T/m/s, 10 µT); bandwidth 700 Hz/pixel; actual TE ≈ 2.1 ms (raster); 400
samples per profile; labels identical to the baseline; RF history identical
(count, starts, centers, scales, phases within 2e-9); for every spatial shot,
differential M0 ≤ 2e-12 T·s/m and first-moment VENC error ≤ 2e-15 T·s²/m;
readout/phase k-space increments within 2e-5 cycles/m. Export candidate ADC
timing in the legacy `legacy_timing.h5` format.

## S1 — references (CPU)

Rebuild the actual-ADC averaged CFD references with the frozen E048 functions
(`timing_weights`, `build_voxel_lookup`, `build_section_lookup`). Identity gate:
with the original legacy timing the rebuild must reproduce E048 `fft_adc` and
`plane_reference` (max abs ≤ 5e-5 m/s). Then build the candidate's own-timing
references. Report the candidate-versus-baseline reference difference as a
timing sensitivity. The candidate's primary comparator is its own actual-ADC
reference; the baseline-timing reference is a disclosed sensitivity.

## S2 — numerical prefix on the real phantom (GPU1, full coupled population)

30-profile prefixes (two full 15-bin cycles after the ramp):

- R0: E055 numerics (Inf, 50 µs) must reproduce E055's saved 15-profile REF
  preflight prefix within relative L2 ≤ 1e-12 (bit-level reproduction expected).
- RF 0.5 µs with precession Inf, 16, 8, 4 µs, REF and X, baseline sequences.
  h* = coarsest step whose relative L2 to the next finer step is ≤ 1e-4 for
  both REF and X. If none up to 4 µs, stop and ask.
- Candidate REF/X prefix at h* and h*/2 must also differ by ≤ 1e-4.
- Record warm prefix timings; if the 8 full encodings extrapolate beyond 12 h of
  GPU time, stop and ask before acquisition.

## S3 — acquisitions (GPU1)

`baseline_fine` then `short_echo_fine`, REF/X/Y/Z each, atomic checkpoints, same
output format as E055 (`{enc}_result.h5` with raw ComplexF64, rawsort, image).

## S4 — analysis and preregistered decision

Reuse the frozen E050/E055 helpers (`load_arm`, `fixed_mask`, `velocities`,
`region_rows`, `partial_flow`) unchanged. Fixed 5% all-bin/all-encoding support;
10/15/20% are sensitivities only. Primary paired support: intersection of all
three arms' 5% masks with eligible geometry; secondary: `baseline_fine` ∩
`short_echo_fine`. Report own coverage beside paired errors, regional vector
RMSE and bias, same-area partial flows (upstream, throat, downstream planes) with
RMSE and cycle transport, peak/low throat profiles, p95/p99 speeds, N-normalized
REF magnitude, and k-space raw-signal identity audits.

Candidate supported on this real case only if, versus `baseline_fine` on primary
identical paired support at 5%: (1) throat paired vector RMSE reduced ≥ 10%;
(2) whole-volume paired vector RMSE not worse; (3) own whole-volume coverage loss
≤ 1 percentage point; (4) same-area throat and upstream flow RMSE not worse.
Anything else is "not supported", reported with every endpoint. The numerics-only
effect (`baseline_fine` versus `baseline_coarse`) is descriptive.

## Limits fixed in advance

Single deterministic spin sample (seed variability unmeasured); noiseless; 0.5-ms
RK4 integration of CFD saved every 10 ms; synthetic final-80-ms closure;
scanner-derived, geometry-adapted, VENC-adjusted separate REF/X/Y/Z encodings;
nearest-neighbor cycle remap; encoded complex-image phase versus CFD mean
velocity are different observables. The candidate changes echo and readout
timing together. No fitting of gain, shift, phase, mask or waveform; no TKE,
wall, pressure or clinical-accuracy claim.
