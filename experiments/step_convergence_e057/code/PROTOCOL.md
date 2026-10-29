# E-057 / D-060 — free-precession step-convergence control for the failed E056 gate

Registered 2026-10-24 before any E-057 Koma signal. Root:
`D:/MRSim/outputs/step_convergence_e057`. New source and results live only here.
E056 sequences, inputs, raw signals, oracle arrays, reports and failed gate stay
byte-for-byte unchanged; both frozen E053 project copies are not edited.
No CFD-driven run, no 1M run, no physical-validation claim.

## Question and evidence class

E056 failed its preregistered 1% complex-signal gate in 8 uniform-motion
700 Hz/pixel encodings (worst 1.389413%). A saved-data diagnostic implicated
Koma's trapezoidal free-precession quadrature of position-dependent `G(t)·x(t)`.
Question: when only Koma's free-precession step is refined, does the Koma signal
converge to the UNCHANGED exact independent oracle and pass the UNCHANGED gates?

Evidence class: numerical verification of a simulator against an independent
numerical implementation that shares the exported physical waveforms. This is
not independent physical validation and not a stenosis-accuracy result.

## What the pinned simulator actually does (read from source, D-060 context)

- KomaMRIBase 0.15.2 `MaxStepSizeRule(Δt, Δt_rf)` refines only inside each
  gradient waveform's own time span (and Δt_rf inside each RF span). Gaps with
  no gradient are not refined, so a global 518.6-s fine grid is never built.
- Motion adds only period-boundary key times, not the 4097 path knots.
- GPU `PrecessionKernel.jl` accumulates `ϕ += (Bz_prev+Bz_next)·(-πγ)·Δt` with
  `Bz = G·x(t)` at the nodes (trapezoid). T2/T1 decay are exact per part.
- `MigrationPerfectSpoiling` integration nodes are (0,1): no interior nodes.
- With the stock rule, a finite Δt ALSO adds nodes to the slice-select gradient
  inside the RF span, changing the RF excitation grid. That would vary RF and
  precession numerics together.

## D-060 design decision

Use a new isolated sampling rule, `PrecessionMaxStepRule(Δt_prec, Δt_rf)`
(`code/precession_step_rule.jl`), passed through Koma's public
`sim_params["sampling_rule"]` extension point. It returns exactly the stock
`MaxStepSizeRule(Inf, Δt_rf)` points plus stock `MaxStepSizeRule(Δt_prec, Inf)`
gradient points that lie strictly outside the open RF span. Preserved samples
stay `(:gradients, :rf_center)`, as in E056. Installed upstream is not modified.

Fixed across every arm: sequences `sequences_v1/bw700_te274/{ref,x}.seq` (and
static), inputs `inputs_v1/controls.h5`, 1344 spins, continuous 4097-knot path,
Float64, Δt_rf = 0.5 µs, `max_block_length` 512, perfect spoiling, physical
GPU1, `direct_sort.jl` reconstruction. Only `Δt_prec` changes.

## Stage 0 — grid certification and resource preflight (CPU, no signal)

For REF and X with the uniform_osc phantom motion:

- C0a: custom rule at Δt_prec = Inf reproduces the primary E056 discretization
  exactly (every DiscreteSequence field, element-wise equality).
- C0b: for each finite Δt_prec: all primary nodes retained exactly; every
  excitation-interval node identical to primary; ADC count 614400 and ADC times
  unchanged; every non-excitation interval with any nonzero gradient endpoint
  has length ≤ Δt_prec·(1+1e-9) + 1e-12 s.
- C0c (reported, not a convergence gate): export the primary node grid and
  compare it with the E056 diagnostic waveform/ADC knot grid in each post-RF
  window; then compute an independent trapezoid on Koma's actual exported grid.
- Resource rule, fixed now: ladder Δt_prec = 16, 8, 4, 2, 1 µs. Skip any step
  whose node count exceeds 20× the primary count or whose part-memory estimate
  exceeds 2 GB. Skipped steps are recorded, never silently replaced.

## Stage 1 — uniform_osc / bw700_te274 REF and X step ladder (GPU1)

- R0 reproducibility: custom rule at Δt_prec = Inf must reproduce the saved
  E056 raw signals with relative L2 ≤ 1e-12.
- Ladder runs at each retained Δt_prec. Compare to the saved, hash-bound
  `analysis_v1/independent_fields.h5` oracle arrays (`raw_signal_independent`,
  `image_independent`); never refit, rescale or re-phase anything.
- Static control: static_bw700_te274 REF and X at the finest retained step.
  For a static spin the integrand is piecewise linear between preserved gradient
  samples, so refinement must be an exact identity: relative L2 to the saved
  E056 static signal ≤ 1e-9, and static core axial velocity RMSE ≤ 0.001 m/s.

Acceptance at the finest retained step (all unchanged from E056 or stricter):

1. raw relative L2 ≤ 1% and image relative L2 ≤ 1% versus the unchanged oracle,
   for both REF and X;
2. late-clock: raw relative L2 over the last 10% of profiles in acquisition
   order ≤ 1%, for REF and X;
3. Koma-versus-oracle axial velocity RMSE ≤ 0.01 m/s on the saved E056
   uniform_osc/bw700_te274 paired core (X/REF phase difference, VENC 5 m/s);
4. convergence: oracle error nonincreasing (within 1e-4 absolute) over the last
   three retained steps; successive self-differences reported with observed order;
5. static identity and static velocity gates above; R0 passes.

Failure of any item stops the workflow at Stage 1 with status FAIL; failed
outputs are preserved and no gate is relaxed after seeing results.

## Stage 2 — conditional full-suite confirmation (only if Stages 0–1 pass)

Run all 36 E056 encodings (3 models × 3 arms × 4 encodings) at the finest
retained Δt_prec with the custom rule. Analyze with a copy of the unchanged E056
analysis whose only change is the acquisition directory/report it reads; assert
its recomputed oracle equals the saved oracle arrays. Apply the unchanged E056
gates: every raw/image relative L2 ≤ 1%, moving Koma–oracle vector RMSE
≤ 0.01 m/s, static vector RMSE ≤ 0.001 m/s; mechanism promotion only with
≥10% paired axial RMSE reduction and ≤1 percentage-point own-coverage loss.
Results are E-057 results. E056's primary status stays FAIL as recorded history.

## Interpretation limits

A pass would show the E056 gate failure was free-precession quadrature in the
simulator, and that the E056 mechanism comparisons can be re-read at a converged
step. It would not change any E050/E055 real-CFD number, would not show that
the shorter-echo arm improves the real stenosis phantom, and would not justify
1M. The next step after a pass is a proposed small paired real-CFD confirmation
with its own registration. Noise/SNR, scanner exactness, turbulence and wall
metrics remain outside this control.
