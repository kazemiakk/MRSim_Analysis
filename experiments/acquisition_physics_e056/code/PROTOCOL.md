# E-056 / D-059 — acquisition-physics mechanism experiment

Registered 2026-10-24 after the user's explicit authorization to test acquisition
physics before another1M run. New root:
`D:/MRSim/outputs/acquisition_physics_e056`. No1M or full CFD accuracy promotion.
Original C:/D: migration snapshots and all completed scientific data stay frozen.

Question: can known nonuniform/accelerating motion produce a correct encoded
signal but biased instantaneous velocity images, and do faster readout/shorter
echo reduce that difference? Distinguish numerical error from image/encoding
response. Do not fix the apparent error by fitting a waveform, shift, gain,
smoothing kernel or mask to the reference.

## Controls and scope

Three compact, deterministic, equal-volume quadrature phantoms share positions,
weights, T1=1.3s and T2=.1s. Cylinder length10mm, radius4.579049785927147mm
(the source throat's minimum-radius scale), with .625/.625/1.25mm subvoxel spacing.
Static; spatially uniform oscillation; and parabolic-profile oscillation:

`x_i(t)=x0_i+(u_i/omega)*(sin(omega*(t-echo_baseline)+phi)-sin(phi))`

`y,z` stay fixed; `phi=-pi/4`; `omega=2pi/(1.012/15)`.
Uniform `u_i=2.2m/s`; parabolic `u_i=2.2*(1-(y_i^2+z_i^2)/R^2)`.
Use continuous periodic Path,4097 Float64 samples/cycle, no resets/remapping.
The field is volume preserving. The compact object remains within the padded
FOV. This deliberately removes inlet/recycling ambiguity. It is synthetic
oscillatory motion, NOT a physical stenosis CFD solution or turbulent-flow test.

The baseline echo is the first actual acquired central-readout echo, not the
nominal start of a cardiac bin. Motion repeats at the physical RF cadence;
all15 phases are stroboscopic samples in this mechanism control, not15 distinct
physiological phases. The10 initial RF ramp shots and actual subsequent RF
timing/phase/flip history must be exported and included in the independent model.

## Paired acquisition arms

True3D matrix80x16x32,FOV100x20x80mm, unchanged1.25x1.25x2.5mm voxels and80mm
RF slab. Separate REF/X/Y/Z,VENC5m/s,3T,8deg nominal flip,TBP1/window0,
10 ramp shots, perfect transverse spoiling and Float64. Keep15 phases,RR1.012s,
nominalTR46.88ms. Exact-RR scheduling gives
approximately67.4667ms acquired RF spacing; do NOT mistake nominalTR for that
spacing or rebuild a one-bin scan with different RF saturation history.

1. baseline:350Hz/pixel,TE2.74ms;
2. fast_readout:700Hz/pixel,TE2.74ms;
3. fast_short_echo:700Hz/pixel,TE2.1ms. The initially requested2.0ms arm was
   rejected before signals: the fixed hardware/VENC/prephaser combination
   requires at least2.034209ms. This explicitly registered2.1ms replacement is
   tested as its own arm; it is not mislabeled as2.0ms.

Use copied builder helpers with only explicit Scanner.limits API adaptation.
Rebuild readout/prephaser/rewinder and enforce Gmax40mT/m,Smax180T/m/s and
B1max10uT. Preserve VENC differential first moment and zero differential area.
Record second moments, actual TE, RF clocks/phases and ADC grids. Refuse
unachievable requestedTE; never silently substitute a different arm. The
third arm changes echo/readout timing together and is not a pure TE-only test.
No claim of exact scanner replication from this smaller research-control FOV.

## Numerical and interpretation gates

Pre-signal numerical amendment: saved E054 production-step signal L2 errors are
about7%, although its phase-oriented production gates passed. Therefore compare
the existing50us RF step against0.5us RF stepping in the repeated-RF prefix.
Require the independent1% signal gate for the fine-RF prefix and use0.5us RF
stepping in all primary physics arms. Keep Inf gradient stepping and measure
the remaining discrepancy against the independent exact phase integration.
The coarse-RF prefix is a declared numerical negative/sensitivity control;
its failure cannot be hidden as a physical acquisition effect. This does not
change or retrospectively correct any E050/E055 simulation or causal attribution.

Before full simulation: bind source/protocol/input/helper/manifest hashes;
check every sequence's hardware/timing, actual RF identities/cadence, matrix,
encoding moments and finite paths; confirm a reviewed idle GPU1 and sufficient
storage. Small complete-population prefix precedes full controlled scans.
Never split coupled spins or launch another process over an active GPU job.

Pre-signal audit correction: a hard-pulse-at-RF-label M0 check exposed a common
0.208333cycles/m slice offset. The first interpretation as half-raster timing
was not supported by inspection and is rejected. The actual RF amplitude
centroid and stored center descriptor are retained separately. A finite
slice-selective RF pulse contributes spatial phase that cannot be certified
using one instantaneous RF time. Require absolute readout/phase grid coordinates
and all Cartesian increments to2e-5cycles/m; retain the common slice offset,
bound it by half the actual RF gradient footprint, and test its signal-level
effect with the independent finite-RF Bloch predictor. Differential M0/M1 and
RF identity gates remain strict. No reconstructed data are shifted or fitted.

Independently predict repeated-RF signals using finite-RF RK4 plus longitudinal
T1 recurrence and free-precession integration of the actual gradients against
the declared paths. Do not normalize by measured REF: this is not only a
REF-conditional phase prediction. Export raw ComplexF64 signals and primary
ComplexF32 sort/FFT and independently reproduce the latter. Verify a fine-step
single-shot or prefix reference before interpreting production-step differences.
Production signal relativeL2 target<=1%, retained-support velocity difference
to independent prediction<=.01m/s; static vectorRMSE<=.001m/s. Failed numerical
gates prevent any physical interpretation or promotion; report failure as such.

Compare each arm against the analytic velocity and geometry at its OWN actual
echo time, preserving physical clock and fixed trajectory across arms. Compare
paired geometric/signal support; report own geometry-weighted5% coverage next
to paired velocityRMSE, peak-profile errors and partial-section flow bias.
Use additional10/15/20% gates only as disclosed sensitivities, not optimization.
No data filling or full-section claim from partial support. Positive mechanism
evidence requires>=10% paired axialRMSE reduction without>1percentage-point own
coverage loss; require the sign/size of partial-flow change to be reported too.
This exploratory improvement threshold is not a clinical accuracy standard.

An independent forward predictor that matches biased Koma images identifies
measurement response; it does not make the original CFD error disappear.
A favorable synthetic result supports a later small CFD-driven confirmation,
not automatic1M. Record negative/mixed results and resource/runtime limits.

## Preservation and governance

Atomically publish new stages and refuse output collisions. Source code stays
outside both frozen project copies. Queue E056/D059 and conclusions in the
external migration journal pending canonical integration after E053 retirement.
Keep every failed attempt and all historical E048–E055 artifacts. No external
publication, email, unrelated CFD archive access or production overwrite.
