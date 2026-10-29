# E-062 / D-065 — aorta cine at 20 cardiac bins (4 mm)

Registered 2026-10-25 by user choice: "20 bins, 4 mm (Recommended)", picked
from the options 20 bins at 4 mm, 3 mm or 2 mm, or no new run.
Root: `D:/MRSim/outputs/aorta_20bin_e062`.

## Question

The 8-bin aorta run (E-027 source, `multibin_8x4mm/simulation_motion_1m_8bin_4mm`)
averages each cardiac bin over 90.8 ms. Does Koma resolve the section-flow
waveform at 20 bins (36.3 ms)? And how does the 20-bin result compare with the
CFD averaged over the same acquisition windows?

## Design: the 8-bin production run, changing only the cardiac binning

**Unchanged:**

- **Phantom:** `multibin_8x4mm/aorta_motion_1m_8bin.phantom`, reused byte for
  byte (SHA256 d23241331c8ec57e0e0396ef5468e1902583e7d2095c6ece83e3c71854242720).
  - 1,000,000 spins, 361 FlowPath samples.
  - Period 0.72616 s.
- **Simulator:** `koma_aorta/simulate_pc.jl` in the frozen D copy, used
  read-only (SHA256 a9cce28c…; identical to the C copy).
  - BlochWithPerfectSpoiling, f32, Δt = Inf, max block length 512, batch size 0.
  - Outputs go only to the E-062 root.
- **Environment:** `env/Project.toml` and `env/Manifest.toml`, byte-identical
  copies of `koma_aorta/`.
  - KomaMRI 0.13.1, KomaMRIBase 0.13.4, KomaMRICore 0.12.3, KomaMRIFiles 0.11.5,
    CUDA 6.3.0, Julia 1.12.7.
  - Depot `C:/Users/lambda-3/.julia`; offline; nothing upgraded.
- **Sequence physics** (frozen `koma_aorta/PC.jl` helpers, included read-only):
  - Matrix 72×32×32; FOV 0.28×0.128×0.128 m; VENC 2.5 m/s.
  - Flip 10°; RF spoil 117°; no added spoiler; 10 ramp shots.
  - `Scanner(Gmax=40e-3, Smax=180, B1=10e-6)`; same ADC dwell.
  - Separate REF/X/Y/Z files.

**Changed (D-065):**

- **Cardiac bins:** 8 → 20.
- **Lines per bin per heartbeat:** 29 → 11.
- **Shot TR:** 3.13 → 3.30 ms, by a 170 µs delay appended after every shot
  (after the readout, so TE and all encoding moments are unchanged).
- **Heartbeat pad:** a 160 µs delay at the start of every heartbeat group.

The heartbeat is then 20 × 11 × 3.30 ms + 0.16 ms = 0.72616 s, equal to the
phantom period, so the phantom is reused unchanged. A 10 µs block raster cannot
divide 0.72616 s into 20 equal bins, which is why the remainder sits in one
per-beat pad. Profiles per encoding go from 8,192 to 20,480. Expected runtime
is about 2.6× the 8-bin run.

## Stages and gates

**S0 sequences** (`code/generate_e062.jl` → `sequences_v1/`).

- **G0a (environment):** the frozen `build_pc_encoding_scans` with the
  production parameters writes REF/X/Y/Z files byte-identical to the production
  8-bin sequences.
- **G0b (faithful copy):** `build_pc_e062` with 8 bins, 29 lines and zero pads
  writes files byte-identical to the production 8-bin sequences.
- **G0c (E-062 set):**
  - `check_timing` and `check_hw_limits` pass.
  - 20,480 unique (LIN, PAR, PHS) labels per encoding.
  - DEF RR = 0.72616 and TR = 0.0033.
- **G0d** (Python, frozen `scripts.aorta_hemodynamic_reference.sequence_timing`):
  - Every file parses, with 20 bins × 1,024 profiles.
  - For each bin, profiles one heartbeat apart are exactly one RR apart (|Δ| < 1e-9 s).
  - REF/X/Y/Z timing weights are identical.
  - Bin phase windows and seam crossings are reported.

**S1 simulation** (`code/run_e062.ps1`).

- Part A runs REF and X on GPU0; part B runs Y and Z on GPU1, concurrently.
  - Same run name, `motion_1m_20bin_4mm`.
  - Separate output roots: `part_gpu0/`, `part_gpu1/`.
  - Batch size 0; result reuse off.
- Combine: copy the four `*_result.jld2` files into `combined/`, then run all
  encodings with reuse on (provenance-checked by the simulator) to build
  `pc_reconstruction.h5`.
- Gate: simulator `status = "PASS"` and all arrays finite.

**S2 analysis** (`code/analyze_e062.py` → `analysis_v1/`).

- **Method:** an E-027-style comparison using the frozen E-027 functions.
  - CFD reference at the actual 20-bin ADC times, on the same 4 mm grid,
    lookup, `core15` mask and section ROIs.
  - Source velocity time scale 0.72/0.72616, as in E-027.
- **Regression gate:** the same code applied to the 8-bin inputs reproduces the
  E-027 `section_series` net flows to within 1e-6 relative.
- **Descriptive outputs** (no pass/fail thresholds):
  - Per-bin section flows, CFD versus Koma, at the four E-027 sections.
  - Cycle net-volume ratio, peak absolute flow ratio, and temporal correlation.
  - Pooled voxel metrics.
  - The 20-bin versus 8-bin comparison of the X+000_C1 waveform.

**S3 figures:** presentation only. The aorta figure is updated with Koma at 20
bins.

## Evidence class

Same-source internal consistency: noiseless, one 1M-spin phantom, no
independent physical validation. The particle tracking and point-cloud support
are approximate (`PASS_APPROXIMATE_POINT_CLOUD`).
