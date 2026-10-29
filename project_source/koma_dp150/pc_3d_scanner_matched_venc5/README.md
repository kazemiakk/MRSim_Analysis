# Scanner-matched dp150 full simulation, VENC 5.0 m/s

This isolated protocol uses the scanner-matched dp150 acquisition settings and
sets VENC to 5.0 m/s. It does not overwrite `pc_3d_scanner_matched` (VENC 4.5
m/s) or the earlier two-bin simulation.

The available scanner metadata reports VENC 3.0 m/s and a 64x64 acquisition
matrix. Here, voxel size, 15 cardiac labels, TR/TE, flip angle, and bandwidth
are metadata-derived,
while VENC is intentionally adjusted to 5.0 m/s. The 400x64x32 matrix and
500x80x80 mm FOV are geometry-adapted for the dp150 model, so this is not an
exact reproduction of the scanner matrix or FOV.

The production wrapper runs all four encodings (`REF`, `X`, `Y`, and `Z`) with
the existing 1,000,000-particle motion phantom. It uses 20,000-spin resumable
batches, 3 T, and maps the stored periodic motion onto the 1012 ms sequence
R-R interval in memory.

E-018 rejects the preserved run's PHS8 because its ADC windows cross the
artificial periodic coordinate seam. The legacy wrappers remain available only
to reproduce/audit historical artifacts. New scientific runs must pass
`-SeamSafeMotion`, which uses `simulate_100k_seam_safe.jl` or
`simulate_1m_seam_safe.jl`, relocates the seam into the half-bin non-ADC gap,
and proves that the controlled reset is applied at a zero-physical-effect
simulation-part boundary. This prevents acquisition across the jump; it does
not claim physical particle-identity continuity.

Generate and audit the sequences:

```powershell
. .\koma_dp150\pc_3d_scanner_matched_venc5\launcher_common.ps1
$julia = (Resolve-Dp150JuliaExecutable).Path
$project = (Resolve-Path .\koma_dp150).Path
& $julia --startup-file=no "--project=$project" `
  .\koma_dp150\pc_3d_scanner_matched_venc5\generate.jl
& $julia --startup-file=no "--project=$project" `
  .\koma_dp150\pc_3d_scanner_matched_venc5\audit_protocol.jl
```

The audit is written beside the sequences with repository-relative filenames,
byte counts, and SHA-256 hashes. Launchers recompute those hashes instead of
trusting a stale `PASS` line.

Run the full simulation:

```powershell
# Read-only audit of the preserved, pre-provenance production run
& .\koma_dp150\pc_3d_scanner_matched_venc5\launch_1m_dual_when_gpus_idle.ps1 `
  -PreflightOnly -AllowLegacyCompletedRun

# A future provenance-bound run must use a fresh run name
& .\koma_dp150\pc_3d_scanner_matched_venc5\launch_1m_dual_when_gpus_idle.ps1 `
  -RunName motion_1m_scanner_matched_venc5_provenance_v1

# Corrected decision pilot; required before corrected 1M production
& .\koma_dp150\pc_3d_scanner_matched_venc5\launch_100k_dual_when_gpus_idle.ps1 `
  -RunName motion_100k_scanner_matched_venc5_seam_safe_v3 `
  -SeamSafeMotion

# Corrected production; launch only after the preregistered 100k gates pass
& .\koma_dp150\pc_3d_scanner_matched_venc5\launch_1m_dual_when_gpus_idle.ps1 `
  -RunName motion_1m_scanner_matched_venc5_seam_safe_v1 `
  -SeamSafeMotion
```

Results are written to
`outputs/dp150_mri_pipeline/koma/simulation_motion_1m_scanner_matched_venc5`.
The preserved production checkpoints predate embedded hashes and are therefore
audit-only. Newly generated batches and encoding results carry strict phantom,
sequence, code, environment, and run-parameter provenance and fail closed on
any reuse mismatch.

When another CUDA workload is active, use `launch_when_gpu_idle.ps1`. It waits
for three stable idle probes and performs an immediate pre-launch recheck. The
queue state is recorded in `queue_status.json` in the run directory.
