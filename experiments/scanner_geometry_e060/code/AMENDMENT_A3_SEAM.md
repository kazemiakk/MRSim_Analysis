# E-060 amendment A3 — periodic motion seam aligned to the E-060 sequence

Registered before any E-060 signal and before the seam is computed.

## Trigger

The S2 prefix failed closed with "Marker overlaps gradient" before any GPU
work. Its log is preserved as `attempt1_prefix.log` and its registration as
`attempt1_prefix_v1`.

The phantom's periodic seam, where spin coordinates wrap at the CFD cycle end,
came from E055: 0.4379376769065857 s plus k periods. E055 placed it at the
half-bin gap of its own 15-bin sequence. In the E-060 sequence the same
absolute time falls about 2.7 ms before an echo, inside a shot. That is the
E-018 seam fault, which D-024 and D-025 forbid.

## Rule

The frozen rule from `koma_dp150/motion_timing.jl` is reused:
`align_periodic_motion_seam_to_first_adc_midbin` places the seam half a
cardiac bin, RR/(2·30), before the first ADC k-space centre.

- The seam is rounded to the nearest multiple of 2^-23 s. This keeps seam
  and seam + period exactly representable in Float32, so the frozen
  `shift_periodic_motion_seam` reproduces the E055 motion period bit for bit.
  The rounding is ≤ 6e-8 s.
- Spins, map, resets and period are unchanged.
- Only the time origin of the periodic CFD cycle moves relative to the
  sequence.

## Gates for the `seam_v1` stage (`code/align_seam_e060.jl`)

1. The frozen `audit_adc_sample_windows_against_periodic_seam` must pass on
   REF, X, Y and Z with zero seam-crossing ADC windows.
   - Required margin: ≥ 0.40 × RR/30 = 13.49 ms.
   - D-025's absolute 0.030 s was 47.6 % of a 15-bin bin. At 30 bins the
     largest possible half-bin-gap margin is about 15.4 ms.
2. The seam-marker count derived from the sequence must equal **1921**. The
   frozen `insert_verified_zero_markers!` must accept every marker on all four
   encodings.
   - With the seam in the half-bin gap, 1920 heartbeats span 1921 seams: one
     before the first bin and one after the last.
   - Attempt 1 expected 1920 and stopped on that check before computing
     anything else. It is preserved as `attempt1_align_seam.log`,
     `attempt1_align_seam_e060.jl` and `attempt1_AMENDMENT_A3_SEAM.md`.
3. New check: for every marker, the next RF starts before the next ADC. The
   marker's perfect-spoiling reset of transverse magnetization therefore cannot
   reach an acquired sample.

## Consequences

- `run_e060.jl` takes the seam from `seam_v1`. The exact bytes of the version
  that failed are preserved as `attempt1_run_e060.jl`.
- The CFD reference must use the same seam:
  - `build_references_e060_v2.py` writes `references_v2`.
  - `references_v1`, built with E055's seam, is preserved as superseded.
- `analyze_e060.py` reads `references_v2`.
- R0, the rotation-invariance check, still uses E055's own seam with E055's
  sequences, which are unchanged.
