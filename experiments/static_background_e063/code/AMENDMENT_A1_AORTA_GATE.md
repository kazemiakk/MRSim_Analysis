# E-063 amendment A1: aorta stationary-identity gate for the single-precision simulator

Recorded 2026-10-26 after the first S2 combine stopped, and by user decision ("Amend gate, continue").

## What failed

The gate required the stationary aortic images to agree across encodings, with X/Y/Z vs REF at relative L2 ≤ 1e-4.

| Encoding | Relative L2 vs REF | Phase difference (median / p99) |
|---|---|---|
| X | 1.98e-3 | 9.5e-4 / 5.3e-3 rad |
| Y | 1.97e-3 | 9.5e-4 / 5.5e-3 rad |
| Z | 1.93e-3 | 9.3e-4 / 5.4e-3 rad |

- **Magnitude ratio:** median 0.999999.
- **Equivalent velocity at VENC 2.5 m/s:** at most 4.4 mm/s (p99); under 0.1 cm/s in the ±30 cm/s display.
- **Logs and output:** `attempt1_combine.log`, `attempt1_analysis_v1.building`.

## Cause

The frozen koma_aorta simulator runs in Float32 (E-062 registered f32). The phantom runner is Float64 and passed G1
at ~1e-8 to 3e-11. The 1e-4 threshold was set without accounting for f32.

## Amended gate (aorta only)

- Relative L2 ≤ 1e-2.
- Phase-difference p99 ≤ 0.01 rad over voxels with |M| > 20 % of the maximum.

The combine adds each encoding's own stationary image; it never substitutes REF. So the figures do not rely on the
identity holding. The phantom gates are unchanged.
