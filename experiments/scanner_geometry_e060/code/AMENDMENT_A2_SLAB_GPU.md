# E-060 amendment A2: slab pulse and GPUs (registered before any E-060 signal)

**Trigger.** The S0b fold-in gate failed for the declared TBP 4 / Hanning
a = 0.46 slab (`slab_profile_v1`, `FAIL_E060_SLAB_FOLD_IN_GATE`):

| Distance from slab centre | Excitation (% of centre) | Reference |
|---|---:|---|
| ±37.5 mm (partition-FOV edge) | 24.6 | gate limit: 2 |
| ±45 mm | 7.9 | — |
| ±25 mm, inside the slab | 67 | — |

A no-signal probe (`probe_slab_tbp_e060.jl`, `probe_slab_profile_e060.py`) used
the same 60 mm slab, the same independent RK4 profile and the Skyra limits:

| TBP | Minimum TE (ms) | Excitation beyond ±37.5 mm (%) | Excitation at ±25 mm (%) |
|---:|---:|---:|---:|
| 8 | 3.2179 | 7.9 | 82 |
| 10 | 3.2729 | 3.3 | 88 |
| 12 | 3.3229 | 0.88 | 92 |
| 16 | 3.4279 | 0.29 | 98 |

## User decisions (2026-10-24)

1. **Slab pulse: TBP 12**, Hanning a = 0.46, 60 mm slab, prephaser 0.95 ms,
   Skyra limits. The minimum TE (about 3.3229 ms) is reported as the actual TE
   and is never relabelled.
   - This is +0.21 ms relative to A1.
   - It is +0.58 ms relative to the scan's 2.74 ms, a declared protocol
     difference.
   - The S0b gate is unchanged: 2 % beyond ±37.5 mm.
   - The scanner's own slab pulse is unknown.
2. **Both GPUs** (identical RTX 6000 Ada cards).
   - GPU1 runs REF and X; GPU0 runs Y and Z, in parallel.
   - GPU0 also drives the user's display and desktop applications. It is
     shared, never cleared.
   - Gate before acquisition: the same 15-profile REF prefix at h* on GPU0
     and GPU1 must agree to relative L2 ≤ 1e-9.
   - Each encoding records its physical GPU UUID.

## Outputs

`sequences_v1` and `slab_profile_v1` are preserved as history. The new outputs
are:

- `code/prepare_sequences_e060_v2.jl` → `sequences_v2`
- `code/slab_profile_v2.py` → `slab_profile_v2`

`run_e060.jl`, `build_references_e060.py` and `analyze_e060.py` had not run.
They are edited in place to point at the v2 stages, and gain the GPU split and
a separate `finalize` step. Their new hashes are journaled.

Everything else in PROTOCOL.md, A1 and S4_ANALYSIS.md is unchanged.
