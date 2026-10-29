# KomaMRI working copy

Working tree for the KomaMRI 4D-flow simulation. Charles's delivered bundle is kept
pristine in `../from charles/` and is the reference; everything here is a copy with
local changes. Diff against it to see exactly what was altered.

## Working assumptions (set by the user 2026-08-18, pending confirmation with Charles)

- **(a) Uniform seeding.** The static phantom is a uniform-density lumen cloud, not a
  flattened particle trajectory. See "Why uniform seeding" below.
- **(b) `dt = 0.01 ms`** for motion simulations (equal to the gradient raster). Not yet
  applied: `simulate_ref.jl` still uses the bundle's `Delta_t = Inf`, which is correct
  and much faster for a static phantom. Change it when Phase 3 starts.
- **(c) Venc 0.3 m/s** instead of 2.5. See `seq/pc_3d_venc03/generate.jl`.

## Changes from the delivered bundle

| File | Change |
|---|---|
| `simulate_ref.jl` | `using Metal` to `using CUDA`; phantom/sequence/result paths resolved and overridable via `KOMA_PHANTOM` / `KOMA_SEQ` / `KOMA_RESULT`; results named per phantom+encoding so REF/X/Y/Z cannot collide |
| `motion_phantom.jl` | `RR_INTERVAL` now **read from the sequence's `DEF["RR"]`** instead of hard-coded; input path points at a one-cycle file |
| `static_phantom.jl` | unchanged (kept for reference/comparison) |
| `uniform_static_phantom.jl` | **new** - builds the phantom from a uniform cloud, same centering / axis-order / assertions as `static_phantom.jl` |
| `seq/pc_3d_venc03/generate.jl` | **new** - Venc 0.3 variant, writes to its own directory; logs realized TR/RR/lines-per-bin |
| `scripts/*.py` | **new** - see below |
| `Project.toml`, `Manifest.toml` | unchanged; CUDA is added at setup time so the pinned manifest resolves properly |

## Why RR_INTERVAL must not be hard-coded

`build_pc` rounds each cardiac bin to a whole number of TRs, so the realized RR moves
whenever TR moves - and lowering Venc lengthens the bipolar, which lengthens TR.
Derived from `_pc_bipolar_timing` in `seq/PC.jl`:

| | Venc 2.5 | Venc 0.3 |
|---|---|---|
| bipolar module | 0.960 ms | 2.240 ms |
| TR | 3.590 ms | ~4.87 ms *(predicted)* |
| lines per bin | 100 | ~74 *(predicted)* |
| realized RR | 0.718 s | ~0.7208 s *(predicted)* |
| scan per encoding | 29.474 s | ~40.4 s *(predicted)* |
| peak phase, aorta dp2 (0.241 m/s peak) | 17.4 deg | 144.6 deg |

The Venc-0.3 numbers are hand-derived from the timing formulas, not measured.
`generate.jl` logs the realized values - check them before trusting anything downstream.

## Why uniform seeding

`static_phantom.jl` flattens every trajectory timestep into one static cloud, so spin
density ends up proportional to residence time, i.e. ~1/speed. Measured on
`aorta_dp2_full_cycle.h5` over 2 mm voxels:

| | trajectory union (3.6M spins) | uniform cloud (2M spins) |
|---|---|---|
| median spins/voxel | 30 | 70 |
| max spins/voxel | 49,954 | 119 |
| max / median | **1665x** | **1.7x** |
| top voxel's share of all spins | 1.39% | 0.006% |

The 1665x range is the saturated blob and the streaks in
`../from charles/ref_static_bin2_projection_slices.png`. Raising the particle count
alone does not fix it - the ratio is structural. The residual 1.7x is Poisson.

## Setup

Julia 1.12 (not yet installed on this machine). From this directory:

```julia
import Pkg
Pkg.activate(".")
Pkg.rm("Metal")      # loading two functional GPU backends silently falls back to CPU
Pkg.add("CUDA")
Pkg.instantiate()
```

## Run order

```bash
# Phase 0 -- data prep (Python, no GPU)
python scripts/fix_track_attrs.py ../aorta_dp2_full_cycle.h5 ../phantom_dp150_tracks_10s.h5
python scripts/slice_one_cycle.py ../phantom_dp150_tracks_10s.h5 \
    --cycle-seconds 1.0 --out ../phantom_dp150_one_cycle.h5 --float32
python scripts/seed_uniform_lumen.py \
    --case "../../PI-GNN(SR)/data/aortic arch/processed/dp2" \
    --n-points 2000000 --out ../aorta_dp2_uniform_2M.h5

# Phase 0 -- regenerate the sequence at Venc 0.3
julia --project=. seq/pc_3d_venc03/generate.jl

# Phase 1 -- static litmus test
julia --project=. uniform_static_phantom.jl
KOMA_SEQ=seq/pc_3d_venc03/ref.seq julia --project=. simulate_ref.jl
```

## Scripts

- `scripts/fix_track_attrs.py` - adds the `n_steps` / `n_particles` attributes the Julia
  phantom builders require. `TrackResult.save()` writes only `dt`
  (`mrsimtracks/core.py:98-100`); the streaming writer used by
  `track(..., output_path=)` writes all three (`core.py:155-157`).
- `scripts/slice_one_cycle.py` - extracts exactly one cardiac cycle. `motion_phantom.jl`
  maps the whole `position` array onto one RR, so a 10 s file replays ~10x too fast
  with no error.
- `scripts/seed_uniform_lumen.py` - uniform-density volumetric seeding by weighting each
  interior node by its local spacing cubed (approximately its Voronoi volume), which
  cancels the mesh's near-wall refinement bias. Wall nodes come from `wall_shear.npy`
  being nonzero, which agrees with the zero-speed node set to 100.00% on dp150.

## Open items

- Julia is not installed; nothing Julia-side has been executed or tested.
- The `Periodic` wrap: at each RR boundary every particle returns to its t=0 position.
  Measured wrap displacement on aorta dp2 is 13.5 mm mean / 269 mm max, and **no reset
  flags are set on the last step**. Whether KomaMRI needs those resets to avoid carrying
  phase across the teleport is unverified - ask Charles.
- `dt = 0.01 ms` over a ~40 s scan is ~4.0M time steps per encoding. Project the real
  cost from the throughput measured in the Phase 1 static run before committing to a
  motion campaign.
- dp150 does not fit this FOV (extent 466 x 25.4 x 25.4 mm vs a 280 mm readout) and
  aliases at any Venc considered so far (peak 4.206 m/s). It needs its own sequence
  geometry - later.
