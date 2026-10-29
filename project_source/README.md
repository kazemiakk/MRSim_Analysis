# MRSimTracks

Generate CFD-derived particle trajectories for MR flow simulation.

MRSimTracks performs Lagrangian particle tracking in time-resolved (pulsatile)
CFD meshes. It operates on mesh velocity fields, not MR image data. It seeds
particles in a tetrahedral flow domain, advects them through a time-periodic
velocity field (RK4), and recycles out-of-bounds particles back to the inflow
boundaries with optional **backflow-aware** reseeding.

Project governance, current production status, experiment/decision logs, and
the ordered scientific backlog are maintained in
[`docs/governance/Project.md`](docs/governance/Project.md). Repository agents
must follow [`AGENTS.md`](AGENTS.md); the single shared MRSim project skill is
under [`.agents/skills/develop-mrsim/`](.agents/skills/develop-mrsim/).

<p align="center">
  <img src="docs/assets/ubend_density_slice.webp" alt="Greyscale center-slice particle animation through the U-bend" width="30%">
  <img src="docs/assets/ubend_tracks.png" alt="Selected speed-colored particle trajectories through the U-bend" width="30%">
  <img src="docs/assets/ubend_particles.webp" alt="Speed-colored particles tracked through the full pulsatile U-bend example" width="30%">
</p>

## Install

MRSimTracks is published on PyPI:

```bash
uv add "mrsimtracks==0.2.1"
```

or with pip:

```bash
python -m pip install "mrsimtracks==0.2.1"
```

To install the latest source from GitHub instead:

```bash
uv add "mrsimtracks @ git+https://github.com/mcgrathcm/MRSimTracks.git"
```

For development from a clone:

```bash
uv sync          # installs the package (editable) and dependencies
```

The distribution name and Python import package are both `mrsimtracks`.
The expanded point-cloud development checkout is versioned `0.2.2.dev0`;
`0.2.1` remains the published release pin shown above.

## Quick start

```python
import numpy as np

import mrsimtracks as mt
from mrsimtracks.seeding import seed_mesh

# 1. Load a time-resolved flow field (.vtu single-file series or .pvd collection)
flow = mt.load_flow("case.pvd", active_key="Velocity")

# 2. (optional) Backflow-aware inflow reseeder from labeled cap surfaces
reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)

# 3. (optional) Near-wall no-penetration projection
wall_slip = mt.WallSlip(flow, caps=["Inlet.vtp", "Outlet.vtp"])

# 4. Seed and track
seeds = seed_mesh(flow.active_mesh, 200_000, rng=np.random.default_rng(0))
result = mt.track(
    flow,
    seeds=seeds,
    dt=0.002,
    reseeder=reseeder,
    wall_slip=wall_slip,
)

# 5. Use / save
result.positions      # (n_steps, n_particles, 3)
result.reset          # (n_steps, n_particles) reseed flags
result.times          # (n_steps,)
result.save("tracks.h5")
```

For large single-process runs, write each timestep directly to HDF5 instead of
keeping all positions in RAM:

```python
result, metrics = mt.track(
    flow,
    seeds=seeds,
    dt=0.002,
    reseeder=reseeder,
    output_path="tracks.h5",
    return_metrics=True,
)

result.is_file_backed  # True until result.positions or result.reset is loaded
metrics["particle_steps_per_s"]
```

Run `example.py` for a complete version using the reduced fixture committed in
`tests/data/`. The full-cycle example flow file is tracked with Git LFS; see
`example/README.md`.

## Visualize tracks

The visualization commands are source-checkout utilities. Run them from the
repository root after `uv sync`. After running `example.py`, open the
interactive PyVista viewer:

```powershell
uv run python visualize_tracks.py
```

The viewer overlays trajectories on the CFD vessel, marks reseed events, colors
particles by speed, and provides a timestep slider. To render a PNG without
opening a window:

```powershell
uv run python visualize_tracks.py --screenshot tracks_preview.png
```

To save a 360-degree rotating animation:

```powershell
uv run python visualize_tracks.py `
  --tracks aorta_dp2_tracks.h5 `
  --point-cloud ignore\dp2\point_coordinates.npy `
  --orbit-gif aorta_dp2_tracks_rotating.gif
```

For large full-cycle results, save a temporal animation using display
subsampling:

```powershell
uv run python visualize_tracks.py `
  --tracks aorta_dp2_full_cycle.h5 `
  --point-cloud ignore\dp2\point_coordinates.npy `
  --particle-stride 10 --time-stride 4 `
  --hide-paths --hide-reseeds `
  --time-gif aorta_dp2_full_cycle_motion.gif
```

Time animations render with fading, speed-colored particle trails, a soft
particle glow, a playback progress bar, and a subtle camera drift by default.
Useful knobs:

- `--interp N` inserts N-1 interpolated frames between stored steps, making
  motion slower and smoother; `--fps` sets playback speed.
- `--trail-frames N` controls comet-trail length (`0` restores the static
  full-path and reseed-marker overlay).
- `--reconstruct-surface` turns a `--point-cloud` geometry into a smooth
  translucent vessel wall (voxel occupancy contour, cached next to the .npy).
- `--window-size W H`, `--camera-zoom`, `--camera-drift`, and `--title` frame
  the shot; `--cmap` overrides the speed colormap.

Example wide-format render of the stenosis phantom:

```powershell
uv run python visualize_tracks.py `
  --tracks outputs\phantom_dp150_rot90cw_tracks_25.h5 `
  --point-cloud outputs\phantom_dp150_rot90cw_point_coordinates.npy `
  --reconstruct-surface --dark-output --view xz `
  --window-size 1920 460 --camera-zoom 3.5 --hide-legend `
  --title "MRSimTracks | phantom dp150" `
  --time-webp outputs\phantom_dp150_cinematic.webp
```

## Point-cloud tracking without connectivity

`PointCloudFlow` provides an approximate fallback when a CFD export contains
coordinates and time-resolved velocities but no volume-cell connectivity. It
uses inverse-distance k-nearest-neighbour interpolation and a support-distance
mask. This API and the aorta helper are part of the `0.2.2.dev0` source
checkout; they are not included in the published `0.2.1` wheel:

```python
flow = mt.PointCloudFlow.from_npy(
    "coordinates.npy",
    "velocities.npy",       # (n_points, 3, n_frames)
    frame_dt=0.01,
    k=8,
)
seeds = flow.seed_points(2_000, rng=np.random.default_rng(0))
result = mt.track(flow, seeds=seeds, inlet=seeds, dt=0.002, tmax=0.04)
```

After restoring the matched aortic source arrays described in
`EXTERNAL_INPUTS_REQUIRED.md`, run the source-checkout utilities:

```powershell
uv run python run_aorta_pointcloud.py --simulation dp2
uv run python visualize_tracks.py `
  --tracks outputs\aorta_dp2_pipeline\aorta_dp2_full_cycle.h5 `
  --point-cloud ..\external_inputs\aorta_pointcloud\processed\dp2\point_coordinates.npy
```

The runner defaults to `../external_inputs/aorta_pointcloud`, resolves both
arrays below `processed/<simulation>`, validates that their node dimensions
match, and records both SHA-256 hashes in its report and HDF5 metadata. Use
`--coordinates PATH` only for an explicitly reviewed coordinate-array
override. A legacy `geometry.pkl` remains supported when the canonical
coordinate array is absent, but the runner prints a reproducibility warning.
Tracking is streamed to a sibling `.partial` file and atomically published only
after HDF5 verification succeeds.

The uniform-seeding audit uses the same canonical input layout:

```powershell
uv run python scripts\audit_aorta_uniform_seeding.py --simulation dp2
```

This mode does not reproduce walls or open boundaries as exactly as a connected
tetrahedral mesh. The aorta runner also recycles escaped particles to its
initial seed pool because labeled inlet caps are unavailable. Verify
`--frame-dt` against the CFD setup before interpreting physical timing.

## Large runs (multiple processes)

Each worker reloads the field, so memory scales with `n_workers`:

```python
result = mt.track_parallel(
    "case.pvd", seeds=seeds, dt=0.002,
    caps=["Inlet.vtp", "Outlet.vtp"], active_key="Velocity",
    n_workers=3, subsamp=1, wall_slip=True,
)
```

## Key functions

| Function | Purpose |
|---|---|
| `load_flow(path, active_key=...)` | Load `.vtu` (one geometry, many time fields) or `.pvd` (series); auto-selects the memory-efficient reader. `subsamp=N` keeps every Nth frame. |
| `track(flow, seeds=..., dt=..., reseeder=...)` | Single-process tracking → `TrackingResult`. Use `output_path=...` for streamed HDF5 output and `return_metrics=True` for timing metrics. |
| `track_parallel(path, ..., caps=..., n_workers=...)` | Multi-process tracking → `TrackingResult`. |
| `BoundaryReseeder(caps, flow, dt=...)` | Flux-weighted, time-resolved inflow reseeder. `caps` = cap surface path(s) or a surface with a `region_id` cell array. |
| `WallSlip(flow, caps=..., band_frac=0.02)` | Optional near-wall no-penetration projection for `track(..., wall_slip=...)`. Excludes cap faces so inlet/outlet flux remains open. |

## Reseeding notes

- The reseeder weights every cap face by `max(-v·n, 0)·area` at the nearest flow
  frame, so particles re-enter only through currently-inflow faces — handling
  backflow and partial inflow/outflow on a single cap.
- `BoundaryReseeder(..., dt=dt)` spreads new particles over a thin inflow
  *volume* (depth `~v_n·dt`) so successive reseeds overlap, keeping spatial
  density smooth (important for MR-style uniform-density use). Omit `dt` for
  plane seeding.
- `flux_waveform()` returns per-cap net flux over the cycle — a conservation /
  validation diagnostic (`Σ caps ≈ 0` for a well-resolved incompressible field).

## Wall-slip notes

`WallSlip` removes only the into-wall velocity component for particles within a
thin wall band. Use it when interpolated near-wall velocities deposit particles
against no-slip walls. Pass the same cap surfaces used for reseeding so open
boundaries are excluded from the wall set:

```python
wall_slip = mt.WallSlip(flow, caps=["Inlet.vtp", "Outlet.vtp"], band_frac=0.02)
result = mt.track(
    flow,
    seeds=seeds,
    dt=0.002,
    reseeder=reseeder,
    wall_slip=wall_slip,
)
```

For `track_parallel`, set `wall_slip=True`; each worker builds its own
projection from the worker-local flow and `caps`.
