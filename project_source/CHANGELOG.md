# Changelog

All notable changes to this project will be documented here.

This project uses semantic versioning while the public API stabilizes.

## [Unreleased]

- Reject the preserved dp150 one-million-spin phase after a timing audit found
  that all 8,192 saved REF/X/Y/Z readouts were acquired across the artificial
  periodic coordinate seam, and reopen dp150 closure rather than treating the
  resulting voids as a display-mask problem.
- Add seam-safe dp150 motion timing that rescales the cycle to the sequence R-R
  interval and places the artificial seam in a non-ADC half-bin gap, with a
  preflight that audits every closed inter-sample interval and requires zero
  seam crossings.
- Add a preregistered cyclic magnitude/phase jump gate whose thresholds and
  implementation hash are registered before the candidate data exists, so a
  corrected run cannot be accepted by tuning a threshold after the fact.
- Add a preregistered support-constrained complex-domain phase reconstruction
  that preserves direct phase where confidence allows and reports every
  regularized voxel as a spatial estimate rather than new signal.
- Freeze a reviewed, versioned evidence release bundle so a source-only
  reviewer can run the manuscript verifier without the private workspace or the
  identifying scanner source. `paper/verify_manuscript_claims.py` gained
  `--evidence-root` and `--list-inputs`, and the builder scans every copied file
  for identifying fields and redacts local absolute paths, failing closed if a
  redaction would corrupt a structured report or if any absolute path survives.

- Add approximate point-cloud tracking for CFD exports without volume-cell
  connectivity, plus reproducible MRI pipeline and visualization tooling.
- Make migrated KomaMRI checkpoints reusable after repository relocation and
  remove machine-specific Julia paths from scanner-matched launchers.
- Constrain source distributions to package code and release metadata so
  local simulation outputs cannot be captured in an oversized archive.
- Ignore generated Koma phantoms, checkpoints, sequences, arrays, logs, and
  media so multi-gigabyte run artifacts cannot be staged accidentally.
- Make production MRI audit/render tools headless-safe and label 100k versus
  1M reconstruction artifacts from the selected run.
- Complete and audit the 1M-spin, 15-bin scanner-derived dp150 VENC5
  REF/X/Y/Z workflow, including all 200 batches, reconstruction integrity,
  descriptive VENC coverage, hash-bound evidence, and verified analysis media.
- Complete the 1M-spin, eight-bin aortic REF/X/Y/Z workflow and add temporal
  response, cine, FFmpeg/ffprobe, and artifact-manifest evidence while keeping
  the historical, 100k-pilot, and two-bin results distinct.
- Harden future Koma production launchers with semantic Julia resolution,
  strict checkpoint provenance, read-only legacy-completion preflight, atomic
  reports, and hash-bound scanner protocol audits.
- Add path-safe fresh-run names to the aortic production launcher and record
  future physical-GPU selection, run-name provenance, and atomic final reports
  without modifying preserved legacy production artifacts.
- Remove the unused `imageio` runtime dependency and repair source-checkout,
  environment, protocol, and evidence-traceability documentation.

## [0.2.1]

- Seed candidate points by probing the volume mesh directly, making volume
  seeding independent of tetrahedron/surface-normal orientation.

## [0.2.0] - 2026-06-24

- Add a `precision` option (`"f64"` default, `"f32"` single) to `load_flow` and
  `track_parallel` that runs the field sampling and advection in single
  precision for a speedup, carrying positions/output at the same precision.
- Add a `time_interp` option (`"linear"` default, `"cubic"`) to `load_flow` and
  `track_parallel`. Cubic uses a uniform Catmull-Rom spline over four frames to
  interpolate the velocity field between stored timesteps, removing the
  per-frame velocity kink and reducing temporal reconstruction error (~38% on
  the example cardiac waveform). Requires uniformly spaced frames.
- Add load-time mesh conditioning (`conform_mesh`, default `True`): split
  non-tetrahedral cells (e.g. boundary-layer wedges/prisms) into tets and drop
  degenerate (near-zero-volume) cells so the fast tet sampler can run on hybrid
  or imperfect meshes. A no-op for already-clean all-tet meshes; pass
  `conform_mesh=False` to load as-is. `_TetSampler` now also guards against
  degenerate cells defensively (no crash; reports the count).
- Add `WallSlip`, an optional near-wall no-penetration projection passed to
  `track`/`track_parallel` (`wall_slip=`). It strips the into-wall velocity
  component within a band of the wall (a fraction of the vessel hydraulic
  diameter, default 2%) so particles slide along walls instead of being
  deposited and trapped by the spurious near-wall divergence of the
  interpolated field. Open boundaries (caps) are excluded so flux still passes.

## [0.1.0rc1] - 2026-06-18

- Rename the import package to `mrsimtracks` and narrow the top-level public
  API.
- Split tracking, parallel execution, flow loading, sampling, seeding, and
  reseeding into focused modules.
- Add explicit seeding control to `track` and `track_parallel`.
- Add streamed HDF5 tracking output with file-backed `TrackingResult` loading.
- Add structured timing metrics via `return_metrics=True`.
- Add input validation for tracking methods, seed shapes, flow array names, and
  seeding arguments.
- Move cap extraction to `mrsimtracks.dev` as a development helper.
- Add MkDocs GitHub Pages documentation.
- Add full-cycle Git LFS release validation with JSON metrics artifacts.
- Add normal-commit coverage reporting and small behavioral regression tests.
- Add GitHub Actions CI for normal tests and release validation.
- Add Git LFS tracking for the full example CFD dataset.
- Add a reduced real-data fixture for fast CI.
- Add initial publication metadata and MIT license.

## [0.1.0rc0] - 2026-06-17

- Initial packaged version of MRSimTracks.
- Add high-level APIs for loading flow fields, tracking particles, parallel
  tracking, cap extraction, and boundary-aware reseeding.
- Add time-resolved static-mesh `.pvd` loading and fast tetrahedral sampling.
