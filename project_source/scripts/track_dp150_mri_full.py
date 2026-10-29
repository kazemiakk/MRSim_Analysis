r"""Track the production one-cycle dp150 MRI particle phantom.

The run uses MRSimTracks 0.2.1, deterministic volume-uniform seeds, cubic
temporal interpolation, RK4 stepping at the timestep selected by the
convergence study, wall slip, and flux-weighted open-boundary reseeding. Raw
CFD coordinates are validated before the trajectory is transformed in-place
to centered MRI coordinates ordered from largest to smallest geometry axis.

Run in an ephemeral, non-project environment so the checkout cannot shadow the
installed wheel:

    uv run --isolated --no-project --with "mrsimtracks==0.2.1" `
        python -I scripts\track_dp150_mri_full.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


DEFAULT_ROOT = Path("outputs/dp150_mri_pipeline")
EXPECTED_VERSION = "0.2.1"
ACTIVE_KEY = "Velocity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--duration", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--wall-band-frac", type=float, default=0.02)
    parser.add_argument("--axial-bins", type=int, default=64)
    parser.add_argument("--qa-frames", type=int, default=17)
    parser.add_argument("--venc", type=float, default=4.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def package_provenance() -> dict[str, str]:
    version = importlib.metadata.version("mrsimtracks")
    import mrsimtracks

    module = Path(mrsimtracks.__file__).resolve()
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Expected MRSimTracks {EXPECTED_VERSION}, found {version}")
    if "site-packages" not in {part.lower() for part in module.parts}:
        raise RuntimeError("Run with Python -I; the checkout is shadowing the wheel")
    return {"distribution_version": version, "module_path": str(module)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def tet_axial_volume_fraction(mesh: pv.UnstructuredGrid, edges: np.ndarray) -> np.ndarray:
    if not np.all(np.asarray(mesh.celltypes) == pv.CellType.TETRA):
        raise ValueError("Tracking QC requires an all-tetrahedral mesh")
    connectivity = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    vertices = np.asarray(mesh.points, dtype=np.float64)[connectivity]
    e0 = vertices[:, 0] - vertices[:, 3]
    e1 = vertices[:, 1] - vertices[:, 3]
    e2 = vertices[:, 2] - vertices[:, 3]
    volumes = np.abs(np.einsum("ij,ij->i", e0, np.cross(e1, e2))) / 6.0
    centres = vertices[:, :, 2].mean(axis=1)
    histogram, _ = np.histogram(centres, bins=edges, weights=volumes)
    return histogram / histogram.sum()


def wall_radius_profile(wall: pv.PolyData, edges: np.ndarray) -> np.ndarray:
    points = np.asarray(wall.points, dtype=np.float64)
    bins = np.clip(
        np.searchsorted(edges, points[:, 2], side="right") - 1,
        0,
        len(edges) - 2,
    )
    radius = np.linalg.norm(points[:, :2], axis=1)
    profile = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    for index in range(profile.size):
        values = radius[bins == index]
        if values.size:
            profile[index] = values.max()
    valid = np.isfinite(profile)
    if not valid.any():
        raise RuntimeError("Could not derive a radial wall profile")
    return np.interp(np.arange(profile.size), np.flatnonzero(valid), profile[valid])


def density_metrics(
    points: np.ndarray,
    edges: np.ndarray,
    expected_axial: np.ndarray,
    wall_radius: np.ndarray,
) -> tuple[float, float]:
    counts, _ = np.histogram(points[:, 2], bins=edges)
    observed = counts.astype(np.float64) / points.shape[0]
    axial_tv = float(0.5 * np.abs(observed - expected_axial).sum())
    bins = np.clip(
        np.searchsorted(edges, points[:, 2], side="right") - 1,
        0,
        wall_radius.size - 1,
    )
    ratio = np.linalg.norm(points[:, :2], axis=1) / wall_radius[bins]
    near_wall = float(np.mean(ratio > 0.95))
    return axial_tv, near_wall


def inspect_raw_track(
    path: Path,
    flow,
    initial: np.ndarray,
    edges: np.ndarray,
    expected_axial: np.ndarray,
    wall_radius: np.ndarray,
    qa_frames: int,
    venc: float,
    axis_order: np.ndarray,
) -> dict[str, object]:
    z_min, z_max = float(flow.active_mesh.bounds[4]), float(flow.active_mesh.bounds[5])
    initial_tv, initial_near_wall = density_metrics(
        initial, edges, expected_axial, wall_radius
    )

    with h5py.File(path, "r") as h5:
        position = h5["position"]
        reset = h5["reset"]
        n_steps, n_particles, n_components = position.shape
        dt = float(h5.attrs["dt"])
        sample_indices = np.unique(
            np.linspace(0, n_steps - 1, min(qa_frames, n_steps), dtype=np.int64)
        )
        sample_lookup = set(sample_indices.tolist())

        finite = True
        reset_per_step = np.zeros(n_steps, dtype=np.int64)
        raw_component_max = np.zeros(3, dtype=np.float64)
        raw_component_exceed = np.zeros(3, dtype=np.int64)
        valid_velocity_vectors = 0
        sampled_outside: list[float] = []
        sampled_lateral_outside: list[float] = []
        sampled_axial_tv: list[float] = []
        sampled_near_wall: list[float] = []
        previous = initial

        for step in range(n_steps):
            current = np.asarray(position[step], dtype=np.float32)
            flags = np.asarray(reset[step], dtype=bool)
            finite = finite and bool(np.isfinite(current).all())
            reset_per_step[step] = int(flags.sum())

            valid = ~flags
            if valid.any():
                velocity = (current[valid] - previous[valid]) / dt
                absolute = np.abs(velocity)
                raw_component_max = np.maximum(
                    raw_component_max, absolute.max(axis=0, initial=0.0)
                )
                raw_component_exceed += (absolute > venc).sum(axis=0)
                valid_velocity_vectors += int(velocity.shape[0])

            if step in sample_lookup:
                cells = flow._sampler.locate(np.ascontiguousarray(current), guess=None)
                outside = cells < 0
                through_cap = outside & (
                    (current[:, 2] < z_min) | (current[:, 2] > z_max)
                )
                lateral = outside & ~through_cap
                axial_tv, near_wall = density_metrics(
                    current, edges, expected_axial, wall_radius
                )
                sampled_outside.append(float(outside.mean()))
                sampled_lateral_outside.append(float(lateral.mean()))
                sampled_axial_tv.append(axial_tv)
                sampled_near_wall.append(near_wall)
                print(
                    f"  QA frame {step + 1}/{n_steps}: axial TV={axial_tv:.4f}, "
                    f"lateral OOB={lateral.mean():.6g}",
                    flush=True,
                )
            previous = current

        shape = tuple(int(value) for value in position.shape)
        chunks = tuple(int(value) for value in position.chunks)
        position_dtype = str(position.dtype)
        reset_dtype = str(reset.dtype)

    mri_component_max = raw_component_max[axis_order]
    mri_component_exceed = raw_component_exceed[axis_order]
    limits = {
        "sampled_lateral_outside_fraction_max": 1e-4,
        "sampled_axial_total_variation_max": 0.06,
        "near_wall_fraction_change_max": 0.03,
        "venc_component_exceedance_count_max": 0,
    }
    checks = {
        "shape": shape == (n_steps, n_particles, 3) and n_components == 3,
        "finite": finite,
        "sampled_lateral_containment": max(sampled_lateral_outside, default=1.0)
        <= limits["sampled_lateral_outside_fraction_max"],
        "density_retention": max(sampled_axial_tv, default=1.0)
        <= limits["sampled_axial_total_variation_max"],
        "no_near_wall_clumping": max(
            (abs(value - initial_near_wall) for value in sampled_near_wall),
            default=1.0,
        )
        <= limits["near_wall_fraction_change_max"],
        "venc_coverage": int(mri_component_exceed.sum())
        <= limits["venc_component_exceedance_count_max"],
    }
    return {
        "datasets": {
            "position_shape_python": list(shape),
            "position_shape_hdf5_julia": [3, n_particles, n_steps],
            "position_dtype": position_dtype,
            "position_chunks": list(chunks),
            "reset_dtype": reset_dtype,
        },
        "finite": finite,
        "reset": {
            "total_events": int(reset_per_step.sum()),
            "steps_with_resets": int(np.count_nonzero(reset_per_step)),
            "maximum_in_one_step": int(reset_per_step.max(initial=0)),
            "fraction_of_particle_steps": float(
                reset_per_step.sum() / (n_steps * n_particles)
            ),
        },
        "sampled_frames": {
            "indices_zero_based": sample_indices.tolist(),
            "times_s": ((sample_indices + 1) * dt).tolist(),
            "outside_fraction": sampled_outside,
            "lateral_outside_fraction": sampled_lateral_outside,
            "axial_total_variation": sampled_axial_tv,
            "near_wall_fraction_r_over_R_gt_0p95": sampled_near_wall,
            "initial_axial_total_variation": initial_tv,
            "initial_near_wall_fraction_r_over_R_gt_0p95": initial_near_wall,
            "maximum_near_wall_fraction_change": max(
                (abs(value - initial_near_wall) for value in sampled_near_wall),
                default=0.0,
            ),
            "note": (
                "Open-cap overshoot may persist for one stored step before recycling; "
                "lateral escape is the wall-failure diagnostic."
            ),
        },
        "velocity_encoding": {
            "venc_m_per_s": venc,
            "mri_component_order": ["readout", "phase", "slice"],
            "maximum_absolute_component_m_per_s": mri_component_max.tolist(),
            "component_exceedance_count": mri_component_exceed.tolist(),
            "valid_nonreset_velocity_vectors": valid_velocity_vectors,
            "minimum_component_margin_m_per_s": float(venc - mri_component_max.max()),
        },
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "limits": limits,
        },
    }


def transform_to_mri_in_place(
    path: Path,
    axis_order: np.ndarray,
    source_center: np.ndarray,
    fov: np.ndarray,
    matrix: np.ndarray,
) -> None:
    ordered_center = source_center[axis_order]
    with h5py.File(path, "r+") as h5:
        initial = np.asarray(h5["initial_position"], dtype=np.float32)
        mri_initial = np.ascontiguousarray(
            initial[:, axis_order] - ordered_center, dtype=np.float32
        )
        h5["initial_position"][:] = mri_initial

        position = h5["position"]
        for step in range(position.shape[0]):
            raw = np.asarray(position[step], dtype=np.float32)
            mri = np.ascontiguousarray(
                raw[:, axis_order] - ordered_center, dtype=np.float32
            )
            if np.any(np.abs(mri) > 0.5 * fov + 1e-6):
                raise RuntimeError(f"MRI coordinates exceed FOV at step {step}")
            position[step] = mri
            if step == 0 or (step + 1) % 120 == 0 or step + 1 == position.shape[0]:
                print(f"  centered MRI frame {step + 1}/{position.shape[0]}", flush=True)

        h5.attrs.update(
            {
                "coordinate_space": "MRI_centered_largest_to_smallest",
                "source_axis_order_zero_based": axis_order,
                "source_geometry_center_m": source_center,
                "fov_m": fov,
                "matrix": matrix,
                "resolution_m": fov / matrix,
            }
        )


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.dt <= 0 or args.duration <= 0 or args.venc <= 0:
        raise ValueError("count, dt, duration, and venc must be positive")
    n_steps = int(round(args.duration / args.dt))
    if not np.isclose(n_steps * args.dt, args.duration, atol=1e-12, rtol=0):
        raise ValueError("duration must be an integer multiple of dt")
    if args.axial_bins < 8 or args.qa_frames < 2:
        raise ValueError("axial-bins must be >= 8 and qa-frames must be >= 2")

    provenance = package_provenance()
    import mrsimtracks as mt

    root = args.root.resolve()
    pvd_path = root / "flow_periodic" / "dp150_periodic.pvd"
    wall_path = root / "flow_periodic" / "wall.vtp"
    inlet_path = root / "flow_periodic" / "inlet.vtp"
    outlet_path = root / "flow_periodic" / "outlet.vtp"
    static_path = root / "dp150_static_2m_mri.h5"
    raw_seeds_path = root / "dp150_uniform_seeds_2m_raw.npy"
    periodic_report_path = root / "periodic_flow_report.json"
    convergence_report_path = root / "tracking_convergence_report.json"
    output_path = root / "dp150_motion_1m_mri.h5"
    temporary_path = root / "dp150_motion_1m_raw.partial.h5"
    report_path = root / "motion_1m_report.json"
    required = [
        pvd_path,
        wall_path,
        inlet_path,
        outlet_path,
        static_path,
        raw_seeds_path,
        periodic_report_path,
        convergence_report_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    for path in (output_path, temporary_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    if args.overwrite and temporary_path.exists():
        temporary_path.unlink()

    periodic_report = json.loads(periodic_report_path.read_text(encoding="utf-8"))
    convergence = json.loads(convergence_report_path.read_text(encoding="utf-8"))
    if not periodic_report["validation"]["passed"]:
        raise RuntimeError("Periodic-flow closure report did not pass")
    selected_dt = float(convergence["decision"]["selected_full_tracking_dt_s"])
    if not np.isclose(args.dt, selected_dt, atol=1e-15, rtol=0):
        raise ValueError(
            f"Requested dt {args.dt} differs from convergence decision {selected_dt}"
        )

    with h5py.File(static_path, "r") as h5:
        axis_order = np.asarray(h5.attrs["source_axis_order_zero_based"], dtype=int)
        source_center = np.asarray(h5.attrs["source_geometry_center_m"], dtype=float)
        fov = np.asarray(h5.attrs["fov_m"], dtype=float)
        matrix = np.asarray(h5.attrs["matrix"], dtype=int)
    if sorted(axis_order.tolist()) != [0, 1, 2]:
        raise RuntimeError(f"Invalid source axis order: {axis_order}")

    all_seeds = np.load(raw_seeds_path, mmap_mode="r")
    if all_seeds.shape[1:] != (3,) or args.count > all_seeds.shape[0]:
        raise ValueError(f"Cannot select {args.count:,} seeds from {all_seeds.shape}")
    selection_rng = np.random.default_rng(args.seed)
    source_indices = np.sort(
        selection_rng.choice(all_seeds.shape[0], args.count, replace=False)
    )
    initial = np.ascontiguousarray(all_seeds[source_indices], dtype=np.float32)

    print(f"Loading periodic flow with MRSimTracks {EXPECTED_VERSION} ...", flush=True)
    load_start = time.perf_counter()
    flow = mt.load_flow(
        pvd_path,
        active_key=ACTIVE_KEY,
        precision="f32",
        time_interp="cubic",
        pbar=True,
    )
    load_seconds = time.perf_counter() - load_start
    if not np.isclose(float(flow.tmax), args.duration, atol=1e-9, rtol=0):
        raise ValueError(f"Flow duration {flow.tmax} does not match {args.duration}")
    initial_cells = flow._sampler.locate(initial, guess=None)
    if np.any(initial_cells < 0):
        raise RuntimeError("One or more production seeds are outside the mesh")

    caps = [inlet_path, outlet_path]
    reseeder = mt.BoundaryReseeder(
        caps, flow, rng=np.random.default_rng(args.seed + 2), dt=args.dt
    )
    wall_slip = mt.WallSlip(flow, caps=caps, band_frac=args.wall_band_frac)
    flux_time, cap_flux = reseeder.flux_waveform()

    print(
        f"Tracking {args.count:,} particles for {args.duration:.3f} s "
        f"({n_steps} RK4 steps at {args.dt * 1e3:.1f} ms) ...",
        flush=True,
    )
    track_start = time.perf_counter()
    result, package_metrics = mt.track(
        flow,
        seeds=initial,
        dt=args.dt,
        tmax=args.duration,
        reseeder=reseeder,
        method="RK4",
        pbar=True,
        rng=np.random.default_rng(args.seed + 1),
        output_path=temporary_path,
        return_metrics=True,
        wall_slip=wall_slip,
    )
    track_seconds = time.perf_counter() - track_start
    if result.n_steps != n_steps or result.n_particles != args.count:
        raise RuntimeError("Unexpected production tracking shape")

    with h5py.File(temporary_path, "r+") as h5:
        h5.create_dataset("initial_position", data=initial, dtype=np.float32)
        h5.create_dataset("source_seed_index", data=source_indices, dtype=np.int64)
        h5.create_dataset(
            "position_time_s",
            data=(np.arange(n_steps, dtype=np.float64) + 1.0) * args.dt,
        )
        h5.attrs.update(
            {
                "mrsimtracks_version": EXPECTED_VERSION,
                "source_pvd": str(pvd_path),
                "active_key": ACTIVE_KEY,
                "method": "RK4",
                "time_interpolation": "cubic",
                "cycle_duration_s": args.duration,
                "periodic_endpoint_closure_passed": True,
                "single_recorded_cycle": True,
                "periodic_repetition_ready": True,
                "wall_slip_band_fraction": args.wall_band_frac,
                "wall_slip_band_m": wall_slip.band,
                "rng_seed_selection": args.seed,
                "rng_seed_tracking": args.seed + 1,
                "rng_seed_reseeding": args.seed + 2,
                "synthetic_periodic_tail": True,
            }
        )

    bounds = np.asarray(flow.active_mesh.bounds, dtype=float).reshape(3, 2)
    edges = np.linspace(bounds[2, 0], bounds[2, 1], args.axial_bins + 1)
    expected_axial = tet_axial_volume_fraction(flow.active_mesh, edges)
    wall_radius = wall_radius_profile(pv.read(wall_path), edges)
    print("Running full-trajectory QC in raw CFD coordinates ...", flush=True)
    qc = inspect_raw_track(
        temporary_path,
        flow,
        initial,
        edges,
        expected_axial,
        wall_radius,
        args.qa_frames,
        args.venc,
        axis_order,
    )
    if not qc["acceptance"]["passed"]:
        raise RuntimeError(f"Production tracking QC failed: {qc['acceptance']['checks']}")

    print("Transforming trajectory to centered MRI coordinates ...", flush=True)
    transform_start = time.perf_counter()
    transform_to_mri_in_place(
        temporary_path, axis_order, source_center, fov, matrix
    )
    transform_seconds = time.perf_counter() - transform_start
    if output_path.exists():
        output_path.unlink()
    temporary_path.replace(output_path)

    net_flux = cap_flux.sum(axis=1)
    flux_scale = np.abs(cap_flux).sum(axis=1)
    relative_imbalance = np.divide(
        np.abs(net_flux), flux_scale, out=np.zeros_like(net_flux), where=flux_scale > 0
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "flow_load_seconds": load_seconds,
            "tracking_wall_seconds": track_seconds,
            "coordinate_transform_seconds": transform_seconds,
            "package_metrics": json_value(package_metrics),
        },
        "mrsimtracks": provenance,
        "inputs": {
            "periodic_pvd": str(pvd_path),
            "periodic_pvd_sha256": sha256(pvd_path),
            "uniform_seed_file": str(raw_seeds_path),
            "static_transform_hdf5": str(static_path),
            "periodic_flow_report": str(periodic_report_path),
            "convergence_report": str(convergence_report_path),
        },
        "run": {
            "particle_count": args.count,
            "source_seed_count": int(all_seeds.shape[0]),
            "dt_s": args.dt,
            "duration_s": args.duration,
            "n_steps": n_steps,
            "method": "RK4",
            "precision": "f32",
            "time_interpolation": "cubic",
            "axis_order_zero_based": axis_order.tolist(),
            "mri_axes_from_source": [["x", "y", "z"][i] for i in axis_order],
            "source_center_m": source_center.tolist(),
            "fov_m": fov.tolist(),
            "matrix": matrix.tolist(),
            "resolution_mm": (fov / matrix * 1e3).tolist(),
        },
        "boundary_flux": {
            "time_samples": int(flux_time.size),
            "cap_count": int(cap_flux.shape[1]),
            "mean_m3_per_s_by_cap": cap_flux.mean(axis=0).tolist(),
            "maximum_relative_net_imbalance": float(
                relative_imbalance.max(initial=0.0)
            ),
        },
        "cycle_scope": {
            "source_cycle_s": args.duration,
            "periodic_velocity_endpoint_closed": True,
            "synthetic_tail_blend_start_s": periodic_report["closure_method"][
                "blend_start_s"
            ],
            "limitation": periodic_report["closure_method"]["scientific_scope"],
        },
        "quality_control": qc,
        "outputs": {
            "motion_hdf5": str(output_path),
            "motion_hdf5_bytes": output_path.stat().st_size,
        },
    }
    report["outputs"]["motion_hdf5_sha256"] = sha256(output_path)
    report_path.write_text(json.dumps(json_value(report), indent=2), encoding="utf-8")
    print(f"Production tracking QC: PASS", flush=True)
    print(f"HDF5: {output_path}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
