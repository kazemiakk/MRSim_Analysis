r"""Track a deterministic one-cycle dp150 pilot with MRSimTracks 0.2.1.

The pilot uses the accepted volume-uniform seed cloud, 1 ms RK4 stepping,
flux-weighted open-boundary reseeding, and the package wall-slip correction.
The HDF5 trajectory is streamed to disk and then checked in chunks.

Run in an ephemeral, non-project environment with Python's isolated mode so
the repository checkout cannot shadow the installed release:

    uv run --isolated --no-project --with "mrsimtracks==0.2.1" `
        python -I scripts\track_dp150_one_cycle_pilot.py --overwrite
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
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


DEFAULT_ROOT = Path("outputs/dp150_volume_pipeline")
EXPECTED_VERSION = "0.2.1"
ACTIVE_KEY = "Velocity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--wall-band-frac", type=float, default=0.02)
    parser.add_argument("--axial-bins", type=int, default=48)
    parser.add_argument("--qa-frames", type=int, default=25)
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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tet_axial_volume_fraction(
    mesh: pv.UnstructuredGrid, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.asarray(mesh.celltypes) == pv.CellType.TETRA):
        raise ValueError("Tracking QC requires an all-tetrahedral mesh")
    conn = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    points = np.asarray(mesh.points, dtype=np.float64)
    vertices = points[conn]
    e0 = vertices[:, 0] - vertices[:, 3]
    e1 = vertices[:, 1] - vertices[:, 3]
    e2 = vertices[:, 2] - vertices[:, 3]
    volumes = np.abs(np.einsum("ij,ij->i", e0, np.cross(e1, e2))) / 6.0
    centers_z = vertices[:, :, 2].mean(axis=1)
    per_bin, _ = np.histogram(centers_z, bins=edges, weights=volumes)
    return per_bin / per_bin.sum(), volumes


def occupancy_fraction(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(points[:, 2], bins=edges)
    return counts.astype(np.float64) / points.shape[0]


def total_variation(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(0.5 * np.abs(observed - expected).sum())


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


def inspect_track(
    h5_path: Path,
    flow,
    initial: np.ndarray,
    edges: np.ndarray,
    expected_axial: np.ndarray,
    qa_frames: int,
    source_speed_max: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    z_min, z_max = float(flow.active_mesh.bounds[4]), float(flow.active_mesh.bounds[5])
    sample_indices: np.ndarray
    speed_samples: list[np.ndarray] = []
    reset_per_step: np.ndarray
    outside_fraction: list[float] = []
    lateral_outside_fraction: list[float] = []

    with h5py.File(h5_path, "r") as h5:
        position = h5["position"]
        reset = h5["reset"]
        n_steps, n_particles, n_xyz = position.shape
        dt = float(h5.attrs["dt"])
        sample_indices = np.unique(
            np.linspace(0, n_steps - 1, min(qa_frames, n_steps), dtype=int)
        )
        finite = True
        reset_per_step = np.zeros(n_steps, dtype=np.int64)
        previous = None
        final = None
        for step in range(n_steps):
            current = np.asarray(position[step], dtype=np.float32)
            flags = np.asarray(reset[step], dtype=bool)
            finite = finite and bool(np.isfinite(current).all())
            reset_per_step[step] = int(flags.sum())
            if previous is not None:
                speed = np.linalg.norm(current - previous, axis=1) / dt
                valid_motion = ~flags
                if valid_motion.any():
                    speed_samples.append(speed[valid_motion])
            previous = current
            if step == n_steps - 1:
                final = current.copy()

        if final is None:
            raise RuntimeError("Tracking file contains no positions")

        for step in sample_indices:
            current = np.ascontiguousarray(position[step], dtype=np.float32)
            cells = flow._sampler.locate(current, guess=None)
            outside = cells < 0
            outside_fraction.append(float(outside.mean()))
            through_open_cap = outside & (
                (current[:, 2] < z_min) | (current[:, 2] > z_max)
            )
            lateral_outside = outside & ~through_open_cap
            lateral_outside_fraction.append(float(lateral_outside.mean()))

        shape = tuple(int(value) for value in position.shape)
        position_dtype = str(position.dtype)
        reset_dtype = str(reset.dtype)
        chunks = tuple(int(value) for value in position.chunks)

    all_speed = np.concatenate(speed_samples) if speed_samples else np.empty(0)
    if all_speed.size:
        speed_quantiles = {
            "median_m_per_s": float(np.quantile(all_speed, 0.5)),
            "p95_m_per_s": float(np.quantile(all_speed, 0.95)),
            "p99_m_per_s": float(np.quantile(all_speed, 0.99)),
            "p999_m_per_s": float(np.quantile(all_speed, 0.999)),
            "max_m_per_s": float(all_speed.max()),
        }
    else:
        speed_quantiles = {key: 0.0 for key in (
            "median_m_per_s", "p95_m_per_s", "p99_m_per_s",
            "p999_m_per_s", "max_m_per_s"
        )}

    initial_axial = occupancy_fraction(initial, edges)
    final_axial = occupancy_fraction(final, edges)
    initial_tv = total_variation(initial_axial, expected_axial)
    final_tv = total_variation(final_axial, expected_axial)
    lateral_max = max(lateral_outside_fraction, default=1.0)
    max_expected_speed = 1.10 * source_speed_max

    limits = {
        "sampled_lateral_outside_fraction_max": 1e-4,
        "final_axial_total_variation_max": 0.06,
        "nonreset_speed_max_multiplier": 1.10,
    }
    checks = {
        "shape": shape == (n_steps, n_particles, 3) and n_xyz == 3,
        "finite": finite,
        "sampled_lateral_containment": lateral_max <= limits[
            "sampled_lateral_outside_fraction_max"
        ],
        "final_axial_density": final_tv <= limits["final_axial_total_variation_max"],
        "nonreset_speed_bound": speed_quantiles["max_m_per_s"] <= max_expected_speed,
    }
    passed = all(checks.values())
    report = {
        "datasets": {
            "position_shape": list(shape),
            "position_dtype": position_dtype,
            "position_chunks": list(chunks),
            "reset_dtype": reset_dtype,
        },
        "finite": finite,
        "reset": {
            "total_events": int(reset_per_step.sum()),
            "steps_with_resets": int(np.count_nonzero(reset_per_step)),
            "maximum_in_one_step": int(reset_per_step.max(initial=0)),
            "mean_per_step": float(reset_per_step.mean()),
            "fraction_of_particle_steps": float(reset_per_step.sum() / (n_steps * n_particles)),
        },
        "sampled_containment": {
            "frame_indices": sample_indices.tolist(),
            "outside_fraction_by_frame": outside_fraction,
            "lateral_outside_fraction_by_frame": lateral_outside_fraction,
            "maximum_outside_fraction": max(outside_fraction, default=0.0),
            "maximum_lateral_outside_fraction": lateral_max,
            "note": "Open-cap overshoot is expected for at most one stored step before recycling; lateral escape is the wall-failure diagnostic.",
        },
        "nonreset_motion": {
            **speed_quantiles,
            "source_point_velocity_max_m_per_s": source_speed_max,
            "acceptance_max_m_per_s": max_expected_speed,
            "note": "Differences ending at reset events are excluded, so boundary recycling is not interpreted as particle speed.",
        },
        "axial_density": {
            "initial_total_variation": initial_tv,
            "final_total_variation": final_tv,
            "change": final_tv - initial_tv,
        },
        "acceptance": {
            "passed": passed,
            "checks": checks,
            "limits": limits,
            "interpretation": (
                "The pilot completed with finite, wall-contained motion and acceptable axial density retention."
                if passed
                else "One or more pilot QC checks failed; inspect the reported diagnostic before scaling particle count."
            ),
        },
    }
    arrays = {
        "reset_per_step": reset_per_step,
        "speed": all_speed,
        "initial_axial": initial_axial,
        "final_axial": final_axial,
        "expected_axial": expected_axial,
        "axial_centers": 0.5 * (edges[:-1] + edges[1:]),
    }
    return report, arrays


def make_figure(
    path: Path,
    arrays: dict[str, np.ndarray],
    frame_times: np.ndarray,
    cap_flux: np.ndarray,
    dt: float,
    report: dict[str, object],
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), constrained_layout=True)

    for cap in range(cap_flux.shape[1]):
        axes[0, 0].plot(frame_times, cap_flux[:, cap] * 1e6, lw=1.7, label=f"Cap {cap}")
    axes[0, 0].axhline(0, color="#64748B", lw=0.8)
    axes[0, 0].set_title("Signed open-boundary flux")
    axes[0, 0].set_xlabel("Cycle time (s)")
    axes[0, 0].set_ylabel(r"Flux ($cm^3/s$)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(color="#D1D5DB", lw=0.5, alpha=0.7)

    steps = np.arange(arrays["reset_per_step"].size)
    axes[0, 1].plot((steps + 1) * dt, arrays["reset_per_step"], color="#B45309", lw=1.2)
    axes[0, 1].set_title("Boundary recycling events")
    axes[0, 1].set_xlabel("Position time (s)")
    axes[0, 1].set_ylabel("Particles reset per step")
    axes[0, 1].grid(color="#D1D5DB", lw=0.5, alpha=0.7)

    centers_mm = arrays["axial_centers"] * 1e3
    axes[1, 0].plot(centers_mm, arrays["expected_axial"], color="#D1495B", lw=1.8, label="Mesh volume")
    axes[1, 0].plot(centers_mm, arrays["initial_axial"], color="#124E78", lw=1.3, label="Initial")
    axes[1, 0].plot(centers_mm, arrays["final_axial"], color="#2A9D8F", lw=1.3, label="Final")
    axes[1, 0].set_title("Axial particle-density retention")
    axes[1, 0].set_xlabel("z (mm)")
    axes[1, 0].set_ylabel("Fraction per bin")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(color="#D1D5DB", lw=0.5, alpha=0.7)

    axes[1, 1].axis("off")
    motion = report["nonreset_motion"]
    density = report["axial_density"]
    reset = report["reset"]
    acceptance = report["acceptance"]
    summary = (
        f"TRACKING QC: {'PASS' if acceptance['passed'] else 'FAIL'}\n\n"
        f"Particles             {report['run']['particle_count']:,}\n"
        f"Steps / dt            {report['run']['n_steps']} / {dt * 1e3:.1f} ms\n"
        f"Reset events          {reset['total_events']:,}\n"
        f"Non-reset p99 speed   {motion['p99_m_per_s']:.3f} m/s\n"
        f"Initial axial TV      {density['initial_total_variation']:.4f}\n"
        f"Final axial TV        {density['final_total_variation']:.4f}\n"
        f"Lateral OOB max       {report['sampled_containment']['maximum_lateral_outside_fraction']:.6f}\n\n"
        "Single recorded cycle; not validated for periodic looping"
    )
    axes[1, 1].text(
        0.02,
        0.96,
        summary,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        color="#111827",
        bbox={"facecolor": "#F8FAFC", "edgecolor": "#CBD5E1", "boxstyle": "square,pad=0.8"},
    )
    fig.suptitle("dp150 one-cycle particle-tracking pilot", fontsize=14)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def write_markdown(path: Path, report: dict[str, object]) -> None:
    run = report["run"]
    reset = report["quality_control"]["reset"]
    density = report["quality_control"]["axial_density"]
    motion = report["quality_control"]["nonreset_motion"]
    acceptance = report["quality_control"]["acceptance"]
    lines = [
        "# dp150 One-Cycle Tracking Pilot",
        "",
        f"- Status: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
        f"- Particles: **{run['particle_count']:,}**",
        f"- Integration: **{run['method']}**, `{run['dt_s'] * 1e3:.1f} ms`, `{run['n_steps']}` steps",
        f"- Recorded interval: **0.000-0.720 s**; stored post-step positions span `{run['dt_s']:.3f}-{run['duration_s']:.3f} s`",
        f"- MRSimTracks: `{report['mrsimtracks']['distribution_version']}`",
        "",
        "## Quality control",
        "",
        f"- Boundary reset events: **{reset['total_events']:,}** ({reset['fraction_of_particle_steps']:.6%} of particle-steps)",
        f"- Maximum sampled lateral outside fraction: **{report['quality_control']['sampled_containment']['maximum_lateral_outside_fraction']:.6f}**",
        f"- Non-reset p99 / maximum speed: **{motion['p99_m_per_s']:.3f} / {motion['max_m_per_s']:.3f} m/s**",
        f"- Initial / final axial TV distance: **{density['initial_total_variation']:.4f} / {density['final_total_variation']:.4f}**",
        "",
        "## Scope",
        "",
        "This is a computational pilot, not the final 1-2 million-particle MR phantom. The source cycle fails the endpoint-closure test, so this trajectory is intentionally labeled single-cycle-only and must not be treated as a seamless periodic loop.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.dt <= 0 or args.duration <= 0:
        raise ValueError("dt and duration must be positive")
    n_steps = int(round(args.duration / args.dt))
    if not np.isclose(n_steps * args.dt, args.duration, rtol=0, atol=1e-12):
        raise ValueError("duration must be an integer multiple of dt")
    if args.qa_frames < 2 or args.axial_bins < 8:
        raise ValueError("qa-frames must be >= 2 and axial-bins must be >= 8")

    provenance = package_provenance()
    import mrsimtracks as mt

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pvd_path = root / "flow" / "dp150_flow.pvd"
    inlet_path = root / "flow" / "inlet.vtp"
    outlet_path = root / "flow" / "outlet.vtp"
    seed_path = root / "dp150_uniform_seeds_100k.npy"
    conversion_path = root / "conversion_report.json"
    output_path = root / f"dp150_one_cycle_pilot_{args.count // 1000}k.h5"
    temporary_path = output_path.with_suffix(".partial.h5")
    report_path = root / "tracking_report.json"
    markdown_path = root / "tracking_report.md"
    figure_path = root / "tracking_qc.png"

    required = [pvd_path, inlet_path, outlet_path, seed_path, conversion_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    if temporary_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Partial output exists: {temporary_path}")
        temporary_path.unlink()

    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if conversion.get("endpoint_closure", {}).get("periodic_closure_passed") is not False:
        raise RuntimeError("Conversion report must explicitly identify failed endpoint closure")

    all_seeds = np.load(seed_path)
    if all_seeds.ndim != 2 or all_seeds.shape[1] != 3 or args.count > all_seeds.shape[0]:
        raise ValueError(f"Cannot select {args.count:,} particles from seed shape {all_seeds.shape}")
    rng_select = np.random.default_rng(args.seed)
    source_indices = np.sort(rng_select.choice(all_seeds.shape[0], args.count, replace=False))
    initial = np.ascontiguousarray(all_seeds[source_indices], dtype=np.float32)

    print(f"Loading {pvd_path} with MRSimTracks {EXPECTED_VERSION} ...", flush=True)
    load_start = time.perf_counter()
    flow = mt.load_flow(
        pvd_path,
        active_key=ACTIVE_KEY,
        precision="f32",
        time_interp="linear",
        pbar=True,
    )
    load_seconds = time.perf_counter() - load_start
    if not np.isclose(float(flow.tmax), args.duration, rtol=0, atol=1e-9):
        raise ValueError(f"Flow duration {flow.tmax!r} does not match requested {args.duration}")
    initial_cells = flow._sampler.locate(initial, guess=None)
    if np.any(initial_cells < 0):
        raise RuntimeError(f"{int(np.count_nonzero(initial_cells < 0))} pilot seeds are outside the mesh")

    rng_track = np.random.default_rng(args.seed + 1)
    rng_reseed = np.random.default_rng(args.seed + 2)
    caps = [inlet_path, outlet_path]
    reseeder = mt.BoundaryReseeder(caps, flow, rng=rng_reseed, dt=args.dt)
    wall_slip = mt.WallSlip(flow, caps=caps, band_frac=args.wall_band_frac)
    flux_time, cap_flux = reseeder.flux_waveform()

    print(
        f"Tracking {args.count:,} particles for {args.duration:.3f} s "
        f"({n_steps} RK4 steps at {args.dt * 1e3:.1f} ms) ...",
        flush=True,
    )
    track_start = time.perf_counter()
    result, metrics = mt.track(
        flow,
        seeds=initial,
        dt=args.dt,
        tmax=args.duration,
        reseeder=reseeder,
        method="RK4",
        pbar=True,
        rng=rng_track,
        output_path=temporary_path,
        return_metrics=True,
        wall_slip=wall_slip,
    )
    track_seconds = time.perf_counter() - track_start
    if result.n_steps != n_steps or result.n_particles != args.count:
        raise RuntimeError(f"Unexpected tracked shape {result.n_steps, result.n_particles}")

    position_times = (np.arange(n_steps, dtype=np.float64) + 1.0) * args.dt
    with h5py.File(temporary_path, "r+") as h5:
        h5.create_dataset("initial_position", data=initial, dtype=np.float32)
        h5.create_dataset("source_seed_index", data=source_indices, dtype=np.int64)
        h5.create_dataset("position_time_s", data=position_times, dtype=np.float64)
        h5.attrs.update(
            {
                "mrsimtracks_version": EXPECTED_VERSION,
                "source_pvd": str(pvd_path),
                "source_pvd_sha256": sha256(pvd_path),
                "source_seed_file": str(seed_path),
                "source_seed_sha256": sha256(seed_path),
                "active_key": ACTIVE_KEY,
                "method": "RK4",
                "time_interpolation": "linear",
                "cycle_duration_s": args.duration,
                "position_time_offset_s": args.dt,
                "reset_detection_time_offset_s": 0.0,
                "rng_seed_selection": args.seed,
                "rng_seed_tracking": args.seed + 1,
                "rng_seed_reseeding": args.seed + 2,
                "wall_slip_band_fraction": args.wall_band_frac,
                "wall_slip_band_m": wall_slip.band,
                "hydraulic_diameter_m": wall_slip.d_hydraulic,
                "periodic_endpoint_closure_passed": False,
                "single_cycle_only": True,
                "scope": "computational pilot; not the final 1-2 million-particle phantom",
            }
        )

    bounds = flow.active_mesh.bounds
    edges = np.linspace(float(bounds[4]), float(bounds[5]), args.axial_bins + 1)
    expected_axial, tet_volumes = tet_axial_volume_fraction(flow.active_mesh, edges)
    source_speed_max = float(
        conversion.get("velocity", {}).get(
            "global_point_speed_max_m_per_s",
            conversion.get("velocity_summary", {}).get("global_point_speed_max_m_per_s", 0.0),
        )
    )
    if source_speed_max <= 0:
        frame_maxima = conversion.get("frames", [])
        candidates = [
            float(frame.get("point_speed_max_m_per_s", 0.0))
            for frame in frame_maxima
            if isinstance(frame, dict)
        ]
        source_speed_max = max(candidates, default=0.0)
    if source_speed_max <= 0:
        source_speed_max = max(float(np.linalg.norm(field, axis=1).max()) for field in flow.fields)

    qc, arrays = inspect_track(
        temporary_path,
        flow,
        initial,
        edges,
        expected_axial,
        args.qa_frames,
        source_speed_max,
    )
    net_flux = cap_flux.sum(axis=1)
    flux_scale = np.abs(cap_flux).sum(axis=1)
    flux_relative = np.divide(
        np.abs(net_flux), flux_scale, out=np.zeros_like(net_flux), where=flux_scale > 0
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mrsimtracks": provenance,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "flow_load_seconds": load_seconds,
            "tracking_wall_seconds": track_seconds,
            "package_metrics": json_value(metrics),
        },
        "inputs": {
            "pvd": str(pvd_path),
            "pvd_sha256": sha256(pvd_path),
            "uniform_seed_file": str(seed_path),
            "uniform_seed_sha256": sha256(seed_path),
            "conversion_report": str(conversion_path),
            "conversion_report_sha256": sha256(conversion_path),
        },
        "run": {
            "particle_count": args.count,
            "source_seed_count": int(all_seeds.shape[0]),
            "selection_without_replacement": True,
            "rng_seed_selection": args.seed,
            "rng_seed_tracking": args.seed + 1,
            "rng_seed_reseeding": args.seed + 2,
            "dt_s": args.dt,
            "duration_s": args.duration,
            "n_steps": n_steps,
            "method": "RK4",
            "precision": "f32",
            "time_interpolation": "linear",
            "wall_slip_band_fraction": args.wall_band_frac,
            "wall_slip_band_m": wall_slip.band,
            "hydraulic_diameter_m": wall_slip.d_hydraulic,
            "tetrahedral_cell_count": int(tet_volumes.size),
        },
        "boundary_flux": {
            "sign_convention": "positive is outward",
            "cap_count": int(cap_flux.shape[1]),
            "mean_m3_per_s_by_cap": cap_flux.mean(axis=0).tolist(),
            "minimum_m3_per_s_by_cap": cap_flux.min(axis=0).tolist(),
            "maximum_m3_per_s_by_cap": cap_flux.max(axis=0).tolist(),
            "maximum_relative_net_imbalance": float(flux_relative.max(initial=0.0)),
            "median_relative_net_imbalance": float(np.median(flux_relative)),
        },
        "cycle_scope": {
            "single_cycle_only": True,
            "periodic_endpoint_closure_passed": False,
            "reason": "The first and last CFD velocity fields are not equivalent; no synthetic wrap is used.",
        },
        "quality_control": qc,
        "outputs": {
            "hdf5": str(output_path),
            "report_json": str(report_path),
            "report_markdown": str(markdown_path),
            "qc_figure": str(figure_path),
        },
    }
    qc["run"] = report["run"]

    if output_path.exists():
        output_path.unlink()
    temporary_path.replace(output_path)
    report["outputs"]["hdf5_bytes"] = output_path.stat().st_size
    report["outputs"]["hdf5_sha256"] = sha256(output_path)
    report_path.write_text(json.dumps(json_value(report), indent=2), encoding="utf-8")
    write_markdown(markdown_path, report)
    make_figure(figure_path, arrays, flux_time, cap_flux, args.dt, qc)

    status = "PASS" if qc["acceptance"]["passed"] else "FAIL"
    print(f"Tracking QC: {status}", flush=True)
    print(f"HDF5: {output_path}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"Figure: {figure_path}", flush=True)


if __name__ == "__main__":
    main()
