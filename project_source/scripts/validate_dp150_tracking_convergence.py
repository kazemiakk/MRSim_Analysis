r"""Compare 1.0 ms and 0.5 ms RK4 tracking on the periodic dp150 flow."""

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


DEFAULT_ROOT = Path("outputs/dp150_mri_pipeline")
EXPECTED_VERSION = "0.2.1"
ACTIVE_KEY = "Velocity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--duration", type=float, default=0.72)
    parser.add_argument("--coarse-dt", type=float, default=0.001)
    parser.add_argument("--fine-dt", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--wall-band-frac", type=float, default=0.02)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Recompute comparison/report from existing coarse and fine HDF5 files.",
    )
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


def track_one(
    mt,
    flow,
    seeds: np.ndarray,
    caps: list[Path],
    path: Path,
    *,
    dt: float,
    duration: float,
    rng_seed: int,
    wall_band_frac: float,
) -> tuple[dict[str, object], float, float]:
    reseeder = mt.BoundaryReseeder(
        caps,
        flow,
        rng=np.random.default_rng(rng_seed + 1),
        dt=dt,
    )
    wall_slip = mt.WallSlip(flow, caps=caps, band_frac=wall_band_frac)
    partial = path.with_suffix(".partial.h5")
    if partial.exists():
        partial.unlink()
    start = time.perf_counter()
    result, metrics = mt.track(
        flow,
        seeds=seeds,
        dt=dt,
        tmax=duration,
        reseeder=reseeder,
        method="RK4",
        pbar=True,
        rng=np.random.default_rng(rng_seed),
        output_path=partial,
        return_metrics=True,
        wall_slip=wall_slip,
    )
    elapsed = time.perf_counter() - start
    n_steps = int(round(duration / dt))
    if result.n_steps != n_steps or result.n_particles != seeds.shape[0]:
        raise RuntimeError("Unexpected tracking result shape")
    with h5py.File(partial, "r+") as h5:
        h5.create_dataset("initial_position", data=seeds, dtype=np.float32)
        h5.create_dataset(
            "position_time_s",
            data=(np.arange(n_steps, dtype=np.float64) + 1.0) * dt,
        )
        h5.attrs.update(
            {
                "mrsimtracks_version": EXPECTED_VERSION,
                "method": "RK4",
                "time_interpolation": "cubic",
                "cycle_duration_s": duration,
                "periodic_flow": True,
                "wall_slip_band_fraction": wall_band_frac,
                "wall_slip_band_m": wall_slip.band,
                "rng_seed_tracking": rng_seed,
                "rng_seed_reseeding": rng_seed + 1,
            }
        )
    if path.exists():
        path.unlink()
    partial.replace(path)
    return metrics, elapsed, wall_slip.band


def compare_tracks(
    coarse_path: Path,
    fine_path: Path,
    coarse_dt: float,
    fine_dt: float,
) -> dict[str, object]:
    ratio = int(round(coarse_dt / fine_dt))
    if ratio < 2 or not np.isclose(ratio * fine_dt, coarse_dt, atol=1e-15):
        raise ValueError("coarse-dt must be an integer multiple of fine-dt")
    with h5py.File(coarse_path, "r") as coarse, h5py.File(fine_path, "r") as fine:
        coarse_reset = np.asarray(coarse["reset"], dtype=bool)
        fine_reset = np.asarray(fine["reset"], dtype=bool)
        expected_fine_steps = coarse_reset.shape[0] * ratio
        if fine_reset.shape != (expected_fine_steps, coarse_reset.shape[1]):
            raise ValueError("Fine trajectory shape does not match the timestep ratio")
        never_reset = ~coarse_reset.any(axis=0) & ~fine_reset.any(axis=0)
        comparison_count = int(never_reset.sum())
        if comparison_count < coarse_reset.shape[1] // 2:
            raise RuntimeError("Fewer than half the particles remain identity-matched")

        rms_by_time = np.zeros(coarse_reset.shape[0], dtype=np.float64)
        q95_by_time = np.zeros_like(rms_by_time)
        q99_by_time = np.zeros_like(rms_by_time)
        axial_tv_by_time = np.zeros_like(rms_by_time)
        radial_tv_by_time = np.zeros_like(rms_by_time)
        initial = np.asarray(coarse["initial_position"], dtype=np.float64)
        axial_edges = np.linspace(initial[:, 2].min(), initial[:, 2].max(), 65)
        radial_initial = np.linalg.norm(initial[:, :2], axis=1)
        radial_edges = np.linspace(0.0, radial_initial.max() * (1 + 1e-6), 49)
        final_error = None
        for coarse_index in range(coarse_reset.shape[0]):
            fine_index = (coarse_index + 1) * ratio - 1
            coarse_frame = np.asarray(coarse["position"][coarse_index])
            fine_frame = np.asarray(fine["position"][fine_index])
            delta = coarse_frame[never_reset] - fine_frame[never_reset]
            error = np.linalg.norm(delta, axis=1)
            rms_by_time[coarse_index] = np.sqrt(np.mean(error**2))
            q95_by_time[coarse_index] = np.quantile(error, 0.95)
            q99_by_time[coarse_index] = np.quantile(error, 0.99)
            coarse_axial, _ = np.histogram(coarse_frame[:, 2], bins=axial_edges)
            fine_axial, _ = np.histogram(fine_frame[:, 2], bins=axial_edges)
            axial_tv_by_time[coarse_index] = 0.5 * np.abs(
                coarse_axial / coarse_frame.shape[0]
                - fine_axial / fine_frame.shape[0]
            ).sum()
            coarse_radial, _ = np.histogram(
                np.linalg.norm(coarse_frame[:, :2], axis=1), bins=radial_edges
            )
            fine_radial, _ = np.histogram(
                np.linalg.norm(fine_frame[:, :2], axis=1), bins=radial_edges
            )
            radial_tv_by_time[coarse_index] = 0.5 * np.abs(
                coarse_radial / coarse_frame.shape[0]
                - fine_radial / fine_frame.shape[0]
            ).sum()
            if coarse_index == coarse_reset.shape[0] - 1:
                final_error = error
        if final_error is None:
            raise RuntimeError("No matched trajectory frames were compared")

        coarse_events = int(coarse_reset.sum())
        fine_events = int(fine_reset.sum())
        event_relative_difference = abs(coarse_events - fine_events) / max(fine_events, 1)

    limits = {
        "final_q95_mm_max": 0.25,
        "final_q99_mm_max": 0.50,
        "maximum_time_rms_mm_max": 0.25,
        "reset_event_relative_difference_max": 0.10,
        "identity_matched_fraction_min": 0.50,
        "ensemble_maximum_axial_tv_max": 0.03,
        "ensemble_maximum_radial_tv_max": 0.03,
    }
    metrics = {
        "identity_matched_particles": comparison_count,
        "identity_matched_fraction": comparison_count / coarse_reset.shape[1],
        "final_error_mm": {
            "rms": float(np.sqrt(np.mean(final_error**2)) * 1e3),
            "median": float(np.median(final_error) * 1e3),
            "q95": float(np.quantile(final_error, 0.95) * 1e3),
            "q99": float(np.quantile(final_error, 0.99) * 1e3),
            "max": float(final_error.max(initial=0.0) * 1e3),
        },
        "over_cycle_error_mm": {
            "maximum_rms": float(rms_by_time.max(initial=0.0) * 1e3),
            "maximum_q95": float(q95_by_time.max(initial=0.0) * 1e3),
            "maximum_q99": float(q99_by_time.max(initial=0.0) * 1e3),
        },
        "reset_events": {
            "coarse": coarse_events,
            "fine": fine_events,
            "relative_difference": event_relative_difference,
        },
        "ensemble_distribution": {
            "axial_bins": 64,
            "radial_bins": 48,
            "maximum_axial_total_variation": float(axial_tv_by_time.max(initial=0.0)),
            "final_axial_total_variation": float(axial_tv_by_time[-1]),
            "median_axial_total_variation": float(np.median(axial_tv_by_time)),
            "maximum_radial_total_variation": float(radial_tv_by_time.max(initial=0.0)),
            "final_radial_total_variation": float(radial_tv_by_time[-1]),
            "median_radial_total_variation": float(np.median(radial_tv_by_time)),
            "interpretation": (
                "Compares particle-density distributions at matched physical times; "
                "it is not an individual-trajectory error metric."
            ),
        },
    }
    checks = {
        "identity_matched_fraction": metrics["identity_matched_fraction"]
        >= limits["identity_matched_fraction_min"],
        "final_q95": metrics["final_error_mm"]["q95"] <= limits["final_q95_mm_max"],
        "final_q99": metrics["final_error_mm"]["q99"] <= limits["final_q99_mm_max"],
        "maximum_time_rms": metrics["over_cycle_error_mm"]["maximum_rms"]
        <= limits["maximum_time_rms_mm_max"],
        "reset_events": event_relative_difference
        <= limits["reset_event_relative_difference_max"],
    }
    ensemble_checks = {
        "maximum_axial_total_variation": metrics["ensemble_distribution"][
            "maximum_axial_total_variation"
        ]
        <= limits["ensemble_maximum_axial_tv_max"],
        "maximum_radial_total_variation": metrics["ensemble_distribution"][
            "maximum_radial_total_variation"
        ]
        <= limits["ensemble_maximum_radial_tv_max"],
        "reset_events": event_relative_difference
        <= limits["reset_event_relative_difference_max"],
    }
    return {
        **metrics,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "limits": limits,
        },
        "ensemble_acceptance": {
            "passed": all(ensemble_checks.values()),
            "checks": ensemble_checks,
            "limits": {
                key: value
                for key, value in limits.items()
                if key.startswith("ensemble_")
                or key == "reset_event_relative_difference_max"
            },
        },
    }


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.coarse_dt <= 0 or args.fine_dt <= 0:
        raise ValueError("count and timesteps must be positive")
    provenance = package_provenance()
    import mrsimtracks as mt

    root = args.root.resolve()
    pvd_path = root / "flow_periodic" / "dp150_periodic.pvd"
    seed_path = root / "dp150_uniform_seeds_2m_raw.npy"
    periodic_report_path = root / "periodic_flow_report.json"
    caps = [root / "flow_periodic" / "inlet.vtp", root / "flow_periodic" / "outlet.vtp"]
    convergence_dir = root / "convergence"
    convergence_dir.mkdir(parents=True, exist_ok=True)
    coarse_path = convergence_dir / "dp150_20k_dt1ms.h5"
    fine_path = convergence_dir / "dp150_20k_dt0p5ms.h5"
    report_path = root / "tracking_convergence_report.json"
    for path in (pvd_path, seed_path, periodic_report_path, *caps):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (coarse_path, fine_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    all_seeds = np.load(seed_path, mmap_mode="r")
    if args.count > all_seeds.shape[0]:
        raise ValueError("Requested convergence count exceeds the static seed cloud")
    select_rng = np.random.default_rng(args.seed)
    source_indices = np.sort(
        select_rng.choice(all_seeds.shape[0], args.count, replace=False)
    )
    seeds = np.ascontiguousarray(all_seeds[source_indices], dtype=np.float32)

    print("Loading exact-closure periodic flow with cubic time interpolation ...", flush=True)
    flow = mt.load_flow(
        pvd_path,
        active_key=ACTIVE_KEY,
        precision="f32",
        time_interp="cubic",
        pbar=True,
    )
    if not np.isclose(flow.tmax, args.duration, rtol=0, atol=1e-12):
        raise ValueError("Flow duration does not match convergence duration")
    if np.any(flow._sampler.locate(seeds, guess=None) < 0):
        raise RuntimeError("Convergence seeds are not all inside the mesh")

    if args.reuse_existing:
        for path in (coarse_path, fine_path):
            if not path.exists():
                raise FileNotFoundError(path)
        previous = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {}
        )
        previous_tracking = previous.get("tracking", {})
        coarse_metrics = previous_tracking.get("coarse_package_metrics", {})
        fine_metrics = previous_tracking.get("fine_package_metrics", {})
        coarse_seconds = float(previous_tracking.get("coarse_wall_seconds", 0.0))
        fine_seconds = float(previous_tracking.get("fine_wall_seconds", 0.0))
        with h5py.File(coarse_path, "r") as h5:
            wall_band = float(h5.attrs["wall_slip_band_m"])
        print("Reusing existing coarse and fine trajectories ...", flush=True)
    else:
        print(f"Tracking coarse dt={args.coarse_dt * 1e3:.3f} ms ...", flush=True)
        coarse_metrics, coarse_seconds, wall_band = track_one(
            mt,
            flow,
            seeds,
            caps,
            coarse_path,
            dt=args.coarse_dt,
            duration=args.duration,
            rng_seed=args.seed + 10,
            wall_band_frac=args.wall_band_frac,
        )
        print(f"Tracking fine dt={args.fine_dt * 1e3:.3f} ms ...", flush=True)
        fine_metrics, fine_seconds, _ = track_one(
            mt,
            flow,
            seeds,
            caps,
            fine_path,
            dt=args.fine_dt,
            duration=args.duration,
            rng_seed=args.seed + 10,
            wall_band_frac=args.wall_band_frac,
        )
    comparison = compare_tracks(
        coarse_path,
        fine_path,
        args.coarse_dt,
        args.fine_dt,
    )
    individual_passed = comparison["acceptance"]["passed"]
    ensemble_passed = comparison["ensemble_acceptance"]["passed"]
    selected_dt = args.coarse_dt if individual_passed else args.fine_dt
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "mrsimtracks": provenance,
        "input": {
            "periodic_pvd": str(pvd_path),
            "periodic_pvd_sha256": sha256(pvd_path),
            "periodic_flow_report": str(periodic_report_path),
            "seed_file": str(seed_path),
            "seed_file_sha256": sha256(seed_path),
            "particle_count": args.count,
            "selection_rng_seed": args.seed,
        },
        "tracking": {
            "method": "RK4",
            "time_interpolation": "cubic",
            "duration_s": args.duration,
            "coarse_dt_s": args.coarse_dt,
            "fine_dt_s": args.fine_dt,
            "wall_slip_band_fraction": args.wall_band_frac,
            "wall_slip_band_m": wall_band,
            "coarse_wall_seconds": coarse_seconds,
            "fine_wall_seconds": fine_seconds,
            "coarse_package_metrics": coarse_metrics,
            "fine_package_metrics": fine_metrics,
        },
        "comparison": comparison,
        "decision": {
            "selected_full_tracking_dt_s": selected_dt,
            "individual_trajectory_convergence_passed": individual_passed,
            "ensemble_distribution_convergence_passed": ensemble_passed,
            "status": (
                "PASS"
                if individual_passed
                else "PASS_ENSEMBLE_WITH_CONSERVATIVE_FINE_DT"
                if ensemble_passed
                else "FAIL"
            ),
            "reason": (
                "1 ms passed the sub-voxel individual comparison against 0.5 ms."
                if individual_passed
                else (
                    "Individual paths diverged over the full cycle, but matched-time "
                    "ensemble density distributions and reset counts passed; use the "
                    "finer 0.5 ms trajectory conservatively."
                    if ensemble_passed
                    else "Both individual and ensemble gates failed; timestep remains unresolved."
                )
            ),
            "source_temporal_limitation": (
                "The CFD fields are saved every 10 ms. Cubic interpolation is smooth but "
                "does not replace a CFD solution saved at finer temporal resolution."
            ),
        },
        "outputs": {
            "coarse_hdf5": str(coarse_path),
            "fine_hdf5": str(fine_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Convergence: {report['decision']['status']}; "
        f"selected dt={selected_dt * 1e3:.3f} ms"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
