r"""Audit density preservation for approximate aorta point-cloud tracking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--sample-particles", type=int, default=100_000)
    parser.add_argument("--sample-steps", type=int, default=9)
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def density_metrics(points: np.ndarray, voxel_m: float) -> dict[str, float]:
    tree = cKDTree(points)
    nearest = tree.query(points, k=2, workers=-1)[0][:, 1]
    origin = points.min(axis=0)
    index = np.floor((points - origin) / voxel_m).astype(np.int64)
    shape = index.max(axis=0) + 1
    counts = np.bincount(np.ravel_multi_index(index.T, shape))
    occupied = counts[counts > 0]
    median_occupancy = float(np.median(occupied))
    return {
        "nearest_p05_m": float(np.quantile(nearest, 0.05)),
        "nearest_median_m": float(np.median(nearest)),
        "nearest_p95_m": float(np.quantile(nearest, 0.95)),
        "duplicate_fraction": float(np.mean(nearest <= 1e-9)),
        "occupied_voxels": int(occupied.size),
        "voxel_occupancy_median": median_occupancy,
        "voxel_occupancy_p99": float(np.quantile(occupied, 0.99)),
        "voxel_occupancy_max": int(occupied.max()),
        "voxel_max_to_median": float(occupied.max() / median_occupancy),
    }


def main() -> None:
    args = parse_args()
    tracks_path = args.tracks.resolve()
    coordinates_path = args.coordinates.resolve()
    output_path = args.output.resolve()
    figure_path = args.figure.resolve()
    source = np.load(coordinates_path, mmap_mode="r")
    source_tree = cKDTree(np.asarray(source, dtype=np.float32))

    rng = np.random.default_rng(args.seed)
    with h5py.File(tracks_path, "r") as h5:
        nsteps, nparticles, dimensions = h5["position"].shape
        if dimensions != 3 or h5["initial_position"].shape != (nparticles, 3):
            raise ValueError("unexpected aorta tracking HDF5 shape")
        sample_count = min(args.sample_particles, nparticles)
        particle_ids = np.sort(
            rng.choice(nparticles, size=sample_count, replace=False)
        )
        step_ids = np.unique(
            np.linspace(0, nsteps - 1, args.sample_steps, dtype=np.int64)
        )
        support_radius = float(h5.attrs["pointcloud_support_radius_m"])
        initial = np.asarray(h5["initial_position"][particle_ids], dtype=np.float32)
        baseline = density_metrics(initial, args.voxel_mm * 1e-3)
        records = []
        for step in step_ids:
            points = np.asarray(
                h5["position"][int(step), particle_ids, :], dtype=np.float32
            )
            metrics = density_metrics(points, args.voxel_mm * 1e-3)
            distance_to_source = source_tree.query(
                points, k=1, workers=-1
            )[0]
            outside_support = distance_to_source > support_radius
            if step + 1 < nsteps:
                next_step_reset = np.asarray(
                    h5["reset"][int(step + 1), particle_ids], dtype=bool
                )
                unexplained_outside = outside_support & ~next_step_reset
                recycled_next_step = outside_support & next_step_reset
                terminal_cycle_reset = False
            else:
                # The Koma builder resets all spin identities at the periodic
                # cycle boundary, including final-step overshoots.
                unexplained_outside = np.zeros_like(outside_support)
                recycled_next_step = outside_support
                terminal_cycle_reset = True
            metrics.update(
                {
                    "step": int(step + 1),
                    "time_s": float((step + 1) * h5.attrs["dt"]),
                    "outside_support_fraction": float(
                        np.mean(outside_support)
                    ),
                    "outside_support_recycled_fraction": float(
                        np.mean(recycled_next_step)
                    ),
                    "unexplained_outside_support_fraction": float(
                        np.mean(unexplained_outside)
                    ),
                    "terminal_cycle_reset": terminal_cycle_reset,
                    "sample_reset_fraction": float(
                        np.mean(h5["reset"][int(step), particle_ids])
                    ),
                    "nearest_p05_ratio": float(
                        metrics["nearest_p05_m"] / baseline["nearest_p05_m"]
                    ),
                    "nearest_median_ratio": float(
                        metrics["nearest_median_m"]
                        / baseline["nearest_median_m"]
                    ),
                    "nearest_p95_ratio": float(
                        metrics["nearest_p95_m"] / baseline["nearest_p95_m"]
                    ),
                    "voxel_max_to_median_relative": float(
                        metrics["voxel_max_to_median"]
                        / baseline["voxel_max_to_median"]
                    ),
                }
            )
            records.append(metrics)

    gates = {
        "unexplained_outside_support_fraction_max": 1e-4,
        "duplicate_fraction_max": 1e-3,
        "nearest_median_ratio_min": 0.70,
        "nearest_median_ratio_max": 1.30,
        "nearest_p05_ratio_min": 0.40,
        "voxel_max_to_median_relative_max": 2.0,
    }
    checks = {
        "support": max(
            r["unexplained_outside_support_fraction"] for r in records
        )
        <= gates["unexplained_outside_support_fraction_max"],
        "duplicates": max(r["duplicate_fraction"] for r in records)
        <= gates["duplicate_fraction_max"],
        "median_spacing": min(r["nearest_median_ratio"] for r in records)
        >= gates["nearest_median_ratio_min"]
        and max(r["nearest_median_ratio"] for r in records)
        <= gates["nearest_median_ratio_max"],
        "lower_tail_spacing": min(r["nearest_p05_ratio"] for r in records)
        >= gates["nearest_p05_ratio_min"],
        "voxel_clumping": max(
            r["voxel_max_to_median_relative"] for r in records
        )
        <= gates["voxel_max_to_median_relative_max"],
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_APPROXIMATE_POINT_CLOUD_DIAGNOSTIC"
            if passed
            else "REVIEW_APPROXIMATE_POINT_CLOUD_DIAGNOSTIC"
        ),
        "scope": (
            "Sampled density-preservation diagnostic; not a substitute for "
            "boundary-resolved validation on a connected volume mesh."
        ),
        "tracks": str(tracks_path),
        "coordinates": str(coordinates_path),
        "sample_particle_count": sample_count,
        "sample_step_count": len(records),
        "voxel_mm": args.voxel_mm,
        "baseline": baseline,
        "gates": gates,
        "checks": checks,
        "samples": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    times = np.asarray([r["time_s"] for r in records])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    axes[0].plot(times, [r["nearest_median_ratio"] for r in records], "o-")
    axes[0].axhline(1.0, color="0.4", linewidth=1)
    axes[0].set(title="Median nearest spacing", xlabel="Time (s)", ylabel="Ratio to t=0")
    axes[1].plot(times, [r["nearest_p05_ratio"] for r in records], "o-")
    axes[1].axhline(1.0, color="0.4", linewidth=1)
    axes[1].set(title="5th percentile spacing", xlabel="Time (s)", ylabel="Ratio to t=0")
    axes[2].plot(
        times,
        [r["voxel_max_to_median_relative"] for r in records],
        "o-",
    )
    axes[2].axhline(1.0, color="0.4", linewidth=1)
    axes[2].set(title="Voxel clumping diagnostic", xlabel="Time (s)", ylabel="Relative ratio")
    fig.suptitle("Aorta dp2 1M particle-density preservation")
    fig.savefig(figure_path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"{report['status']}\nReport: {output_path}\nFigure: {figure_path}")


if __name__ == "__main__":
    main()
