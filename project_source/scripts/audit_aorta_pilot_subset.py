r"""Audit whether a leading particle subset represents the full aorta seeds."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "outputs" / "aorta_dp2_mri_pipeline"
DEFAULT_TRACKS = PIPELINE / "aorta_motion_1m.h5"
DEFAULT_PILOT = PIPELINE / "koma" / "multibin_8x4mm" / "pilot_100k"
DEFAULT_MOTION_REPORT = DEFAULT_PILOT / "motion_phantom_report.toml"
DEFAULT_SEQUENCE_REPORT = (
    PIPELINE
    / "koma"
    / "multibin_8x4mm"
    / "sequences"
    / "sequence_report.toml"
)
DEFAULT_OUTPUT = DEFAULT_PILOT / "pilot_subset_density_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument(
        "--motion-report", type=Path, default=DEFAULT_MOTION_REPORT
    )
    parser.add_argument(
        "--sequence-report", type=Path, default=DEFAULT_SEQUENCE_REPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def voxel_density(
    points_source_m: np.ndarray,
    source_center_m: np.ndarray,
    axis_order: np.ndarray,
    fov_m: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, int]:
    points_mri = points_source_m[:, axis_order] - source_center_m[axis_order]
    indices = np.floor((points_mri + fov_m / 2) / fov_m * matrix).astype(
        np.int64
    )
    inside = np.all((indices >= 0) & (indices < matrix), axis=1)
    selected = indices[inside]
    flat = (
        (selected[:, 2] * matrix[1] + selected[:, 1]) * matrix[0]
        + selected[:, 0]
    )
    density = np.bincount(flat, minlength=int(np.prod(matrix)))
    return density, int(inside.sum())


def main() -> None:
    args = parse_args()
    with args.motion_report.open("rb") as stream:
        motion_report = tomllib.load(stream)
    with args.sequence_report.open("rb") as stream:
        sequence_report = tomllib.load(stream)

    particle_count = int(motion_report["particle_count"])
    source_center_m = np.asarray(
        motion_report["source_coordinate_center_m"], dtype=float
    )
    axis_order = (
        np.asarray(
            motion_report[
                "source_axis_order_largest_to_smallest_one_based"
            ],
            dtype=np.int64,
        )
        - 1
    )
    fov_m = np.asarray(sequence_report["FOV_m"], dtype=float)
    matrix = np.asarray(sequence_report["matrix"], dtype=np.int64)

    with h5py.File(args.tracks, "r") as source:
        available = int(source.attrs["n_particles"])
        if not 0 < particle_count <= available:
            raise ValueError("Pilot particle count is outside the source range")
        points = np.asarray(source["initial_position"], dtype=np.float32)
    if points.shape != (available, 3) or not np.isfinite(points).all():
        raise ValueError("Initial particle coordinates are invalid")

    full_density, full_inside = voxel_density(
        points, source_center_m, axis_order, fov_m, matrix
    )
    pilot_density, pilot_inside = voxel_density(
        points[:particle_count],
        source_center_m,
        axis_order,
        fov_m,
        matrix,
    )
    occupied = full_density > 0
    expected_fraction = particle_count / available
    expected = full_density * expected_fraction
    stable = expected >= 5
    if not stable.any():
        raise ValueError("No voxels meet the expected-count diagnostic floor")
    ratio = np.divide(
        pilot_density,
        expected,
        out=np.full_like(expected, np.nan, dtype=float),
        where=expected > 0,
    )

    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DESCRIPTIVE_PILOT_SUBSET_AUDIT",
        "scope": (
            "Descriptive 4 mm grid comparison of the deterministic leading "
            "pilot rows with the full tracked seed set; not proof of exact "
            "physical-volume uniformity or MRI convergence."
        ),
        "inputs": {
            "tracks": str(args.tracks.resolve()),
            "motion_report": str(args.motion_report.resolve()),
            "sequence_report": str(args.sequence_report.resolve()),
        },
        "particle_count_full": available,
        "particle_count_pilot": particle_count,
        "expected_fraction": expected_fraction,
        "selection": "leading particle rows, matching the pilot phantom builder",
        "matrix": matrix.tolist(),
        "FOV_m": fov_m.tolist(),
        "particles_inside_fov": {
            "full": full_inside,
            "pilot": pilot_inside,
        },
        "occupied_voxels": {
            "full": int(np.count_nonzero(occupied)),
            "pilot": int(np.count_nonzero(pilot_density)),
            "pilot_coverage_fraction_of_full": float(
                np.mean(pilot_density[occupied] > 0)
            ),
        },
        "density_map_correlation_full_vs_pilot": float(
            np.corrcoef(full_density[occupied], pilot_density[occupied])[0, 1]
        ),
        "expected_count_at_least_5_voxels": int(np.count_nonzero(stable)),
        "pilot_to_expected_ratio_expected_count_at_least_5": {
            "median": float(np.nanmedian(ratio[stable])),
            "p05": float(np.nanquantile(ratio[stable], 0.05)),
            "p95": float(np.nanquantile(ratio[stable], 0.95)),
            "p99": float(np.nanquantile(ratio[stable], 0.99)),
        },
        "normalized_rmse_expected_count_at_least_5": float(
            np.sqrt(np.mean((pilot_density[stable] - expected[stable]) ** 2))
            / np.mean(expected[stable])
        ),
        "coordinate_extent_m": {
            "full": np.ptp(points, axis=0).astype(float).tolist(),
            "pilot": np.ptp(points[:particle_count], axis=0)
            .astype(float)
            .tolist(),
        },
        "limitations": [
            "The grid includes partial boundary voxels.",
            "The audit evaluates initial seed density, not temporal MRI noise convergence.",
            "The source is a point cloud without tetrahedral cell-volume ground truth.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
