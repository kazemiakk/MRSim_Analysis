r"""Voxelize tracked aorta particles into cardiac-bin preview volumes.

This produces a tracking-derived preview on a Koma-compatible grid. It is not
an MR signal simulation or a substitute for a multi-bin KomaMRI reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "outputs" / "aorta_dp2_mri_pipeline"
DEFAULT_TRACKS = PIPELINE / "aorta_motion_1m.h5"
DEFAULT_MOTION_REPORT = PIPELINE / "koma" / "motion_phantom_report.toml"
DEFAULT_SEQUENCE_REPORT = (
    PIPELINE / "koma" / "multibin_8x4mm" / "sequences" / "sequence_report.toml"
)
DEFAULT_OUTPUT = (
    PIPELINE
    / "koma"
    / "multibin_8x4mm"
    / "tracking_preview"
    / "aorta_tracking_8bin_preview.h5"
)


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
    parser.add_argument(
        "--minimum-particles-per-voxel",
        type=int,
        default=8,
        help="Minimum density and valid-velocity count used for the preview mask.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_step_indices(
    bin_count: int, source_cycle_s: float, step_dt_s: float, step_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_phase_times = (
        np.arange(bin_count, dtype=float) + 0.5
    ) * source_cycle_s / bin_count
    indices = np.rint(source_phase_times / step_dt_s).astype(int) - 1
    indices = np.clip(indices, 1, step_count - 2)
    sampled_times = (indices + 1) * step_dt_s
    return indices, source_phase_times, sampled_times


def voxel_indices(
    points_mri_m: np.ndarray, fov_m: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaled = (points_mri_m + fov_m / 2) / fov_m
    indices = np.floor(scaled * matrix).astype(np.int64)
    inside = np.all((indices >= 0) & (indices < matrix), axis=1)
    return indices, inside


def voxelize_phase(
    points_mri_m: np.ndarray,
    velocity_mri_m_per_s: np.ndarray,
    velocity_valid: np.ndarray,
    fov_m: np.ndarray,
    matrix: np.ndarray,
    minimum_particles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    indices, inside = voxel_indices(points_mri_m, fov_m, matrix)
    nx, ny, nz = (int(value) for value in matrix)
    voxel_count = nx * ny * nz

    inside_indices = indices[inside]
    density_flat = (
        (inside_indices[:, 2] * ny + inside_indices[:, 1]) * nx
        + inside_indices[:, 0]
    )
    density = np.bincount(density_flat, minlength=voxel_count).reshape(
        nz, ny, nx
    )

    valid = inside & velocity_valid
    valid_indices = indices[valid]
    valid_flat = (
        (valid_indices[:, 2] * ny + valid_indices[:, 1]) * nx
        + valid_indices[:, 0]
    )
    valid_count = np.bincount(valid_flat, minlength=voxel_count).reshape(
        nz, ny, nx
    )
    velocity = np.zeros((3, nz, ny, nx), dtype=np.float32)
    for component in range(3):
        component_sum = np.bincount(
            valid_flat,
            weights=velocity_mri_m_per_s[valid, component],
            minlength=voxel_count,
        ).reshape(nz, ny, nx)
        velocity[component] = np.divide(
            component_sum,
            valid_count,
            out=np.zeros_like(component_sum, dtype=np.float64),
            where=valid_count > 0,
        )

    particle_speed = np.linalg.norm(velocity_mri_m_per_s[valid], axis=1)
    speed_sum = np.bincount(
        valid_flat, weights=particle_speed, minlength=voxel_count
    ).reshape(nz, ny, nx)
    speed = np.divide(
        speed_sum,
        valid_count,
        out=np.zeros_like(speed_sum, dtype=np.float64),
        where=valid_count > 0,
    ).astype(np.float32)
    mask = (density >= minimum_particles) & (
        valid_count >= minimum_particles
    )
    report = {
        "particles": int(points_mri_m.shape[0]),
        "particles_inside_fov": int(inside.sum()),
        "particles_with_valid_centered_velocity": int(velocity_valid.sum()),
        "valid_velocity_particles_inside_fov": int(valid.sum()),
        "occupied_voxels": int(np.count_nonzero(density)),
        "masked_voxels": int(mask.sum()),
        "density_median_in_mask": float(np.median(density[mask])),
        "density_p99_in_mask": float(np.quantile(density[mask], 0.99)),
        "speed_median_m_per_s": float(np.median(particle_speed)),
        "speed_p99_m_per_s": float(np.quantile(particle_speed, 0.99)),
        "speed_max_m_per_s": float(particle_speed.max()),
    }
    return density.astype(np.float32), velocity, speed, mask, report


def main() -> None:
    args = parse_args()
    if args.minimum_particles_per_voxel < 1:
        raise ValueError("--minimum-particles-per-voxel must be positive")
    with args.motion_report.open("rb") as stream:
        motion_report = tomllib.load(stream)
    with args.sequence_report.open("rb") as stream:
        sequence_report = tomllib.load(stream)

    fov_m = np.asarray(sequence_report["FOV_m"], dtype=float)
    matrix = np.asarray(sequence_report["matrix"], dtype=np.int64)
    bin_count = int(sequence_report["cardiac_bins"])
    sequence_rr_s = float(sequence_report["realized_RR_s"])
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

    with h5py.File(args.tracks, "r") as source:
        positions = source["position"]
        resets = source["reset"]
        step_dt_s = float(source.attrs["dt"])
        step_count = int(source.attrs["n_steps"])
        particle_count = int(source.attrs["n_particles"])
        source_cycle_s = step_dt_s * step_count
        phase_indices, source_phase_times, sampled_times = phase_step_indices(
            bin_count, source_cycle_s, step_dt_s, step_count
        )

        nx, ny, nz = (int(value) for value in matrix)
        magnitude = np.zeros((bin_count, nz, ny, nx), dtype=np.float32)
        velocity = np.zeros(
            (3, bin_count, nz, ny, nx), dtype=np.float32
        )
        speed = np.zeros_like(magnitude)
        mask = np.zeros_like(magnitude, dtype=bool)
        phase_reports = []

        for cardiac_bin, index in enumerate(phase_indices):
            previous = np.asarray(positions[index - 1], dtype=np.float32)
            current = np.asarray(positions[index], dtype=np.float32)
            following = np.asarray(positions[index + 1], dtype=np.float32)
            reset_into_current = np.asarray(resets[index], dtype=bool)
            reset_into_following = np.asarray(resets[index + 1], dtype=bool)
            velocity_valid = ~(reset_into_current | reset_into_following)
            source_velocity = (following - previous) / (2 * step_dt_s)
            points_mri = current[:, axis_order] - source_center_m[axis_order]
            velocity_mri = source_velocity[:, axis_order]
            (
                magnitude[cardiac_bin],
                velocity[:, cardiac_bin],
                speed[cardiac_bin],
                mask[cardiac_bin],
                phase_report,
            ) = voxelize_phase(
                points_mri,
                velocity_mri,
                velocity_valid,
                fov_m,
                matrix,
                args.minimum_particles_per_voxel,
            )
            phase_report.update(
                {
                    "cardiac_bin": cardiac_bin + 1,
                    "source_step_index_zero_based": int(index),
                    "requested_source_phase_time_s": float(
                        source_phase_times[cardiac_bin]
                    ),
                    "sampled_source_time_s": float(
                        sampled_times[cardiac_bin]
                    ),
                    "sequence_phase_center_s": float(
                        (cardiac_bin + 0.5) * sequence_rr_s / bin_count
                    ),
                }
            )
            phase_reports.append(phase_report)

    if not (
        np.isfinite(magnitude).all()
        and np.isfinite(velocity).all()
        and np.isfinite(speed).all()
    ):
        raise RuntimeError("Non-finite preview values were generated")
    if not np.all(mask.reshape(bin_count, -1).any(axis=1)):
        raise RuntimeError("At least one cardiac bin has an empty preview mask")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with h5py.File(temporary, "w") as target:
        target.create_dataset(
            "reference_magnitude", data=magnitude, compression="gzip"
        )
        target.create_dataset("velocity", data=velocity, compression="gzip")
        target.create_dataset("speed", data=speed, compression="gzip")
        target.create_dataset("mask", data=mask, compression="gzip")
        target.create_dataset(
            "source_phase_time_s", data=source_phase_times.astype(np.float32)
        )
        target.create_dataset(
            "sampled_source_time_s", data=sampled_times.astype(np.float32)
        )
        target.attrs["data_kind"] = (
            "tracking_derived_voxel_preview_not_koma_mri"
        )
        target.attrs["dimension_order_python"] = (
            "cardiac_bin,slice,phase,readout"
        )
        target.attrs["FOV_m"] = fov_m.astype(np.float32)
        target.attrs["matrix"] = matrix.astype(np.int32)
        target.attrs["RR_s"] = np.float32(sequence_rr_s)
        target.attrs["source_cycle_s"] = np.float32(source_cycle_s)
        target.attrs["Venc_m_per_s"] = np.float32(
            sequence_report["Venc_m_per_s"]
        )
        target.attrs["n_particles"] = particle_count
        target.attrs["cardiac_bins"] = bin_count
        target.attrs["minimum_particles_per_voxel"] = (
            args.minimum_particles_per_voxel
        )
        target.attrs["velocity_method"] = (
            "centered finite difference of tracked positions; reset-adjacent samples excluded"
        )
        target.attrs["axis_mapping"] = (
            "MRI readout,phase,slice = source Z,X,Y"
        )
    temporary.replace(args.output)

    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TRACKING_DERIVED_MULTIBIN_PREVIEW",
        "scientific_scope": (
            "Particle-density and finite-difference velocity preview sampled "
            "at cardiac phase centers; not KomaMRI signal simulation or reconstruction."
        ),
        "tracks": str(args.tracks.resolve()),
        "motion_report": str(args.motion_report.resolve()),
        "sequence_report": str(args.sequence_report.resolve()),
        "output_hdf5": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "particle_count": particle_count,
        "cardiac_bins": bin_count,
        "source_cycle_s": source_cycle_s,
        "sequence_RR_s": sequence_rr_s,
        "FOV_m": fov_m.tolist(),
        "matrix": matrix.tolist(),
        "voxel_size_mm": (1e3 * fov_m / matrix).tolist(),
        "minimum_particles_per_voxel": args.minimum_particles_per_voxel,
        "phase_reports": phase_reports,
        "limitations": [
            "Voxel magnitude is particle count, not simulated MR magnitude.",
            "Velocity is derived from centered particle displacement, not phase contrast.",
            "Reset-adjacent finite differences are excluded.",
            "The full multi-bin Koma simulation still requires regenerated motion phantom and available GPU resources.",
        ],
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{report['status']}\nPreview: {args.output}\nReport: {report_path}"
    )


if __name__ == "__main__":
    main()
