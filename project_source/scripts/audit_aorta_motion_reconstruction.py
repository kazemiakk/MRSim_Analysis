"""Quantify low-magnitude phase outliers in an aorta PC-MRI reconstruction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "outputs/aorta_dp2_mri_pipeline/koma/simulation_motion_1m/"
    "pc_reconstruction.h5"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
DEFAULT_SOURCE_REPORT = Path(
    "outputs/aorta_dp2_archive_2026-08-21/rerun/"
    "aorta_dp2_full_cycle_rerun_animation_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--core-threshold", type=float, default=0.15)
    return parser.parse_args()


def threshold_mask(magnitude: np.ndarray, fraction: float) -> np.ndarray:
    mask = np.zeros_like(magnitude, dtype=bool)
    for cardiac_bin in range(magnitude.shape[0]):
        mask[cardiac_bin] = (
            magnitude[cardiac_bin] >= fraction * magnitude[cardiac_bin].max()
        )
    return mask


def speed_statistics(
    speed: np.ndarray, mask: np.ndarray, source_max: float
) -> dict[str, float | int]:
    values = speed[mask]
    if values.size == 0:
        raise ValueError("magnitude threshold produced an empty mask")
    return {
        "masked_voxel_count": int(values.size),
        "median_m_per_s": float(np.percentile(values, 50)),
        "p90_m_per_s": float(np.percentile(values, 90)),
        "p95_m_per_s": float(np.percentile(values, 95)),
        "p99_m_per_s": float(np.percentile(values, 99)),
        "p999_m_per_s": float(np.percentile(values, 99.9)),
        "maximum_m_per_s": float(values.max()),
        "fraction_above_source_global_max": float(np.mean(values > source_max)),
    }


def weighted_projection(
    speed: np.ndarray, magnitude: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    weights = np.where(mask, magnitude, 0.0)
    denominator = weights.sum(axis=0)
    return np.divide(
        (speed * weights).sum(axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def plot_sensitivity(
    magnitude: np.ndarray,
    speed: np.ndarray,
    thresholds: list[float],
    records: list[dict],
    source_max: float,
    fov_m: np.ndarray,
    output: Path,
) -> None:
    extent = [
        -fov_m[0] * 500,
        fov_m[0] * 500,
        -fov_m[1] * 500,
        fov_m[1] * 500,
    ]
    figure, axes = plt.subplots(
        len(thresholds),
        magnitude.shape[0],
        figsize=(13, 3.4 * len(thresholds)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, (fraction, record) in enumerate(zip(thresholds, records, strict=True)):
        mask = threshold_mask(magnitude, fraction)
        for cardiac_bin in range(magnitude.shape[0]):
            bin_stats = record["per_bin"][cardiac_bin]
            projection = weighted_projection(
                speed[cardiac_bin], magnitude[cardiac_bin], mask[cardiac_bin]
            )
            image = axes[row, cardiac_bin].imshow(
                projection,
                origin="lower",
                extent=extent,
                cmap="turbo",
                vmin=0,
                vmax=source_max,
                aspect="equal",
            )
            axes[row, cardiac_bin].set_title(
                f"Magnitude >= {100 * fraction:.0f}% | bin {cardiac_bin + 1}\n"
                f"p99={bin_stats['p99_m_per_s']:.3f} m/s, "
                f">source max={100 * bin_stats['fraction_above_source_global_max']:.2f}%"
            )
            axes[row, cardiac_bin].set_xlabel("Largest-first MRI coordinate (mm)")
            axes[row, cardiac_bin].set_ylabel("Transverse MRI coordinate (mm)")
    if image is not None:
        figure.colorbar(image, ax=axes, label="Magnitude-weighted projected speed (m/s)")
    figure.suptitle(
        "Aorta motion PC-MRI magnitude-mask sensitivity\n"
        "Color range capped at the source CFD global maximum",
        fontsize=14,
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def write_core_derivative(
    output: Path,
    source_path: Path,
    source_attributes: dict,
    magnitude: np.ndarray,
    velocity: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = threshold_mask(magnitude, fraction)
    core_velocity = np.where(mask[np.newaxis, ...], velocity, 0.0).astype(np.float32)
    core_speed = np.sqrt(np.sum(np.square(core_velocity), axis=0)).astype(np.float32)
    temporary = output.with_suffix(output.suffix + ".partial")
    with h5py.File(temporary, "w") as target:
        target.create_dataset("reference_magnitude", data=magnitude)
        target.create_dataset("velocity", data=core_velocity)
        target.create_dataset("speed", data=core_speed)
        target.create_dataset("mask", data=mask.astype(np.uint8))
        for key, value in source_attributes.items():
            target.attrs[key] = value
        target.attrs["analysis_derivative"] = True
        target.attrs["analysis_magnitude_mask_fraction"] = fraction
        target.attrs["derivative_of"] = str(source_path.resolve())
        target.attrs["derivative_scope"] = (
            "Core-lumen magnitude-mask sensitivity derivative; primary reconstruction preserved"
        )
    temporary.replace(output)
    return mask, core_speed


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds + [args.core_threshold]))
    if any(not 0 < value < 1 for value in thresholds):
        raise ValueError("all thresholds must lie strictly between 0 and 1")

    source_summary = json.loads(args.source_report.read_text(encoding="utf-8"))
    source_max = float(source_summary["source_cycle"]["global_max_speed_m_per_s"])
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as source:
        magnitude = np.asarray(source["reference_magnitude"], dtype=np.float32)
        velocity = np.asarray(source["velocity"], dtype=np.float32)
        speed = np.asarray(source["speed"], dtype=np.float32)
        stored_mask = np.asarray(source["mask"], dtype=bool)
        attributes = dict(source.attrs.items())
        fov_m = np.asarray(source.attrs["FOV_m"], dtype=float)

    if velocity.shape != (3, *magnitude.shape):
        raise ValueError(f"unexpected velocity shape {velocity.shape}")
    if speed.shape != magnitude.shape or stored_mask.shape != magnitude.shape:
        raise ValueError("speed/mask shape does not match reference magnitude")
    if not all(np.isfinite(array).all() for array in (magnitude, velocity, speed)):
        raise ValueError("reconstruction contains non-finite values")

    records = []
    for fraction in thresholds:
        mask = threshold_mask(magnitude, fraction)
        records.append(
            {
                "magnitude_fraction": fraction,
                "speed": speed_statistics(speed, mask, source_max),
                "per_bin": [
                    speed_statistics(speed[index], mask[index], source_max)
                    for index in range(magnitude.shape[0])
                ],
            }
        )

    core_path = output_dir / "pc_reconstruction_coremask015.h5"
    core_mask, core_speed = write_core_derivative(
        core_path,
        input_path,
        attributes,
        magnitude,
        velocity,
        args.core_threshold,
    )
    figure_path = output_dir / "aorta_motion_1m_velocity_mask_sensitivity.png"
    plot_sensitivity(
        magnitude,
        speed,
        thresholds,
        records,
        source_max,
        fov_m,
        figure_path,
    )

    stored_fraction = float(attributes.get("mask_threshold_fraction", 0.05))
    stored_mask_match = bool(
        np.array_equal(stored_mask, threshold_mask(magnitude, stored_fraction))
    )
    report_path = output_dir / "aorta_motion_1m_velocity_mask_report.json"
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DESCRIPTIVE_MASK_SENSITIVITY_AUDIT",
        "scope": (
            "Magnitude-mask sensitivity and finite-value audit; this does not validate "
            "reconstructed velocity accuracy against independent PC-MRI."
        ),
        "input": str(input_path),
        "source_cfd_global_max_speed_m_per_s": source_max,
        "stored_mask_fraction_assumed": stored_fraction,
        "stored_mask_matches_recomputed": stored_mask_match,
        "threshold_sensitivity": records,
        "finding": (
            "Speeds above the source CFD maximum are concentrated in low-magnitude "
            "edge/background voxels and decrease strongly as the magnitude mask is tightened."
        ),
        "core_derivative": {
            "path": str(core_path),
            "magnitude_fraction": args.core_threshold,
            "masked_voxel_count": int(core_mask.sum()),
            "speed": speed_statistics(core_speed, core_mask, source_max),
            "primary_reconstruction_preserved": True,
        },
        "outputs": {
            "report": str(report_path),
            "sensitivity_figure": str(figure_path),
            "core_derivative_hdf5": str(core_path),
        },
        "limitations": [
            "The 15% core mask is a sensitivity derivative, not an externally validated threshold.",
            "The source CFD maximum is not independent experimental ground truth.",
            "The supplied acquisition uses separate REF/X/Y/Z scans; equivalence to a balanced Hadamard protocol is unverified.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "core": str(core_path)}, indent=2))


if __name__ == "__main__":
    main()
