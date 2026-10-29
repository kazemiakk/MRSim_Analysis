r"""Audit temporal response in an aorta KomaMRI reconstruction.

The audit compares reconstructed cardiac-bin speed summaries with a
tracking-derived preview sampled over the same normalized cardiac cycle. It
does not treat particle tracking as independent MRI ground truth.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "outputs" / "aorta_dp2_mri_pipeline" / "koma"
DEFAULT_RECONSTRUCTION = (
    PIPELINE / "simulation_motion_1m" / "pc_reconstruction.h5"
)
DEFAULT_TRACKING_PREVIEW = (
    PIPELINE
    / "multibin_8x4mm"
    / "tracking_preview"
    / "aorta_tracking_8bin_preview.h5"
)
DEFAULT_OUTPUT_DIR = PIPELINE / "temporal_response_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION
    )
    parser.add_argument(
        "--tracking-preview", type=Path, default=DEFAULT_TRACKING_PREVIEW
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="aorta_koma_temporal_response")
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.15,
        help="Per-bin reference-magnitude fraction used for vessel support.",
    )
    return parser.parse_args()


def scalar_attr(attributes: h5py.AttributeManager, name: str) -> float:
    value = np.asarray(attributes[name])
    return float(value.reshape(-1)[0])


def temporal_relative_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    return float(np.ptp(values) / mean) if mean > 0 else float("nan")


def summarize(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot summarize an empty finite sample")
    return {
        "voxel_count": int(finite.size),
        "mean_m_per_s": float(np.mean(finite)),
        "median_m_per_s": float(np.median(finite)),
        "p90_m_per_s": float(np.quantile(finite, 0.90)),
        "p99_m_per_s": float(np.quantile(finite, 0.99)),
        "maximum_m_per_s": float(np.max(finite)),
    }


def cardiac_bin_indices(
    source_times_s: np.ndarray, source_cycle_s: float, bin_count: int
) -> np.ndarray:
    if source_cycle_s <= 0 or bin_count <= 0:
        raise ValueError("source_cycle_s and bin_count must be positive")
    phase = np.mod(np.asarray(source_times_s, dtype=float), source_cycle_s)
    indices = np.floor(phase * bin_count / source_cycle_s).astype(int)
    return np.clip(indices, 0, bin_count - 1)


def group_tracking_profile(
    source_times_s: np.ndarray,
    values: np.ndarray,
    source_cycle_s: float,
    bin_count: int,
) -> tuple[np.ndarray, list[list[int]]]:
    indices = cardiac_bin_indices(source_times_s, source_cycle_s, bin_count)
    grouped = np.empty(bin_count, dtype=float)
    members: list[list[int]] = []
    for cardiac_bin in range(bin_count):
        selected = np.flatnonzero(indices == cardiac_bin)
        if selected.size == 0:
            raise ValueError(
                f"Tracking preview has no sample in cardiac bin {cardiac_bin + 1}"
            )
        grouped[cardiac_bin] = float(np.mean(values[selected]))
        members.append(selected.astype(int).tolist())
    return grouped, members


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.size < 3 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def normalized_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if mean <= 0:
        raise ValueError("Temporal profile mean must be positive")
    return values / mean


def load_reconstruction(
    path: Path, mask_fraction: float
) -> dict[str, object]:
    with h5py.File(path, "r") as source:
        magnitude = np.asarray(source["reference_magnitude"], dtype=np.float32)
        speed = np.asarray(source["speed"], dtype=np.float32)
        velocity = np.asarray(source["velocity"], dtype=np.float32)
        rr_s = scalar_attr(source.attrs, "RR_s")
        fov_m = np.asarray(source.attrs["FOV_m"], dtype=float)
        matrix = np.asarray(source.attrs["matrix"], dtype=int)

    if magnitude.shape != speed.shape:
        raise ValueError("Magnitude and speed volumes must have identical shapes")
    if velocity.shape[1:] != speed.shape:
        raise ValueError("Velocity must have shape (component, bin, z, y, x)")
    if magnitude.ndim != 4 or magnitude.shape[0] < 2:
        raise ValueError("Expected at least two cardiac bins")
    if not (
        np.isfinite(magnitude).all()
        and np.isfinite(speed).all()
        and np.isfinite(velocity).all()
    ):
        raise ValueError("Reconstruction contains non-finite values")

    masks = np.stack(
        [
            magnitude[index] >= mask_fraction * float(magnitude[index].max())
            for index in range(magnitude.shape[0])
        ]
    )
    if not np.all(masks.reshape(masks.shape[0], -1).any(axis=1)):
        raise ValueError("At least one cardiac-bin mask is empty")
    common_mask = np.all(masks, axis=0)
    union_mask = np.any(masks, axis=0)
    if not common_mask.any():
        raise ValueError("Cardiac bins have no common vessel support")

    per_bin = [summarize(speed[index][masks[index]]) for index in range(speed.shape[0])]
    common_per_bin = [
        summarize(speed[index][common_mask]) for index in range(speed.shape[0])
    ]
    common_speed = speed[:, common_mask]
    common_velocity = velocity[:, :, common_mask]

    return {
        "magnitude": magnitude,
        "speed": speed,
        "velocity": velocity,
        "masks": masks,
        "common_mask": common_mask,
        "union_mask": union_mask,
        "rr_s": rr_s,
        "fov_m": fov_m,
        "matrix": matrix,
        "per_bin": per_bin,
        "common_per_bin": common_per_bin,
        "common_speed": common_speed,
        "common_velocity": common_velocity,
    }


def load_tracking_preview(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as source:
        speed = np.asarray(source["speed"], dtype=np.float32)
        mask = np.asarray(source["mask"], dtype=bool)
        source_times_s = np.asarray(
            source["source_phase_time_s"], dtype=float
        )
        source_cycle_s = scalar_attr(source.attrs, "source_cycle_s")
    if speed.shape != mask.shape or speed.shape[0] != source_times_s.size:
        raise ValueError("Tracking preview datasets have inconsistent shapes")
    per_phase = [summarize(speed[index][mask[index]]) for index in range(speed.shape[0])]
    return {
        "speed": speed,
        "mask": mask,
        "source_times_s": source_times_s,
        "source_cycle_s": source_cycle_s,
        "per_phase": per_phase,
    }


def make_figure(
    output_path: Path,
    reconstruction: dict[str, object],
    tracking: dict[str, object],
    grouped_tracking: np.ndarray,
    mask_fraction: float,
) -> None:
    speed = np.asarray(reconstruction["speed"])
    common_mask = np.asarray(reconstruction["common_mask"])
    rr_s = float(reconstruction["rr_s"])
    source_times_s = np.asarray(tracking["source_times_s"])
    source_cycle_s = float(tracking["source_cycle_s"])
    tracking_medians = np.asarray(
        [item["median_m_per_s"] for item in tracking["per_phase"]]
    )
    koma_medians = np.asarray(
        [item["median_m_per_s"] for item in reconstruction["common_per_bin"]]
    )
    bin_count = speed.shape[0]
    koma_times = (np.arange(bin_count) + 0.5) * rr_s / bin_count
    mapped_tracking_times = source_times_s * rr_s / source_cycle_s

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(
        mapped_tracking_times,
        tracking_medians,
        "o-",
        color="#0f766e",
        lw=2,
        label="Tracking preview",
    )
    axes[0].plot(
        koma_times,
        koma_medians,
        "s-",
        color="#b91c1c",
        lw=2,
        label="KomaMRI",
    )
    axes[0].set_xlabel("Sequence time within RR (s)")
    axes[0].set_ylabel("Median speed (m/s)")
    axes[0].set_title("Native temporal profiles")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(
        np.arange(1, bin_count + 1),
        normalized_profile(grouped_tracking),
        "o-",
        color="#0f766e",
        lw=2,
        label="Tracking averaged in Koma windows",
    )
    axes[1].plot(
        np.arange(1, bin_count + 1),
        normalized_profile(koma_medians),
        "s-",
        color="#b91c1c",
        lw=2,
        label="KomaMRI common-mask median",
    )
    axes[1].axhline(1.0, color="#64748b", lw=1)
    axes[1].set_xticks(np.arange(1, bin_count + 1))
    axes[1].set_xlabel("Cardiac bin")
    axes[1].set_ylabel("Speed normalized by temporal mean")
    axes[1].set_title("Matched acquisition windows")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    temporal_range = np.ptp(speed, axis=0)
    display = np.max(np.where(common_mask, temporal_range, 0.0), axis=0)
    image = axes[2].imshow(
        display,
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=float(np.quantile(display[display > 0], 0.99)),
    )
    axes[2].set_title(
        f"KomaMRI temporal speed range\ncommon {100 * mask_fraction:.0f}% support"
    )
    axes[2].set_xlabel("MRI readout index")
    axes[2].set_ylabel("MRI phase index")
    figure.colorbar(image, ax=axes[2], label="max - min speed (m/s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.mask_fraction < 1:
        raise ValueError("--mask-fraction must be between zero and one")
    reconstruction = load_reconstruction(
        args.reconstruction.resolve(), args.mask_fraction
    )
    tracking = load_tracking_preview(args.tracking_preview.resolve())
    bin_count = np.asarray(reconstruction["speed"]).shape[0]
    tracking_medians = np.asarray(
        [item["median_m_per_s"] for item in tracking["per_phase"]],
        dtype=float,
    )
    grouped_tracking, grouped_members = group_tracking_profile(
        np.asarray(tracking["source_times_s"]),
        tracking_medians,
        float(tracking["source_cycle_s"]),
        bin_count,
    )
    koma_medians = np.asarray(
        [item["median_m_per_s"] for item in reconstruction["common_per_bin"]],
        dtype=float,
    )
    common_speed = np.asarray(reconstruction["common_speed"])
    common_velocity = np.asarray(reconstruction["common_velocity"])

    consecutive_speed_correlations = [
        float(np.corrcoef(common_speed[index], common_speed[index + 1])[0, 1])
        for index in range(bin_count - 1)
    ]
    component_temporal_mae = {}
    for component, name in enumerate(("X", "Y", "Z")):
        differences = np.diff(common_velocity[component], axis=0)
        component_temporal_mae[name] = float(np.mean(np.abs(differences)))

    native_tracking_contrast = temporal_relative_range(tracking_medians)
    grouped_tracking_contrast = temporal_relative_range(grouped_tracking)
    koma_contrast = temporal_relative_range(koma_medians)
    profile_correlation = safe_correlation(grouped_tracking, koma_medians)
    normalized_tracking = normalized_profile(grouped_tracking)
    normalized_koma = normalized_profile(koma_medians)
    normalized_rmse = float(
        np.sqrt(np.mean((normalized_tracking - normalized_koma) ** 2))
    )
    phase_order_comparison_meaningful = bin_count >= 3
    tracking_peak_bin = (
        int(np.argmax(grouped_tracking)) + 1
        if phase_order_comparison_meaningful
        else None
    )
    tracking_trough_bin = (
        int(np.argmin(grouped_tracking)) + 1
        if phase_order_comparison_meaningful
        else None
    )
    koma_peak_bin = (
        int(np.argmax(koma_medians)) + 1
        if phase_order_comparison_meaningful
        else None
    )
    koma_trough_bin = (
        int(np.argmin(koma_medians)) + 1
        if phase_order_comparison_meaningful
        else None
    )
    if bin_count == tracking_medians.size:
        finding = (
            "The KomaMRI reconstruction resolves the same discrete cardiac "
            "phases as the tracking preview. Temporal agreement should be "
            "judged from the reported profile correlation, normalized RMSE, "
            "relative ranges, and peak/trough bins; this is internal simulator "
            "validation rather than independent PC-MRI validation."
        )
    else:
        finding = (
            "The source motion has substantially more temporal contrast at the "
            "native tracking phases than after averaging those samples into the "
            "Koma cardiac windows. A low-bin reconstruction can therefore look "
            "nearly static even when temporal labels and motion are active."
        )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DESCRIPTIVE_TEMPORAL_RESPONSE_AUDIT",
        "scope": (
            "Internal consistency comparison between tracked particle motion and "
            "KomaMRI cardiac bins; not independent PC-MRI accuracy validation."
        ),
        "inputs": {
            "koma_reconstruction": str(args.reconstruction.resolve()),
            "tracking_preview": str(args.tracking_preview.resolve()),
        },
        "koma": {
            "cardiac_bins": bin_count,
            "RR_s": float(reconstruction["rr_s"]),
            "phase_interval_s": float(reconstruction["rr_s"]) / bin_count,
            "matrix": np.asarray(reconstruction["matrix"]).astype(int).tolist(),
            "mask_fraction_of_phase_max": args.mask_fraction,
            "common_mask_voxels": int(
                np.count_nonzero(reconstruction["common_mask"])
            ),
            "union_mask_voxels": int(
                np.count_nonzero(reconstruction["union_mask"])
            ),
            "common_to_union_fraction": float(
                np.count_nonzero(reconstruction["common_mask"])
                / np.count_nonzero(reconstruction["union_mask"])
            ),
            "per_bin_core": reconstruction["per_bin"],
            "per_bin_common_support": reconstruction["common_per_bin"],
            "median_speed_profile_m_per_s": koma_medians.tolist(),
            "median_speed_temporal_relative_range": koma_contrast,
            "consecutive_speed_map_correlations": consecutive_speed_correlations,
            "component_consecutive_bin_mae_m_per_s": component_temporal_mae,
        },
        "tracking_preview": {
            "native_phase_count": int(tracking_medians.size),
            "source_cycle_s": float(tracking["source_cycle_s"]),
            "source_phase_times_s": np.asarray(
                tracking["source_times_s"]
            ).tolist(),
            "native_median_speed_profile_m_per_s": tracking_medians.tolist(),
            "native_median_speed_temporal_relative_range": native_tracking_contrast,
            "samples_grouped_into_koma_bins_zero_based": grouped_members,
            "window_averaged_median_speed_profile_m_per_s": grouped_tracking.tolist(),
            "window_averaged_temporal_relative_range": grouped_tracking_contrast,
        },
        "profile_comparison": {
            "correlation_requires_at_least_three_bins": profile_correlation,
            "temporal_mean_normalized_rmse": normalized_rmse,
            "peak_and_trough_comparison_requires_at_least_three_bins": (
                phase_order_comparison_meaningful
            ),
            "tracking_peak_bin_one_based": tracking_peak_bin,
            "koma_peak_bin_one_based": koma_peak_bin,
            "peak_bin_matches": (
                tracking_peak_bin == koma_peak_bin
                if phase_order_comparison_meaningful
                else None
            ),
            "tracking_trough_bin_one_based": tracking_trough_bin,
            "koma_trough_bin_one_based": koma_trough_bin,
            "trough_bin_matches": (
                tracking_trough_bin == koma_trough_bin
                if phase_order_comparison_meaningful
                else None
            ),
            "koma_to_tracking_relative_range_ratio": float(
                koma_contrast / grouped_tracking_contrast
            )
            if grouped_tracking_contrast > 0
            else None,
            "tracking_native_to_windowed_contrast_ratio": float(
                native_tracking_contrast / grouped_tracking_contrast
            )
            if grouped_tracking_contrast > 0
            else None,
        },
        "finding": finding,
        "limitations": [
            "Tracking-derived voxel speed is an internal reference, not independent measured PC-MRI ground truth.",
            "Window averaging of tracking phase summaries is an approximation to k-space signal formation.",
            "The magnitude threshold defines a sensitivity mask and is not externally validated.",
            "Separate REF/X/Y/Z scans are not verified equivalent to a balanced Hadamard protocol.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.label}_report.json"
    figure_path = args.output_dir / f"{args.label}.png"
    make_figure(
        figure_path,
        reconstruction,
        tracking,
        grouped_tracking,
        args.mask_fraction,
    )
    report["outputs"] = {
        "report": str(report_path.resolve()),
        "figure": str(figure_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
