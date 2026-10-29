r"""Render and audit a scanner-matched dp150 KomaMRI reconstruction.

The default input is the archived 100,000-particle pilot. Pass
``--reconstruction`` explicitly when auditing a production reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from scripts.plot_dp150_koma_publication import (
        choose_cross_sections,
        coordinate_centers,
        coordinate_edges,
        find_throat,
        masked_max_projection,
        support_crop,
        voxelize_particle_support,
        weighted_projection,
    )
except ModuleNotFoundError:
    from plot_dp150_koma_publication import (
        choose_cross_sections,
        coordinate_centers,
        coordinate_edges,
        find_throat,
        masked_max_projection,
        support_crop,
        voxelize_particle_support,
        weighted_projection,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECONSTRUCTION = (
    ROOT
    / "outputs"
    / "dp150_mri_pipeline"
    / "koma"
    / "simulation_motion_100k_scanner_matched_venc5"
    / "pc_reconstruction.h5"
)
DEFAULT_SUPPORT = (
    ROOT / "outputs" / "dp150_mri_pipeline" / "dp150_static_2m_mri.h5"
)
DEFAULT_FOV_M = (0.5, 0.08, 0.08)
DEFAULT_MATRIX = (400, 64, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction",
        type=Path,
        default=DEFAULT_RECONSTRUCTION,
        help=(
            "Input pc_reconstruction.h5. The default is the archived 100k "
            "pilot; pass an explicit path for production analysis."
        ),
    )
    parser.add_argument("--support-h5", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fov-m",
        type=float,
        nargs=3,
        metavar=("READOUT", "PHASE", "SLICE"),
        default=DEFAULT_FOV_M,
    )
    parser.add_argument(
        "--matrix",
        type=int,
        nargs=3,
        metavar=("READOUT", "PHASE", "SLICE"),
        default=DEFAULT_MATRIX,
    )
    parser.add_argument("--frame-duration-ms", type=int, default=420)
    parser.add_argument("--mp4-cycles", type=int, default=2)
    parser.add_argument(
        "--particle-label",
        default=None,
        help=(
            "Particle-count label used in figures and filenames (for example, "
            "100k or 1M). By default it is inferred from the reconstruction path."
        ),
    )
    return parser.parse_args()


def resolve_particle_label(path: Path, explicit: str | None = None) -> tuple[str, str]:
    """Return a display label and filesystem-safe slug for a reconstruction."""
    label = explicit.strip() if explicit is not None else ""
    if not label:
        match = re.search(
            r"(?<![a-z0-9])(\d+(?:\.\d+)?\s*[km])(?=$|[^a-z0-9])",
            str(path).lower(),
        )
        label = match.group(1).replace(" ", "") if match else "unspecified-count"
    suffix = label[-1:].lower()
    if suffix == "k":
        display_label = f"{label[:-1]}k"
    elif suffix == "m":
        display_label = f"{label[:-1]}M"
    else:
        display_label = label
    slug = re.sub(r"[^a-z0-9]+", "_", display_label.lower()).strip("_")
    if not slug:
        raise ValueError("--particle-label must contain at least one letter or digit")
    return display_label, slug


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reconstruction(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        required = ("reference_magnitude", "velocity", "speed", "mask")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing reconstruction datasets: {missing}")
        result: dict[str, Any] = {
            "magnitude": np.asarray(
                source["reference_magnitude"], dtype=np.float32
            ),
            "velocity": np.asarray(source["velocity"], dtype=np.float32),
            "speed": np.asarray(source["speed"], dtype=np.float32),
            "mask": np.asarray(source["mask"], dtype=bool),
            "RR_s": float(source.attrs["RR_s"]),
            "Venc_m_per_s": float(source.attrs["Venc_m_per_s"]),
        }
    magnitude = result["magnitude"]
    if magnitude.ndim != 4:
        raise ValueError(f"Expected magnitude (bin,slice,phase,readout), got {magnitude.shape}")
    if result["velocity"].shape != (3, *magnitude.shape):
        raise ValueError(f"Unexpected velocity shape {result['velocity'].shape}")
    for name in ("speed", "mask"):
        if result[name].shape != magnitude.shape:
            raise ValueError(f"{name} does not match magnitude shape")
    for name in ("magnitude", "velocity", "speed"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    return result


def load_support(
    path: Path, fov_m: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as source:
        if "position" not in source or source["position"].shape[0] < 1:
            raise KeyError("Support HDF5 must contain at least one position frame")
        if not bool(source.attrs.get("static_no_motion", False)):
            raise ValueError("Support HDF5 is not marked static_no_motion")
        points = np.asarray(source["position"][0], dtype=np.float64)
        coordinate_space = str(source.attrs.get("coordinate_space", "unknown"))
        source_order = np.asarray(
            source.attrs.get("source_axis_order_zero_based", [-1, -1, -1]),
            dtype=np.int64,
        )
    support, metadata = voxelize_particle_support(points, fov_m, matrix)
    metadata.update(
        {
            "coordinate_space": coordinate_space,
            "source_axis_order_zero_based": source_order.tolist(),
            "target_fov_m": fov_m.tolist(),
            "target_matrix_readout_phase_slice": matrix.tolist(),
            "target_voxel_size_mm": (1000.0 * fov_m / matrix).tolist(),
        }
    )
    return support, metadata


def masked_colormap(name: str):
    colormap = plt.get_cmap(name).copy()
    colormap.set_bad("white")
    return colormap


def finite_quantile(values: np.ndarray, quantile: float) -> float:
    selected = np.asarray(values)[np.isfinite(values)]
    if not selected.size:
        raise ValueError("Cannot calculate a quantile from an empty array")
    return float(np.quantile(selected, quantile))


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return None
    selected_first = first[valid]
    selected_second = second[valid]
    if np.std(selected_first) == 0 or np.std(selected_second) == 0:
        return None
    return float(np.corrcoef(selected_first, selected_second)[0, 1])


def prepare_analysis(
    reconstruction: dict[str, Any], support: np.ndarray
) -> dict[str, Any]:
    magnitude = reconstruction["magnitude"]
    velocity = reconstruction["velocity"]
    speed = reconstruction["speed"]
    stored_mask = reconstruction["mask"]
    venc = reconstruction["Venc_m_per_s"]
    if magnitude.shape[1:] != support.shape:
        raise ValueError(
            f"Reconstruction shape {magnitude.shape[1:]} and support {support.shape} disagree"
        )

    signal_mask = stored_mask & support[None, ...]
    support_projection = support.any(axis=0)
    magnitude_projections = []
    velocity_projections = []
    speed_projections = []
    metrics = []

    for cardiac_bin in range(magnitude.shape[0]):
        valid = signal_mask[cardiac_bin]
        stored = stored_mask[cardiac_bin]
        magnitude_values = magnitude[cardiac_bin][support]
        speed_values = speed[cardiac_bin][valid]
        component_values = velocity[:, cardiac_bin][:, valid]
        if not speed_values.size:
            raise ValueError(f"Cardiac bin {cardiac_bin + 1} has no valid lumen voxels")

        magnitude_projections.append(
            masked_max_projection(magnitude[cardiac_bin], support, axis=0)
        )
        velocity_projections.append(
            weighted_projection(
                velocity[0, cardiac_bin], magnitude[cardiac_bin], valid, axis=0
            )
        )
        speed_projections.append(
            weighted_projection(
                speed[cardiac_bin], magnitude[cardiac_bin], valid, axis=0
            )
        )

        outside_count = int((stored & ~support).sum())
        metrics.append(
            {
                "cardiac_bin": cardiac_bin + 1,
                "phase_center_s": float(
                    (cardiac_bin + 0.5)
                    * reconstruction["RR_s"]
                    / magnitude.shape[0]
                ),
                "stored_mask_voxels": int(stored.sum()),
                "valid_lumen_voxels": int(valid.sum()),
                "support_coverage_fraction": float(valid.sum() / support.sum()),
                "stored_mask_outside_support_voxels": outside_count,
                "stored_mask_outside_support_fraction": float(
                    outside_count / stored.sum()
                ),
                "magnitude_au": {
                    "mean": float(magnitude_values.mean()),
                    "coefficient_of_variation": float(
                        magnitude_values.std() / magnitude_values.mean()
                    ),
                    "p05": float(np.quantile(magnitude_values, 0.05)),
                    "p95": float(np.quantile(magnitude_values, 0.95)),
                },
                "speed_m_per_s": {
                    "p50": float(np.quantile(speed_values, 0.50)),
                    "p95": float(np.quantile(speed_values, 0.95)),
                    "p99": float(np.quantile(speed_values, 0.99)),
                    "maximum": float(speed_values.max()),
                },
                "readout_velocity_m_per_s": {
                    "median": float(np.median(component_values[0])),
                    "p95_absolute": float(
                        np.quantile(np.abs(component_values[0]), 0.95)
                    ),
                },
                "speed_above_venc_voxels": int((speed_values > venc).sum()),
                "any_component_above_0_9_venc_voxels": int(
                    (np.max(np.abs(component_values), axis=0) > 0.9 * venc).sum()
                ),
            }
        )

    magnitude_projections = np.stack(magnitude_projections)
    velocity_projections = np.stack(velocity_projections)
    speed_projections = np.stack(speed_projections)
    scales = {
        "magnitude_p99_5_au": finite_quantile(magnitude_projections, 0.995),
        "speed_projection_p99_5_m_per_s": finite_quantile(
            speed_projections, 0.995
        ),
        "readout_velocity_projection_p99_5_absolute_m_per_s": finite_quantile(
            np.abs(velocity_projections), 0.995
        ),
    }
    scales["speed_projection_p99_5_m_per_s"] = max(
        scales["speed_projection_p99_5_m_per_s"], 1e-6
    )
    scales["readout_velocity_projection_p99_5_absolute_m_per_s"] = max(
        scales["readout_velocity_projection_p99_5_absolute_m_per_s"], 1e-6
    )

    correlations = [
        safe_correlation(speed_projections[index - 1], speed_projections[index])
        for index in range(1, speed_projections.shape[0])
    ]
    p95_values = np.asarray(
        [item["speed_m_per_s"]["p95"] for item in metrics], dtype=float
    )
    low_index = int(np.argmin(p95_values))
    peak_index = int(np.argmax(p95_values))
    median_target = float(np.median(p95_values))
    intermediate_index = min(
        (
            index
            for index in range(p95_values.size)
            if index not in {low_index, peak_index}
        ),
        key=lambda index: abs(p95_values[index] - median_target),
    )
    representatives = [
        {"role": "Low flow", "index": low_index},
        {"role": "Intermediate flow", "index": intermediate_index},
        {"role": "Peak flow", "index": peak_index},
    ]
    return {
        "signal_mask": signal_mask,
        "support_projection": support_projection,
        "magnitude_projections": magnitude_projections,
        "velocity_projections": velocity_projections,
        "speed_projections": speed_projections,
        "scales": scales,
        "metrics": metrics,
        "consecutive_speed_projection_correlations": correlations,
        "representatives": representatives,
    }


def draw_support_outline(
    axis: plt.Axes,
    support_projection: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> None:
    axis.contour(
        horizontal,
        vertical,
        support_projection.astype(float),
        levels=[0.5],
        colors=["#64748b"],
        linewidths=0.55,
    )


def render_representatives(
    analysis: dict[str, Any],
    readout: np.ndarray,
    phase: np.ndarray,
    row_slice: slice,
    column_slice: slice,
    extent: list[float],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        3, 3, figsize=(18, 8.8), constrained_layout=True, facecolor="white"
    )
    support_projection = analysis["support_projection"][row_slice, column_slice]
    horizontal = readout[column_slice]
    vertical = phase[row_slice]
    scales = analysis["scales"]
    images = [None, None, None]
    column_titles = (
        "Reference magnitude",
        "MRI readout velocity",
        "Speed",
    )
    colormaps = ("Greys", "RdBu_r", "viridis")
    limits = (
        (0.0, scales["magnitude_p99_5_au"]),
        (
            -scales["readout_velocity_projection_p99_5_absolute_m_per_s"],
            scales["readout_velocity_projection_p99_5_absolute_m_per_s"],
        ),
        (0.0, scales["speed_projection_p99_5_m_per_s"]),
    )

    for row, representative in enumerate(analysis["representatives"]):
        index = representative["index"]
        metric = analysis["metrics"][index]
        panels = (
            analysis["magnitude_projections"][index],
            analysis["velocity_projections"][index],
            analysis["speed_projections"][index],
        )
        for column, values in enumerate(panels):
            axis = axes[row, column]
            cropped = values[row_slice, column_slice]
            images[column] = axis.imshow(
                np.ma.masked_invalid(cropped),
                cmap=masked_colormap(colormaps[column]),
                origin="lower",
                extent=extent,
                vmin=limits[column][0],
                vmax=limits[column][1],
                interpolation="nearest",
                aspect="auto",
            )
            draw_support_outline(axis, support_projection, horizontal, vertical)
            axis.grid(False)
            axis.tick_params(labelsize=8, colors="#334155")
            if row == 0:
                axis.set_title(column_titles[column], fontsize=12, color="#172033")
            if column == 0:
                axis.set_ylabel(
                    f"{representative['role']}\nBin {index + 1}, "
                    f"t={1000 * metric['phase_center_s']:.0f} ms\nSource X (mm)",
                    fontsize=9,
                )
            if row == 2:
                axis.set_xlabel("MRI readout / source Z (mm)", fontsize=9)

    labels = ("Normalized a.u.", "m/s", "m/s")
    for column, image in enumerate(images):
        figure.colorbar(
            image,
            ax=axes[:, column],
            orientation="horizontal",
            shrink=0.78,
            pad=0.04,
            label=labels[column],
        )
    figure.suptitle(
        "dp150 scanner-matched KomaMRI: representative cardiac bins",
        fontsize=17,
        color="#172033",
    )
    figure.text(
        0.5,
        0.002,
        "Exact seed-derived lumen support; magnitude-weighted through-slice velocity. "
        "No data smoothing; transverse display scale is expanded.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_all_bins(
    analysis: dict[str, Any],
    readout: np.ndarray,
    phase: np.ndarray,
    row_slice: slice,
    column_slice: slice,
    extent: list[float],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        5, 3, figsize=(18, 11), constrained_layout=True, facecolor="white"
    )
    support_projection = analysis["support_projection"][row_slice, column_slice]
    horizontal = readout[column_slice]
    vertical = phase[row_slice]
    speed_limit = analysis["scales"]["speed_projection_p99_5_m_per_s"]
    image = None
    for index, axis in enumerate(axes.flat):
        values = analysis["speed_projections"][index][row_slice, column_slice]
        image = axis.imshow(
            np.ma.masked_invalid(values),
            cmap=masked_colormap("viridis"),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=speed_limit,
            interpolation="nearest",
            aspect="auto",
        )
        draw_support_outline(axis, support_projection, horizontal, vertical)
        metric = analysis["metrics"][index]
        axis.set_title(
            f"Bin {index + 1} | {1000 * metric['phase_center_s']:.0f} ms | "
            f"p95={metric['speed_m_per_s']['p95']:.2f} m/s",
            fontsize=9,
            color="#172033",
        )
        axis.grid(False)
        axis.tick_params(labelsize=7, colors="#334155")
        if index // 3 == 4:
            axis.set_xlabel("Source Z (mm)", fontsize=8)
        if index % 3 == 0:
            axis.set_ylabel("Source X (mm)", fontsize=8)
    assert image is not None
    figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.55,
        pad=0.025,
        label="Magnitude-weighted speed (m/s)",
    )
    figure.suptitle(
        "dp150 scanner-matched KomaMRI: all 15 reconstructed cardiac bins",
        fontsize=17,
        color="#172033",
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_cross_sections(
    reconstruction: dict[str, Any],
    support: np.ndarray,
    analysis: dict[str, Any],
    readout: np.ndarray,
    phase: np.ndarray,
    slice_: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    peak_index = next(
        item["index"]
        for item in analysis["representatives"]
        if item["role"] == "Peak flow"
    )
    speed_limit = max(
        analysis["scales"]["speed_projection_p99_5_m_per_s"],
        analysis["metrics"][peak_index]["speed_m_per_s"]["p99"],
    )
    throat_index, support_area = find_throat(support)
    section_indices = choose_cross_sections(throat_index, readout, support_area)
    combined_support = np.logical_or.reduce(
        [support[:, :, index] for index in section_indices]
    )
    rows, columns = np.where(combined_support)
    row0, row1 = max(0, rows.min() - 2), min(support.shape[0], rows.max() + 3)
    col0, col1 = max(0, columns.min() - 2), min(support.shape[1], columns.max() + 3)
    slice_edges = coordinate_edges(slice_)
    phase_edges = coordinate_edges(phase)
    extent = [
        phase_edges[col0],
        phase_edges[col1],
        slice_edges[row0],
        slice_edges[row1],
    ]

    figure, axes = plt.subplots(
        1, 4, figsize=(15, 4.4), constrained_layout=True, facecolor="white"
    )
    image = None
    for axis, index in zip(axes, section_indices, strict=True):
        valid = analysis["signal_mask"][peak_index, :, :, index]
        values = reconstruction["speed"][peak_index, :, :, index]
        displayed = np.where(valid, values, np.nan)[row0:row1, col0:col1]
        local_support = support[row0:row1, col0:col1, index]
        image = axis.imshow(
            np.ma.masked_invalid(displayed),
            cmap=masked_colormap("viridis"),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=speed_limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.contour(
            phase[col0:col1],
            slice_[row0:row1],
            local_support.astype(float),
            levels=[0.5],
            colors=["#64748b"],
            linewidths=0.7,
        )
        axis.set_title(f"Z = {readout[index]:.0f} mm", fontsize=11)
        axis.set_xlabel("Source X (mm)")
        axis.grid(False)
        axis.tick_params(labelsize=8, colors="#334155")
    axes[0].set_ylabel("Source Y (mm)")
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.8, pad=0.02, label="Speed (m/s)")
    figure.suptitle(
        f"Peak-flow cross-sections | bin {peak_index + 1}",
        fontsize=16,
        color="#172033",
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return {
        "peak_cardiac_bin": peak_index + 1,
        "display_speed_limit_m_per_s": float(speed_limit),
        "throat_readout_index": throat_index,
        "throat_source_Z_mm": float(readout[throat_index]),
        "section_readout_indices": section_indices,
        "section_source_Z_mm": [float(readout[index]) for index in section_indices],
    }


def render_temporal_summary(
    analysis: dict[str, Any], output: Path, particle_label: str
) -> None:
    metrics = analysis["metrics"]
    times_ms = np.asarray([1000 * item["phase_center_s"] for item in metrics])
    p50 = np.asarray([item["speed_m_per_s"]["p50"] for item in metrics])
    p95 = np.asarray([item["speed_m_per_s"]["p95"] for item in metrics])
    p99 = np.asarray([item["speed_m_per_s"]["p99"] for item in metrics])
    coverage = 100 * np.asarray(
        [item["support_coverage_fraction"] for item in metrics]
    )
    leakage = 100 * np.asarray(
        [item["stored_mask_outside_support_fraction"] for item in metrics]
    )
    above_venc = np.asarray([item["speed_above_venc_voxels"] for item in metrics])
    near_limit = np.asarray(
        [item["any_component_above_0_9_venc_voxels"] for item in metrics]
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(13, 8), constrained_layout=True, facecolor="white"
    )
    axes[0, 0].plot(times_ms, p50, marker="o", label="Median")
    axes[0, 0].plot(times_ms, p95, marker="o", label="p95")
    axes[0, 0].plot(times_ms, p99, marker="o", label="p99")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].set_title("Lumen speed percentiles")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(times_ms, coverage, marker="o", color="#167d75")
    axes[0, 1].set_ylabel("Support represented (%)")
    axes[0, 1].set_title("Stored signal-mask coverage inside lumen")

    axes[1, 0].plot(times_ms, leakage, marker="o", color="#c43c5b")
    axes[1, 0].set_ylabel("Stored mask outside lumen (%)")
    axes[1, 0].set_title("Magnitude-mask leakage")

    width = 0.32 * np.median(np.diff(times_ms))
    axes[1, 1].bar(
        times_ms - width / 2,
        above_venc,
        width=width,
        label="Vector speed > VENC",
        color="#d97706",
    )
    axes[1, 1].bar(
        times_ms + width / 2,
        near_limit,
        width=width,
        label="Any component > 0.9 VENC",
        color="#7c3aed",
    )
    axes[1, 1].set_ylabel("Voxel count")
    axes[1, 1].set_title("Near-limit phase outliers")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Cardiac phase center (ms)")
        axis.grid(True, color="#d8dee8", linewidth=0.6, alpha=0.7)
        axis.tick_params(colors="#334155")
        for spine in axis.spines.values():
            spine.set_color("#94a3b8")
    figure.suptitle(
        f"dp150 {particle_label} reconstruction: temporal and mask-quality audit",
        fontsize=16,
        color="#172033",
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_cine_frame(
    analysis: dict[str, Any],
    index: int,
    readout: np.ndarray,
    phase: np.ndarray,
    row_slice: slice,
    column_slice: slice,
    extent: list[float],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(16, 5.2), constrained_layout=True, facecolor="white"
    )
    support_projection = analysis["support_projection"][row_slice, column_slice]
    horizontal = readout[column_slice]
    vertical = phase[row_slice]
    scales = analysis["scales"]
    panels = (
        analysis["magnitude_projections"][index],
        analysis["velocity_projections"][index],
        analysis["speed_projections"][index],
    )
    titles = ("Reference magnitude", "Readout velocity", "Speed")
    maps = ("Greys", "RdBu_r", "viridis")
    limits = (
        (0.0, scales["magnitude_p99_5_au"]),
        (
            -scales["readout_velocity_projection_p99_5_absolute_m_per_s"],
            scales["readout_velocity_projection_p99_5_absolute_m_per_s"],
        ),
        (0.0, scales["speed_projection_p99_5_m_per_s"]),
    )
    for axis, values, title, cmap, limit in zip(
        axes, panels, titles, maps, limits, strict=True
    ):
        image = axis.imshow(
            np.ma.masked_invalid(values[row_slice, column_slice]),
            cmap=masked_colormap(cmap),
            origin="lower",
            extent=extent,
            vmin=limit[0],
            vmax=limit[1],
            interpolation="nearest",
            aspect="auto",
        )
        draw_support_outline(axis, support_projection, horizontal, vertical)
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Source Z (mm)")
        axis.set_ylabel("Source X (mm)")
        axis.grid(False)
        axis.tick_params(labelsize=8)
        figure.colorbar(image, ax=axis, shrink=0.68, pad=0.02)
    metric = analysis["metrics"][index]
    figure.suptitle(
        f"dp150 KomaMRI | cardiac bin {index + 1}/15 | "
        f"phase center {1000 * metric['phase_center_s']:.0f} ms",
        fontsize=16,
        color="#172033",
    )
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def save_gif(frame_paths: list[Path], output: Path, duration_ms: int) -> None:
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    paletted = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
        for frame in frames
    ]
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"probe_available": False}
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    container = payload.get("format", {})
    frame_count = stream.get("nb_frames")
    return {
        "probe_available": True,
        "dimensions_px": [int(stream["width"]), int(stream["height"])],
        "average_frame_rate": stream.get("avg_frame_rate"),
        "frames": int(frame_count) if frame_count not in (None, "N/A") else None,
        "duration_s": float(container.get("duration", stream.get("duration"))),
        "bytes": int(container.get("size", path.stat().st_size)),
    }


def save_mp4(
    frame_paths: list[Path],
    output: Path,
    duration_ms: int,
    cycles: int,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"created": False, "reason": "ffmpeg was not found"}

    concat_path = output.with_suffix(".frames.txt")
    duration_s = duration_ms / 1000.0
    lines = []
    for _ in range(cycles):
        for frame in frame_paths:
            normalized = frame.resolve().as_posix().replace("'", "'\\''")
            lines.extend((f"file '{normalized}'", f"duration {duration_s:.6f}"))
    normalized = frame_paths[-1].resolve().as_posix().replace("'", "'\\''")
    lines.append(f"file '{normalized}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    expected_duration_s = len(frame_paths) * duration_s * cycles
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-t",
        f"{expected_duration_s:.6f}",
        "-vf",
        "fps=24,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return {
        "created": True,
        "path": str(output),
        "sha256": sha256_file(output),
        "source_unique_bins": len(frame_paths),
        "cycles": cycles,
        "frame_hold_ms": duration_ms,
        "expected_duration_s": expected_duration_s,
        "temporal_interpolation": False,
        "ffmpeg_command": command,
        "ffmpeg_stderr_tail": completed.stderr.splitlines()[-4:],
        **probe_video(output),
    }


def image_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "path": str(path),
            "dimensions_px": list(image.size),
            "frames": int(getattr(image, "n_frames", 1)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def main() -> None:
    args = parse_args()
    reconstruction_path = args.reconstruction.resolve()
    support_path = args.support_h5.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else reconstruction_path.parent / "final_image_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "cine_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    fov_m = np.asarray(args.fov_m, dtype=np.float64)
    matrix = np.asarray(args.matrix, dtype=np.int64)
    if fov_m.shape != (3,) or matrix.shape != (3,):
        raise ValueError("FOV and matrix must each contain three values")
    if np.any(fov_m <= 0) or np.any(matrix <= 0):
        raise ValueError("FOV and matrix values must be positive")
    if args.frame_duration_ms < 20:
        raise ValueError("--frame-duration-ms must be at least 20")
    if args.mp4_cycles < 1:
        raise ValueError("--mp4-cycles must be at least 1")

    reconstruction = load_reconstruction(reconstruction_path)
    particle_label, particle_slug = resolve_particle_label(
        reconstruction_path, args.particle_label
    )
    expected_shape = tuple(int(value) for value in matrix[::-1])
    if reconstruction["magnitude"].shape[1:] != expected_shape:
        raise ValueError(
            f"Reconstruction spatial shape {reconstruction['magnitude'].shape[1:]} "
            f"does not match requested matrix {matrix.tolist()}"
        )
    support, support_metadata = load_support(support_path, fov_m, matrix)
    analysis = prepare_analysis(reconstruction, support)
    readout, phase, slice_ = coordinate_centers(support.shape, fov_m * 1000.0)
    row_slice, column_slice, extent = support_crop(
        analysis["support_projection"], readout, phase, pad_voxels=3
    )

    bin_count = int(reconstruction["magnitude"].shape[0])
    artifact_prefix = f"dp150_{particle_slug}"
    representative_path = output_dir / f"{artifact_prefix}_representative_bins.png"
    all_bins_path = output_dir / f"{artifact_prefix}_all_{bin_count}_bins_speed.png"
    cross_sections_path = output_dir / f"{artifact_prefix}_peak_cross_sections.png"
    temporal_path = output_dir / f"{artifact_prefix}_temporal_quality.png"
    gif_path = output_dir / f"{artifact_prefix}_all_{bin_count}_bins.gif"
    mp4_path = output_dir / f"{artifact_prefix}_all_{bin_count}_bins.mp4"
    render_representatives(
        analysis,
        readout,
        phase,
        row_slice,
        column_slice,
        extent,
        representative_path,
    )
    render_all_bins(
        analysis,
        readout,
        phase,
        row_slice,
        column_slice,
        extent,
        all_bins_path,
    )
    cross_section_metadata = render_cross_sections(
        reconstruction,
        support,
        analysis,
        readout,
        phase,
        slice_,
        cross_sections_path,
    )
    render_temporal_summary(analysis, temporal_path, particle_label)

    frame_paths = []
    for index in range(reconstruction["magnitude"].shape[0]):
        frame_path = frame_dir / f"frame_{index + 1:03d}.png"
        render_cine_frame(
            analysis,
            index,
            readout,
            phase,
            row_slice,
            column_slice,
            extent,
            frame_path,
        )
        frame_paths.append(frame_path)
    save_gif(frame_paths, gif_path, args.frame_duration_ms)
    video = save_mp4(
        frame_paths,
        mp4_path,
        args.frame_duration_ms,
        args.mp4_cycles,
    )

    outputs = [
        image_record(path)
        for path in (
            representative_path,
            all_bins_path,
            cross_sections_path,
            temporal_path,
            gif_path,
        )
    ]
    metrics = analysis["metrics"]
    all_valid_speed = reconstruction["speed"][analysis["signal_mask"]]
    report = {
        "schema_version": 1,
        "status": "PASS_WITH_QUALITATIVE_LIMITATIONS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "support-constrained display and numerical audit of the completed "
            f"{particle_label} reconstruction"
        ),
        "inputs": {
            "reconstruction": {
                "path": str(reconstruction_path),
                "sha256": sha256_file(reconstruction_path),
            },
            "seed_derived_support": {
                "path": str(support_path),
                "sha256": sha256_file(support_path),
            },
        },
        "grid": {
            "FOV_m": fov_m.tolist(),
            "matrix_readout_phase_slice": matrix.tolist(),
            "array_order_python": ["cardiac_bin", "slice", "phase", "readout"],
            "voxel_size_mm": (1000.0 * fov_m / matrix).tolist(),
        },
        "acquisition": {
            "particle_count_label": particle_label,
            "cardiac_bins": int(reconstruction["magnitude"].shape[0]),
            "RR_s": reconstruction["RR_s"],
            "Venc_per_component_m_per_s": reconstruction["Venc_m_per_s"],
        },
        "support": support_metadata,
        "display": {
            "magnitude_projection": "maximum within exact seed-derived support",
            "velocity_and_speed_projection": "magnitude-weighted through-slice mean within stored signal mask AND exact support",
            "data_smoothing": False,
            "image_interpolation": "nearest",
            "temporal_interpolation": False,
            "transverse_scale_expanded": True,
            "shared_scales": analysis["scales"],
        },
        "global_metrics": {
            "support_voxels": int(support.sum()),
            "valid_lumen_voxels_all_bins": int(analysis["signal_mask"].sum()),
            "speed_p50_m_per_s": float(np.quantile(all_valid_speed, 0.50)),
            "speed_p95_m_per_s": float(np.quantile(all_valid_speed, 0.95)),
            "speed_p99_m_per_s": float(np.quantile(all_valid_speed, 0.99)),
            "speed_maximum_m_per_s": float(all_valid_speed.max()),
            "speed_above_venc_voxels": int(
                (all_valid_speed > reconstruction["Venc_m_per_s"]).sum()
            ),
            "stored_mask_outside_support_fraction_range": [
                min(item["stored_mask_outside_support_fraction"] for item in metrics),
                max(item["stored_mask_outside_support_fraction"] for item in metrics),
            ],
            "support_coverage_fraction_range": [
                min(item["support_coverage_fraction"] for item in metrics),
                max(item["support_coverage_fraction"] for item in metrics),
            ],
            "magnitude_mean_au_range": [
                min(item["magnitude_au"]["mean"] for item in metrics),
                max(item["magnitude_au"]["mean"] for item in metrics),
            ],
            "magnitude_coefficient_of_variation_range": [
                min(
                    item["magnitude_au"]["coefficient_of_variation"]
                    for item in metrics
                ),
                max(
                    item["magnitude_au"]["coefficient_of_variation"]
                    for item in metrics
                ),
            ],
            "consecutive_speed_projection_correlations": analysis[
                "consecutive_speed_projection_correlations"
            ],
        },
        "representative_bins": [
            {
                "role": item["role"],
                "cardiac_bin": item["index"] + 1,
                "phase_center_s": metrics[item["index"]]["phase_center_s"],
                "lumen_speed_p95_m_per_s": metrics[item["index"]][
                    "speed_m_per_s"
                ]["p95"],
            }
            for item in analysis["representatives"]
        ],
        "cross_sections": cross_section_metadata,
        "per_bin_metrics": metrics,
        "outputs": outputs,
        "video": video,
        "scientific_limits": [
            "The figures are reconstructed KomaMRI displays, not independent experimental validation.",
            "The seed-derived support changes display masking only; the source reconstruction is unchanged.",
            "Vector speed may exceed a per-component VENC without proving component aliasing.",
            f"The {bin_count} cardiac bins are discrete; no temporal frames were interpolated.",
            "Particle-density convergence against a higher-count reconstruction has not been established.",
        ],
    }
    report_path = output_dir / f"{artifact_prefix}_final_image_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "PASS_WITH_QUALITATIVE_LIMITATIONS\n"
        f"Representative figure: {representative_path}\n"
        f"All-bin figure: {all_bins_path}\n"
        f"GIF: {gif_path}\n"
        f"MP4: {mp4_path if video['created'] else video['reason']}\n"
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
