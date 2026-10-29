r"""Create support-constrained, publication-quality dp150 KomaMRI figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import binary_fill_holes, label as connected_components


DEFAULT_MOTION = Path(
    "outputs/dp150_mri_pipeline/koma/"
    "simulation_motion_1m/pc_reconstruction.h5"
)
DEFAULT_STATIC = Path(
    "outputs/dp150_mri_pipeline/koma/"
    "simulation_static_2m_ref/pc_reconstruction.h5"
)
DEFAULT_SUPPORT = Path("outputs/dp150_mri_pipeline/dp150_static_2m_mri.h5")
DEFAULT_OUTPUT = Path(
    "outputs/dp150_mri_pipeline/koma/publication_figures_support_corrected"
)
LEGACY_PRESENTATION_FILES = (
    Path(
        "outputs/dp150_mri_pipeline/koma/simulation_motion_1m/"
        "motion_pc_velocity_projections.png"
    ),
    Path(
        "outputs/dp150_mri_pipeline/koma/simulation_motion_1m/"
        "pc_reconstruction_qc.png"
    ),
    Path(
        "outputs/dp150_mri_pipeline/koma/simulation_motion_1m/"
        "motion_bin1_projection_slices.png"
    ),
    Path(
        "outputs/dp150_mri_pipeline/koma/simulation_motion_1m/"
        "motion_bin2_projection_slices.png"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--support-h5", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reconstruction(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        required = ("reference_magnitude", "velocity", "speed", "mask")
        missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(f"Missing reconstruction datasets in {path}: {missing}")
        data: dict[str, Any] = {
            "magnitude": np.asarray(h5["reference_magnitude"], dtype=np.float32),
            "velocity": np.asarray(h5["velocity"], dtype=np.float32),
            "speed": np.asarray(h5["speed"], dtype=np.float32),
            "mask": np.asarray(h5["mask"], dtype=bool),
            "RR_s": float(h5.attrs["RR_s"]),
            "Venc_m_per_s": float(h5.attrs["Venc_m_per_s"]),
        }
    magnitude = data["magnitude"]
    if magnitude.ndim != 4:
        raise ValueError(
            "Expected magnitude order (bin, slice, phase, readout), "
            f"got {magnitude.shape}"
        )
    if data["velocity"].shape != (3, *magnitude.shape):
        raise ValueError(f"Unexpected velocity shape {data['velocity'].shape}")
    if data["speed"].shape != magnitude.shape or data["mask"].shape != magnitude.shape:
        raise ValueError("Speed or mask shape does not match the magnitude volume")
    for name in ("magnitude", "velocity", "speed"):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"{name} contains non-finite values in {path}")
    return data


def voxelize_particle_support(
    points_mri_m: np.ndarray,
    fov_m: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Voxelize MRI-ordered points into Python (slice, phase, readout) order."""
    points = np.asarray(points_mri_m, dtype=np.float64)
    fov = np.asarray(fov_m, dtype=np.float64)
    shape_rps = np.asarray(matrix, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if fov.shape != (3,) or shape_rps.shape != (3,) or np.any(shape_rps <= 0):
        raise ValueError("FOV and matrix must each contain three positive values")

    indices = np.floor((points + 0.5 * fov) / (fov / shape_rps)).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < shape_rps), axis=1)
    support = np.zeros(
        (int(shape_rps[2]), int(shape_rps[1]), int(shape_rps[0])), dtype=bool
    )
    selected = indices[valid]
    support[selected[:, 2], selected[:, 1], selected[:, 0]] = True
    raw_voxels = int(support.sum())

    for readout_index in range(support.shape[2]):
        support[:, :, readout_index] = binary_fill_holes(
            support[:, :, readout_index]
        )

    _, component_count = connected_components(
        support, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )

    return support, {
        "input_particles": int(points.shape[0]),
        "particles_inside_fov": int(valid.sum()),
        "particles_outside_fov": int((~valid).sum()),
        "raw_occupied_voxels": raw_voxels,
        "filled_support_voxels": int(support.sum()),
        "connected_components_after_transverse_hole_fill": int(component_count),
        "occupied_components_removed": 0,
        "support_processing": (
            "voxel occupancy; enclosed holes filled independently in each "
            "transverse plane; every occupied component retained"
        ),
        "dilation_or_smoothing_applied": False,
    }


def load_particle_support(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as h5:
        if "position" not in h5:
            raise KeyError(f"Missing position dataset in {path}")
        if h5["position"].shape[0] < 1:
            raise ValueError(f"No particle frames in {path}")
        points = np.asarray(h5["position"][0], dtype=np.float64)
        fov_m = np.asarray(h5.attrs["fov_m"], dtype=np.float64)
        matrix = np.asarray(h5.attrs["matrix"], dtype=np.int64)
        source_order = np.asarray(
            h5.attrs.get("source_axis_order_zero_based", [-1, -1, -1]),
            dtype=np.int64,
        )
        static_no_motion = bool(h5.attrs.get("static_no_motion", False))
    if not static_no_motion:
        raise ValueError("Support source is not marked static_no_motion")
    support, metadata = voxelize_particle_support(points, fov_m, matrix)
    metadata.update(
        {
            "coordinate_space": "MRI centered",
            "source_axis_order_zero_based": source_order.tolist(),
            "fov_m": fov_m.tolist(),
            "matrix_readout_phase_slice": matrix.tolist(),
        }
    )
    return support, fov_m * 1000.0, matrix, metadata


def coordinate_centers(
    shape_spr: tuple[int, int, int], fov_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ns, np_, nr = shape_spr
    readout = -fov_mm[0] / 2 + (np.arange(nr) + 0.5) * fov_mm[0] / nr
    phase = -fov_mm[1] / 2 + (np.arange(np_) + 0.5) * fov_mm[1] / np_
    slice_ = -fov_mm[2] / 2 + (np.arange(ns) + 0.5) * fov_mm[2] / ns
    return readout, phase, slice_


def coordinate_edges(centers: np.ndarray) -> np.ndarray:
    if centers.size < 2:
        raise ValueError("At least two coordinate centers are required")
    spacing = float(centers[1] - centers[0])
    return np.concatenate(
        ([centers[0] - spacing / 2], centers + spacing / 2)
    )


def support_crop(
    support_projection: np.ndarray,
    readout: np.ndarray,
    transverse: np.ndarray,
    pad_voxels: int,
) -> tuple[slice, slice, list[float]]:
    rows, columns = np.where(support_projection)
    if rows.size == 0:
        raise ValueError("Support projection is empty")
    row0 = max(0, int(rows.min()) - pad_voxels)
    row1 = min(support_projection.shape[0], int(rows.max()) + pad_voxels + 1)
    col0 = max(0, int(columns.min()) - pad_voxels)
    col1 = min(support_projection.shape[1], int(columns.max()) + pad_voxels + 1)
    x_edges = coordinate_edges(readout)
    y_edges = coordinate_edges(transverse)
    return (
        slice(row0, row1),
        slice(col0, col1),
        [x_edges[col0], x_edges[col1], y_edges[row0], y_edges[row1]],
    )


def masked_max_projection(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    if values.shape != mask.shape:
        raise ValueError("Values and mask must have identical shapes")
    projected = np.where(mask, values, -np.inf).max(axis=axis)
    valid = mask.any(axis=axis)
    projected = projected.astype(np.float64, copy=False)
    projected[~valid] = np.nan
    return projected


def weighted_projection(
    values: np.ndarray,
    magnitude: np.ndarray,
    mask: np.ndarray,
    axis: int,
) -> np.ndarray:
    if values.shape != magnitude.shape or values.shape != mask.shape:
        raise ValueError("Values, magnitude, and mask must have identical shapes")
    weights = np.where(mask, magnitude, 0.0).astype(np.float64, copy=False)
    denominator = weights.sum(axis=axis)
    numerator = (np.where(mask, values, 0.0) * weights).sum(axis=axis)
    projected = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=projected, where=denominator > 0)
    return projected


def robust_quantile(values: np.ndarray, mask: np.ndarray, quantile: float) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    return float(np.quantile(selected, quantile)) if selected.size else 1.0


def style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.tick_params(colors="#334155", labelsize=8, length=3, width=0.7)
    for spine in axis.spines.values():
        spine.set_color("#64748b")
        spine.set_linewidth(0.7)


def add_support_outline(
    axis: plt.Axes,
    support_image: np.ndarray,
    extent: list[float],
    color: str = "#475569",
) -> None:
    rows, columns = support_image.shape
    x = extent[0] + (np.arange(columns) + 0.5) * (
        extent[1] - extent[0]
    ) / columns
    y = extent[2] + (np.arange(rows) + 0.5) * (
        extent[3] - extent[2]
    ) / rows
    if support_image.any() and not support_image.all():
        axis.contour(
            x,
            y,
            support_image.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=0.65,
            alpha=0.9,
        )


def prepare_longitudinal_panels(
    reconstruction: dict[str, Any],
    support: np.ndarray,
) -> tuple[list[list[np.ndarray]], np.ndarray, dict[str, float]]:
    magnitude = reconstruction["magnitude"]
    velocity = reconstruction["velocity"]
    speed = reconstruction["speed"]
    stored_mask = reconstruction["mask"]
    signal_mask = stored_mask & support[None, ...]

    magnitude_limit = max(
        robust_quantile(magnitude, np.broadcast_to(support, magnitude.shape), 0.995),
        1e-8,
    )
    velocity_limit = max(robust_quantile(np.abs(velocity[0]), signal_mask, 0.995), 1e-6)
    speed_limit = max(robust_quantile(speed, signal_mask, 0.995), 1e-6)

    panels: list[list[np.ndarray]] = []
    for cardiac_bin in range(magnitude.shape[0]):
        mag_projection = masked_max_projection(
            magnitude[cardiac_bin], support, axis=0
        )
        mag_projection = np.clip(mag_projection / magnitude_limit, 0.0, 1.0)
        readout_velocity = weighted_projection(
            velocity[0, cardiac_bin],
            magnitude[cardiac_bin],
            signal_mask[cardiac_bin],
            axis=0,
        )
        projected_speed = weighted_projection(
            speed[cardiac_bin],
            magnitude[cardiac_bin],
            signal_mask[cardiac_bin],
            axis=0,
        )
        panels.append([mag_projection, readout_velocity, projected_speed])
    return panels, signal_mask, {
        "magnitude_common_p99_5": magnitude_limit,
        "absolute_readout_velocity_common_p99_5_m_per_s": velocity_limit,
        "speed_common_p99_5_m_per_s": speed_limit,
    }


def render_longitudinal(
    panels: list[list[np.ndarray]],
    support: np.ndarray,
    readout: np.ndarray,
    phase: np.ndarray,
    scales: dict[str, float],
    path: Path,
    zoom: bool,
    throat_index: int,
) -> None:
    support_projection = support.any(axis=0)
    row_slice, col_slice, full_extent = support_crop(
        support_projection, readout, phase, pad_voxels=3
    )
    if zoom:
        half_width_voxels = max(1, int(round(45.0 / (readout[1] - readout[0]))))
        col0 = max(0, throat_index - half_width_voxels)
        col1 = min(readout.size, throat_index + half_width_voxels + 1)
        occupied_rows = np.where(support_projection[:, col0:col1].any(axis=1))[0]
        row0 = max(0, int(occupied_rows.min()) - 3)
        row1 = min(phase.size, int(occupied_rows.max()) + 4)
        row_slice, col_slice = slice(row0, row1), slice(col0, col1)
        x_edges, y_edges = coordinate_edges(readout), coordinate_edges(phase)
        extent = [x_edges[col0], x_edges[col1], y_edges[row0], y_edges[row1]]
    else:
        extent = full_extent

    magnitude_cmap = plt.get_cmap("gray_r").copy()
    velocity_cmap = plt.get_cmap("RdBu_r").copy()
    speed_cmap = plt.get_cmap("viridis").copy()
    for cmap in (magnitude_cmap, velocity_cmap, speed_cmap):
        cmap.set_bad("white")

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "semibold"}):
        fig, axes = plt.subplots(
            len(panels),
            3,
            figsize=(14.2, 5.2 if not zoom else 6.2),
            constrained_layout=True,
            squeeze=False,
        )
        images: list[Any] = [None, None, None]
        for cardiac_bin, row in enumerate(panels):
            display_values = [
                (row[0], magnitude_cmap, 0.0, 1.0),
                (
                    row[1],
                    velocity_cmap,
                    -scales["absolute_readout_velocity_common_p99_5_m_per_s"],
                    scales["absolute_readout_velocity_common_p99_5_m_per_s"],
                ),
                (
                    row[2],
                    speed_cmap,
                    0.0,
                    scales["speed_common_p99_5_m_per_s"],
                ),
            ]
            for column, (values, cmap, vmin, vmax) in enumerate(display_values):
                axis = axes[cardiac_bin, column]
                images[column] = axis.imshow(
                    values[row_slice, col_slice],
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    aspect="equal" if zoom else "auto",
                    rasterized=True,
                )
                add_support_outline(
                    axis,
                    support_projection[row_slice, col_slice],
                    extent,
                )
                style_axis(axis)
                axis.set_ylabel(f"Bin {cardiac_bin + 1}\nSource X (mm)", fontsize=9)
                if cardiac_bin == len(panels) - 1:
                    axis.set_xlabel("MRI readout / source Z (mm)", fontsize=9)

        titles = (
            "Reference magnitude",
            "Readout velocity",
            "Speed",
        )
        for column, title in enumerate(titles):
            axes[0, column].set_title(title, fontsize=11, color="#172033")
        labels = ("Normalized a.u.", "m/s", "m/s")
        for column, label in enumerate(labels):
            colorbar = fig.colorbar(
                images[column], ax=axes[:, column], location="bottom", shrink=0.82, pad=0.08
            )
            colorbar.set_label(label, fontsize=8)
            colorbar.ax.tick_params(labelsize=7)

        title = (
            "dp150 stenosis: support-constrained cardiac-bin comparison"
            if zoom
            else "dp150 phantom: support-constrained cardiac-bin comparison"
        )
        fig.suptitle(title, fontsize=14, color="#172033")
        note = (
            "Exact seed-derived lumen outline; velocity shown only where the "
            "stored 5% signal mask intersects the lumen support."
        )
        if not zoom:
            note += " Transverse scale is expanded; axes remain in mm."
        fig.text(0.5, -0.045, note, ha="center", fontsize=7.5, color="#475569")
        fig.savefig(
            path,
            dpi=240,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={"Software": "MRSimTracks dp150 support-constrained renderer"},
        )
        plt.close(fig)


def find_throat(support: np.ndarray) -> tuple[int, np.ndarray]:
    area = support.sum(axis=(0, 1))
    occupied = np.where(area > 0)[0]
    if occupied.size < 5:
        raise ValueError("Support is too short to identify an interior stenosis")
    margin = max(2, int(round(0.08 * occupied.size)))
    candidates = occupied[margin:-margin]
    throat = int(candidates[np.argmin(area[candidates])])
    return throat, area


def choose_cross_sections(
    throat_index: int,
    readout: np.ndarray,
    support_area: np.ndarray,
) -> list[int]:
    offsets_mm = (-35.0, 0.0, 30.0, 100.0)
    indices = [
        int(np.argmin(np.abs(readout - (readout[throat_index] + offset))))
        for offset in offsets_mm
    ]
    occupied = np.where(support_area > 0)[0]
    return [int(np.clip(index, occupied.min(), occupied.max())) for index in indices]


def render_cross_sections(
    reconstruction: dict[str, Any],
    support: np.ndarray,
    signal_mask: np.ndarray,
    readout: np.ndarray,
    phase: np.ndarray,
    slice_: np.ndarray,
    section_indices: list[int],
    speed_limit: float,
    path: Path,
) -> None:
    speed_cmap = plt.get_cmap("viridis").copy()
    speed_cmap.set_bad("white")
    section_support = np.logical_or.reduce(
        [support[:, :, index] for index in section_indices]
    )
    rows, columns = np.where(section_support)
    pad = 3
    row0 = max(0, int(rows.min()) - pad)
    row1 = min(support.shape[0], int(rows.max()) + pad + 1)
    column0 = max(0, int(columns.min()) - pad)
    column1 = min(support.shape[1], int(columns.max()) + pad + 1)
    row_slice = slice(row0, row1)
    column_slice = slice(column0, column1)
    x_edges, y_edges = coordinate_edges(phase), coordinate_edges(slice_)
    extent = [x_edges[column0], x_edges[column1], y_edges[row0], y_edges[row1]]
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "semibold"}):
        fig, axes = plt.subplots(
            reconstruction["magnitude"].shape[0],
            len(section_indices),
            figsize=(11.8, 5.7),
            constrained_layout=True,
            squeeze=False,
        )
        image = None
        for cardiac_bin in range(axes.shape[0]):
            for column, readout_index in enumerate(section_indices):
                axis = axes[cardiac_bin, column]
                section = np.where(
                    signal_mask[cardiac_bin, :, :, readout_index],
                    reconstruction["speed"][cardiac_bin, :, :, readout_index],
                    np.nan,
                )[row_slice, column_slice]
                image = axis.imshow(
                    section,
                    origin="lower",
                    extent=extent,
                    cmap=speed_cmap,
                    vmin=0.0,
                    vmax=speed_limit,
                    interpolation="nearest",
                    aspect="equal",
                    rasterized=True,
                )
                add_support_outline(
                    axis,
                    support[row_slice, column_slice, readout_index],
                    extent,
                    color="#334155",
                )
                style_axis(axis)
                if cardiac_bin == 0:
                    axis.set_title(
                        f"Z = {readout[readout_index]:.0f} mm", fontsize=10
                    )
                if cardiac_bin == axes.shape[0] - 1:
                    axis.set_xlabel("Source X (mm)", fontsize=8)
                if column == 0:
                    axis.set_ylabel(
                        f"Bin {cardiac_bin + 1}\nSource Y (mm)", fontsize=8
                    )
        if image is not None:
            colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.78, pad=0.02)
            colorbar.set_label("Speed (m/s)", fontsize=9)
            colorbar.ax.tick_params(labelsize=8)
        fig.suptitle(
            "dp150 through-plane speed at selected axial locations",
            fontsize=14,
            color="#172033",
        )
        fig.text(
            0.5,
            -0.035,
            "Gray contours show the seed-derived lumen; white pixels are outside the valid signal support.",
            ha="center",
            fontsize=7.5,
            color="#475569",
        )
        fig.savefig(
            path,
            dpi=240,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)


def render_static_control(
    reconstruction: dict[str, Any],
    support: np.ndarray,
    readout: np.ndarray,
    phase: np.ndarray,
    path: Path,
) -> dict[str, float]:
    panels, signal_mask, scales = prepare_longitudinal_panels(reconstruction, support)
    support_projection = support.any(axis=0)
    row_slice, col_slice, extent = support_crop(
        support_projection, readout, phase, pad_voxels=3
    )
    velocity_limit_mm_s = 1000.0 * scales[
        "absolute_readout_velocity_common_p99_5_m_per_s"
    ]
    speed_limit_mm_s = 1000.0 * scales["speed_common_p99_5_m_per_s"]
    magnitude_cmap = plt.get_cmap("gray_r").copy()
    velocity_cmap = plt.get_cmap("RdBu_r").copy()
    speed_cmap = plt.get_cmap("viridis").copy()
    for cmap in (magnitude_cmap, velocity_cmap, speed_cmap):
        cmap.set_bad("white")

    values = [panels[0][0], 1000.0 * panels[0][1], 1000.0 * panels[0][2]]
    display = [
        (magnitude_cmap, 0.0, 1.0, "Normalized a.u."),
        (velocity_cmap, -velocity_limit_mm_s, velocity_limit_mm_s, "mm/s"),
        (speed_cmap, 0.0, speed_limit_mm_s, "mm/s"),
    ]
    titles = (
        "Reference magnitude",
        "Residual readout velocity",
        "Residual apparent speed",
    )
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "semibold"}):
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.2), constrained_layout=True)
        for axis, image_data, settings, title in zip(
            axes, values, display, titles, strict=True
        ):
            cmap, vmin, vmax, units = settings
            image = axis.imshow(
                image_data[row_slice, col_slice],
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
                rasterized=True,
            )
            add_support_outline(
                axis, support_projection[row_slice, col_slice], extent
            )
            style_axis(axis)
            axis.set_title(title, fontsize=10, color="#172033")
            axis.set_xlabel("MRI readout / source Z (mm)", fontsize=8)
            axis.set_ylabel("Source X (mm)", fontsize=8)
            colorbar = fig.colorbar(image, ax=axis, location="bottom", shrink=0.8, pad=0.15)
            colorbar.set_label(units, fontsize=7)
            colorbar.ax.tick_params(labelsize=7)
        fig.suptitle(
            "dp150 static control, cardiac bin 1",
            fontsize=13,
            color="#172033",
        )
        fig.text(
            0.5,
            -0.105,
            "Residual velocities are reconstruction noise, displayed in mm/s; transverse scale is expanded.",
            ha="center",
            fontsize=7.5,
            color="#475569",
        )
        fig.savefig(
            path,
            dpi=240,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)
    selected_speed = reconstruction["speed"][signal_mask]
    return {
        "absolute_readout_velocity_common_p99_5_mm_per_s": velocity_limit_mm_s,
        "speed_common_p99_5_mm_per_s": speed_limit_mm_s,
        "speed_max_mm_per_s": float(selected_speed.max(initial=0.0) * 1000.0),
    }


def render_mask_diagnostic(
    stored_mask: np.ndarray,
    support: np.ndarray,
    readout: np.ndarray,
    phase: np.ndarray,
    leakage: list[dict[str, Any]],
    path: Path,
) -> None:
    extent = [
        coordinate_edges(readout)[0],
        coordinate_edges(readout)[-1],
        coordinate_edges(phase)[0],
        coordinate_edges(phase)[-1],
    ]
    binary_cmaps = [
        ListedColormap(["white", "#c2415d"]),
        ListedColormap(["white", "#0f766e"]),
        ListedColormap(["white", "#1d4ed8"]),
    ]
    support_projection = support.any(axis=0)
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "semibold"}):
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 5.5), constrained_layout=True)
        for cardiac_bin in range(stored_mask.shape[0]):
            maps = [
                stored_mask[cardiac_bin].any(axis=0),
                support_projection,
                (stored_mask[cardiac_bin] & support).any(axis=0),
            ]
            for column, (axis, image_data, cmap) in enumerate(
                zip(axes[cardiac_bin], maps, binary_cmaps, strict=True)
            ):
                axis.imshow(
                    image_data,
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    aspect="auto",
                    rasterized=True,
                )
                style_axis(axis)
                axis.set_xlabel("MRI readout / source Z (mm)", fontsize=8)
                axis.set_ylabel(f"Bin {cardiac_bin + 1}\nSource X (mm)", fontsize=8)
                if column == 0:
                    axis.text(
                        0.02,
                        0.95,
                        f"{100 * leakage[cardiac_bin]['outside_support_fraction']:.1f}% outside support",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=8,
                        color="#881337",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
                    )
        for axis, title in zip(
            axes[0],
            ("Stored 5% MRI mask", "Seed-derived lumen support", "Intersection used"),
            strict=True,
        ):
            axis.set_title(title, fontsize=10, color="#172033")
        fig.suptitle(
            "dp150 display-mask correction audit",
            fontsize=14,
            color="#172033",
        )
        fig.text(
            0.5,
            -0.035,
            "Binary projections are diagnostic only; the reconstruction HDF5 files remain unchanged.",
            ha="center",
            fontsize=7.5,
            color="#475569",
        )
        fig.savefig(
            path,
            dpi=240,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)


def mask_audit(
    reconstruction: dict[str, Any], support: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leakage: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    magnitude = reconstruction["magnitude"]
    stored_mask = reconstruction["mask"]
    speed = reconstruction["speed"]
    for cardiac_bin in range(magnitude.shape[0]):
        stored = stored_mask[cardiac_bin]
        outside = stored & ~support
        inside = stored & support
        stored_count = int(stored.sum())
        leakage.append(
            {
                "cardiac_bin": cardiac_bin + 1,
                "stored_mask_voxels": stored_count,
                "inside_seed_support_voxels": int(inside.sum()),
                "outside_seed_support_voxels": int(outside.sum()),
                "outside_support_fraction": (
                    float(outside.sum() / stored_count) if stored_count else 0.0
                ),
            }
        )
        for fraction in (0.05, 0.10, 0.15, 0.20, 0.25):
            threshold_mask = (magnitude[cardiac_bin] >= fraction * magnitude[cardiac_bin].max()) & support
            values = speed[cardiac_bin][threshold_mask]
            sensitivity.append(
                {
                    "cardiac_bin": cardiac_bin + 1,
                    "magnitude_fraction": fraction,
                    "retained_voxels": int(threshold_mask.sum()),
                    "support_coverage_fraction": float(threshold_mask.sum() / support.sum()),
                    "speed_m_per_s": {
                        "median": float(np.median(values)) if values.size else None,
                        "p95": float(np.quantile(values, 0.95)) if values.size else None,
                        "p99": float(np.quantile(values, 0.99)) if values.size else None,
                        "maximum": float(values.max()) if values.size else None,
                    },
                }
            )
    return leakage, sensitivity


def image_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dimensions = list(image.size)
        mode = image.mode
    return {
        "path": str(path.resolve()),
        "dimensions_px": dimensions,
        "mode": mode,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_method_note(
    path: Path,
    report_path: Path,
    outputs: list[dict[str, Any]],
    leakage: list[dict[str, Any]],
    legacy_outputs: list[dict[str, Any]],
) -> None:
    lines = [
        "# dp150 Support-Corrected Figures",
        "",
        "## Status",
        "",
        "These are diagnostic support-audit figures, not recommended result figures.",
        "They are superseded by `../publication_figures_v2_mri` for presentation.",
        "The original PNG and HDF5 files were not overwritten.",
        "",
        "## Correction",
        "",
        "- Lumen support was voxelized from the centered 2-million-particle static phantom.",
        "- Enclosed transverse holes were filled; no dilation or spatial smoothing was used.",
        "- Velocity and speed are displayed only where the stored 5% MRI signal mask intersects the seed-derived support.",
        "- Longitudinal velocity and speed are magnitude-weighted through-slice means, not maximum-intensity projections.",
        "- Magnitude is a support-constrained maximum projection.",
        "",
        "## Why the old views failed",
        "",
    ]
    for item in leakage:
        lines.append(
            f"- Bin {item['cardiac_bin']}: {100 * item['outside_support_fraction']:.1f}% "
            "of stored-mask voxels were outside the seed-derived lumen support."
        )
    lines.extend(
        [
            "",
            "Out-of-support low-signal phase estimates became false inlet flares, repeated circles, and stripes when projected through the full field of view.",
            "The corrected figures change only the display support; they do not repair or alter the reconstruction arrays.",
            "",
            "## Superseded presentation figures",
            "",
            "The following files are retained as raw reconstruction QC but should not be used as publication figures:",
            "",
        ]
    )
    for item in legacy_outputs:
        lines.append(f"- `{item['path']}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for item in outputs:
        lines.append(f"- `{Path(item['path']).name}` ({item['dimensions_px'][0]} x {item['dimensions_px'][1]} px)")
    lines.extend(
        [
            "",
            f"Machine-readable provenance and metrics: `{report_path.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    motion_path = args.motion.resolve()
    static_path = args.static.resolve()
    support_path = args.support_h5.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    support, fov_mm, matrix, support_metadata = load_particle_support(support_path)
    motion = load_reconstruction(motion_path)
    static = load_reconstruction(static_path)
    expected_shape = tuple(int(value) for value in matrix[::-1])
    for name, reconstruction in (("motion", motion), ("static", static)):
        if reconstruction["magnitude"].shape[1:] != expected_shape:
            raise ValueError(
                f"{name} reconstruction shape {reconstruction['magnitude'].shape[1:]} "
                f"does not match support shape {expected_shape}"
            )

    readout, phase, slice_ = coordinate_centers(support.shape, fov_mm)
    panels, signal_mask, scales = prepare_longitudinal_panels(motion, support)
    throat_index, support_area = find_throat(support)
    section_indices = choose_cross_sections(throat_index, readout, support_area)
    leakage, threshold_sensitivity = mask_audit(motion, support)

    output_paths = {
        "full_length": output_dir / "dp150_motion_full_length_support_corrected.png",
        "stenosis_zoom": output_dir / "dp150_motion_stenosis_zoom_support_corrected.png",
        "cross_sections": output_dir / "dp150_motion_cross_sections_support_corrected.png",
        "static_control": output_dir / "dp150_static_control_support_corrected.png",
        "mask_diagnostic": output_dir / "dp150_mask_leakage_diagnostic.png",
    }
    render_longitudinal(
        panels,
        support,
        readout,
        phase,
        scales,
        output_paths["full_length"],
        zoom=False,
        throat_index=throat_index,
    )
    render_longitudinal(
        panels,
        support,
        readout,
        phase,
        scales,
        output_paths["stenosis_zoom"],
        zoom=True,
        throat_index=throat_index,
    )
    render_cross_sections(
        motion,
        support,
        signal_mask,
        readout,
        phase,
        slice_,
        section_indices,
        scales["speed_common_p99_5_m_per_s"],
        output_paths["cross_sections"],
    )
    static_metrics = render_static_control(
        static,
        support,
        readout,
        phase,
        output_paths["static_control"],
    )
    render_mask_diagnostic(
        motion["mask"],
        support,
        readout,
        phase,
        leakage,
        output_paths["mask_diagnostic"],
    )

    outputs = [image_record(path) for path in output_paths.values()]
    legacy_outputs = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in LEGACY_PRESENTATION_FILES
        if path.exists()
    ]
    report = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "support-constrained display correction; raw reconstructions unchanged",
        "inputs": {
            "motion_reconstruction": {
                "path": str(motion_path),
                "sha256": sha256_file(motion_path),
            },
            "static_reconstruction": {
                "path": str(static_path),
                "sha256": sha256_file(static_path),
            },
            "particle_support": {
                "path": str(support_path),
                "sha256": sha256_file(support_path),
            },
        },
        "array_order_python": ["cardiac_bin", "slice", "phase", "readout"],
        "support": support_metadata,
        "motion": {
            "RR_s": motion["RR_s"],
            "Venc_m_per_s": motion["Venc_m_per_s"],
            "display_signal_mask": "stored 5% MRI mask AND seed-derived support",
            "shared_display_scales": scales,
            "stored_mask_leakage": leakage,
            "threshold_sensitivity_within_support": threshold_sensitivity,
            "throat": {
                "readout_index": throat_index,
                "source_Z_mm": float(readout[throat_index]),
                "support_cross_section_voxels": int(support_area[throat_index]),
            },
            "cross_sections": [
                {
                    "readout_index": index,
                    "source_Z_mm": float(readout[index]),
                    "support_cross_section_voxels": int(support_area[index]),
                }
                for index in section_indices
            ],
        },
        "static_control": static_metrics,
        "supersedes_for_presentation": legacy_outputs,
        "outputs": outputs,
    }
    report_path = output_dir / "dp150_support_corrected_figure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_method_note(
        output_dir / "README.md",
        report_path,
        outputs,
        leakage,
        legacy_outputs,
    )
    print(f"Support-corrected dp150 figures: PASS\nReport: {report_path}")


if __name__ == "__main__":
    main()
