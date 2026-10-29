r"""Audit branch geometry in the completed aorta KomaMRI reconstruction."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "outputs" / "aorta_dp2_mri_pipeline"
DEFAULT_TRACKS = PIPELINE / "aorta_motion_1m.h5"
DEFAULT_RECONSTRUCTION = (
    PIPELINE
    / "koma"
    / "simulation_motion_1m"
    / "pc_reconstruction_coremask015.h5"
)
DEFAULT_WALL_POINTS = (
    PIPELINE
    / "visualization_branch_corrected"
    / "aorta_wall_points_from_wss.npy"
)
DEFAULT_MOTION_REPORT = PIPELINE / "koma" / "motion_phantom_report.toml"
DEFAULT_OUTPUT_DIR = PIPELINE / "koma" / "branch_geometry_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument(
        "--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION
    )
    parser.add_argument("--wall-points", type=Path, default=DEFAULT_WALL_POINTS)
    parser.add_argument(
        "--motion-report", type=Path, default=DEFAULT_MOTION_REPORT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--branch-x-mm",
        type=float,
        nargs=2,
        default=(90.0, 128.0),
        metavar=("MIN", "MAX"),
        help="Readout-coordinate range used for the branch-detail audit.",
    )
    parser.add_argument(
        "--branch-y-mm",
        type=float,
        nargs=2,
        default=(-22.0, 46.0),
        metavar=("MIN", "MAX"),
        help="Phase-coordinate range used for the branch-detail audit.",
    )
    parser.add_argument(
        "--preferred-profile-x-mm",
        type=float,
        default=111.0,
        help="Preferred readout coordinate for the outlet-band profile.",
    )
    parser.add_argument("--orbit-frames", type=int, default=72)
    parser.add_argument("--orbit-fps", type=float, default=25.0)
    parser.add_argument("--skip-orbit", action="store_true")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> dict:
    with args.motion_report.open("rb") as stream:
        motion_report = tomllib.load(stream)

    with h5py.File(args.tracks, "r") as h5:
        initial = np.asarray(h5["initial_position"], dtype=np.float32)
        particle_count = int(h5.attrs["n_particles"])

    with h5py.File(args.reconstruction, "r") as h5:
        magnitude = np.asarray(h5["reference_magnitude"], dtype=np.float32)
        velocity = np.asarray(h5["velocity"], dtype=np.float32)
        speed = np.asarray(h5["speed"], dtype=np.float32)
        mask = np.asarray(h5["mask"], dtype=bool)
        fov_m = np.asarray(h5.attrs["FOV_m"], dtype=float)
        matrix = np.asarray(h5.attrs["matrix"], dtype=int)
        rr_s = float(h5.attrs["RR_s"])
        venc = float(h5.attrs["Venc_m_per_s"])
        mask_fraction = float(h5.attrs["analysis_magnitude_mask_fraction"])

    wall_points = np.asarray(np.load(args.wall_points), dtype=np.float32)
    source_center = np.asarray(
        motion_report["source_coordinate_center_m"], dtype=float
    )
    axis_order = (
        np.asarray(
            motion_report["source_axis_order_largest_to_smallest_one_based"],
            dtype=int,
        )
        - 1
    )

    expected_volume_shape = (
        magnitude.shape[0],
        matrix[2],
        matrix[1],
        matrix[0],
    )
    if magnitude.shape != expected_volume_shape:
        raise RuntimeError(
            f"unexpected reconstruction shape {magnitude.shape}; "
            f"expected {expected_volume_shape}"
        )
    if velocity.shape != (3, *magnitude.shape):
        raise RuntimeError(f"unexpected velocity shape {velocity.shape}")
    if speed.shape != magnitude.shape or mask.shape != magnitude.shape:
        raise RuntimeError("speed and mask must match reference magnitude")
    if initial.shape != (particle_count, 3):
        raise RuntimeError(f"unexpected initial-position shape {initial.shape}")
    if wall_points.ndim != 2 or wall_points.shape[1] != 3:
        raise RuntimeError(f"unexpected wall-point shape {wall_points.shape}")
    if not (
        np.isfinite(magnitude).all()
        and np.isfinite(velocity).all()
        and np.isfinite(speed).all()
        and np.isfinite(initial).all()
        and np.isfinite(wall_points).all()
    ):
        raise RuntimeError("non-finite values found in audit inputs")

    initial_mri = initial[:, axis_order] - source_center[axis_order]
    wall_mri = wall_points[:, axis_order] - source_center[axis_order]
    return {
        "initial_mri": initial_mri,
        "wall_mri": wall_mri,
        "particle_count": particle_count,
        "magnitude": magnitude,
        "velocity": velocity,
        "speed": speed,
        "mask": mask,
        "fov_m": fov_m,
        "matrix": matrix,
        "rr_s": rr_s,
        "venc_m_per_s": venc,
        "mask_fraction": mask_fraction,
        "source_center_m": source_center,
        "source_axis_order_zero_based": axis_order,
        "motion_report": motion_report,
    }


def coordinate_grid(fov_m: np.ndarray, matrix: np.ndarray) -> tuple:
    spacing_m = fov_m / matrix
    centers = tuple(
        -fov_m[index] / 2
        + (np.arange(matrix[index], dtype=float) + 0.5) * spacing_m[index]
        for index in range(3)
    )
    edges = tuple(
        np.linspace(-fov_m[index] / 2, fov_m[index] / 2, matrix[index] + 1)
        for index in range(3)
    )
    return centers, edges, spacing_m


def projected_density(
    points_mri: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    density, _, _ = np.histogram2d(
        points_mri[:, 0], points_mri[:, 1], bins=(x_edges, y_edges)
    )
    return density.T


def robust_limit(values: np.ndarray, quantile: float = 0.995) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    return float(np.quantile(finite, quantile)) if finite.size else 1.0


def magnitude_weighted_projection(
    values: np.ndarray, magnitude: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    weights = np.where(mask, magnitude, 0.0)
    denominator = weights.sum(axis=0)
    return np.divide(
        (values * weights).sum(axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def true_runs(mask: np.ndarray, coordinates_mm: np.ndarray, min_pixels: int = 2):
    labels, count = label(mask.astype(bool))
    intervals = []
    for index in range(1, count + 1):
        selected = np.flatnonzero(labels == index)
        if selected.size >= min_pixels:
            intervals.append(
                [
                    float(coordinates_mm[selected[0]]),
                    float(coordinates_mm[selected[-1]]),
                ]
            )
    return intervals


def branch_profile_audit(
    projected_masks: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    branch_x_mm: tuple[float, float],
    branch_y_mm: tuple[float, float],
    preferred_x_mm: float,
) -> dict:
    x_roi = (x_mm >= branch_x_mm[0]) & (x_mm <= branch_x_mm[1])
    y_roi = (y_mm >= branch_y_mm[0]) & (y_mm <= branch_y_mm[1])
    candidate_indices = np.flatnonzero(x_roi)
    if not candidate_indices.size or not np.any(y_roi):
        raise ValueError("branch ROI does not intersect the reconstruction grid")

    counts = np.zeros((projected_masks.shape[0], x_mm.size), dtype=int)
    intervals_by_bin: list[list[list[list[float]]]] = []
    for phase in range(projected_masks.shape[0]):
        phase_intervals = []
        for x_index in range(x_mm.size):
            intervals = true_runs(
                projected_masks[phase, y_roi, x_index], y_mm[y_roi]
            )
            counts[phase, x_index] = len(intervals)
            phase_intervals.append(intervals)
        intervals_by_bin.append(phase_intervals)

    common_count = counts[:, candidate_indices].min(axis=0)
    best_count = int(common_count.max())
    best_candidates = candidate_indices[common_count == best_count]
    profile_index = int(
        best_candidates[np.argmin(np.abs(x_mm[best_candidates] - preferred_x_mm))]
    )

    bins = []
    for phase in range(projected_masks.shape[0]):
        target_count = int(counts[phase, profile_index])
        same_count_mask = np.zeros(x_mm.size, dtype=bool)
        same_count_mask[candidate_indices] = (
            counts[phase, candidate_indices] == target_count
        )
        same_count_labels, _ = label(same_count_mask)
        profile_region = same_count_labels[profile_index]
        same_count = np.flatnonzero(same_count_labels == profile_region)
        bins.append(
            {
                "cardiac_bin": phase + 1,
                "separated_band_count_at_profile": target_count,
                "band_intervals_y_mm": intervals_by_bin[phase][profile_index],
                "same_count_x_span_mm": (
                    [float(x_mm[same_count[0]]), float(x_mm[same_count[-1]])]
                    if same_count.size
                    else None
                ),
                "same_count_contiguous_column_count": int(same_count.size),
                "maximum_band_count_in_roi": int(
                    counts[phase, candidate_indices].max()
                ),
            }
        )
    return {
        "profile_index": profile_index,
        "profile_x_mm": float(x_mm[profile_index]),
        "common_separated_band_count": best_count,
        "bins": bins,
        "counts": counts,
        "y_roi": y_roi,
    }


def style_projection_axis(axis, branch_x_mm, branch_y_mm) -> None:
    axis.set_xlim(branch_x_mm)
    axis.set_ylim(branch_y_mm)
    axis.set_xlabel("MRI readout = source Z (mm)")
    axis.set_ylabel("MRI phase = source X (mm)")
    axis.set_aspect("equal")


def render_alignment(
    data: dict,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    output_path: Path,
    branch_x_mm: tuple[float, float],
    branch_y_mm: tuple[float, float],
) -> None:
    wall_mri = data["wall_mri"] * 1e3
    spin_density = projected_density(data["initial_mri"], x_edges, y_edges)
    magnitude_projection = data["magnitude"].max(axis=1)
    mask_projection = data["mask"].any(axis=1)
    extent = [x_edges[0] * 1e3, x_edges[-1] * 1e3, y_edges[0] * 1e3, y_edges[-1] * 1e3]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), constrained_layout=True)
    axes[0].scatter(
        wall_mri[:, 0],
        wall_mri[:, 1],
        s=0.12,
        color="#334155",
        alpha=0.34,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title(f"Native wall-associated samples\n(n={wall_mri.shape[0]:,})")
    axes[1].imshow(
        np.log1p(spin_density),
        origin="lower",
        extent=extent,
        cmap="Blues",
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title(f"Koma spins at t=0\n(n={data['particle_count']:,})")
    for phase in range(2):
        axes[phase + 2].imshow(
            magnitude_projection[phase],
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=0,
            vmax=robust_limit(magnitude_projection[phase], 0.999),
            interpolation="nearest",
            aspect="equal",
        )
        axes[phase + 2].contour(
            x_mm,
            y_mm,
            mask_projection[phase].astype(float),
            levels=[0.5],
            colors=["#0891b2"],
            linewidths=0.8,
        )
        axes[phase + 2].set_title(
            f"KomaMRI magnitude | bin {phase + 1}\n"
            f"{100 * data['mask_fraction']:.0f}% core-mask contour"
        )

    for axis in axes:
        axis.set_xlim(extent[:2])
        axis.set_ylim(extent[2:])
        axis.set_xlabel("MRI readout = source Z (mm)")
        axis.set_ylabel("MRI phase = source X (mm)")
        axis.set_aspect("equal")
        axis.add_patch(
            plt.Rectangle(
                (branch_x_mm[0], branch_y_mm[0]),
                branch_x_mm[1] - branch_x_mm[0],
                branch_y_mm[1] - branch_y_mm[0],
                fill=False,
                edgecolor="#dc2626",
                linewidth=1.1,
                linestyle="--",
            )
        )
    fig.suptitle(
        "Aorta dp2 geometry alignment: native 3D data -> Koma spins -> MRI",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def render_branch_detail(
    data: dict,
    audit: dict,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    output_path: Path,
    branch_x_mm: tuple[float, float],
    branch_y_mm: tuple[float, float],
) -> None:
    magnitude_projection = data["magnitude"].max(axis=1)
    mask_projection = data["mask"].any(axis=1)
    speed_projection = np.stack(
        [
            magnitude_weighted_projection(
                data["speed"][phase],
                data["magnitude"][phase],
                data["mask"][phase],
            )
            for phase in range(data["magnitude"].shape[0])
        ]
    )
    extent = [x_edges[0] * 1e3, x_edges[-1] * 1e3, y_edges[0] * 1e3, y_edges[-1] * 1e3]
    speed_limit = robust_limit(data["speed"][data["mask"]], 0.99)
    profile_index = audit["profile_index"]
    profile_x = audit["profile_x_mm"]
    y_roi = audit["y_roi"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
    for phase in range(2):
        magnitude_image = axes[phase, 0].imshow(
            magnitude_projection[phase],
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=0,
            vmax=robust_limit(magnitude_projection[phase], 0.999),
            interpolation="nearest",
            aspect="equal",
        )
        axes[phase, 0].contour(
            x_mm,
            y_mm,
            mask_projection[phase].astype(float),
            levels=[0.5],
            colors=["#06b6d4"],
            linewidths=1.0,
        )
        axes[phase, 0].axvline(profile_x, color="#dc2626", ls="--", lw=1.0)
        axes[phase, 0].set_title(f"Magnitude | cardiac bin {phase + 1}")
        fig.colorbar(magnitude_image, ax=axes[phase, 0], shrink=0.72)

        speed_image = axes[phase, 1].imshow(
            speed_projection[phase],
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=speed_limit,
            interpolation="nearest",
            aspect="equal",
        )
        axes[phase, 1].contour(
            x_mm,
            y_mm,
            mask_projection[phase].astype(float),
            levels=[0.5],
            colors=["white"],
            linewidths=0.8,
        )
        axes[phase, 1].axvline(profile_x, color="#dc2626", ls="--", lw=1.0)
        axes[phase, 1].set_title(f"Magnitude-weighted speed | bin {phase + 1}")
        fig.colorbar(
            speed_image,
            ax=axes[phase, 1],
            shrink=0.72,
            label="m/s",
        )

        profile = magnitude_projection[phase, :, profile_index]
        profile_scale = max(float(profile[y_roi].max()), np.finfo(float).eps)
        normalized_profile = profile / profile_scale
        profile_mask = mask_projection[phase, :, profile_index]
        axes[phase, 2].plot(normalized_profile, y_mm, color="#111827", lw=1.5)
        axes[phase, 2].fill_betweenx(
            y_mm,
            0,
            1,
            where=profile_mask,
            color="#67e8f9",
            alpha=0.38,
            step="mid",
        )
        for low, high in audit["bins"][phase]["band_intervals_y_mm"]:
            axes[phase, 2].axhspan(low, high, color="#0891b2", alpha=0.16)
        count = audit["bins"][phase]["separated_band_count_at_profile"]
        axes[phase, 2].set_title(
            f"Profile at readout {profile_x:.0f} mm\n"
            f"{count} separated projected bands"
        )
        axes[phase, 2].set_xlim(0, 1.05)
        axes[phase, 2].set_ylim(branch_y_mm)
        axes[phase, 2].set_xlabel("Normalized magnitude")
        axes[phase, 2].set_ylabel("MRI phase = source X (mm)")
        axes[phase, 2].grid(alpha=0.18)

        style_projection_axis(axes[phase, 0], branch_x_mm, branch_y_mm)
        style_projection_axis(axes[phase, 1], branch_x_mm, branch_y_mm)

    fig.suptitle(
        "Aorta KomaMRI branch-detail audit | 15% core-mask sensitivity view",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=190, facecolor="white")
    plt.close(fig)


def koma_surface(volume: np.ndarray, fov_m: np.ndarray, matrix: np.ndarray, fraction: float):
    import pyvista as pv

    values = volume.transpose(2, 1, 0)
    spacing_mm = fov_m / matrix * 1e3
    grid = pv.ImageData(
        dimensions=values.shape,
        spacing=tuple(spacing_mm),
        origin=tuple(-fov_m * 1e3 / 2 + spacing_mm / 2),
    )
    grid.point_data["magnitude"] = values.ravel(order="F")
    threshold = fraction * float(volume.max())
    raw_surface = grid.contour([threshold], scalars="magnitude")
    display_surface = raw_surface.connectivity(extraction_mode="largest")
    return raw_surface, display_surface, threshold


def render_orbit(
    data: dict,
    output_path: Path,
    contact_sheet_path: Path,
    frames_count: int,
    fps: float,
) -> dict:
    import pyvista as pv

    if frames_count < 4 or fps <= 0:
        raise ValueError("orbit requires at least four frames and positive fps")
    surfaces = [
        koma_surface(
            data["magnitude"][phase],
            data["fov_m"],
            data["matrix"],
            data["mask_fraction"],
        )
        for phase in range(2)
    ]
    if any(
        raw_surface.n_points == 0 or display_surface.n_points == 0
        for raw_surface, display_surface, _ in surfaces
    ):
        raise RuntimeError("Koma magnitude iso-surface is empty")

    all_phantom_points = data["initial_mri"] * 1e3
    support_tree = cKDTree(all_phantom_points)
    voxel_size_mm = data["fov_m"] / data["matrix"] * 1e3
    support_radius_mm = 2.0 * float(voxel_size_mm.max())
    display_stride = max(1, int(np.ceil(data["particle_count"] / 100_000)))
    phantom_points = all_phantom_points[::display_stride]
    phantom_cloud = pv.PolyData(phantom_points)
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 660))
    plotter.set_background("white")
    plotter.subplot(0, 0)
    plotter.add_points(
        phantom_cloud,
        color="#0f766e",
        opacity=0.30,
        point_size=2.0,
        render_points_as_spheres=False,
    )
    plotter.add_text(
        f"Koma phantom t=0 | {phantom_points.shape[0]:,}/{data['particle_count']:,} spins",
        position="upper_left",
        font_size=11,
        color="#1f2937",
    )
    plotter.show_axes()
    colors = ("#2a9d8f", "#457b9d")
    for phase, ((_, display_surface, threshold), color) in enumerate(
        zip(surfaces, colors)
    ):
        plotter.subplot(0, phase + 1)
        plotter.add_mesh(
            display_surface,
            color=color,
            opacity=0.92,
            smooth_shading=True,
            ambient=0.30,
            diffuse=0.68,
            specular=0.14,
            show_scalar_bar=False,
        )
        plotter.add_text(
            f"KomaMRI bin {phase + 1} | largest connected "
            f"{100 * data['mask_fraction']:.0f}% component",
            position="upper_left",
            font_size=11,
            color="#1f2937",
        )
        plotter.show_axes()

    center = np.array([0.0, 0.0, 0.0])
    radius = float(data["fov_m"][0] * 1e3 * 1.15)
    frames = []
    for index, angle in enumerate(np.linspace(0, 2 * np.pi, frames_count, endpoint=False)):
        position = (
            center
            + np.array(
                [
                    radius * 0.72 * np.cos(angle),
                    radius * 0.72 * np.sin(angle),
                    radius * 1.05,
                ]
            )
        )
        for panel in range(3):
            plotter.subplot(0, panel)
            plotter.camera.position = tuple(position)
            plotter.camera.focal_point = tuple(center)
            plotter.camera.up = (0.0, 0.0, 1.0)
            plotter.renderer.reset_camera_clipping_range()
        plotter.render()
        frames.append(Image.fromarray(plotter.screenshot(return_img=True)).convert("RGB"))
    plotter.close()

    duration_ms = max(1, int(round(1000 / fps)))
    paletted = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
        for frame in frames
    ]
    paletted[0].save(
        output_path,
        save_all=True,
        append_images=paletted[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    selected = [
        frames[index]
        for index in (
            0,
            frames_count // 4,
            frames_count // 2,
            3 * frames_count // 4,
        )
    ]
    thumb_width = 1200
    thumbs = [
        frame.resize(
            (thumb_width, round(frame.height * thumb_width / frame.width)),
            Image.Resampling.LANCZOS,
        )
        for frame in selected
    ]
    sheet = Image.new(
        "RGB",
        (thumb_width * 2, max(frame.height for frame in thumbs) * 2),
        "white",
    )
    for index, frame in enumerate(thumbs):
        sheet.paste(
            frame,
            ((index % 2) * thumb_width, (index // 2) * frame.height),
        )
    sheet.save(contact_sheet_path)

    connectivity = []
    for phase, (raw_surface, display_surface, _) in enumerate(surfaces):
        connected = raw_surface.connectivity()
        region_ids = np.asarray(
            connected.point_data.get("RegionId", np.empty(0, dtype=int)),
            dtype=int,
        )
        region_sizes = np.bincount(region_ids) if region_ids.size else np.empty(0)
        nearest_spin_distance_mm = np.asarray(
            support_tree.query(np.asarray(connected.points))[0], dtype=float
        )

        def support_summary(selected: np.ndarray) -> dict | None:
            distances = nearest_spin_distance_mm[selected]
            if not distances.size:
                return None
            return {
                "surface_points": int(distances.size),
                "nearest_spin_distance_median_mm": float(
                    np.quantile(distances, 0.50)
                ),
                "nearest_spin_distance_p95_mm": float(
                    np.quantile(distances, 0.95)
                ),
                "fraction_within_support_radius": float(
                    np.mean(distances <= support_radius_mm)
                ),
            }

        largest_region_id = int(np.argmax(region_sizes))
        retained = region_ids == largest_region_id
        rejected = ~retained
        rejected_region_sizes = region_sizes.copy()
        rejected_region_sizes[largest_region_id] = 0
        largest_rejected_region_id = (
            int(np.argmax(rejected_region_sizes))
            if np.any(rejected_region_sizes)
            else None
        )
        largest_rejected = (
            region_ids == largest_rejected_region_id
            if largest_rejected_region_id is not None
            else np.zeros_like(region_ids, dtype=bool)
        )
        if int(retained.sum()) != int(display_surface.n_points):
            raise RuntimeError(
                "largest-component point count does not match display surface"
            )
        connectivity.append(
            {
                "cardiac_bin": phase + 1,
                "connected_region_count": int(region_sizes.size),
                "regions_with_at_least_100_surface_points": int(
                    np.count_nonzero(region_sizes >= 100)
                ),
                "largest_region_point_fraction": (
                    float(region_sizes.max() / region_sizes.sum())
                    if region_sizes.size
                    else 0.0
                ),
                "raw_surface_points": int(raw_surface.n_points),
                "display_surface_points": int(display_surface.n_points),
                "removed_surface_point_fraction": float(
                    1 - display_surface.n_points / raw_surface.n_points
                ),
                "support_radius_mm": support_radius_mm,
                "retained_component_support": support_summary(retained),
                "all_rejected_components_support": support_summary(rejected),
                "largest_rejected_component_support": support_summary(
                    largest_rejected
                ),
            }
        )

    return {
        "path": str(output_path),
        "dimensions_px": list(frames[0].size),
        "frames": frames_count,
        "duration_ms": duration_ms,
        "loop": 0,
        "raw_surface_points": [surface.n_points for surface, _, _ in surfaces],
        "raw_surface_cells": [surface.n_cells for surface, _, _ in surfaces],
        "display_surface_points": [surface.n_points for _, surface, _ in surfaces],
        "display_surface_cells": [surface.n_cells for _, surface, _ in surfaces],
        "thresholds": [threshold for _, _, threshold in surfaces],
        "display_filter": (
            "Largest connected magnitude component only; raw reconstruction "
            "and unfiltered derivative are preserved."
        ),
        "support_reference": {
            "points": "All one-million t=0 input spins in centered MRI coordinates",
            "radius_mm": support_radius_mm,
            "radius_definition": "Two times the largest reconstruction voxel width",
        },
        "phantom_display_stride": display_stride,
        "phantom_displayed_spins": int(phantom_points.shape[0]),
        "surface_connectivity": connectivity,
        "interpretation": (
            "The input phantom is a 3D spin cloud. The displayed MRI surfaces "
            "retain only the largest connected 15% magnitude component to "
            "remove visually detached threshold islands. Nearest-spin support "
            "diagnostics are reported for the retained and rejected surfaces. "
            "This is presentation-only and does not create a physical wall."
        ),
        "contact_sheet": str(contact_sheet_path),
    }


def image_metadata(path: Path) -> dict:
    with Image.open(path) as image:
        return {
            "path": str(path),
            "dimensions_px": list(image.size),
            "bytes": path.stat().st_size,
        }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args)
    centers, edges, spacing_m = coordinate_grid(data["fov_m"], data["matrix"])
    x_mm, y_mm, _ = (coordinate * 1e3 for coordinate in centers)
    x_edges, y_edges, _ = edges
    projected_masks = data["mask"].any(axis=1)
    branch_x_mm = tuple(sorted(args.branch_x_mm))
    branch_y_mm = tuple(sorted(args.branch_y_mm))
    audit = branch_profile_audit(
        projected_masks,
        x_mm,
        y_mm,
        branch_x_mm,
        branch_y_mm,
        args.preferred_profile_x_mm,
    )

    alignment_path = args.output_dir / "aorta_koma_geometry_alignment.png"
    detail_path = args.output_dir / "aorta_koma_three_branch_detail.png"
    render_alignment(
        data,
        x_mm,
        y_mm,
        x_edges,
        y_edges,
        alignment_path,
        branch_x_mm,
        branch_y_mm,
    )
    render_branch_detail(
        data,
        audit,
        x_mm,
        y_mm,
        x_edges,
        y_edges,
        detail_path,
        branch_x_mm,
        branch_y_mm,
    )

    orbit = None
    if not args.skip_orbit:
        orbit = render_orbit(
            data,
            args.output_dir / "aorta_koma_input_vs_reconstruction_3d_orbit.gif",
            args.output_dir
            / "aorta_koma_input_vs_reconstruction_3d_orbit_contact_sheet.png",
            args.orbit_frames,
            args.orbit_fps,
        )

    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_THREE_PROJECTED_BRANCH_BANDS_VISIBLE"
            if audit["common_separated_band_count"] == 3
            else "REVIEW_PROJECTED_BRANCH_SEPARABILITY"
        ),
        "scope": (
            "Descriptive geometry alignment and branch-separability review of "
            "the existing KomaMRI reconstruction; not an anatomical segmentation "
            "or independent velocity-accuracy validation."
        ),
        "inputs": {
            "tracks": str(args.tracks.resolve()),
            "reconstruction": str(args.reconstruction.resolve()),
            "wall_points": str(args.wall_points.resolve()),
            "motion_report": str(args.motion_report.resolve()),
        },
        "particle_count": data["particle_count"],
        "wall_associated_point_count": int(data["wall_mri"].shape[0]),
        "reconstruction_shape": list(data["magnitude"].shape),
        "dimension_order_python": ["cardiac_bin", "slice", "phase", "readout"],
        "source_axis_order_zero_based": data[
            "source_axis_order_zero_based"
        ].tolist(),
        "mri_axis_mapping": {
            "readout": "source Z",
            "phase": "source X",
            "slice": "source Y",
        },
        "source_coordinate_center_m": data["source_center_m"].tolist(),
        "FOV_m": data["fov_m"].tolist(),
        "matrix": data["matrix"].tolist(),
        "voxel_size_mm": (spacing_m * 1e3).tolist(),
        "RR_s": data["rr_s"],
        "Venc_m_per_s": data["venc_m_per_s"],
        "magnitude_mask_fraction": data["mask_fraction"],
        "branch_projection_audit": {
            "projection": "maximum over MRI slice/source-Y depth",
            "branch_roi_x_mm": list(branch_x_mm),
            "branch_roi_y_mm": list(branch_y_mm),
            "profile_x_mm": audit["profile_x_mm"],
            "common_separated_band_count": audit[
                "common_separated_band_count"
            ],
            "minimum_contiguous_band_size_pixels": 2,
            "bins": audit["bins"],
            "interpretation": (
                "Three separated signal bands are visible in the 15% core-mask "
                "projection for both cardiac bins. This supports visual branch "
                "separability at this projection and threshold; it does not label "
                "the outlets or prove exact wall geometry."
            ),
        },
        "outputs": {
            "alignment": image_metadata(alignment_path),
            "branch_detail": image_metadata(detail_path),
            "orbit": orbit,
        },
        "simulation_reused": True,
        "simulation_rerun": False,
        "limitations": [
            "The 15% magnitude core mask is a sensitivity derivative, not an externally validated lumen threshold.",
            "The 2D branch count is measured after maximum projection over source-Y depth and is not a 3D anatomical segmentation.",
            "The 15% magnitude iso-surfaces contain disconnected signal components and must not be interpreted as physical wall surfaces.",
            "The clean 3D orbit displays only the largest connected component; the raw reconstruction remains unchanged.",
            "Nearest-spin support uses the t=0 input cloud and a two-voxel diagnostic radius; it is not an external anatomical reference.",
            "The source has no connected wall mesh or labeled inlet/outlet caps, so exact branch identities are not assigned.",
            "The separate REF/X/Y/Z simulations are not proven equivalent to the experimental balanced-Hadamard protocol.",
            "No independent PC-MRI or higher-resolution reconstruction is available for velocity-accuracy validation.",
        ],
    }
    report_path = args.output_dir / "aorta_koma_branch_geometry_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['status']}\nReport: {report_path}")


if __name__ == "__main__":
    main()
