r"""Render a compact MRI-focused dp150 figure set without visible QC overlays."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.plot_dp150_koma_publication import (
    DEFAULT_MOTION,
    DEFAULT_SUPPORT,
    coordinate_centers,
    coordinate_edges,
    find_throat,
    load_particle_support,
    load_reconstruction,
    masked_max_projection,
    robust_quantile,
    sha256_file,
)


DEFAULT_OUTPUT = Path(
    "outputs/dp150_mri_pipeline/koma/publication_figures_v2_mri"
)
EXPECTED_CARDIAC_BINS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--support-h5", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_two_bin_input(reconstruction: dict[str, Any]) -> int:
    """Enforce the legacy clean renderer's explicit two-bin contract."""
    magnitude = np.asarray(reconstruction["magnitude"])
    cardiac_bins = int(magnitude.shape[0]) if magnitude.ndim else 0
    if cardiac_bins != EXPECTED_CARDIAC_BINS:
        raise ValueError(
            "plot_dp150_koma_clean.py requires exactly two reconstructed "
            f"cardiac bins; found {cardiac_bins}. Use "
            "analyze_dp150_scanner_reconstruction.py for multi-bin inputs."
        )
    return cardiac_bins


def center_plane_indices(support: np.ndarray) -> tuple[int, int]:
    slice_index = int(np.argmax(support.sum(axis=(1, 2))))
    phase_index = int(np.argmax(support.sum(axis=(0, 2))))
    return slice_index, phase_index


def coordinate_slice(
    coordinates: np.ndarray, minimum: float, maximum: float
) -> slice:
    selected = np.where((coordinates >= minimum) & (coordinates <= maximum))[0]
    if selected.size == 0:
        raise ValueError(f"No coordinates lie between {minimum} and {maximum}")
    return slice(int(selected.min()), int(selected.max()) + 1)


def image_extent(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    horizontal_slice: slice,
    vertical_slice: slice,
) -> list[float]:
    horizontal_edges = coordinate_edges(horizontal)
    vertical_edges = coordinate_edges(vertical)
    return [
        float(horizontal_edges[horizontal_slice.start]),
        float(horizontal_edges[horizontal_slice.stop]),
        float(vertical_edges[vertical_slice.start]),
        float(vertical_edges[vertical_slice.stop]),
    ]


def masked_plane(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[~mask] = np.nan
    return result


def support_rms_projection(
    values: np.ndarray, support: np.ndarray, axis: int
) -> np.ndarray:
    if values.shape != support.shape:
        raise ValueError("Values and support must have identical shapes")
    denominator = support.sum(axis=axis)
    numerator = np.square(np.where(support, values, 0.0)).sum(axis=axis)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    np.sqrt(result, out=result)
    return result


def mri_colormap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#05070a")
    return cmap


def style_image_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#05070a")
    axis.grid(False)
    axis.tick_params(colors="#334155", labelsize=8, length=3, width=0.7)
    for spine in axis.spines.values():
        spine.set_color("#64748b")
        spine.set_linewidth(0.7)


def render_pc_comparison(
    reconstruction: dict[str, Any],
    support: np.ndarray,
    readout: np.ndarray,
    phase: np.ndarray,
    slice_: np.ndarray,
    throat_index: int,
    path: Path,
) -> dict[str, Any]:
    magnitude = reconstruction["magnitude"]
    velocity = reconstruction["velocity"]
    speed = reconstruction["speed"]
    signal_mask = reconstruction["mask"] & support[None, ...]
    slice_index, _ = center_plane_indices(support)

    longitudinal_readout = coordinate_slice(
        readout, readout[throat_index] - 58.0, readout[throat_index] + 58.0
    )
    longitudinal_phase = coordinate_slice(phase, -18.0, 18.0)
    transverse_phase = coordinate_slice(phase, -9.0, 9.0)
    transverse_slice = coordinate_slice(slice_, -9.0, 9.0)
    longitudinal_extent = image_extent(
        readout,
        phase,
        longitudinal_readout,
        longitudinal_phase,
    )
    transverse_extent = image_extent(
        phase,
        slice_,
        transverse_phase,
        transverse_slice,
    )

    support_all_bins = np.broadcast_to(support, magnitude.shape)
    magnitude_limit = max(
        robust_quantile(magnitude, support_all_bins, 0.995), 1e-8
    )
    velocity_limit = max(
        robust_quantile(np.abs(velocity[0]), signal_mask, 0.995), 1e-6
    )
    speed_limit = max(robust_quantile(speed, signal_mask, 0.995), 1e-6)

    magnitude_cmap = mri_colormap("gray")
    velocity_cmap = mri_colormap("RdBu_r")
    speed_cmap = mri_colormap("magma")
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            magnitude.shape[0],
            3,
            figsize=(13.2, 5.3),
            constrained_layout=True,
            squeeze=False,
            gridspec_kw={"width_ratios": (2.1, 2.1, 0.85)},
        )
        images: list[Any] = [None, None, None]
        for cardiac_bin in range(magnitude.shape[0]):
            anatomy = magnitude[cardiac_bin, slice_index][
                longitudinal_phase, longitudinal_readout
            ]
            axial_velocity = masked_plane(
                velocity[0, cardiac_bin, slice_index],
                signal_mask[cardiac_bin, slice_index],
            )[longitudinal_phase, longitudinal_readout]
            throat_speed = masked_plane(
                speed[cardiac_bin, :, :, throat_index],
                signal_mask[cardiac_bin, :, :, throat_index],
            )[transverse_slice, transverse_phase]

            panel_data = (
                (anatomy, magnitude_cmap, 0.0, magnitude_limit, longitudinal_extent),
                (
                    axial_velocity,
                    velocity_cmap,
                    -velocity_limit,
                    velocity_limit,
                    longitudinal_extent,
                ),
                (throat_speed, speed_cmap, 0.0, speed_limit, transverse_extent),
            )
            for column, (data, cmap, vmin, vmax, extent) in enumerate(panel_data):
                axis = axes[cardiac_bin, column]
                images[column] = axis.imshow(
                    data,
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                    interpolation_stage="rgba",
                    aspect="equal",
                    rasterized=True,
                )
                style_image_axis(axis)
                if cardiac_bin == magnitude.shape[0] - 1:
                    if column < 2:
                        axis.set_xlabel("MRI readout / source Z (mm)", fontsize=8)
                    else:
                        axis.set_xlabel("Source X (mm)", fontsize=8)
                if column < 2:
                    axis.set_ylabel(
                        f"Bin {cardiac_bin + 1}\nSource X (mm)", fontsize=8
                    )
                else:
                    axis.set_ylabel(
                        f"Bin {cardiac_bin + 1}\nSource Y (mm)", fontsize=8
                    )

        titles = (
            "Magnitude, central plane",
            "Axial velocity, central plane",
            f"Speed at stenosis (Z = {readout[throat_index]:.0f} mm)",
        )
        for axis, title in zip(axes[0], titles, strict=True):
            axis.set_title(title, fontsize=10, color="#172033")
        labels = ("Magnitude (a.u.)", "Axial velocity (m/s)", "Speed (m/s)")
        for column, label in enumerate(labels):
            colorbar = fig.colorbar(
                images[column],
                ax=axes[:, column],
                location="bottom",
                shrink=0.78,
                pad=0.09,
            )
            colorbar.set_label(label, fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
        fig.suptitle(
            "dp150 phase-contrast MRI reconstruction",
            fontsize=14,
            color="#172033",
        )
        fig.savefig(
            path,
            dpi=260,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)
    return {
        "center_slice_index": slice_index,
        "center_slice_source_Y_mm": float(slice_[slice_index]),
        "stenosis_readout_index": throat_index,
        "stenosis_source_Z_mm": float(readout[throat_index]),
        "display_limits": {
            "magnitude_p99_5": magnitude_limit,
            "absolute_axial_velocity_p99_5_m_per_s": velocity_limit,
            "speed_p99_5_m_per_s": speed_limit,
        },
        "longitudinal_crop_mm": longitudinal_extent,
        "transverse_crop_mm": transverse_extent,
    }


def render_anatomy(
    reconstruction: dict[str, Any],
    support: np.ndarray,
    readout: np.ndarray,
    phase: np.ndarray,
    slice_: np.ndarray,
    throat_index: int,
    path: Path,
) -> dict[str, Any]:
    magnitude = reconstruction["magnitude"][0]
    slice_index, _ = center_plane_indices(support)
    magnitude_limit = max(robust_quantile(magnitude, support, 0.995), 1e-8)
    magnitude_cmap = mri_colormap("gray")

    full_readout = coordinate_slice(readout, readout.min(), readout.max())
    longitudinal_phase = coordinate_slice(phase, -18.0, 18.0)
    zoom_readout = coordinate_slice(
        readout, readout[throat_index] - 58.0, readout[throat_index] + 58.0
    )
    transverse_phase = coordinate_slice(phase, -9.0, 9.0)
    transverse_slice = coordinate_slice(slice_, -9.0, 9.0)

    full_extent = image_extent(
        readout, phase, full_readout, longitudinal_phase
    )
    zoom_extent = image_extent(
        readout, phase, zoom_readout, longitudinal_phase
    )
    transverse_extent = image_extent(
        phase, slice_, transverse_phase, transverse_slice
    )

    center_plane = magnitude[slice_index]
    throat_plane = magnitude[:, :, throat_index]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
        }
    ):
        fig = plt.figure(figsize=(12.8, 5.9), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, height_ratios=(0.72, 1.0))
        full_axis = fig.add_subplot(grid[0, :])
        zoom_axis = fig.add_subplot(grid[1, :2])
        transverse_axis = fig.add_subplot(grid[1, 2])

        panels = (
            (
                full_axis,
                center_plane[longitudinal_phase, full_readout],
                full_extent,
                "Full phantom",
            ),
            (
                zoom_axis,
                center_plane[longitudinal_phase, zoom_readout],
                zoom_extent,
                "Stenosis region",
            ),
            (
                transverse_axis,
                throat_plane[transverse_slice, transverse_phase],
                transverse_extent,
                f"Stenosis cross-section (Z = {readout[throat_index]:.0f} mm)",
            ),
        )
        image = None
        for axis, data, extent, title in panels:
            image = axis.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=magnitude_cmap,
                vmin=0.0,
                vmax=magnitude_limit,
                interpolation="bilinear",
                interpolation_stage="rgba",
                aspect="auto" if axis is full_axis else "equal",
                rasterized=True,
            )
            style_image_axis(axis)
            axis.set_title(title, fontsize=10, color="#172033")
            axis.set_ylabel("Source X (mm)" if axis is not transverse_axis else "Source Y (mm)", fontsize=8)
            axis.set_xlabel(
                "MRI readout / source Z (mm)"
                if axis is not transverse_axis
                else "Source X (mm)",
                fontsize=8,
            )
        if image is not None:
            colorbar = fig.colorbar(
                image,
                ax=(full_axis, zoom_axis, transverse_axis),
                location="right",
                shrink=0.75,
                pad=0.02,
            )
            colorbar.set_label("Magnitude (a.u.)", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
        fig.suptitle(
            "dp150 KomaMRI reference magnitude, cardiac bin 1",
            fontsize=14,
            color="#172033",
        )
        fig.savefig(
            path,
            dpi=260,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)
    return {
        "cardiac_bin": 1,
        "center_slice_index": slice_index,
        "magnitude_display_p99_5": magnitude_limit,
        "full_crop_mm": full_extent,
        "stenosis_crop_mm": zoom_extent,
        "transverse_crop_mm": transverse_extent,
        "full_panel_transverse_scale_expanded": True,
    }


def render_charles_style(
    reconstruction: dict[str, Any],
    support: np.ndarray,
    readout: np.ndarray,
    phase: np.ndarray,
    slice_: np.ndarray,
    throat_index: int,
    cardiac_bin: int,
    path: Path,
) -> dict[str, Any]:
    magnitude = reconstruction["magnitude"][cardiac_bin]
    slice_index, phase_index = center_plane_indices(support)
    magnitude_limit = max(robust_quantile(magnitude, support, 0.995), 1e-8)
    magnitude_cmap = mri_colormap("gray")

    occupied_readout = np.where(support.any(axis=(0, 1)))[0]
    readout_min = readout[max(0, int(occupied_readout.min()) - 2)]
    readout_max = readout[min(readout.size - 1, int(occupied_readout.max()) + 2)]
    readout_crop = coordinate_slice(readout, readout_min, readout_max)
    phase_crop = coordinate_slice(phase, -18.0, 18.0)
    slice_crop = coordinate_slice(slice_, -18.0, 18.0)
    transverse_phase = coordinate_slice(phase, -16.0, 16.0)
    transverse_slice = coordinate_slice(slice_, -16.0, 16.0)

    xy_extent = image_extent(readout, phase, readout_crop, phase_crop)
    xz_extent = image_extent(readout, slice_, readout_crop, slice_crop)
    yz_extent = image_extent(
        phase, slice_, transverse_phase, transverse_slice
    )

    projection_xy = support_rms_projection(magnitude, support, axis=0)
    projection_xz = support_rms_projection(magnitude, support, axis=1)
    projection_yz = support_rms_projection(magnitude, support, axis=2)
    panels = (
        (
            projection_xy[phase_crop, readout_crop],
            xy_extent,
            "Projection - XY",
            "MRI readout / source Z (mm)",
            "Source X (mm)",
            "auto",
        ),
        (
            projection_xz[slice_crop, readout_crop],
            xz_extent,
            "Projection - XZ",
            "MRI readout / source Z (mm)",
            "Source Y (mm)",
            "auto",
        ),
        (
            projection_yz[transverse_slice, transverse_phase],
            yz_extent,
            "Projection - YZ",
            "Source X (mm)",
            "Source Y (mm)",
            "equal",
        ),
        (
            magnitude[slice_index][phase_crop, readout_crop],
            xy_extent,
            "Central slice - XY",
            "MRI readout / source Z (mm)",
            "Source X (mm)",
            "auto",
        ),
        (
            magnitude[:, phase_index, :][slice_crop, readout_crop],
            xz_extent,
            "Central slice - XZ",
            "MRI readout / source Z (mm)",
            "Source Y (mm)",
            "auto",
        ),
        (
            magnitude[:, :, throat_index][transverse_slice, transverse_phase],
            yz_extent,
            f"Stenosis slice - YZ (Z = {readout[throat_index]:.0f} mm)",
            "Source X (mm)",
            "Source Y (mm)",
            "equal",
        ),
    )

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(13.8, 6.2),
            constrained_layout=True,
            gridspec_kw={"width_ratios": (2.55, 2.55, 1.0)},
        )
        for axis, (data, extent, title, xlabel, ylabel, aspect) in zip(
            axes.ravel(), panels, strict=True
        ):
            axis.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=magnitude_cmap,
                vmin=0.0,
                vmax=magnitude_limit,
                interpolation="bilinear",
                interpolation_stage="rgba",
                aspect=aspect,
                rasterized=True,
            )
            style_image_axis(axis)
            axis.set_title(title, fontsize=10, color="#172033")
            axis.set_xlabel(xlabel, fontsize=8)
            axis.set_ylabel(ylabel, fontsize=8)
        fig.suptitle(
            f"dp150 KomaMRI magnitude | cardiac bin {cardiac_bin + 1}",
            fontsize=14,
            color="#172033",
        )
        fig.savefig(
            path,
            dpi=260,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)
    return {
        "cardiac_bin": cardiac_bin + 1,
        "projection_method": "root mean square within seed-derived lumen support",
        "slice_method": "raw reconstructed magnitude",
        "center_slice_source_Y_mm": float(slice_[slice_index]),
        "center_phase_source_X_mm": float(phase[phase_index]),
        "stenosis_source_Z_mm": float(readout[throat_index]),
        "magnitude_display_p99_5": magnitude_limit,
        "longitudinal_transverse_scale_expanded": True,
    }


def image_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dimensions = list(image.size)
        mode = image.mode
        alpha = (
            list(image.getchannel("A").getextrema())
            if "A" in image.getbands()
            else None
        )
    return {
        "path": str(path.resolve()),
        "dimensions_px": dimensions,
        "mode": mode,
        "alpha_extrema": alpha,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    motion_path = args.motion.resolve()
    support_path = args.support_h5.resolve()
    output_dir = args.output_dir.resolve()

    reconstruction = load_reconstruction(motion_path)
    cardiac_bins = validate_two_bin_input(reconstruction)
    output_dir.mkdir(parents=True, exist_ok=True)
    support, fov_mm, matrix, support_metadata = load_particle_support(support_path)
    if reconstruction["magnitude"].shape[1:] != support.shape:
        raise ValueError("Reconstruction and particle support shapes do not match")
    readout, phase, slice_ = coordinate_centers(support.shape, fov_mm)
    throat_index, support_area = find_throat(support)

    comparison_path = output_dir / "dp150_pc_mri_clean_comparison.png"
    anatomy_path = output_dir / "dp150_mri_clean_anatomy.png"
    charles_paths = [
        output_dir / "dp150_mri_charles_style_bin1.png",
        output_dir / "dp150_mri_charles_style_bin2.png",
    ]
    comparison_metadata = render_pc_comparison(
        reconstruction,
        support,
        readout,
        phase,
        slice_,
        throat_index,
        comparison_path,
    )
    anatomy_metadata = render_anatomy(
        reconstruction,
        support,
        readout,
        phase,
        slice_,
        throat_index,
        anatomy_path,
    )
    charles_metadata = [
        render_charles_style(
            reconstruction,
            support,
            readout,
            phase,
            slice_,
            throat_index,
            cardiac_bin,
            charles_paths[cardiac_bin],
        )
        for cardiac_bin in range(cardiac_bins)
    ]
    outputs = [
        image_record(path)
        for path in (comparison_path, anatomy_path, *charles_paths)
    ]
    report = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "compact MRI-focused publication figures",
        "inputs": {
            "motion_reconstruction": {
                "path": str(motion_path),
                "sha256": sha256_file(motion_path),
            },
            "particle_support": {
                "path": str(support_path),
                "sha256": sha256_file(support_path),
            },
        },
        "spatial_resolution_mm": (fov_mm / matrix).tolist(),
        "cardiac_bins_available": cardiac_bins,
        "cardiac_bins_required_by_renderer": EXPECTED_CARDIAC_BINS,
        "display_processing": {
            "visible_wall_contours": False,
            "visible_mask_panels": False,
            "data_smoothing": False,
            "image_interpolation": "bilinear display interpolation only",
            "magnitude_support": "raw reconstructed magnitude; crop only",
            "velocity_support": "stored 5% signal mask intersected with lumen support",
        },
        "support": support_metadata,
        "throat": {
            "readout_index": throat_index,
            "source_Z_mm": float(readout[throat_index]),
            "cross_section_voxels": int(support_area[throat_index]),
        },
        "comparison_figure": comparison_metadata,
        "anatomy_figure": anatomy_metadata,
        "charles_style_figures": charles_metadata,
        "outputs": outputs,
        "interpretation_limit": (
            "Only two reconstructed cardiac bins are available; these figures "
            "must not be presented as a continuous temporal cine."
        ),
    }
    report_path = output_dir / "dp150_clean_mri_figure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    readme = "\n".join(
        [
            "# dp150 Clean MRI Figures",
            "",
            "Recommended result figures:",
            "",
            "- `dp150_mri_charles_style_bin1.png`: primary Charles-style magnitude figure.",
            "- `dp150_pc_mri_clean_comparison.png`: primary two-bin PC-MRI figure.",
            "- `dp150_mri_clean_anatomy.png`: compact reference-magnitude anatomy figure.",
            "- `dp150_mri_charles_style_bin2.png`: Charles-style second-bin magnitude figure.",
            "",
            "The support mask is not drawn. It is used only to suppress invalid background phase.",
            "No data smoothing was applied; bilinear interpolation affects display pixels only.",
            "The reconstruction has 2 mm isotropic voxels and only two cardiac bins.",
            "The previous mask-audit and multi-cross-section figures are QC, not recommended result figures.",
            "",
            "Machine-readable provenance: `dp150_clean_mri_figure_report.json`",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Clean dp150 MRI figures: PASS\nReport: {report_path}")


if __name__ == "__main__":
    main()
