r"""Render aorta Koma reference or phase-contrast reconstruction panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="aorta")
    parser.add_argument(
        "--display-mask-fraction",
        type=float,
        default=0.15,
        help=(
            "Per-bin magnitude fraction used only to suppress background "
            "artifacts in the six-panel display (default: 0.15)."
        ),
    )
    parser.add_argument(
        "--slice-mode",
        choices=("max-support", "center"),
        default="max-support",
        help=(
            "Select representative planes by maximum vessel area or use the "
            "array midpoint (default: max-support)."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_limit(volume: np.ndarray, mask: np.ndarray) -> float:
    positive = volume[mask & (volume > 0)]
    return float(np.quantile(positive, 0.995)) if positive.size else 1.0


def coordinate_centers(
    volume_shape: tuple[int, int, int], fov_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz, ny, nx = volume_shape
    x_mm = -fov_mm[0] / 2 + (np.arange(nx) + 0.5) * fov_mm[0] / nx
    y_mm = -fov_mm[1] / 2 + (np.arange(ny) + 0.5) * fov_mm[1] / ny
    z_mm = -fov_mm[2] / 2 + (np.arange(nz) + 0.5) * fov_mm[2] / nz
    return x_mm, y_mm, z_mm


def build_display_mask(
    volume: np.ndarray, mask_fraction: float
) -> tuple[np.ndarray, dict[str, object]]:
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"Expected a finite 3D magnitude volume, got {volume.shape}")
    maximum = float(volume.max())
    if maximum <= 0:
        raise ValueError("Magnitude volume is blank")
    threshold = mask_fraction * maximum
    threshold_mask = volume >= threshold
    regions, region_count = connected_components(
        threshold_mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    region_sizes = np.bincount(regions.ravel())[1:]
    if not region_sizes.size:
        raise RuntimeError("Display threshold produced no connected components")
    largest_region = int(np.argmax(region_sizes)) + 1
    display_mask = regions == largest_region
    threshold_voxels = int(threshold_mask.sum())
    retained_voxels = int(display_mask.sum())
    return display_mask, {
        "mask_fraction_of_bin_maximum": mask_fraction,
        "threshold_magnitude": threshold,
        "connectivity": 26,
        "threshold_component_count": int(region_count),
        "threshold_voxels": threshold_voxels,
        "retained_largest_component_voxels": retained_voxels,
        "removed_threshold_voxels": threshold_voxels - retained_voxels,
        "retained_threshold_voxel_fraction": retained_voxels / threshold_voxels,
    }


def representative_indices(mask: np.ndarray, mode: str) -> tuple[int, int, int]:
    if mode == "center":
        return tuple(int(size // 2) for size in mask.shape)
    return (
        int(np.argmax(mask.sum(axis=(1, 2)))),
        int(np.argmax(mask.sum(axis=(0, 2)))),
        int(np.argmax(mask.sum(axis=(0, 1)))),
    )


def panel_data(
    volume: np.ndarray,
    display_mask: np.ndarray,
    fov_mm: np.ndarray,
    slice_mode: str,
) -> tuple[list[tuple[str, np.ndarray, str, str, list[float]]], dict[str, object]]:
    if display_mask.shape != volume.shape:
        raise ValueError("Display mask must match magnitude volume")
    nz, ny, nx = volume.shape
    x_extent = [-fov_mm[0] / 2, fov_mm[0] / 2]
    y_extent = [-fov_mm[1] / 2, fov_mm[1] / 2]
    z_extent = [-fov_mm[2] / 2, fov_mm[2] / 2]
    x_mm, y_mm, z_mm = coordinate_centers(volume.shape, fov_mm)
    z_index, y_index, x_index = representative_indices(display_mask, slice_mode)
    display_volume = np.where(display_mask, volume, 0.0)
    slice_prefix = "Representative" if slice_mode == "max-support" else "Center"
    panels = [
        (
            "Projection - XY",
            display_volume.max(axis=0),
            "MRI readout / source Z (mm)",
            "MRI phase / source X (mm)",
            [*x_extent, *y_extent],
        ),
        (
            "Projection - XZ",
            display_volume.max(axis=1),
            "MRI readout / source Z (mm)",
            "MRI slice / source Y (mm)",
            [*x_extent, *z_extent],
        ),
        (
            "Projection - YZ",
            display_volume.max(axis=2),
            "MRI phase / source X (mm)",
            "MRI slice / source Y (mm)",
            [*y_extent, *z_extent],
        ),
        (
            f"{slice_prefix} slice - XY (Z = {z_mm[z_index]:.1f} mm)",
            display_volume[z_index],
            "MRI readout / source Z (mm)",
            "MRI phase / source X (mm)",
            [*x_extent, *y_extent],
        ),
        (
            f"{slice_prefix} slice - XZ (Y = {y_mm[y_index]:.1f} mm)",
            display_volume[:, y_index, :],
            "MRI readout / source Z (mm)",
            "MRI slice / source Y (mm)",
            [*x_extent, *z_extent],
        ),
        (
            f"{slice_prefix} slice - YZ (X = {x_mm[x_index]:.1f} mm)",
            display_volume[:, :, x_index],
            "MRI phase / source X (mm)",
            "MRI slice / source Y (mm)",
            [*y_extent, *z_extent],
        ),
    ]
    return panels, {
        "mode": slice_mode,
        "indices_python_slice_phase_readout": [z_index, y_index, x_index],
        "coordinates_mm": {
            "XY_plane_source_Y": float(z_mm[z_index]),
            "XZ_plane_source_X": float(y_mm[y_index]),
            "YZ_plane_source_Z": float(x_mm[x_index]),
        },
        "support_voxels_in_planes": {
            "XY": int(display_mask[z_index].sum()),
            "XZ": int(display_mask[:, y_index, :].sum()),
            "YZ": int(display_mask[:, :, x_index].sum()),
        },
    }


def render_bin(
    volume: np.ndarray,
    fov_mm: np.ndarray,
    phase: int,
    path: Path,
    label: str,
    mask_fraction: float,
    slice_mode: str,
) -> dict[str, object]:
    display_mask, processing = build_display_mask(volume, mask_fraction)
    panels, slice_selection = panel_data(
        volume, display_mask, fov_mm, slice_mode
    )
    limit = robust_limit(volume, display_mask)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.6), constrained_layout=True)
    for axis, (title, image, xlabel, ylabel, extent) in zip(
        axes.ravel(), panels, strict=True
    ):
        axis.imshow(
            image,
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title, fontsize=12, color="#172033")
        axis.set_xlabel(xlabel, fontsize=10)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.tick_params(labelsize=9, colors="#334155")
        for spine in axis.spines.values():
            spine.set_color("#64748b")
            spine.set_linewidth(0.7)
        axis.grid(False)
    title_label = label.replace("_", " ")
    fig.suptitle(
        f"{title_label} KomaMRI magnitude | cardiac bin {phase}",
        fontsize=16,
        color="#172033",
    )
    fig.text(
        0.5,
        0.008,
        (
            f"Display only: {100 * mask_fraction:.0f}% of per-bin maximum; "
            "largest 26-connected component; raw HDF5 unchanged"
        ),
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return {
        "display_processing": processing,
        "slice_selection": slice_selection,
        "display_intensity_vmax": limit,
    }


def render_pc_summary(magnitude, velocity, speed, mask, fov_mm, path, label):
    if velocity.shape != (3, *magnitude.shape):
        raise ValueError(f"unexpected velocity shape {velocity.shape}")
    valid_velocity = np.abs(velocity[0][mask])
    valid_speed = speed[mask]
    velocity_limit = max(
        float(np.quantile(valid_velocity, 0.995)) if valid_velocity.size else 0.0,
        1e-6,
    )
    speed_limit = max(
        float(np.quantile(valid_speed, 0.995)) if valid_speed.size else 0.0,
        1e-6,
    )
    extent = [-fov_mm[0] / 2, fov_mm[0] / 2, -fov_mm[1] / 2, fov_mm[1] / 2]
    fig, axes = plt.subplots(
        magnitude.shape[0],
        3,
        figsize=(18, 4.2 * magnitude.shape[0]),
        constrained_layout=True,
        squeeze=False,
    )
    for phase in range(magnitude.shape[0]):
        mag = magnitude[phase]
        weights = np.where(mask[phase], mag, 0.0)
        masked_magnitude = np.where(mask[phase], mag, 0.0)
        denominator = weights.sum(axis=0)
        readout_velocity = np.divide(
            (velocity[0, phase] * weights).sum(axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        projected_speed = np.divide(
            (speed[phase] * weights).sum(axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        images = [
            axes[phase, 0].imshow(
                masked_magnitude.max(axis=0),
                cmap="gray",
                origin="lower",
                extent=extent,
                vmin=0,
                vmax=robust_limit(mag, mask[phase]),
                aspect="equal",
            ),
            axes[phase, 1].imshow(
                readout_velocity,
                cmap="RdBu_r",
                origin="lower",
                extent=extent,
                vmin=-velocity_limit,
                vmax=velocity_limit,
                aspect="equal",
            ),
            axes[phase, 2].imshow(
                projected_speed,
                cmap="turbo",
                origin="lower",
                extent=extent,
                vmin=0,
                vmax=speed_limit,
                aspect="equal",
            ),
        ]
        titles = ["Reference magnitude", "Readout velocity", "Speed"]
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            axes[phase, column].set_title(f"{title} | bin {phase + 1}")
            axes[phase, column].set_xlabel("MRI readout / source Z (mm)")
            axes[phase, column].set_ylabel("MRI phase / source X (mm)")
            fig.colorbar(image, ax=axes[phase, column], shrink=0.72)
    fig.suptitle(f"{label} phase-contrast MRI projections")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 < args.display_mask_fraction <= 1:
        raise ValueError("--display-mask-fraction must be in (0, 1]")
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as h5:
        magnitude = np.asarray(h5["reference_magnitude"], dtype=np.float32)
        fov_mm = np.asarray(h5.attrs["FOV_m"], dtype=float) * 1e3
        rr = float(h5.attrs["RR_s"])
        venc = float(h5.attrs["Venc_m_per_s"])
        velocity = (
            np.asarray(h5["velocity"], dtype=np.float32)
            if "velocity" in h5
            else None
        )
        speed = (
            np.asarray(h5["speed"], dtype=np.float32)
            if "speed" in h5
            else None
        )
        mask = np.asarray(h5["mask"], dtype=bool) if "mask" in h5 else None
    if magnitude.ndim != 4 or not np.isfinite(magnitude).all():
        raise RuntimeError(f"invalid reference magnitude: {magnitude.shape}")
    outputs = []
    for index in range(magnitude.shape[0]):
        path = output_dir / f"{args.label}_bin{index + 1}_projection_slices.png"
        render_metadata = render_bin(
            magnitude[index],
            fov_mm,
            index + 1,
            path,
            args.label,
            args.display_mask_fraction,
            args.slice_mode,
        )
        with Image.open(path) as image:
            outputs.append(
                {
                    "cardiac_bin": index + 1,
                    "path": str(path),
                    "dimensions_px": list(image.size),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    **render_metadata,
                }
            )
    if velocity is not None and speed is not None and mask is not None:
        pc_path = output_dir / f"{args.label}_pc_velocity_projections.png"
        render_pc_summary(
            magnitude, velocity, speed, mask, fov_mm, pc_path, args.label
        )
        with Image.open(pc_path) as image:
            outputs.append(
                {
                    "type": "phase_contrast_velocity_summary",
                    "path": str(pc_path),
                    "dimensions_px": list(image.size),
                    "bytes": pc_path.stat().st_size,
                    "sha256": sha256_file(pc_path),
                }
            )
    report = {
        "status": "PASS_ARTIFACT_CLEANED_DISPLAY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_bytes": input_path.stat().st_size,
        "shape_python": list(magnitude.shape),
        "dimension_order_python": ["cardiac_bin", "slice", "phase", "readout"],
        "FOV_mm": fov_mm.tolist(),
        "RR_s": rr,
        "Venc_m_per_s": venc,
        "display_only": True,
        "raw_hdf5_modified": False,
        "display_mask_fraction": args.display_mask_fraction,
        "display_component_filter": "largest 26-connected threshold component",
        "slice_mode": args.slice_mode,
        "corrections": [
            "Suppress below-threshold background before maximum-intensity projection",
            "Reject disconnected 3D threshold components outside the largest component",
            "Select representative maximum-vessel-area planes and report their coordinates",
            "Label MRI readout, phase, and slice axes with their source-coordinate mapping",
        ],
        "scientific_limit": (
            "The 15% threshold is a presentation choice, not an externally "
            "validated lumen or physical-wall segmentation."
        ),
        "axis_mapping": {
            "MRI_X_readout": "source Z",
            "MRI_Y_phase": "source X",
            "MRI_Z_slice": "source Y",
        },
        "outputs": outputs,
    }
    report_path = output_dir / f"{args.label}_projection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Projection/slice QC: {report['status']}\nReport: {report_path}")


if __name__ == "__main__":
    main()
