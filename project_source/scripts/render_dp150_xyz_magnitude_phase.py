r"""Render dp150 X/Y/Z encoded magnitude and phase-difference images."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
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
    from scripts.analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        DEFAULT_RECONSTRUCTION,
        DEFAULT_SUPPORT,
        draw_support_outline,
        image_record,
        load_support,
        masked_colormap,
        resolve_particle_label,
        save_gif,
        save_mp4,
        sha256_file,
    )
    from scripts.plot_dp150_koma_publication import (
        coordinate_centers,
        masked_max_projection,
        support_crop,
    )
except ModuleNotFoundError:
    from analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        DEFAULT_RECONSTRUCTION,
        DEFAULT_SUPPORT,
        draw_support_outline,
        image_record,
        load_support,
        masked_colormap,
        resolve_particle_label,
        save_gif,
        save_mp4,
        sha256_file,
    )
    from plot_dp150_koma_publication import (
        coordinate_centers,
        masked_max_projection,
        support_crop,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = DEFAULT_RECONSTRUCTION.parent
ENCODINGS = ("X", "Y", "Z")
ENCODING_LABELS = {
    "X": "PC-X: readout / source Z",
    "Y": "PC-Y: phase / source X",
    "Z": "PC-Z: slice / source Y",
}
RAW_FILENAMES = {
    "REF": "ref_result.jld2",
    "X": "x_result.jld2",
    "Y": "y_result.jld2",
    "Z": "z_result.jld2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
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
    parser.add_argument("--cardiac-bin", type=int, default=None)
    parser.add_argument("--frame-duration-ms", type=int, default=420)
    parser.add_argument("--mp4-cycles", type=int, default=2)
    parser.add_argument(
        "--particle-label",
        default=None,
        help="Particle-count label for figures (for example, 100k or 1M).",
    )
    return parser.parse_args()


def compound_to_complex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    fields = values.dtype.fields
    if fields is not None and {"re", "im"}.issubset(fields):
        return values["re"].astype(np.float32) + 1j * values["im"].astype(
            np.float32
        )
    if np.iscomplexobj(values):
        return values.astype(np.complex64, copy=False)
    raise TypeError(f"Expected complex or re/im compound data, got {values.dtype}")


def load_reconstruction(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        required = ("reference_magnitude", "velocity", "speed", "mask")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing reconstruction datasets: {missing}")
        result = {
            "magnitude": np.asarray(source["reference_magnitude"], dtype=np.float32),
            "velocity": np.asarray(source["velocity"], dtype=np.float32),
            "speed": np.asarray(source["speed"], dtype=np.float32),
            "mask": np.asarray(source["mask"], dtype=bool),
            "RR_s": float(source.attrs["RR_s"]),
            "Venc_m_per_s": float(source.attrs["Venc_m_per_s"]),
            "phase_formula": source.attrs["phase_formula"].decode()
            if isinstance(source.attrs["phase_formula"], bytes)
            else str(source.attrs["phase_formula"]),
        }
    expected = result["magnitude"].shape
    if len(expected) != 4:
        raise ValueError(f"Expected (bin,slice,phase,readout), got {expected}")
    if result["velocity"].shape != (3, *expected):
        raise ValueError(f"Unexpected velocity shape {result['velocity'].shape}")
    if result["speed"].shape != expected or result["mask"].shape != expected:
        raise ValueError("Speed or mask shape does not match reference magnitude")
    return result


def open_raw_datasets(
    stack: ExitStack, run_dir: Path, expected_shape: tuple[int, ...]
) -> tuple[dict[str, h5py.Dataset], dict[str, Path]]:
    datasets: dict[str, h5py.Dataset] = {}
    paths: dict[str, Path] = {}
    for encoding, filename in RAW_FILENAMES.items():
        path = (run_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source = stack.enter_context(h5py.File(path, "r"))
        if "image" not in source:
            raise KeyError(f"Missing image dataset in {path}")
        dataset = source["image"]
        if dataset.shape != expected_shape:
            raise ValueError(
                f"{encoding} image shape {dataset.shape} != {expected_shape}"
            )
        datasets[encoding] = dataset
        paths[encoding] = path
    return datasets, paths


def circular_phase_projection(
    phase: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    axis: int,
) -> np.ndarray:
    if phase.shape != weights.shape or phase.shape != mask.shape:
        raise ValueError("Phase, weights, and mask must have identical shapes")
    weighted_phasor = np.where(
        mask,
        weights.astype(np.float64) * np.exp(1j * phase.astype(np.float64)),
        0.0,
    )
    summed = weighted_phasor.sum(axis=axis)
    denominator = np.where(mask, weights, 0.0).sum(axis=axis)
    projected = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    projected[valid] = np.angle(summed[valid])
    return projected


def calculate_global_scales_and_validate(
    raw: dict[str, h5py.Dataset],
    reconstruction: dict[str, Any],
    support: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    magnitude_values = []
    phase_values = []
    maximum_reference_error = 0.0
    maximum_phase_error = {encoding: 0.0 for encoding in ENCODINGS}
    phase_clipping_counts = {encoding: 0 for encoding in ENCODINGS}
    valid_phase_count = 0
    venc = reconstruction["Venc_m_per_s"]

    for index in range(reconstruction["magnitude"].shape[0]):
        reference = compound_to_complex(raw["REF"][index])
        reference_magnitude = np.abs(reference)
        maximum_reference_error = max(
            maximum_reference_error,
            float(
                np.max(
                    np.abs(reference_magnitude - reconstruction["magnitude"][index])
                )
            ),
        )
        signal_mask = support & reconstruction["mask"][index]
        valid_phase_count += int(signal_mask.sum())
        for component, encoding in enumerate(ENCODINGS):
            encoded = compound_to_complex(raw[encoding][index])
            encoded_magnitude = np.abs(encoded)
            phase = np.angle(encoded * np.conj(reference)).astype(np.float32)
            expected_phase = (
                reconstruction["velocity"][component, index] * np.pi / venc
            )
            maximum_phase_error[encoding] = max(
                maximum_phase_error[encoding],
                float(np.max(np.abs(phase[reconstruction["mask"][index]] - expected_phase[reconstruction["mask"][index]]))),
            )
            magnitude_values.append(encoded_magnitude[support])
            phase_values.append(np.abs(phase[signal_mask]))

    all_magnitude = np.concatenate(magnitude_values)
    all_phase = np.concatenate(phase_values)
    magnitude_limit = max(float(np.quantile(all_magnitude, 0.995)), 1e-6)
    phase_limit = max(float(np.quantile(all_phase, 0.995)), 1e-6)
    phase_limit = min(float(np.pi), phase_limit)

    for index in range(reconstruction["magnitude"].shape[0]):
        reference = compound_to_complex(raw["REF"][index])
        signal_mask = support & reconstruction["mask"][index]
        for encoding in ENCODINGS:
            encoded = compound_to_complex(raw[encoding][index])
            phase = np.angle(encoded * np.conj(reference))
            phase_clipping_counts[encoding] += int(
                (np.abs(phase[signal_mask]) > phase_limit).sum()
            )

    return (
        {
            "encoded_magnitude_p99_5_au": magnitude_limit,
            "absolute_phase_p99_5_rad": phase_limit,
        },
        {
            "maximum_reference_magnitude_absolute_error": maximum_reference_error,
            "maximum_phase_difference_absolute_error_rad": maximum_phase_error,
            "phase_display_clipped_voxels": phase_clipping_counts,
            "phase_display_clipped_fraction": {
                encoding: count / valid_phase_count
                for encoding, count in phase_clipping_counts.items()
            },
            "valid_lumen_voxels_per_encoding_all_bins": valid_phase_count,
        },
    )


def load_bin(
    raw: dict[str, h5py.Dataset],
    reconstruction: dict[str, Any],
    support: np.ndarray,
    index: int,
) -> dict[str, Any]:
    reference = compound_to_complex(raw["REF"][index])
    encoded = {
        encoding: compound_to_complex(raw[encoding][index])
        for encoding in ENCODINGS
    }
    phases = {
        encoding: np.angle(values * np.conj(reference)).astype(np.float32)
        for encoding, values in encoded.items()
    }
    return {
        "reference": reference,
        "encoded": encoded,
        "phases": phases,
        "signal_mask": support & reconstruction["mask"][index],
    }


def long_axis_images(
    bin_data: dict[str, Any], support: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    reference = bin_data["reference"]
    for encoding in ENCODINGS:
        encoded = bin_data["encoded"][encoding]
        result[encoding] = {
            "magnitude": masked_max_projection(np.abs(encoded), support, axis=0),
            "phase": circular_phase_projection(
                bin_data["phases"][encoding],
                np.abs(encoded) * np.abs(reference),
                bin_data["signal_mask"],
                axis=0,
            ),
        }
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.tick_params(colors="#334155", labelsize=8, length=3, width=0.7)
    for spine in axis.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(0.7)


def render_encoding_summary(
    bin_data: dict[str, Any],
    support: np.ndarray,
    readout: np.ndarray,
    phase_coordinate: np.ndarray,
    index: int,
    rr_s: float,
    cardiac_bins: int,
    particle_label: str,
    scales: dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    images = long_axis_images(bin_data, support)
    support_projection = support.any(axis=0)
    row_slice, column_slice, extent = support_crop(
        support_projection, readout, phase_coordinate, pad_voxels=3
    )
    horizontal = readout[column_slice]
    vertical = phase_coordinate[row_slice]
    outline = support_projection[row_slice, column_slice]

    figure, axes = plt.subplots(
        3, 2, figsize=(15.5, 9.2), constrained_layout=True, facecolor="white"
    )
    magnitude_image = None
    phase_image = None
    for row, encoding in enumerate(ENCODINGS):
        magnitude_image = axes[row, 0].imshow(
            np.ma.masked_invalid(
                images[encoding]["magnitude"][row_slice, column_slice]
            ),
            cmap=masked_colormap("Greys"),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=scales["encoded_magnitude_p99_5_au"],
            interpolation="nearest",
            aspect="auto",
        )
        phase_image = axes[row, 1].imshow(
            np.ma.masked_invalid(images[encoding]["phase"][row_slice, column_slice]),
            cmap=masked_colormap("RdBu_r"),
            origin="lower",
            extent=extent,
            vmin=-scales["absolute_phase_p99_5_rad"],
            vmax=scales["absolute_phase_p99_5_rad"],
            interpolation="nearest",
            aspect="auto",
        )
        for axis in axes[row]:
            draw_support_outline(axis, outline, horizontal, vertical)
            style_axis(axis)
            axis.set_xlabel("MRI readout / source Z (mm)")
            axis.set_ylabel("MRI phase / source X (mm)")
        axes[row, 0].set_title(f"{ENCODING_LABELS[encoding]} | magnitude")
        axes[row, 1].set_title(
            f"{ENCODING_LABELS[encoding]} | phase difference vs REF"
        )
    assert magnitude_image is not None and phase_image is not None
    figure.colorbar(
        magnitude_image,
        ax=axes[:, 0],
        shrink=0.62,
        pad=0.015,
        label="Encoded magnitude (a.u.)",
    )
    figure.colorbar(
        phase_image,
        ax=axes[:, 1],
        shrink=0.62,
        pad=0.015,
        label="Wrapped phase difference (rad)",
    )
    phase_center_ms = 1000.0 * (index + 0.5) * rr_s / cardiac_bins
    figure.suptitle(
        f"dp150 {particle_label} KomaMRI | cardiac bin {index + 1}/{cardiac_bins} | "
        f"phase center {phase_center_ms:.0f} ms",
        fontsize=16,
        color="#172033",
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def render_standalone_pairs(
    bin_data: dict[str, Any],
    support: np.ndarray,
    readout: np.ndarray,
    phase_coordinate: np.ndarray,
    index: int,
    scales: dict[str, float],
    output_dir: Path,
) -> list[Path]:
    images = long_axis_images(bin_data, support)
    support_projection = support.any(axis=0)
    row_slice, column_slice, extent = support_crop(
        support_projection, readout, phase_coordinate, pad_voxels=3
    )
    horizontal = readout[column_slice]
    vertical = phase_coordinate[row_slice]
    outline = support_projection[row_slice, column_slice]
    outputs = []
    for encoding in ENCODINGS:
        figure, axes = plt.subplots(
            1, 2, figsize=(15, 4.5), constrained_layout=True, facecolor="white"
        )
        magnitude_image = axes[0].imshow(
            np.ma.masked_invalid(
                images[encoding]["magnitude"][row_slice, column_slice]
            ),
            cmap=masked_colormap("Greys"),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=scales["encoded_magnitude_p99_5_au"],
            interpolation="nearest",
            aspect="auto",
        )
        phase_image = axes[1].imshow(
            np.ma.masked_invalid(images[encoding]["phase"][row_slice, column_slice]),
            cmap=masked_colormap("RdBu_r"),
            origin="lower",
            extent=extent,
            vmin=-scales["absolute_phase_p99_5_rad"],
            vmax=scales["absolute_phase_p99_5_rad"],
            interpolation="nearest",
            aspect="auto",
        )
        for axis in axes:
            draw_support_outline(axis, outline, horizontal, vertical)
            style_axis(axis)
            axis.set_xlabel("MRI readout / source Z (mm)")
            axis.set_ylabel("MRI phase / source X (mm)")
        axes[0].set_title("Encoded magnitude")
        axes[1].set_title("Phase difference relative to REF")
        figure.colorbar(magnitude_image, ax=axes[0], shrink=0.72, label="a.u.")
        figure.colorbar(phase_image, ax=axes[1], shrink=0.72, label="rad")
        figure.suptitle(
            f"dp150 | {ENCODING_LABELS[encoding]} | peak-flow bin {index + 1}",
            fontsize=15,
            color="#172033",
        )
        path = output_dir / f"dp150_bin{index + 1:02d}_{encoding}_magnitude_phase.png"
        figure.savefig(path, dpi=180, facecolor="white")
        plt.close(figure)
        outputs.append(path)
    return outputs


def render_orthogonal_summary(
    bin_data: dict[str, Any],
    support: np.ndarray,
    readout: np.ndarray,
    phase_coordinate: np.ndarray,
    slice_coordinate: np.ndarray,
    index: int,
    particle_label: str,
    scales: dict[str, float],
    output: Path,
) -> None:
    reference_magnitude = np.abs(bin_data["reference"])
    plane_specs = (
        (
            "Source X-Z projection (through Y)",
            0,
            readout,
            phase_coordinate,
            "Source Z (mm)",
            "Source X (mm)",
        ),
        (
            "Source Y-Z projection (through X)",
            1,
            readout,
            slice_coordinate,
            "Source Z (mm)",
            "Source Y (mm)",
        ),
        (
            "Source X-Y projection (through Z)",
            2,
            phase_coordinate,
            slice_coordinate,
            "Source X (mm)",
            "Source Y (mm)",
        ),
    )
    rows = ("Reference magnitude", *[f"Phase {name}" for name in ENCODINGS])
    figure, axes = plt.subplots(
        4, 3, figsize=(18, 12), constrained_layout=True, facecolor="white"
    )
    magnitude_image = None
    phase_image = None
    for column, (
        title,
        collapse_axis,
        horizontal,
        vertical,
        xlabel,
        ylabel,
    ) in enumerate(plane_specs):
        support_projection = support.any(axis=collapse_axis)
        row_slice, column_slice, extent = support_crop(
            support_projection, horizontal, vertical, pad_voxels=2
        )
        cropped_outline = support_projection[row_slice, column_slice]
        cropped_horizontal = horizontal[column_slice]
        cropped_vertical = vertical[row_slice]
        magnitude_projection = masked_max_projection(
            reference_magnitude, support, axis=collapse_axis
        )
        magnitude_image = axes[0, column].imshow(
            np.ma.masked_invalid(magnitude_projection[row_slice, column_slice]),
            cmap=masked_colormap("Greys"),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=scales["encoded_magnitude_p99_5_au"],
            interpolation="nearest",
            aspect="auto" if column < 2 else "equal",
        )
        for row, encoding in enumerate(ENCODINGS, start=1):
            encoded = bin_data["encoded"][encoding]
            phase_projection = circular_phase_projection(
                bin_data["phases"][encoding],
                np.abs(encoded) * reference_magnitude,
                bin_data["signal_mask"],
                axis=collapse_axis,
            )
            phase_image = axes[row, column].imshow(
                np.ma.masked_invalid(phase_projection[row_slice, column_slice]),
                cmap=masked_colormap("RdBu_r"),
                origin="lower",
                extent=extent,
                vmin=-scales["absolute_phase_p99_5_rad"],
                vmax=scales["absolute_phase_p99_5_rad"],
                interpolation="nearest",
                aspect="auto" if column < 2 else "equal",
            )
        for row in range(4):
            axis = axes[row, column]
            draw_support_outline(
                axis, cropped_outline, cropped_horizontal, cropped_vertical
            )
            style_axis(axis)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
        axes[0, column].set_title(title, fontsize=12)
    for row, label in enumerate(rows):
        axes[row, 0].text(
            -0.2,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            color="#172033",
        )
    assert magnitude_image is not None and phase_image is not None
    figure.colorbar(
        magnitude_image,
        ax=axes[0, :],
        shrink=0.7,
        pad=0.01,
        label="Reference magnitude (a.u.)",
    )
    figure.colorbar(
        phase_image,
        ax=axes[1:, :],
        shrink=0.45,
        pad=0.01,
        label="Wrapped phase difference (rad)",
    )
    figure.suptitle(
        f"dp150 {particle_label} orthogonal magnitude and X/Y/Z phase | bin {index + 1}",
        fontsize=17,
        color="#172033",
    )
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    reconstruction_path = run_dir / "pc_reconstruction.h5"
    support_path = args.support_h5.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "final_image_analysis" / "magnitude_phase_xyz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "cine_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    if args.frame_duration_ms < 20 or args.mp4_cycles < 1:
        raise ValueError("Frame duration must be >=20 ms and MP4 cycles >=1")
    fov_m = np.asarray(args.fov_m, dtype=np.float64)
    matrix = np.asarray(args.matrix, dtype=np.int64)
    reconstruction = load_reconstruction(reconstruction_path)
    particle_label, _ = resolve_particle_label(reconstruction_path, args.particle_label)
    cardiac_bins = int(reconstruction["magnitude"].shape[0])
    expected_spatial_shape = tuple(int(value) for value in matrix[::-1])
    if reconstruction["magnitude"].shape[1:] != expected_spatial_shape:
        raise ValueError(
            f"Reconstruction shape {reconstruction['magnitude'].shape[1:]} "
            f"does not match matrix {matrix.tolist()}"
        )
    support, support_metadata = load_support(support_path, fov_m, matrix)
    readout, phase_coordinate, slice_coordinate = coordinate_centers(
        support.shape, fov_m * 1000.0
    )

    signal_mask = reconstruction["mask"] & support[None, ...]
    p95_speed = np.asarray(
        [
            np.quantile(reconstruction["speed"][index][signal_mask[index]], 0.95)
            for index in range(reconstruction["speed"].shape[0])
        ]
    )
    peak_index = int(np.argmax(p95_speed))
    selected_index = peak_index if args.cardiac_bin is None else args.cardiac_bin - 1
    if selected_index not in range(reconstruction["magnitude"].shape[0]):
        raise ValueError("--cardiac-bin is outside the reconstructed bin range")

    with ExitStack() as stack:
        raw, raw_paths = open_raw_datasets(
            stack, run_dir, reconstruction["magnitude"].shape
        )
        scales, validation = calculate_global_scales_and_validate(
            raw, reconstruction, support
        )
        selected = load_bin(raw, reconstruction, support, selected_index)

        summary_path = output_dir / (
            f"dp150_bin{selected_index + 1:02d}_xyz_encoding_magnitude_phase.png"
        )
        render_encoding_summary(
            selected,
            support,
            readout,
            phase_coordinate,
            selected_index,
            reconstruction["RR_s"],
            cardiac_bins,
            particle_label,
            scales,
            summary_path,
            dpi=180,
        )
        standalone_paths = render_standalone_pairs(
            selected,
            support,
            readout,
            phase_coordinate,
            selected_index,
            scales,
            output_dir,
        )
        orthogonal_path = output_dir / (
            f"dp150_bin{selected_index + 1:02d}_orthogonal_magnitude_phase.png"
        )
        render_orthogonal_summary(
            selected,
            support,
            readout,
            phase_coordinate,
            slice_coordinate,
            selected_index,
            particle_label,
            scales,
            orthogonal_path,
        )

        frame_paths = []
        for index in range(reconstruction["magnitude"].shape[0]):
            frame_path = frame_dir / f"frame_{index + 1:03d}.png"
            render_encoding_summary(
                load_bin(raw, reconstruction, support, index),
                support,
                readout,
                phase_coordinate,
                index,
                reconstruction["RR_s"],
                cardiac_bins,
                particle_label,
                scales,
                frame_path,
                dpi=120,
            )
            frame_paths.append(frame_path)

    gif_path = output_dir / f"dp150_xyz_magnitude_phase_{cardiac_bins}_bins.gif"
    mp4_path = output_dir / f"dp150_xyz_magnitude_phase_{cardiac_bins}_bins.mp4"
    save_gif(frame_paths, gif_path, args.frame_duration_ms)
    video = save_mp4(
        frame_paths,
        mp4_path,
        args.frame_duration_ms,
        args.mp4_cycles,
    )

    image_paths = [summary_path, orthogonal_path, *standalone_paths, gif_path]
    report = {
        "schema_version": 1,
        "status": "PASS_WITH_QUALITATIVE_LIMITATIONS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "reconstruction": {
                "path": str(reconstruction_path),
                "sha256": sha256_file(reconstruction_path),
            },
            "seed_derived_support": {
                "path": str(support_path),
                "sha256": sha256_file(support_path),
            },
            "raw_complex_images": {
                encoding: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for encoding, path in raw_paths.items()
            },
        },
        "acquisition": {
            "particle_count_label": particle_label,
            "cardiac_bins": int(reconstruction["magnitude"].shape[0]),
            "RR_s": reconstruction["RR_s"],
            "Venc_m_per_s": reconstruction["Venc_m_per_s"],
            "phase_formula": reconstruction["phase_formula"],
        },
        "grid": {
            "FOV_m": fov_m.tolist(),
            "matrix_readout_phase_slice": matrix.tolist(),
            "array_order_python": ["cardiac_bin", "slice", "phase", "readout"],
            "voxel_size_mm": (1000.0 * fov_m / matrix).tolist(),
        },
        "axis_mapping": {
            "PC_X": "MRI readout / source Z",
            "PC_Y": "MRI phase / source X",
            "PC_Z": "MRI slice / source Y",
        },
        "selected_image": {
            "selection": "maximum exact-lumen p95 speed"
            if args.cardiac_bin is None
            else "user-selected cardiac bin",
            "cardiac_bin": selected_index + 1,
            "phase_center_s": (selected_index + 0.5)
            * reconstruction["RR_s"]
            / reconstruction["magnitude"].shape[0],
            "p95_speed_m_per_s": float(p95_speed[selected_index]),
        },
        "support": support_metadata,
        "display": {
            "magnitude": "maximum projection of each raw encoded magnitude within exact seed-derived support",
            "phase": "magnitude-product-weighted circular projection of angle(encoded * conj(REF)) within stored signal mask AND exact support",
            "phase_wrapping_range_rad": [-float(np.pi), float(np.pi)],
            "shared_display_scales": scales,
            "image_interpolation": "nearest",
            "temporal_interpolation": False,
            "transverse_scale_expanded_in_long_axis_views": True,
        },
        "validation": validation,
        "outputs": [image_record(path) for path in image_paths],
        "video": video,
        "scientific_limits": [
            "Phase panels are phase differences relative to REF, not absolute MRI phase.",
            "Projection panels are not individual acquired slices; phase uses circular projection to preserve wrapping.",
            "The seed-derived lumen support changes display masking only and does not alter reconstruction data.",
            "The REF/X/Y/Z acquisition is separate Cartesian encoding; equivalence to a balanced Hadamard acquisition is not established.",
            "Particle-density convergence against a higher-count reconstruction has not been established.",
        ],
    }
    report_path = output_dir / "dp150_xyz_magnitude_phase_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "PASS_WITH_QUALITATIVE_LIMITATIONS\n"
        f"Selected cardiac bin: {selected_index + 1}\n"
        f"Summary: {summary_path}\n"
        f"Orthogonal: {orthogonal_path}\n"
        f"GIF: {gif_path}\n"
        f"MP4: {mp4_path if video['created'] else video['reason']}\n"
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
