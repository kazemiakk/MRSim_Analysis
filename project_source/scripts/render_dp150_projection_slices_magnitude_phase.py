r"""Render dp150 magnitude and directional phase in Charles-style six-panel views."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        DEFAULT_RECONSTRUCTION,
        DEFAULT_SUPPORT,
        image_record,
        load_support,
        resolve_particle_label,
        save_mp4,
        sha256_file,
    )
except ModuleNotFoundError:
    from analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        DEFAULT_RECONSTRUCTION,
        DEFAULT_SUPPORT,
        image_record,
        load_support,
        resolve_particle_label,
        save_mp4,
        sha256_file,
    )


ENCODINGS = ("X", "Y", "Z")
ENCODING_LABELS = {
    "X": "PC-X / readout / source Z",
    "Y": "PC-Y / phase / source X",
    "Z": "PC-Z / slice / source Y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
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
        "--still-dpi",
        type=int,
        default=180,
        help="Raster resolution for the selected still figures (default: 180).",
    )
    parser.add_argument(
        "--video-dpi",
        type=int,
        default=100,
        help=(
            "Raster resolution for cine source frames (default: 100). "
            "Use 180 to match the 3240x1620 selected-still resolution."
        ),
    )
    parser.add_argument(
        "--particle-label",
        default=None,
        help="Particle-count label for figures (for example, 100k or 1M).",
    )
    parser.add_argument(
        "--phase-colormap",
        choices=("gray", "diverging"),
        default="gray",
        help="Scanner-style grayscale or signed red-blue phase display.",
    )
    return parser.parse_args()


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
    magnitude_shape = result["magnitude"].shape
    if len(magnitude_shape) != 4:
        raise ValueError(
            f"Expected magnitude (bin,slice,phase,readout), got {magnitude_shape}"
        )
    if result["velocity"].shape != (3, *magnitude_shape):
        raise ValueError(f"Unexpected velocity shape {result['velocity'].shape}")
    if result["speed"].shape != magnitude_shape:
        raise ValueError("Speed shape does not match magnitude")
    if result["mask"].shape != magnitude_shape:
        raise ValueError("Mask shape does not match magnitude")
    return result


def plane_definitions(
    shape: tuple[int, int, int], fov_mm: np.ndarray
) -> list[dict[str, Any]]:
    slice_count, phase_count, readout_count = shape
    readout_extent = [-fov_mm[0] / 2, fov_mm[0] / 2]
    phase_extent = [-fov_mm[1] / 2, fov_mm[1] / 2]
    slice_extent = [-fov_mm[2] / 2, fov_mm[2] / 2]
    return [
        {
            "title": "Projection - XY",
            "kind": "projection",
            "axis": 0,
            "xlabel": "Readout / source z (mm)",
            "ylabel": "Phase / source x (mm)",
            "extent": [*readout_extent, *phase_extent],
        },
        {
            "title": "Projection - XZ",
            "kind": "projection",
            "axis": 1,
            "xlabel": "Readout / source z (mm)",
            "ylabel": "Slice / source y (mm)",
            "extent": [*readout_extent, *slice_extent],
        },
        {
            "title": "Projection - YZ",
            "kind": "projection",
            "axis": 2,
            "xlabel": "Phase / source x (mm)",
            "ylabel": "Slice / source y (mm)",
            "extent": [*phase_extent, *slice_extent],
        },
        {
            "title": "Center slice - XY",
            "kind": "slice",
            "index": slice_count // 2,
            "axis": 0,
            "xlabel": "Readout / source z (mm)",
            "ylabel": "Phase / source x (mm)",
            "extent": [*readout_extent, *phase_extent],
        },
        {
            "title": "Center slice - XZ",
            "kind": "slice",
            "index": phase_count // 2,
            "axis": 1,
            "xlabel": "Readout / source z (mm)",
            "ylabel": "Slice / source y (mm)",
            "extent": [*readout_extent, *slice_extent],
        },
        {
            "title": "Center slice - YZ",
            "kind": "slice",
            "index": readout_count // 2,
            "axis": 2,
            "xlabel": "Phase / source x (mm)",
            "ylabel": "Slice / source y (mm)",
            "extent": [*phase_extent, *slice_extent],
        },
    ]


def take_slice(values: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return values[index, :, :]
    if axis == 1:
        return values[:, index, :]
    if axis == 2:
        return values[:, :, index]
    raise ValueError(f"Unsupported axis {axis}")


def phase_at_magnitude_max(
    phase: np.ndarray,
    magnitude: np.ndarray,
    valid: np.ndarray,
    axis: int,
) -> np.ndarray:
    if phase.shape != magnitude.shape or phase.shape != valid.shape:
        raise ValueError("Phase, magnitude, and valid mask shapes must match")
    score = np.where(valid, magnitude, -np.inf)
    maximum_indices = np.argmax(score, axis=axis)
    selected = np.take_along_axis(
        phase, np.expand_dims(maximum_indices, axis=axis), axis=axis
    ).squeeze(axis=axis)
    selected = selected.astype(np.float64, copy=False)
    selected[~valid.any(axis=axis)] = np.nan
    return selected


def magnitude_panels(
    magnitude: np.ndarray, definitions: list[dict[str, Any]]
) -> list[np.ndarray]:
    panels = []
    for definition in definitions:
        if definition["kind"] == "projection":
            panels.append(magnitude.max(axis=definition["axis"]))
        else:
            panels.append(
                take_slice(magnitude, definition["axis"], definition["index"])
            )
    return panels


def phase_panels(
    phase: np.ndarray,
    magnitude: np.ndarray,
    valid: np.ndarray,
    definitions: list[dict[str, Any]],
) -> list[np.ndarray]:
    panels = []
    for definition in definitions:
        axis = definition["axis"]
        if definition["kind"] == "projection":
            panels.append(phase_at_magnitude_max(phase, magnitude, valid, axis))
        else:
            selected_phase = take_slice(phase, axis, definition["index"])
            selected_valid = take_slice(valid, axis, definition["index"])
            panels.append(np.where(selected_valid, selected_phase, np.nan))
    return panels


def phase_colormap(style: str):
    name = "gray" if style == "gray" else "RdBu_r"
    colormap = plt.get_cmap(name).copy()
    colormap.set_bad("black")
    return colormap


def render_six_panels(
    panels: list[np.ndarray],
    definitions: list[dict[str, Any]],
    output: Path,
    title: str,
    mode: str,
    display_limit: float,
    dpi: int,
    phase_cmap: str = "gray",
) -> None:
    if len(panels) != 6:
        raise ValueError("Six panels are required")
    figure, axes = plt.subplots(
        2, 3, figsize=(18, 9), constrained_layout=True, facecolor="white"
    )
    for axis, image, definition in zip(
        axes.ravel(), panels, definitions, strict=True
    ):
        if mode == "magnitude":
            axis.imshow(
                image,
                cmap="gray",
                origin="lower",
                extent=definition["extent"],
                vmin=0,
                vmax=display_limit,
                interpolation="nearest",
                aspect="equal",
            )
        else:
            axis.set_facecolor("black")
            axis.imshow(
                np.ma.masked_invalid(image),
                cmap=phase_colormap(phase_cmap),
                origin="lower",
                extent=definition["extent"],
                vmin=-display_limit,
                vmax=display_limit,
                interpolation="nearest",
                aspect="equal",
            )
        axis.set_title(definition["title"], fontsize=12)
        axis.set_xlabel(definition["xlabel"])
        axis.set_ylabel(definition["ylabel"])
        axis.grid(False)
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def render_bin_outputs(
    reconstruction: dict[str, Any],
    support: np.ndarray,
    definitions: list[dict[str, Any]],
    index: int,
    magnitude_limit: float,
    phase_limits: dict[str, float],
    output_paths: dict[str, Path],
    dpi: int,
    phase_cmap: str,
    particle_label: str,
) -> None:
    magnitude = reconstruction["magnitude"][index]
    valid = support & reconstruction["mask"][index]
    render_six_panels(
        magnitude_panels(magnitude, definitions),
        definitions,
        output_paths["magnitude"],
        f"dp150 {particle_label} motion KomaMRI reference magnitude | cardiac bin {index + 1}",
        "magnitude",
        magnitude_limit,
        dpi,
    )
    venc = reconstruction["Venc_m_per_s"]
    for component, encoding in enumerate(ENCODINGS):
        phase = reconstruction["velocity"][component, index] * np.pi / venc
        render_six_panels(
            phase_panels(phase, magnitude, valid, definitions),
            definitions,
            output_paths[encoding],
            (
                f"dp150 {particle_label} motion KomaMRI phase {encoding} vs REF | "
                f"cardiac bin {index + 1} | display +/-{phase_limits[encoding]:.3f} rad"
            ),
            "phase",
            phase_limits[encoding],
            dpi,
            phase_cmap,
        )


def main() -> None:
    args = parse_args()
    reconstruction_path = args.reconstruction.resolve()
    support_path = args.support_h5.resolve()
    default_output_name = (
        "projection_slices_magnitude_phase_grayscale"
        if args.phase_colormap == "gray"
        else "projection_slices_magnitude_phase"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else reconstruction_path.parent
        / "final_image_analysis"
        / default_output_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.frame_duration_ms < 20 or args.mp4_cycles < 1:
        raise ValueError("Frame duration must be >=20 ms and MP4 cycles >=1")
    if args.still_dpi < 72 or args.video_dpi < 72:
        raise ValueError("Still and video DPI must each be >=72")

    fov_m = np.asarray(args.fov_m, dtype=np.float64)
    matrix = np.asarray(args.matrix, dtype=np.int64)
    reconstruction = load_reconstruction(reconstruction_path)
    particle_label, _ = resolve_particle_label(reconstruction_path, args.particle_label)
    cardiac_bins = int(reconstruction["magnitude"].shape[0])
    expected_shape = tuple(int(value) for value in matrix[::-1])
    if reconstruction["magnitude"].shape[1:] != expected_shape:
        raise ValueError(
            f"Reconstruction shape {reconstruction['magnitude'].shape[1:]} "
            f"does not match matrix {matrix.tolist()}"
        )
    support, support_metadata = load_support(support_path, fov_m, matrix)
    definitions = plane_definitions(support.shape, fov_m * 1000.0)
    valid_all = reconstruction["mask"] & support[None, ...]

    positive_magnitude = reconstruction["magnitude"][
        reconstruction["magnitude"] > 0
    ]
    magnitude_limit = max(float(np.quantile(positive_magnitude, 0.999)), 1e-6)
    phase_limits = {}
    phase_statistics = {}
    for component, encoding in enumerate(ENCODINGS):
        phase = reconstruction["velocity"][component] * np.pi / reconstruction[
            "Venc_m_per_s"
        ]
        absolute = np.abs(phase[valid_all])
        phase_limits[encoding] = max(float(np.quantile(absolute, 0.995)), 1e-6)
        phase_statistics[encoding] = {
            "median_absolute_rad": float(np.quantile(absolute, 0.5)),
            "p95_absolute_rad": float(np.quantile(absolute, 0.95)),
            "p99_5_absolute_rad": phase_limits[encoding],
            "maximum_absolute_rad": float(absolute.max()),
            "display_clipped_fraction": float(
                (absolute > phase_limits[encoding]).mean()
            ),
        }

    p95_speed = np.asarray(
        [
            np.quantile(reconstruction["speed"][index][valid_all[index]], 0.95)
            for index in range(reconstruction["magnitude"].shape[0])
        ]
    )
    peak_index = int(np.argmax(p95_speed))
    selected_index = peak_index if args.cardiac_bin is None else args.cardiac_bin - 1
    if selected_index not in range(reconstruction["magnitude"].shape[0]):
        raise ValueError("--cardiac-bin is outside the reconstructed bin range")

    selected_paths = {
        "magnitude": output_dir
        / f"dp150_bin{selected_index + 1:02d}_reference_magnitude_projection_slices.png",
        **{
            encoding: output_dir
            / f"dp150_bin{selected_index + 1:02d}_phase_{encoding}_projection_slices.png"
            for encoding in ENCODINGS
        },
    }
    render_bin_outputs(
        reconstruction,
        support,
        definitions,
        selected_index,
        magnitude_limit,
        phase_limits,
        selected_paths,
        dpi=args.still_dpi,
        phase_cmap=args.phase_colormap,
        particle_label=particle_label,
    )

    frame_paths: dict[str, list[Path]] = {
        key: [] for key in ("magnitude", *ENCODINGS)
    }
    frame_dirs = {}
    for key in frame_paths:
        frame_dir = output_dir / f"{key.lower()}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_dirs[key] = frame_dir
    for index in range(reconstruction["magnitude"].shape[0]):
        paths = {
            "magnitude": frame_dirs["magnitude"] / f"frame_{index + 1:03d}.png",
            **{
                encoding: frame_dirs[encoding] / f"frame_{index + 1:03d}.png"
                for encoding in ENCODINGS
            },
        }
        render_bin_outputs(
            reconstruction,
            support,
            definitions,
            index,
            magnitude_limit,
            phase_limits,
            paths,
            dpi=args.video_dpi,
            phase_cmap=args.phase_colormap,
            particle_label=particle_label,
        )
        for key, path in paths.items():
            frame_paths[key].append(path)

    videos = {}
    for key, paths in frame_paths.items():
        label = "reference_magnitude" if key == "magnitude" else f"phase_{key}"
        video_path = output_dir / (
            f"dp150_{label}_projection_slices_{cardiac_bins}_bins.mp4"
        )
        videos[key] = save_mp4(
            paths,
            video_path,
            args.frame_duration_ms,
            args.mp4_cycles,
        )

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
        "support": support_metadata,
        "display": {
            "layout": "top row orthogonal projections; bottom row geometric center slices",
            "magnitude_projection": "maximum intensity",
            "phase_projection": "phase at the maximum-reference-magnitude valid voxel along the projection axis",
            "phase_center_slices": "stored signal mask intersected with exact seed-derived support",
            "magnitude_display_p99_9_au": magnitude_limit,
            "component_specific_phase_limits_rad": phase_limits,
            "phase_colormap": args.phase_colormap,
            "image_interpolation": "nearest",
            "temporal_interpolation": False,
            "still_dpi": args.still_dpi,
            "video_frame_dpi": args.video_dpi,
            "video_frame_size_px": [18 * args.video_dpi, 9 * args.video_dpi],
        },
        "phase_statistics": phase_statistics,
        "selected_outputs": [
            image_record(path) for path in selected_paths.values()
        ],
        "videos": videos,
        "scientific_limits": [
            "Phase panels show wrapped phase differences relative to REF, not absolute phase.",
            "Each directional phase figure uses its own labeled robust display scale; colors should not be compared directly across components.",
            "Projection phase is selected at the strongest valid magnitude voxel and is not a simple signed maximum.",
            "The exact seed-derived support changes phase display masking only; reconstruction arrays are unchanged.",
            "Particle-density convergence against a higher-count reconstruction has not been established.",
        ],
    }
    report_path = output_dir / "dp150_projection_slices_magnitude_phase_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "PASS_WITH_QUALITATIVE_LIMITATIONS\n"
        f"Selected cardiac bin: {selected_index + 1}\n"
        f"Magnitude: {selected_paths['magnitude']}\n"
        f"Phase X: {selected_paths['X']}\n"
        f"Phase Y: {selected_paths['Y']}\n"
        f"Phase Z: {selected_paths['Z']}\n"
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
