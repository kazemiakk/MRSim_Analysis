r"""Render provenance-bound dp150 phase cines for lumen-boundary QC.

This is a presentation derivative of the preserved one-million-spin,
15-bin reconstruction.  It does not modify the reconstruction, infer wall
material, or interpolate cardiac bins.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np

try:
    from scripts.analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        image_record,
        load_support,
        probe_video,
        sha256_file,
    )
    from scripts.plot_dp150_koma_publication import (
        coordinate_centers,
        coordinate_edges,
        find_throat,
    )
    from scripts.render_dp150_projection_slices_magnitude_phase import (
        load_reconstruction,
    )
    from scripts.render_dp150_xyz_magnitude_phase import (
        compound_to_complex,
        open_raw_datasets,
    )
except ModuleNotFoundError:
    from analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        image_record,
        load_support,
        probe_video,
        sha256_file,
    )
    from plot_dp150_koma_publication import (
        coordinate_centers,
        coordinate_edges,
        find_throat,
    )
    from render_dp150_projection_slices_magnitude_phase import (
        load_reconstruction,
    )
    from render_dp150_xyz_magnitude_phase import (
        compound_to_complex,
        open_raw_datasets,
    )


ENCODINGS = ("X", "Y", "Z")
PANEL_CLASSES = (
    "projection_xy",
    "projection_xz",
    "projection_yz",
    "upstream",
    "stenosis_throat",
    "immediate_post_stenotic",
)
EXPECTED_BINS = 15
FRAME_DPI = 180
FRAME_SIZE_PX = (3240, 1620)
CANONICAL_RECONSTRUCTION_SHA256 = (
    "a3b6306398cb98dab0ef02473692d76e7bc57026b3f6d5d090d71624bdacb376"
)
CANONICAL_SUPPORT_SHA256 = (
    "5c15d5df641187cada1f5f4c2895213245a722d362c6cf08ddcf581ea34278a1"
)
CANONICAL_RAW_SHA256 = {
    "REF": "f5865f9ebe8b2f137c80bee24d5695871a02bf17916fa71e9c68a8fa8af27b48",
    "X": "ad9886bb9cbd8e9a1dc32916cce6d2862c8950bfae443e66a3dd04d87bc319b5",
    "Y": "2b6e5b3ef2b2118567d48b6437c0e9ea28893829837211536b0d1c9af32e04f0",
    "Z": "15d31b6b0d572a7440a8e215a71949f274ca4fa0c687766e5c272a8b398ee2ad",
}


@dataclass(frozen=True)
class Section:
    role: str
    index: int
    readout_mm: float
    offset_from_throat_mm: float


@dataclass(frozen=True)
class Panel:
    title: str
    values: np.ndarray
    support: np.ndarray
    no_data: np.ndarray
    extent: tuple[float, float, float, float]
    xlabel: str
    ylabel: str
    panel_class: str


@dataclass(frozen=True)
class PlaybackPlan:
    slug: str
    label: str
    fps: Fraction
    cycles: int
    source_bins: int

    @property
    def frame_period_s(self) -> Fraction:
        return 1 / self.fps

    @property
    def encoded_frames(self) -> int:
        return self.source_bins * self.cycles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", required=True, type=Path)
    parser.add_argument("--support-h5", required=True, type=Path)
    parser.add_argument("--raw-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fov-m", nargs=3, type=float, default=DEFAULT_FOV_M)
    parser.add_argument("--matrix", nargs=3, type=int, default=DEFAULT_MATRIX)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--slow-hold-ms", type=int, default=420)
    parser.add_argument(
        "--expected-reconstruction-sha256",
        default=CANONICAL_RECONSTRUCTION_SHA256,
    )
    parser.add_argument(
        "--expected-support-sha256", default=CANONICAL_SUPPORT_SHA256
    )
    return parser.parse_args()


def circular_phase_projection(
    phase: np.ndarray, weights: np.ndarray, valid: np.ndarray, axis: int
) -> np.ndarray:
    """Magnitude-weighted circular mean; never averages wrapped angles linearly."""
    if phase.shape != weights.shape or phase.shape != valid.shape:
        raise ValueError("phase, weights, and valid must have identical shapes")
    finite = np.isfinite(phase) & np.isfinite(weights) & (weights > 0)
    use = valid & finite
    weighted = np.where(use, weights * np.exp(1j * phase), 0.0)
    phasor = weighted.sum(axis=axis)
    denominator = np.where(use, weights, 0.0).sum(axis=axis)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    present = denominator > 0
    result[present] = np.angle(phasor[present])
    return result


def select_fixed_sections(
    support: np.ndarray, readout_mm: np.ndarray
) -> tuple[Section, Section, Section]:
    """Use the established dp150 throat-35/throat/throat+30 mm convention."""
    throat, area = find_throat(support)
    occupied = np.flatnonzero(area > 0)
    if occupied.size == 0:
        raise ValueError("Exact seed-derived support is empty")
    roles_offsets = (
        ("Upstream", -35.0),
        ("Stenosis throat", 0.0),
        ("Immediate post-stenotic", 30.0),
    )
    sections = []
    for role, offset in roles_offsets:
        target = readout_mm[throat] + offset
        index = int(occupied[np.argmin(np.abs(readout_mm[occupied] - target))])
        sections.append(
            Section(role, index, float(readout_mm[index]), float(offset))
        )
    if len({item.index for item in sections}) != 3:
        raise ValueError("Fixed upstream/throat/post-stenotic sections overlap")
    return tuple(sections)  # type: ignore[return-value]


def physical_and_slow_plans(
    rr_s: float, bins: int, cycles: int, slow_hold_ms: int
) -> tuple[PlaybackPlan, PlaybackPlan]:
    if rr_s <= 0 or bins <= 0 or cycles <= 0 or slow_hold_ms <= 0:
        raise ValueError("Playback timing inputs must be positive")
    # The scanner-derived protocol records RR=1012 ms.  The HDF5 Float32 value
    # differs only by 35 ns, so rationalize at the protocol's 1-ms precision.
    rr = Fraction(round(1000.0 * rr_s), 1000)
    physical_fps = Fraction(bins, 1) / rr
    slow_fps = Fraction(1000, slow_hold_ms)
    slowdown = physical_fps / slow_fps
    return (
        PlaybackPlan(
            "physical_realtime",
            "PHYSICAL REAL-TIME | RR=1012 ms / 15",
            physical_fps,
            cycles,
            bins,
        ),
        PlaybackPlan(
            "slow_motion_qc",
            f"SLOW-MOTION QC | {float(slowdown):.2f}x SLOW",
            slow_fps,
            cycles,
            bins,
        ),
    )


def playback_indices(plan: PlaybackPlan) -> list[int]:
    """Every discrete source bin occurs exactly once in each repeated cycle."""
    return list(range(plan.source_bins)) * plan.cycles


def _extent(horizontal: np.ndarray, vertical: np.ndarray) -> tuple[float, ...]:
    x, y = coordinate_edges(horizontal), coordinate_edges(vertical)
    return float(x[0]), float(x[-1]), float(y[0]), float(y[-1])


def _section_crop(
    support: np.ndarray,
    sections: tuple[Section, Section, Section],
    phase_mm: np.ndarray,
    slice_mm: np.ndarray,
) -> tuple[slice, slice, tuple[float, float, float, float]]:
    union = np.logical_or.reduce([support[:, :, item.index] for item in sections])
    rows, columns = np.where(union)
    if rows.size == 0:
        raise ValueError("Selected exact-support sections are empty")
    row0, row1 = max(0, int(rows.min()) - 3), min(support.shape[0], int(rows.max()) + 4)
    col0, col1 = max(0, int(columns.min()) - 3), min(support.shape[1], int(columns.max()) + 4)
    x, y = coordinate_edges(phase_mm), coordinate_edges(slice_mm)
    return (
        slice(row0, row1),
        slice(col0, col1),
        (float(x[col0]), float(x[col1]), float(y[row0]), float(y[row1])),
    )


def build_panels(
    phase: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
    sections: tuple[Section, Section, Section],
    readout_mm: np.ndarray,
    phase_mm: np.ndarray,
    slice_mm: np.ndarray,
) -> list[Panel]:
    projection_specs = (
        ("Projection - XY", 0, readout_mm, phase_mm, "Readout / source Z (mm)", "Phase / source X (mm)", "projection_xy"),
        ("Projection - XZ", 1, readout_mm, slice_mm, "Readout / source Z (mm)", "Slice / source Y (mm)", "projection_xz"),
        ("Projection - YZ", 2, phase_mm, slice_mm, "Phase / source X (mm)", "Slice / source Y (mm)", "projection_yz"),
    )
    panels: list[Panel] = []
    for title, axis, horizontal, vertical, xlabel, ylabel, panel_class in projection_specs:
        projected = circular_phase_projection(phase, weights, valid, axis)
        projected_support = support.any(axis=axis)
        panels.append(
            Panel(
                title,
                projected,
                projected_support,
                projected_support & ~np.isfinite(projected),
                _extent(horizontal, vertical),
                xlabel,
                ylabel,
                panel_class,
            )
        )
    row_slice, column_slice, section_extent = _section_crop(
        support, sections, phase_mm, slice_mm
    )
    for section in sections:
        local_support = support[:, :, section.index][row_slice, column_slice]
        local_valid = valid[:, :, section.index][row_slice, column_slice]
        local_phase = phase[:, :, section.index][row_slice, column_slice]
        displayed = np.where(local_valid, local_phase, np.nan)
        panels.append(
            Panel(
                f"{section.role} - YZ | Z={section.readout_mm:.3f} mm",
                displayed,
                local_support,
                local_support & ~local_valid,
                section_extent,
                "Phase / source X (mm)",
                "Slice / source Y (mm)",
                section.role.lower().replace(" ", "_").replace("-", "_"),
            )
        )
    return panels


def displayed_p999(
    panel_values: dict[str, list[np.ndarray]],
) -> tuple[float, dict[str, Any]]:
    """Return the maximum of the six per-panel-class p99.9 values."""
    if set(panel_values) != set(PANEL_CLASSES):
        raise ValueError(
            f"Expected exactly the six panel classes {PANEL_CLASSES}, got "
            f"{tuple(panel_values)}"
        )
    per_class: dict[str, dict[str, float | int]] = {}
    for panel_class, arrays in panel_values.items():
        finite = np.concatenate(
            [np.abs(values[np.isfinite(values)]) for values in arrays]
        )
        if finite.size == 0:
            raise ValueError(f"No finite displayed values for {panel_class}")
        class_limit = min(float(np.quantile(finite, 0.999)), float(np.pi))
        per_class[panel_class] = {
            "displayed_values": int(finite.size),
            "p99_9_absolute_rad": class_limit,
            "maximum_absolute_rad": float(finite.max()),
        }
    limit = max(
        1e-6,
        max(float(item["p99_9_absolute_rad"]) for item in per_class.values()),
    )
    total = sum(int(item["displayed_values"]) for item in per_class.values())
    clipped = 0
    for panel_class, arrays in panel_values.items():
        finite = np.concatenate(
            [np.abs(values[np.isfinite(values)]) for values in arrays]
        )
        count = int((finite > limit).sum())
        per_class[panel_class]["clipped_at_component_limit"] = count
        per_class[panel_class]["clipped_fraction_at_component_limit"] = float(
            count / finite.size
        )
        clipped += count
    return limit, {
        "displayed_values": total,
        "limit_rule": "maximum p99.9 across the six displayed panel classes",
        "p99_9_component_limit_rad": limit,
        "clipped_fraction_at_component_limit": float(clipped / total),
        "per_panel_class": per_class,
    }


def _centers(extent: tuple[float, ...], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = shape
    dx = (extent[1] - extent[0]) / columns
    dy = (extent[3] - extent[2]) / rows
    return (
        extent[0] + (np.arange(columns) + 0.5) * dx,
        extent[2] + (np.arange(rows) + 0.5) * dy,
    )


def signed_phase_colormap() -> LinearSegmentedColormap:
    """Keep valid negative phase above black; reserve pure black for no-data."""
    return LinearSegmentedColormap.from_list(
        "signed_phase_gray", ("#202020", "#8c8c8c", "#ffffff")
    ).with_extremes(bad="#000000")


def render_frame(
    panels: list[Panel],
    output: Path,
    encoding: str,
    cardiac_bin: int,
    bins: int,
    phase_center_ms: float,
    limit: float,
    dpi: int = FRAME_DPI,
) -> None:
    if len(panels) != 6:
        raise ValueError("Exactly six panels are required")
    cmap = signed_phase_colormap()
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        figure = plt.figure(figsize=(18, 9), layout="constrained", facecolor="white")
        grid = figure.add_gridspec(3, 4, height_ratios=(1, 1, 0.07), width_ratios=(1, 1, 1, 0.035))
        axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
        colorbar_axis = figure.add_subplot(grid[:2, 3])
        note_axis = figure.add_subplot(grid[2, :3])
        note_axis.axis("off")
        image = None
        for axis, panel in zip(axes, panels, strict=True):
            image = axis.imshow(
                np.ma.masked_invalid(panel.values),
                cmap=cmap,
                origin="lower",
                extent=panel.extent,
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="equal",
            )
            x, y = _centers(panel.extent, panel.support.shape)
            if panel.support.any() and not panel.support.all():
                axis.contour(
                    x,
                    y,
                    panel.support.astype(float),
                    levels=(0.5,),
                    colors=("0.05",),
                    linewidths=2.2,
                    zorder=4,
                )
                axis.contour(
                    x,
                    y,
                    panel.support.astype(float),
                    levels=(0.5,),
                    colors=("white",),
                    linewidths=0.8,
                    zorder=5,
                )
            axis.set_title(panel.title, fontsize=11)
            axis.set_xlabel(panel.xlabel, fontsize=9)
            axis.set_ylabel(panel.ylabel, fontsize=9)
            axis.tick_params(labelsize=8)
            axis.grid(False)
        assert image is not None
        figure.colorbar(
            image,
            cax=colorbar_axis,
            label=f"Wrapped phase {encoding} vs REF (rad)",
            ticks=(-limit, 0.0, limit),
        )
        note_axis.legend(
            handles=(
                Patch(facecolor="white", edgecolor="0.05", label="exact seed-derived lumen-support contour"),
                Patch(facecolor="#000000", label="invalid/no reconstructed phase (pure black)"),
                Patch(facecolor="#202020", label="most-negative valid phase (dark gray, not black)"),
            ),
            loc="center",
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        figure.suptitle(
            f"dp150 1M scanner-derived KomaMRI phase {encoding} vs REF | "
            f"cardiac bin {cardiac_bin:02d}/{bins} | center {phase_center_ms:.3f} ms",
            fontsize=15,
        )
        temporary = output.with_name(f".{output.stem}.partial.png")
        figure.savefig(temporary, dpi=dpi, facecolor="white", format="png")
        plt.close(figure)
        temporary.replace(output)


def _write_manifest(paths: list[Path], output: Path, period: Fraction) -> None:
    lines = ["ffconcat version 1.0"]
    duration = f"{float(period):.15f}"
    for path in paths:
        normalized = path.resolve().as_posix().replace("'", "'\\''")
        lines.extend((f"file '{normalized}'", f"duration {duration}"))
    normalized = paths[-1].resolve().as_posix().replace("'", "'\\''")
    lines.append(f"file '{normalized}'")
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(output)


def encode_video(
    unique_frames: list[Path], output: Path, plan: PlaybackPlan, title: str
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the requested cines")
    playlist = [unique_frames[index] for index in playback_indices(plan)]
    manifest = output.with_suffix(".ffconcat")
    _write_manifest(playlist, manifest, plan.frame_period_s)
    fps_text = f"{plan.fps.numerator}/{plan.fps.denominator}"
    timescale = plan.fps.numerator * max(1, math.ceil(1000 / plan.fps.numerator))
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    video_filter = (
        "drawtext="
        f"text='{plan.label}':"
        "fontcolor=white:fontsize=42:x=36:y=36:"
        "box=1:boxcolor=black@0.65:boxborderw=12,"
        "format=yuv420p"
    )
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-vf",
        video_filter,
        "-r",
        fps_text,
        "-fps_mode",
        "cfr",
        "-frames:v",
        str(plan.encoded_frames),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-video_track_timescale",
        str(timescale),
        "-metadata",
        f"title={title} | {plan.label}",
        "-metadata",
        "comment=Discrete reconstructed bins only; no temporal interpolation",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    probe = probe_video(temporary)
    if not probe.get("probe_available"):
        raise RuntimeError("ffprobe is required for fail-closed video validation")
    if probe.get("dimensions_px") != list(FRAME_SIZE_PX):
        raise RuntimeError(f"Unexpected video dimensions: {probe}")
    if probe.get("frames") != plan.encoded_frames:
        raise RuntimeError(f"Unexpected encoded frame count: {probe}")
    if Fraction(str(probe.get("average_frame_rate"))) != plan.fps:
        raise RuntimeError(f"Unexpected encoded frame rate: {probe}")
    expected_duration = plan.encoded_frames * float(plan.frame_period_s)
    actual_duration = float(probe["duration_s"])
    if not math.isclose(actual_duration, expected_duration, abs_tol=1e-6):
        raise RuntimeError(f"Unexpected encoded duration: {probe}")
    temporary.replace(output)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "playback_label": plan.label,
        "fps_exact": fps_text,
        "frame_period_s_exact": f"{plan.frame_period_s.numerator}/{plan.frame_period_s.denominator}",
        "frame_period_ms": 1000.0 * float(plan.frame_period_s),
        "source_unique_bins": plan.source_bins,
        "cycles": plan.cycles,
        "encoded_frames": plan.encoded_frames,
        "expected_duration_s": expected_duration,
        "actual_duration_s": actual_duration,
        "actual_minus_expected_duration_s": actual_duration - expected_duration,
        "uniform_source_frames_per_bin_per_cycle": 1,
        "temporal_interpolation": False,
        "manifest": str(manifest),
        "ffmpeg_command": command,
        "ffmpeg_stderr_tail": completed.stderr.splitlines()[-4:],
        **probe_video(output),
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _raw_bin(
    raw: dict[str, Any], reconstruction: dict[str, Any], support: np.ndarray, index: int, encoding: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = compound_to_complex(raw["REF"][index])
    encoded = compound_to_complex(raw[encoding][index])
    phase = np.angle(encoded * np.conj(reference)).astype(np.float32)
    weights = (np.abs(encoded) * np.abs(reference)).astype(np.float32)
    finite = np.isfinite(phase) & np.isfinite(weights) & (weights > 0)
    valid = support & reconstruction["mask"][index] & finite
    return phase, weights, valid


def main() -> None:
    args = parse_args()
    reconstruction_path = args.reconstruction.resolve()
    support_path = args.support_h5.resolve()
    raw_run_dir = args.raw_run_dir.resolve()
    final_output_dir = args.output_dir.resolve()
    output_dir = final_output_dir.with_name(f".{final_output_dir.name}.partial")
    if final_output_dir.exists() or output_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite a final or partial presentation derivative: "
            f"{final_output_dir}"
        )
    reconstruction_hash = sha256_file(reconstruction_path)
    support_hash = sha256_file(support_path)
    if reconstruction_hash != args.expected_reconstruction_sha256:
        raise ValueError("Reconstruction SHA-256 does not match the explicit expected 1M input")
    if support_hash != args.expected_support_sha256:
        raise ValueError("Support SHA-256 does not match the explicit expected input")
    if args.cycles != 2 or args.slow_hold_ms != 420:
        raise ValueError("E-016 requires exactly two cycles and a 420-ms slow-QC hold")

    fov_m = np.asarray(args.fov_m, dtype=np.float64)
    matrix = np.asarray(args.matrix, dtype=np.int64)
    if not np.array_equal(matrix, np.asarray(DEFAULT_MATRIX, dtype=np.int64)):
        raise ValueError(f"E-016 requires canonical matrix {list(DEFAULT_MATRIX)}")
    if not np.array_equal(fov_m, np.asarray(DEFAULT_FOV_M, dtype=np.float64)):
        raise ValueError(f"E-016 requires canonical FOV {list(DEFAULT_FOV_M)} m")
    reconstruction = load_reconstruction(reconstruction_path)
    if reconstruction["magnitude"].shape[0] != EXPECTED_BINS:
        raise ValueError("This derivative requires all 15 reconstructed bins")
    expected_shape = tuple(int(value) for value in matrix[::-1])
    if reconstruction["magnitude"].shape[1:] != expected_shape:
        raise ValueError("Reconstruction shape does not match the explicit matrix")
    support, support_metadata = load_support(support_path, fov_m, matrix)
    readout_mm, phase_mm, slice_mm = coordinate_centers(support.shape, 1000.0 * fov_m)
    sections = select_fixed_sections(support, readout_mm)
    canonical_indices = [70, 98, 122]
    if [item.index for item in sections] != canonical_indices:
        raise ValueError(f"Canonical section indices changed: {sections}")
    canonical_coordinates_mm = [-161.875, -126.875, -96.875]
    if [item.readout_mm for item in sections] != canonical_coordinates_mm:
        raise ValueError(f"Canonical section coordinates changed: {sections}")
    plans = physical_and_slow_plans(
        reconstruction["RR_s"], EXPECTED_BINS, args.cycles, args.slow_hold_ms
    )

    with ExitStack() as stack:
        raw, raw_paths = open_raw_datasets(
            stack, raw_run_dir, reconstruction["magnitude"].shape
        )
        input_records_before = {
            "reconstruction": _file_record(reconstruction_path),
            "seed_derived_support": _file_record(support_path),
            "raw_complex_images": {
                name: _file_record(path) for name, path in raw_paths.items()
            },
        }
        raw_hashes = {
            name: record["sha256"]
            for name, record in input_records_before["raw_complex_images"].items()
        }
        if raw_hashes != CANONICAL_RAW_SHA256:
            raise ValueError(
                "Raw REF/X/Y/Z hashes do not match the audited E-010 inputs"
            )
        displayed: dict[str, dict[str, list[np.ndarray]]] = {
            name: {} for name in ENCODINGS
        }
        max_phase_error = {name: 0.0 for name in ENCODINGS}
        max_reference_error = 0.0
        for index in range(EXPECTED_BINS):
            reference = compound_to_complex(raw["REF"][index])
            max_reference_error = max(
                max_reference_error,
                float(np.max(np.abs(np.abs(reference) - reconstruction["magnitude"][index]))),
            )
            for component, encoding in enumerate(ENCODINGS):
                phase, weights, valid = _raw_bin(raw, reconstruction, support, index, encoding)
                expected = reconstruction["velocity"][component, index] * np.pi / reconstruction["Venc_m_per_s"]
                delta = np.angle(np.exp(1j * (phase - expected)))
                max_phase_error[encoding] = max(
                    max_phase_error[encoding],
                    float(np.max(np.abs(delta[reconstruction["mask"][index]]))),
                )
                panels = build_panels(
                    phase, weights, valid, support, sections, readout_mm, phase_mm, slice_mm
                )
                for panel in panels:
                    displayed[encoding].setdefault(panel.panel_class, []).append(
                        panel.values
                    )
        limits_statistics = {
            encoding: displayed_p999(displayed[encoding]) for encoding in ENCODINGS
        }
        if max_reference_error > 1e-4:
            raise ValueError("Raw REF magnitude does not match the reconstruction")
        if max(max_phase_error.values()) > 1e-4:
            raise ValueError("Raw complex phase does not match reconstructed phase")

        output_dir.mkdir(parents=True)
        frame_paths: dict[str, list[Path]] = {name: [] for name in ENCODINGS}
        for encoding in ENCODINGS:
            (output_dir / f"phase_{encoding.lower()}_frames").mkdir()
        for index in range(EXPECTED_BINS):
            for encoding in ENCODINGS:
                phase, weights, valid = _raw_bin(raw, reconstruction, support, index, encoding)
                panels = build_panels(
                    phase, weights, valid, support, sections, readout_mm, phase_mm, slice_mm
                )
                path = output_dir / f"phase_{encoding.lower()}_frames" / f"frame_{index + 1:03d}.png"
                render_frame(
                    panels,
                    path,
                    encoding,
                    index + 1,
                    EXPECTED_BINS,
                    1000.0 * (index + 0.5) * reconstruction["RR_s"] / EXPECTED_BINS,
                    limits_statistics[encoding][0],
                )
                frame_paths[encoding].append(path)

    videos: dict[str, dict[str, Any]] = {}
    for encoding in ENCODINGS:
        videos[encoding] = {}
        for plan in plans:
            path = output_dir / f"dp150_phase_{encoding}_{plan.slug}_15_bins.mp4"
            videos[encoding][plan.slug] = encode_video(
                frame_paths[encoding], path, plan, f"dp150 phase {encoding} vs REF"
            )

    staging_text = output_dir.resolve().as_posix()
    final_text = final_output_dir.resolve().as_posix()
    for component_videos in videos.values():
        for video in component_videos.values():
            manifest_path = Path(video["manifest"])
            manifest_text = manifest_path.read_text(encoding="ascii")
            temporary_manifest = manifest_path.with_name(
                f".{manifest_path.name}.partial"
            )
            temporary_manifest.write_text(
                manifest_text.replace(staging_text, final_text), encoding="ascii"
            )
            temporary_manifest.replace(manifest_path)
            video["path"] = str(
                final_output_dir / Path(video["path"]).relative_to(output_dir)
            )
            video["manifest"] = str(
                final_output_dir / manifest_path.relative_to(output_dir)
            )
            video["manifest_record"] = {
                **_file_record(manifest_path),
                "path": video["manifest"],
            }

    input_records_after = {
        "reconstruction": _file_record(reconstruction_path),
        "seed_derived_support": _file_record(support_path),
        "raw_complex_images": {
            name: _file_record(path) for name, path in raw_paths.items()
        },
    }
    input_hashes_unchanged = input_records_before == input_records_after
    if not input_hashes_unchanged:
        raise RuntimeError("A scientific input changed while rendering")

    report = {
        "schema_version": 2,
        "status": "PASS_WITH_QUALITATIVE_LIMITATIONS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "E-016",
        "parent_experiment_id": "E-010",
        "evidence_class": "presentation derivative of preserved E-010 production reconstruction",
        "provenance": {
            "script": _file_record(Path(__file__).resolve()),
            "command_line": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "input_records_before_render": input_records_before,
            "input_records_after_render": input_records_after,
            "input_hashes_unchanged": input_hashes_unchanged,
        },
        "inputs": input_records_before,
        "acquisition": {
            "particle_count": 1_000_000,
            "cardiac_bins": EXPECTED_BINS,
            "RR_s": reconstruction["RR_s"],
            "physical_bin_spacing_ms": 1000.0 * reconstruction["RR_s"] / EXPECTED_BINS,
            "Venc_m_per_s": reconstruction["Venc_m_per_s"],
            "encoding": "separate REF/X/Y/Z",
        },
        "grid": {
            "FOV_m_readout_phase_slice": fov_m.tolist(),
            "matrix_readout_phase_slice": matrix.tolist(),
            "array_order": ["cardiac_bin", "slice", "phase", "readout"],
            "voxel_size_mm": (1000.0 * fov_m / matrix).tolist(),
        },
        "axis_mapping": {"PC_X": "readout/source Z", "PC_Y": "phase/source X", "PC_Z": "slice/source Y"},
        "support": support_metadata,
        "fixed_lumen_boundary_qc_sections": [
            {
                **item.__dict__,
                "exact_support_area_voxels": int(
                    support[:, :, item.index].sum()
                ),
            }
            for item in sections
        ],
        "display": {
            "layout": "top: XY/XZ/YZ projections; bottom: fixed upstream/throat/immediate-post-stenotic YZ slices",
            "phase_source": "angle(encoded * conj(REF)) from raw complex images",
            "projection": "abs(encoded)*abs(REF)-weighted circular mean within stored reconstruction mask AND exact seed-derived support",
            "slice_mask": "stored reconstruction mask AND exact seed-derived support",
            "lumen_boundary": "unsmoothed seed-derived support contour; not simulated wall material",
            "wall_material_simulated": False,
            "no_data": "pure black; the most-negative valid phase is clipped to #202020 so it remains distinguishable",
            "colormap": "signed grayscale #202020 -> #8c8c8c -> #ffffff; pure black reserved for invalid/no-data",
            "component_limits": {
                name: {
                    "limit_population": "per-class finite displayed values over all 15 bins; component limit is the maximum of six class p99.9 values",
                    **stats,
                }
                for name, (_, stats) in limits_statistics.items()
            },
            "phase_limit_quantile": 0.999,
            "phase_limit_quantile_method": "NumPy linear",
            "image_interpolation": "nearest",
            "frame_dpi": FRAME_DPI,
            "frame_dimensions_px": list(FRAME_SIZE_PX),
            "temporal_interpolation": False,
        },
        "raw_reconstruction_identity": {
            "maximum_reference_magnitude_absolute_error": max_reference_error,
            "maximum_circular_phase_error_rad": max_phase_error,
            "acceptance_magnitude_au": 1e-4,
            "acceptance_phase_rad": 1e-4,
        },
        "playback": {
            plan.slug: {
                "label": plan.label,
                "fps_exact": f"{plan.fps.numerator}/{plan.fps.denominator}",
                "frame_period_ms": 1000.0 * float(plan.frame_period_s),
                "cycles": plan.cycles,
                "one_source_frame_per_bin_per_cycle": True,
                "temporal_interpolation": False,
                "expected_duration_s": plan.encoded_frames
                * float(plan.frame_period_s),
                **(
                    {
                        "slowdown_factor_exact": "1575/253",
                        "slowdown_factor": 1575 / 253,
                    }
                    if plan.slug == "slow_motion_qc"
                    else {}
                ),
            }
            for plan in plans
        },
        "frames": {
            name: [
                {
                    **image_record(path),
                    "path": str(final_output_dir / path.relative_to(output_dir)),
                }
                for path in paths
            ]
            for name, paths in frame_paths.items()
        },
        "videos": videos,
        "scientific_limits": [
            "These are presentation-only wrapped-phase displays, not independent velocity validation.",
            "The contour is seed-derived lumen support, not wall material or a wall simulation.",
            "Separate REF/X/Y/Z encodings are not established as balanced-Hadamard equivalent.",
            "No cardiac bins are interpolated; repeated cycles are playback only.",
        ],
    }
    report_path = output_dir / "dp150_phase_boundary_qc_report.json"
    temporary = report_path.with_name(f".{report_path.name}.partial")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    output_dir.replace(final_output_dir)
    published_report = final_output_dir / report_path.name
    print(f"PASS_WITH_QUALITATIVE_LIMITATIONS\nReport: {published_report}")


if __name__ == "__main__":
    main()
