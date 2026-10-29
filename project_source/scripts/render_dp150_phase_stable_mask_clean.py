r"""Render a clean, fixed-mask dp150 phase presentation derivative.

E-017 is a presentation-only child of E-016 and the preserved E-010
reconstruction.  It replaces time-varying no-data holes with one conservative
all-bin display domain.  Voxels excluded from that domain are shown as
neutral-gray hatching; their phase is neither interpolated nor imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

try:
    from scripts import render_dp150_phase_boundary_qc as e016
except ModuleNotFoundError:
    import render_dp150_phase_boundary_qc as e016


EXPERIMENT_ID = "E-017"
EXPECTED_E016_REPORT_SHA256 = (
    "b968e0d6aa236e459bb7c7a9a36fa977f4b8e2e8708b85fea1c3ccb72f256124"
)
INHERITED_E016_LIMITS_RAD = {
    "X": 1.8401588255166987,
    "Y": 0.1981773659586905,
    "Z": 0.2359500242769718,
}
EXCLUDED_GRAY = "#b8b8b8"
HATCH_GRAY = "#4a4a4a"
HATCH_PATTERN = "///"
PLAYBACK_BANNER_X_PX = 36
PLAYBACK_BANNER_Y_PX = 105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", required=True, type=Path)
    parser.add_argument("--support-h5", required=True, type=Path)
    parser.add_argument("--raw-run-dir", required=True, type=Path)
    parser.add_argument("--e016-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fov-m", nargs=3, type=float, default=e016.DEFAULT_FOV_M)
    parser.add_argument("--matrix", nargs=3, type=int, default=e016.DEFAULT_MATRIX)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--slow-hold-ms", type=int, default=420)
    parser.add_argument(
        "--expected-reconstruction-sha256",
        default=e016.CANONICAL_RECONSTRUCTION_SHA256,
    )
    parser.add_argument(
        "--expected-support-sha256",
        default=e016.CANONICAL_SUPPORT_SHA256,
    )
    parser.add_argument(
        "--expected-e016-report-sha256",
        default=EXPECTED_E016_REPORT_SHA256,
    )
    return parser.parse_args()


def stable_reconstruction_mask(
    support: np.ndarray, reconstruction_mask: np.ndarray
) -> np.ndarray:
    """Return the exact-support voxels valid in every reconstructed bin."""
    support = np.asarray(support, dtype=bool)
    reconstruction_mask = np.asarray(reconstruction_mask)
    if reconstruction_mask.ndim != support.ndim + 1:
        raise ValueError("Reconstruction mask must have one leading cardiac-bin axis")
    if reconstruction_mask.shape[1:] != support.shape:
        raise ValueError("Reconstruction mask and support shapes do not match")
    if reconstruction_mask.shape[0] != e016.EXPECTED_BINS:
        raise ValueError("Stable-mask derivative requires all 15 cardiac bins")
    return support & np.all(reconstruction_mask.astype(bool), axis=0)


def refine_stable_mask_with_signal(
    initial_stable: np.ndarray, signal_valid_masks: Iterable[np.ndarray]
) -> tuple[np.ndarray, int]:
    """Intersect any all-bin/all-encoding signal failures without filling data."""
    initial_stable = np.asarray(initial_stable, dtype=bool)
    final_stable = initial_stable.copy()
    found = False
    for signal_valid in signal_valid_masks:
        found = True
        signal_valid = np.asarray(signal_valid, dtype=bool)
        if signal_valid.shape != initial_stable.shape:
            raise ValueError("Signal-valid mask and stable mask shapes do not match")
        final_stable &= signal_valid
    if not found:
        raise ValueError("At least one signal-valid mask is required")
    removed = int(np.count_nonzero(initial_stable & ~final_stable))
    return final_stable, removed


def _complex_signal_valid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    magnitude = np.abs(values)
    return (
        np.isfinite(values.real)
        & np.isfinite(values.imag)
        & np.isfinite(magnitude)
        & (magnitude > 0)
    )


def scan_all_signal_validity(
    raw: dict[str, Any], shape: tuple[int, ...], initial_stable: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find a component-independent phase domain valid for every raw image."""
    all_valid = np.ones(shape[1:], dtype=bool)
    per_bin: dict[str, list[dict[str, int]]] = {
        name: [] for name in ("REF", *e016.ENCODINGS)
    }
    for index in range(shape[0]):
        reference = e016.compound_to_complex(raw["REF"][index])
        reference_valid = _complex_signal_valid(reference)
        per_bin["REF"].append(
            {
                "cardiac_bin": index + 1,
                "invalid_inside_initial_stable": int(
                    np.count_nonzero(initial_stable & ~reference_valid)
                ),
            }
        )
        all_valid &= reference_valid
        for encoding in e016.ENCODINGS:
            encoded = e016.compound_to_complex(raw[encoding][index])
            encoded_valid = _complex_signal_valid(encoded)
            phase = np.angle(encoded * np.conj(reference))
            weights = np.abs(encoded) * np.abs(reference)
            combined = (
                reference_valid
                & encoded_valid
                & np.isfinite(phase)
                & np.isfinite(weights)
                & (weights > 0)
            )
            per_bin[encoding].append(
                {
                    "cardiac_bin": index + 1,
                    "invalid_inside_initial_stable": int(
                        np.count_nonzero(initial_stable & ~combined)
                    ),
                }
            )
            all_valid &= combined
    return all_valid, {
        "rule": "finite positive complex signal and finite positive magnitude-product weight for REF/X/Y/Z in every bin",
        "per_raw_image_checks": per_bin,
    }


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Coverage denominator must be positive")
    return float(numerator / denominator)


def stable_mask_coverage(
    support: np.ndarray,
    stable: np.ndarray,
    reconstruction_mask: np.ndarray,
    sections: tuple[e016.Section, e016.Section, e016.Section],
) -> dict[str, Any]:
    """Report voxel, bin, section, and projection line-of-sight coverage."""
    support = np.asarray(support, dtype=bool)
    stable = np.asarray(stable, dtype=bool)
    if support.shape != stable.shape:
        raise ValueError("Support and stable mask shapes do not match")
    if np.any(stable & ~support):
        raise ValueError("Stable mask must be a subset of exact support")
    full_count = int(support.sum())
    stable_count = int(stable.sum())
    section_records = []
    for section in sections:
        full = int(support[:, :, section.index].sum())
        kept = int(stable[:, :, section.index].sum())
        section_records.append(
            {
                **section.__dict__,
                "full_support_voxels": full,
                "stable_voxels": kept,
                "excluded_voxels": full - kept,
                "stable_fraction_of_full_support": _fraction(kept, full),
            }
        )
    projection_records: dict[str, Any] = {}
    for name, axis in (("XY", 0), ("XZ", 1), ("YZ", 2)):
        full_los = support.any(axis=axis)
        stable_los = stable.any(axis=axis)
        full = int(full_los.sum())
        kept = int(stable_los.sum())
        projection_records[name] = {
            "collapsed_axis": axis,
            "full_support_lines_of_sight": full,
            "stable_lines_of_sight": kept,
            "excluded_lines_of_sight": full - kept,
            "stable_fraction_of_full_support": _fraction(kept, full),
            "static_excluded_pattern_sha256": _mask_hash(full_los & ~stable_los),
        }
    reconstruction_mask = np.asarray(reconstruction_mask, dtype=bool)
    in_support_by_bin = reconstruction_mask & support[None, ...]
    ever_valid = np.any(in_support_by_bin, axis=0)
    always_valid = np.all(in_support_by_bin, axis=0) & support
    temporally_variable = support & ever_valid & ~always_valid
    never_valid = support & ~ever_valid
    transition_counts = [
        int(np.count_nonzero(in_support_by_bin[index] ^ in_support_by_bin[index - 1]))
        for index in range(1, reconstruction_mask.shape[0])
    ]
    bin_records = []
    for index in range(reconstruction_mask.shape[0]):
        full_bin = in_support_by_bin[index]
        count = int(full_bin.sum())
        bin_records.append(
            {
                "cardiac_bin": index + 1,
                "original_mask_voxels_inside_support": count,
                "original_mask_fraction_of_full_support": _fraction(count, full_count),
                "fixed_stable_voxels": stable_count,
                "fixed_stable_fraction_of_original_mask": _fraction(
                    stable_count, count
                ),
            }
        )
    dynamic_counts = [
        item["original_mask_voxels_inside_support"] for item in bin_records
    ]
    return {
        "full_exact_support_voxels": full_count,
        "stable_voxels": stable_count,
        "excluded_voxels": full_count - stable_count,
        "stable_fraction_of_full_support": _fraction(stable_count, full_count),
        "excluded_fraction_of_full_support": _fraction(
            full_count - stable_count, full_count
        ),
        "temporally_variable_voxels": int(temporally_variable.sum()),
        "temporally_variable_fraction_of_full_support": _fraction(
            int(temporally_variable.sum()), full_count
        ),
        "never_valid_voxels": int(never_valid.sum()),
        "never_valid_fraction_of_full_support": _fraction(
            int(never_valid.sum()), full_count
        ),
        "original_dynamic_mask_summary": {
            "mean_voxels_inside_support": float(np.mean(dynamic_counts)),
            "minimum_voxels_inside_support": int(min(dynamic_counts)),
            "maximum_voxels_inside_support": int(max(dynamic_counts)),
            "mean_fraction_of_full_support": float(
                np.mean(dynamic_counts) / full_count
            ),
            "adjacent_bin_transition_counts": transition_counts,
            "total_adjacent_bin_voxel_switch_events": int(sum(transition_counts)),
            "mean_adjacent_bin_voxel_switch_events": float(
                np.mean(transition_counts)
            ),
        },
        "fixed_mask_transition_summary": {
            "adjacent_bin_pairs": reconstruction_mask.shape[0] - 1,
            "voxel_switch_events": 0,
            "component_switch_events": {name: 0 for name in e016.ENCODINGS},
        },
        "full_support_mask_sha256": _mask_hash(support),
        "stable_mask_sha256": _mask_hash(stable),
        "stable_mask_canonicalization": {
            "dtype": "uint8",
            "memory_order": "C",
            "shape_slice_phase_readout": list(stable.shape),
            "hash_algorithm": "SHA-256 over contiguous uint8 bytes",
            "cardiac_bins_intersected": reconstruction_mask.shape[0],
        },
        "fixed_across_all_bins_and_components": True,
        "sections": section_records,
        "projection_lines_of_sight": projection_records,
        "original_per_bin_mask_coverage": bin_records,
    }


def stable_image_record(path: Path) -> dict[str, Any]:
    """Record frame identity plus fail-closed grayscale presentation QA."""
    record = e016.image_record(path)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    channels_identical = bool(
        np.array_equal(rgb[..., 0], rgb[..., 1])
        and np.array_equal(rgb[..., 1], rgb[..., 2])
    )
    if not channels_identical:
        raise RuntimeError(f"Presentation frame is not true grayscale: {path}")
    excluded_value = int(EXCLUDED_GRAY[1:3], 16)
    record.update(
        {
            "rgb_channels_identical": True,
            "pure_black_pixels": int(np.count_nonzero(np.all(rgb == 0, axis=-1))),
            "neutral_exclusion_fill_pixels": int(
                np.count_nonzero(np.all(rgb == excluded_value, axis=-1))
            ),
        }
    )
    if record["pure_black_pixels"] <= 0:
        raise RuntimeError(f"Frame does not preserve black outside support: {path}")
    if record["neutral_exclusion_fill_pixels"] <= 0:
        raise RuntimeError(f"Frame does not contain the neutral exclusion layer: {path}")
    return record


def validate_canonical_coverage(coverage: dict[str, Any]) -> None:
    """Fail if the audited canonical mask or fixed sections silently change."""
    if coverage["full_exact_support_voxels"] != 65_589:
        raise ValueError("Canonical full-support voxel count changed")
    if coverage["stable_voxels"] != 52_314:
        raise ValueError("Canonical all-bin stable voxel count changed")
    if coverage["temporally_variable_voxels"] != 10_646:
        raise ValueError("Canonical temporally-variable voxel count changed")
    if coverage["never_valid_voxels"] != 2_629:
        raise ValueError("Canonical never-valid voxel count changed")
    if (
        coverage["original_dynamic_mask_summary"]
        ["total_adjacent_bin_voxel_switch_events"]
        != 31_683
    ):
        raise ValueError("Canonical dynamic-mask transition count changed")
    expected_sections = [(185, 162), (28, 20), (188, 128)]
    actual_sections = [
        (item["full_support_voxels"], item["stable_voxels"])
        for item in coverage["sections"]
    ]
    if actual_sections != expected_sections:
        raise ValueError(f"Canonical fixed-section coverage changed: {actual_sections}")
    expected_los = {"XY": 0.9345, "XZ": 0.9017, "YZ": 0.8827}
    for name, expected in expected_los.items():
        actual = coverage["projection_lines_of_sight"][name][
            "stable_fraction_of_full_support"
        ]
        if not np.isclose(actual, expected, atol=5e-5):
            raise ValueError(f"Canonical {name} line-of-sight coverage changed: {actual}")


def load_e016_limit_source(
    path: Path, expected_sha256: str | None
) -> tuple[dict[str, float], dict[str, Any]]:
    """Bind display scaling to the immutable E-016 machine report."""
    path = path.resolve()
    record = e016._file_record(path)
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise ValueError("E-016 report SHA-256 does not match the expected source")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("experiment_id") != "E-016":
        raise ValueError("Display-limit source is not the E-016 report")
    limits = {
        name: float(
            report["display"]["component_limits"][name][
                "p99_9_component_limit_rad"
            ]
        )
        for name in e016.ENCODINGS
    }
    if limits != INHERITED_E016_LIMITS_RAD:
        raise ValueError(f"E-016 component limits changed: {limits}")
    inputs = report.get("inputs", {})
    if inputs.get("reconstruction", {}).get("sha256") != e016.CANONICAL_RECONSTRUCTION_SHA256:
        raise ValueError("E-016 report is not bound to the canonical reconstruction")
    if inputs.get("seed_derived_support", {}).get("sha256") != e016.CANONICAL_SUPPORT_SHA256:
        raise ValueError("E-016 report is not bound to the canonical support")
    raw_hashes = {
        name: inputs.get("raw_complex_images", {}).get(name, {}).get("sha256")
        for name in ("REF", *e016.ENCODINGS)
    }
    if raw_hashes != e016.CANONICAL_RAW_SHA256:
        raise ValueError("E-016 report is not bound to the canonical raw images")
    return limits, record


def stable_display_statistics(
    panel_values: dict[str, list[np.ndarray]], inherited_limit: float
) -> dict[str, Any]:
    """Keep E-016 scaling while reporting the E-017 stable-domain p99.9."""
    stable_limit, stable_stats = e016.displayed_p999(panel_values)
    per_class: dict[str, Any] = {}
    displayed = 0
    clipped = 0
    for panel_class in e016.PANEL_CLASSES:
        arrays = panel_values[panel_class]
        finite = np.concatenate(
            [np.abs(values[np.isfinite(values)]) for values in arrays]
        )
        count = int(np.count_nonzero(finite > inherited_limit))
        per_class[panel_class] = {
            "displayed_values": int(finite.size),
            "clipped_at_inherited_e016_limit": count,
            "clipped_fraction_at_inherited_e016_limit": float(count / finite.size),
        }
        displayed += int(finite.size)
        clipped += count
    return {
        "display_limit_rad": inherited_limit,
        "display_limit_source": "exact inherited E-016 component limit",
        "rescaled_for_e017": False,
        "stable_domain_maximum_six_class_p99_9_rad": stable_limit,
        "stable_domain_p99_9_statistics": stable_stats,
        "displayed_values": displayed,
        "clipped_at_inherited_e016_limit": clipped,
        "clipped_fraction_at_inherited_e016_limit": float(clipped / displayed),
        "per_panel_class_at_inherited_limit": per_class,
    }


def _stable_raw_bin(
    raw: dict[str, Any], stable: np.ndarray, index: int, encoding: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = e016.compound_to_complex(raw["REF"][index])
    encoded = e016.compound_to_complex(raw[encoding][index])
    phase = np.angle(encoded * np.conj(reference)).astype(np.float32)
    weights = (np.abs(encoded) * np.abs(reference)).astype(np.float32)
    finite_positive = np.isfinite(phase) & np.isfinite(weights) & (weights > 0)
    if not np.all(finite_positive[stable]):
        raise RuntimeError("Final stable mask contains an invalid raw phase/weight")
    return phase, weights, stable


def _overlay_excluded_hatching(axis: Any, panel: e016.Panel) -> None:
    """Show excluded in-support pixels without fabricating a phase value."""
    excluded = np.asarray(panel.no_data, dtype=bool)
    if not excluded.any():
        return
    layer = np.ma.masked_where(~excluded, np.ones(excluded.shape, dtype=float))
    axis.imshow(
        layer,
        cmap=ListedColormap([EXCLUDED_GRAY]),
        origin="lower",
        extent=panel.extent,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )
    x, y = e016._centers(panel.extent, panel.support.shape)
    axis.contourf(
        x,
        y,
        excluded.astype(float),
        levels=(0.5, 1.5),
        colors="none",
        hatches=(HATCH_PATTERN,),
        antialiased=False,
        zorder=3,
    )


def render_frame(
    panels: list[e016.Panel],
    output: Path,
    encoding: str,
    cardiac_bin: int,
    bins: int,
    phase_center_ms: float,
    limit: float,
    dpi: int = e016.FRAME_DPI,
) -> None:
    if len(panels) != 6:
        raise ValueError("Exactly six panels are required")
    cmap = e016.signed_phase_colormap()
    rc = {
        "font.family": "DejaVu Sans",
        "hatch.color": HATCH_GRAY,
        "hatch.linewidth": 0.8,
    }
    with plt.rc_context(rc):
        figure = plt.figure(figsize=(18, 9), layout="constrained", facecolor="white")
        grid = figure.add_gridspec(
            3,
            4,
            height_ratios=(1, 1, 0.07),
            width_ratios=(1, 1, 1, 0.035),
        )
        axes = [
            figure.add_subplot(grid[row, column])
            for row in range(2)
            for column in range(3)
        ]
        colorbar_axis = figure.add_subplot(grid[:2, 3])
        note_axis = figure.add_subplot(grid[2, :3])
        note_axis.axis("off")
        image = None
        for axis, panel in zip(axes, panels, strict=True):
            axis.set_facecolor("#000000")
            image = axis.imshow(
                np.ma.masked_invalid(panel.values),
                cmap=cmap,
                origin="lower",
                extent=panel.extent,
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="equal",
                zorder=1,
            )
            _overlay_excluded_hatching(axis, panel)
            x, y = e016._centers(panel.extent, panel.support.shape)
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
                Patch(
                    facecolor="white",
                    edgecolor="0.05",
                    label="exact seed-derived lumen-support contour",
                ),
                Patch(
                    facecolor=EXCLUDED_GRAY,
                    edgecolor=HATCH_GRAY,
                    hatch=HATCH_PATTERN,
                    label="excluded by fixed all-bin mask (no imputation)",
                ),
                Patch(
                    facecolor="#000000",
                    label="outside exact support",
                ),
                Patch(
                    facecolor="#202020",
                    label="most-negative valid phase (nonblack)",
                ),
            ),
            loc="center",
            ncol=4,
            frameon=False,
            fontsize=8.5,
        )
        figure.suptitle(
            f"dp150 1M scanner-derived KomaMRI phase {encoding} vs REF | "
            f"fixed all-bin stable mask | cardiac bin {cardiac_bin:02d}/{bins} | "
            f"center {phase_center_ms:.3f} ms",
            fontsize=15,
        )
        temporary = output.with_name(f".{output.stem}.partial.png")
        figure.savefig(temporary, dpi=dpi, facecolor="white", format="png")
        plt.close(figure)
        temporary.replace(output)


def _replace_staging_paths(
    videos: dict[str, dict[str, Any]],
    output_dir: Path,
    final_output_dir: Path,
) -> None:
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
                **e016._file_record(manifest_path),
                "path": video["manifest"],
            }


def encode_video(
    unique_frames: list[Path],
    output: Path,
    plan: e016.PlaybackPlan,
    title: str,
) -> dict[str, Any]:
    """Encode with a banner below, rather than over, the figure title."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the requested cines")
    playlist = [unique_frames[index] for index in e016.playback_indices(plan)]
    manifest = output.with_suffix(".ffconcat")
    e016._write_manifest(playlist, manifest, plan.frame_period_s)
    fps_text = f"{plan.fps.numerator}/{plan.fps.denominator}"
    timescale = plan.fps.numerator * max(
        1, math.ceil(1000 / plan.fps.numerator)
    )
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    video_filter = (
        "drawtext="
        f"text='{plan.label}':"
        "fontcolor=white:fontsize=42:"
        f"x={PLAYBACK_BANNER_X_PX}:y={PLAYBACK_BANNER_Y_PX}:"
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
    probe = e016.probe_video(temporary)
    if not probe.get("probe_available"):
        raise RuntimeError("ffprobe is required for fail-closed video validation")
    if probe.get("dimensions_px") != list(e016.FRAME_SIZE_PX):
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
        "sha256": e016.sha256_file(output),
        "playback_label": plan.label,
        "playback_banner_position_px": {
            "x": PLAYBACK_BANNER_X_PX,
            "y": PLAYBACK_BANNER_Y_PX,
            "placement": "unused upper-left margin below the centered figure title",
        },
        "fps_exact": fps_text,
        "frame_period_s_exact": (
            f"{plan.frame_period_s.numerator}/{plan.frame_period_s.denominator}"
        ),
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
        **e016.probe_video(output),
    }


def main() -> None:
    args = parse_args()
    reconstruction_path = args.reconstruction.resolve()
    support_path = args.support_h5.resolve()
    raw_run_dir = args.raw_run_dir.resolve()
    e016_report_path = args.e016_report.resolve()
    final_output_dir = args.output_dir.resolve()
    # Keep the atomic staging pathname short enough for Windows' legacy
    # MAX_PATH handling when descriptive video/manifest names are appended.
    output_dir = final_output_dir.with_name(".e017.partial")
    if final_output_dir.exists() or output_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite a final or partial presentation derivative: "
            f"{final_output_dir}"
        )

    reconstruction_hash = e016.sha256_file(reconstruction_path)
    support_hash = e016.sha256_file(support_path)
    if reconstruction_hash != args.expected_reconstruction_sha256:
        raise ValueError("Reconstruction SHA-256 does not match canonical E-010")
    if support_hash != args.expected_support_sha256:
        raise ValueError("Support SHA-256 does not match the canonical input")
    inherited_limits, e016_report_record = load_e016_limit_source(
        e016_report_path, args.expected_e016_report_sha256
    )
    if args.cycles != 2 or args.slow_hold_ms != 420:
        raise ValueError(f"{EXPERIMENT_ID} requires two cycles and 420-ms slow-QC hold")

    fov_m = np.asarray(args.fov_m, dtype=np.float64)
    matrix = np.asarray(args.matrix, dtype=np.int64)
    if not np.array_equal(matrix, np.asarray(e016.DEFAULT_MATRIX, dtype=np.int64)):
        raise ValueError(f"{EXPERIMENT_ID} requires canonical matrix {list(e016.DEFAULT_MATRIX)}")
    if not np.array_equal(fov_m, np.asarray(e016.DEFAULT_FOV_M, dtype=np.float64)):
        raise ValueError(f"{EXPERIMENT_ID} requires canonical FOV {list(e016.DEFAULT_FOV_M)} m")

    reconstruction = e016.load_reconstruction(reconstruction_path)
    if reconstruction["magnitude"].shape[0] != e016.EXPECTED_BINS:
        raise ValueError("This derivative requires all 15 reconstructed bins")
    expected_shape = tuple(int(value) for value in matrix[::-1])
    if reconstruction["magnitude"].shape[1:] != expected_shape:
        raise ValueError("Reconstruction shape does not match the explicit matrix")
    support, support_metadata = e016.load_support(support_path, fov_m, matrix)
    readout_mm, phase_mm, slice_mm = e016.coordinate_centers(
        support.shape, 1000.0 * fov_m
    )
    sections = e016.select_fixed_sections(support, readout_mm)
    if [item.index for item in sections] != [70, 98, 122]:
        raise ValueError(f"Canonical fixed sections changed: {sections}")
    if [item.readout_mm for item in sections] != [-161.875, -126.875, -96.875]:
        raise ValueError(f"Canonical fixed coordinates changed: {sections}")
    plans = e016.physical_and_slow_plans(
        reconstruction["RR_s"],
        e016.EXPECTED_BINS,
        args.cycles,
        args.slow_hold_ms,
    )
    initial_stable = stable_reconstruction_mask(support, reconstruction["mask"])

    with ExitStack() as stack:
        raw, raw_paths = e016.open_raw_datasets(
            stack, raw_run_dir, reconstruction["magnitude"].shape
        )
        input_records_before = {
            "reconstruction": e016._file_record(reconstruction_path),
            "seed_derived_support": e016._file_record(support_path),
            "raw_complex_images": {
                name: e016._file_record(path) for name, path in raw_paths.items()
            },
            "e016_display_limit_report": e016_report_record,
        }
        raw_hashes = {
            name: record["sha256"]
            for name, record in input_records_before["raw_complex_images"].items()
        }
        if raw_hashes != e016.CANONICAL_RAW_SHA256:
            raise ValueError("Raw REF/X/Y/Z hashes do not match audited E-010 inputs")

        all_signal_valid, signal_scan = scan_all_signal_validity(
            raw, reconstruction["magnitude"].shape, initial_stable
        )
        stable, signal_removed = refine_stable_mask_with_signal(
            initial_stable, [all_signal_valid]
        )
        if signal_removed != 0:
            raise ValueError(
                "The declared all-bin reconstruction mask contains an invalid "
                "raw phase/weight sample; refusing to redefine the display mask"
            )
        if not np.array_equal(stable, initial_stable):
            raise RuntimeError("Signal validation unexpectedly changed the stable mask")
        coverage = stable_mask_coverage(
            support, stable, reconstruction["mask"], sections
        )
        validate_canonical_coverage(coverage)

        displayed: dict[str, dict[str, list[np.ndarray]]] = {
            name: {} for name in e016.ENCODINGS
        }
        max_phase_error = {name: 0.0 for name in e016.ENCODINGS}
        max_reference_error = 0.0
        for index in range(e016.EXPECTED_BINS):
            reference = e016.compound_to_complex(raw["REF"][index])
            max_reference_error = max(
                max_reference_error,
                float(
                    np.max(
                        np.abs(
                            np.abs(reference)
                            - reconstruction["magnitude"][index]
                        )
                    )
                ),
            )
            for component, encoding in enumerate(e016.ENCODINGS):
                phase, weights, valid = _stable_raw_bin(
                    raw, stable, index, encoding
                )
                expected = (
                    reconstruction["velocity"][component, index]
                    * np.pi
                    / reconstruction["Venc_m_per_s"]
                )
                delta = np.angle(np.exp(1j * (phase - expected)))
                max_phase_error[encoding] = max(
                    max_phase_error[encoding],
                    float(
                        np.max(
                            np.abs(delta[reconstruction["mask"][index]])
                        )
                    ),
                )
                panels = e016.build_panels(
                    phase,
                    weights,
                    valid,
                    support,
                    sections,
                    readout_mm,
                    phase_mm,
                    slice_mm,
                )
                for panel in panels:
                    displayed[encoding].setdefault(panel.panel_class, []).append(
                        panel.values
                    )
        display_statistics = {
            name: stable_display_statistics(displayed[name], inherited_limits[name])
            for name in e016.ENCODINGS
        }
        if max_reference_error > 1e-4:
            raise ValueError("Raw REF magnitude does not match the reconstruction")
        if max(max_phase_error.values()) > 1e-4:
            raise ValueError("Raw complex phase does not match reconstructed phase")

        output_dir.mkdir(parents=True)
        frame_paths: dict[str, list[Path]] = {
            name: [] for name in e016.ENCODINGS
        }
        for encoding in e016.ENCODINGS:
            (output_dir / f"phase_{encoding.lower()}_frames").mkdir()
        for index in range(e016.EXPECTED_BINS):
            for encoding in e016.ENCODINGS:
                phase, weights, valid = _stable_raw_bin(
                    raw, stable, index, encoding
                )
                panels = e016.build_panels(
                    phase,
                    weights,
                    valid,
                    support,
                    sections,
                    readout_mm,
                    phase_mm,
                    slice_mm,
                )
                path = (
                    output_dir
                    / f"phase_{encoding.lower()}_frames"
                    / f"frame_{index + 1:03d}.png"
                )
                render_frame(
                    panels,
                    path,
                    encoding,
                    index + 1,
                    e016.EXPECTED_BINS,
                    1000.0
                    * (index + 0.5)
                    * reconstruction["RR_s"]
                    / e016.EXPECTED_BINS,
                    inherited_limits[encoding],
                )
                frame_paths[encoding].append(path)

    videos: dict[str, dict[str, Any]] = {}
    for encoding in e016.ENCODINGS:
        videos[encoding] = {}
        for plan in plans:
            path = (
                output_dir
                / f"dp150_phase_{encoding}_stable_mask_{plan.slug}_15_bins.mp4"
            )
            videos[encoding][plan.slug] = encode_video(
                frame_paths[encoding],
                path,
                plan,
                f"dp150 stable-mask phase {encoding} vs REF",
            )
    _replace_staging_paths(videos, output_dir, final_output_dir)

    input_records_after = {
        "reconstruction": e016._file_record(reconstruction_path),
        "seed_derived_support": e016._file_record(support_path),
        "raw_complex_images": {
            name: e016._file_record(path) for name, path in raw_paths.items()
        },
        "e016_display_limit_report": e016._file_record(e016_report_path),
    }
    input_hashes_unchanged = input_records_before == input_records_after
    if not input_hashes_unchanged:
        raise RuntimeError("A scientific or display-limit input changed while rendering")

    report = {
        "schema_version": 1,
        "status": "PASS_PRESENTATION_DERIVATIVE_WITH_EXCLUSIONS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_ids": ["E-016", "E-010"],
        "evidence_class": "clean presentation derivative of preserved E-010 reconstruction and E-016 display convention",
        "provenance": {
            "script": e016._file_record(Path(__file__).resolve()),
            "e016_helper_dependency": e016._file_record(Path(e016.__file__).resolve()),
            "command_line": [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            "input_records_before_render": input_records_before,
            "input_records_after_render": input_records_after,
            "input_hashes_unchanged": input_hashes_unchanged,
        },
        "inputs": input_records_before,
        "acquisition": {
            "particle_count": 1_000_000,
            "cardiac_bins": e016.EXPECTED_BINS,
            "RR_s": reconstruction["RR_s"],
            "physical_bin_spacing_ms": 1000.0
            * reconstruction["RR_s"]
            / e016.EXPECTED_BINS,
            "Venc_m_per_s": reconstruction["Venc_m_per_s"],
            "encoding": "separate REF/X/Y/Z",
        },
        "grid": {
            "FOV_m_readout_phase_slice": fov_m.tolist(),
            "matrix_readout_phase_slice": matrix.tolist(),
            "array_order": ["cardiac_bin", "slice", "phase", "readout"],
            "voxel_size_mm": (1000.0 * fov_m / matrix).tolist(),
        },
        "axis_mapping": {
            "PC_X": "readout/source Z",
            "PC_Y": "phase/source X",
            "PC_Z": "slice/source Y",
        },
        "support": support_metadata,
        "stable_phase_domain": {
            "definition": "exact seed-derived support AND stored reconstruction mask true in all 15 bins",
            "raw_signal_acceptance_rule": signal_scan["rule"],
            "initial_stable_voxels": int(initial_stable.sum()),
            "raw_signal_validation_passed_without_mask_change": signal_removed == 0,
            "removed_by_signal_validation": signal_removed,
            "final_stable_voxels": int(stable.sum()),
            "component_independent": True,
            "fixed_across_all_bins": True,
            "phase_imputation": False,
            "spatial_smoothing": False,
            "temporal_smoothing": False,
            "morphology": False,
            "spatial_interpolation": False,
            "temporal_interpolation": False,
            "raw_signal_checks": signal_scan["per_raw_image_checks"],
            "coverage": coverage,
        },
        "fixed_lumen_boundary_qc_sections": coverage["sections"],
        "display": {
            "layout": "top: XY/XZ/YZ projections; bottom: fixed upstream/throat/immediate-post-stenotic YZ slices",
            "phase_source": "angle(encoded * conj(REF)) from raw complex images",
            "projection": "abs(encoded)*abs(REF)-weighted circular mean on the fixed all-bin stable phase domain",
            "projection_exclusion": "hatch only when exact support has no stable line of sight",
            "slice_exclusion": "exact support outside the fixed stable phase domain",
            "lumen_boundary": "unsmoothed seed-derived support contour; not simulated wall material",
            "wall_material_simulated": False,
            "excluded_in_support": f"neutral gray {EXCLUDED_GRAY} with {HATCH_PATTERN} hatch {HATCH_GRAY}; no phase shown",
            "outside_support": "pure black",
            "valid_negative_phase": "dark gray down to #202020, never pure black",
            "colormap": "signed grayscale #202020 -> #8c8c8c -> #ffffff",
            "component_limits": display_statistics,
            "component_limits_exactly_equal_to_bound_e016_report": True,
            "limit_rule": "exact E-016 component limits retained for single-factor comparability; stable-domain six-class p99.9 also reported",
            "phase_limit_quantile": 0.999,
            "phase_limit_quantile_method": "NumPy linear",
            "image_interpolation": "nearest",
            "phase_imputation": False,
            "spatial_smoothing": False,
            "temporal_smoothing": False,
            "morphology": False,
            "frame_dpi": e016.FRAME_DPI,
            "frame_dimensions_px": list(e016.FRAME_SIZE_PX),
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
                    **stable_image_record(path),
                    "path": str(final_output_dir / path.relative_to(output_dir)),
                }
                for path in paths
            ]
            for name, paths in frame_paths.items()
        },
        "videos": videos,
        "scientific_limits": [
            "This is a presentation-only display derivative, not independent velocity validation.",
            "Excluded voxels are visibly hatched and are not assigned, interpolated, or imputed phase.",
            "The fixed mask is conservative: it discards any support voxel absent from one or more reconstructed bins.",
            "The clean view removes mask flicker by hiding intermittent data; it does not recover missing phase or improve wall coverage.",
            "Fixed exclusions are largest at the throat and immediate post-stenotic sections, so E-016 remains the dynamic-mask diagnostic companion.",
            "The contour is seed-derived lumen support, not wall material or a wall simulation.",
            "Voxel stair-stepping at the contour is retained and is not evidence of wall roughness.",
            "Separate REF/X/Y/Z encodings are not established as balanced-Hadamard equivalent.",
            "Component-specific inherited display limits preclude brightness comparison across X/Y/Z.",
            "Peak bin 14 partly overlaps the controlled synthetic final-80-ms tail in the source motion schedule.",
            "No cardiac bins are interpolated; repeated cycles are playback only.",
        ],
    }
    report_path = output_dir / "dp150_phase_stable_mask_clean_report.json"
    temporary = report_path.with_name(f".{report_path.name}.partial")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    output_dir.replace(final_output_dir)
    published_report = final_output_dir / report_path.name
    print(f"PASS_PRESENTATION_DERIVATIVE_WITH_EXCLUSIONS\nReport: {published_report}")


if __name__ == "__main__":
    main()
