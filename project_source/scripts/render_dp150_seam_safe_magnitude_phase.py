r"""Render corrected dp150 magnitude and grayscale phase overview cines."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
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
        image_record,
        probe_video,
        resolve_particle_label,
        sha256_file,
    )
    from scripts.render_dp150_phase_boundary_qc import (
        FRAME_DPI,
        FRAME_SIZE_PX,
        playback_indices,
        physical_and_slow_plans,
        signed_phase_colormap,
    )
    from scripts.render_dp150_projection_slices_magnitude_phase import (
        magnitude_panels,
        phase_panels,
        plane_definitions,
    )
except ModuleNotFoundError:
    from analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        image_record,
        probe_video,
        resolve_particle_label,
        sha256_file,
    )
    from render_dp150_phase_boundary_qc import (
        FRAME_DPI,
        FRAME_SIZE_PX,
        playback_indices,
        physical_and_slow_plans,
        signed_phase_colormap,
    )
    from render_dp150_projection_slices_magnitude_phase import (
        magnitude_panels,
        phase_panels,
        plane_definitions,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT
    / "outputs"
    / "dp150_mri_pipeline"
    / "koma"
    / "simulation_motion_100k_scanner_matched_venc5_seam_safe_v3"
)
DEFAULT_RECONSTRUCTION = (
    DEFAULT_RUN_DIR
    / "final_image_analysis"
    / "support_constrained_phase_reconstruction_seam_safe_v3"
    / "dp150_seam_safe_support_phase.h5"
)
DEFAULT_RECONSTRUCTION_REPORT = (
    DEFAULT_RECONSTRUCTION.parent / "dp150_seam_safe_support_phase_report.json"
)
COMPONENTS = ("X", "Y", "Z")
EXPECTED_BINS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument(
        "--reconstruction-report", type=Path, default=DEFAULT_RECONSTRUCTION_REPORT
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fov-m", nargs=3, type=float, default=DEFAULT_FOV_M)
    parser.add_argument("--matrix", nargs=3, type=int, default=DEFAULT_MATRIX)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--slow-hold-ms", type=int, default=420)
    parser.add_argument(
        "--particle-label",
        default=None,
        help="Display label such as 100k or 1M; inferred from the path by default.",
    )
    return parser.parse_args()


def load_corrected_reconstruction(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        required = (
            "reference_magnitude",
            "wrapped_phase_corrected_rad",
            "velocity_corrected_m_per_s",
            "speed_corrected_m_per_s",
            "support_mask",
            "direct_mask",
            "regularized_mask",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing corrected reconstruction datasets: {missing}")
        result = {
            "magnitude": np.asarray(source["reference_magnitude"], np.float32),
            "phase": np.asarray(source["wrapped_phase_corrected_rad"], np.float32),
            "velocity": np.asarray(source["velocity_corrected_m_per_s"], np.float32),
            "speed": np.asarray(source["speed_corrected_m_per_s"], np.float32),
            "support": np.asarray(source["support_mask"], bool),
            "direct": np.asarray(source["direct_mask"], bool),
            "regularized": np.asarray(source["regularized_mask"], bool),
            "RR_s": float(source.attrs["RR_s"]),
            "Venc_m_per_s": float(source.attrs["Venc_m_per_s"]),
            "evidence_class": str(source.attrs["evidence_class"]),
        }
    shape = result["magnitude"].shape
    if len(shape) != 4 or shape[0] != EXPECTED_BINS:
        raise ValueError(f"Expected magnitude (15,slice,phase,readout), got {shape}")
    if result["phase"].shape != (3, *shape):
        raise ValueError(f"Unexpected corrected phase shape {result['phase'].shape}")
    if result["velocity"].shape != (3, *shape):
        raise ValueError(f"Unexpected corrected velocity shape {result['velocity'].shape}")
    if result["speed"].shape != shape:
        raise ValueError(f"Unexpected corrected speed shape {result['speed'].shape}")
    for name in ("support", "direct", "regularized"):
        if result[name].shape != shape[1:]:
            raise ValueError(f"Unexpected {name} shape {result[name].shape}")
    if not np.array_equal(result["direct"] | result["regularized"], result["support"]):
        raise RuntimeError("Direct and regularized masks do not cover exact support")
    if np.any(result["direct"] & result["regularized"]):
        raise RuntimeError("Direct and regularized masks overlap")
    if not np.isfinite(result["magnitude"]).all():
        raise ValueError("Reference magnitude contains non-finite values")
    if not np.isfinite(result["phase"][:, :, result["support"]]).all():
        raise ValueError("Corrected phase is non-finite inside support")
    return result


def render_six_panels(
    panels: list[np.ndarray],
    definitions: list[dict[str, Any]],
    output: Path,
    title: str,
    mode: str,
    limit: float,
    *,
    dpi: int = FRAME_DPI,
) -> None:
    if len(panels) != 6 or mode not in {"magnitude", "phase"}:
        raise ValueError("Expected six panels and magnitude or phase mode")
    figure, axes = plt.subplots(
        2, 3, figsize=(18, 9), constrained_layout=True, facecolor="white"
    )
    for axis, panel, definition in zip(
        axes.ravel(), panels, definitions, strict=True
    ):
        axis.set_facecolor("black")
        if mode == "magnitude":
            image = axis.imshow(
                panel,
                cmap="gray",
                origin="lower",
                extent=definition["extent"],
                vmin=0,
                vmax=limit,
                interpolation="nearest",
                aspect="equal",
            )
        else:
            image = axis.imshow(
                np.ma.masked_invalid(panel),
                cmap=signed_phase_colormap(),
                origin="lower",
                extent=definition["extent"],
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="equal",
            )
        axis.set_title(definition["title"], fontsize=12)
        axis.set_xlabel(definition["xlabel"])
        axis.set_ylabel(definition["ylabel"])
        axis.grid(False)
        if mode == "phase":
            figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label="rad")
    figure.suptitle(title, fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def one_voxel_boundary(support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support, bool)
    interior = np.zeros_like(support)
    interior[1:-1, 1:-1, 1:-1] = (
        support[1:-1, 1:-1, 1:-1]
        & support[:-2, 1:-1, 1:-1]
        & support[2:, 1:-1, 1:-1]
        & support[1:-1, :-2, 1:-1]
        & support[1:-1, 2:, 1:-1]
        & support[1:-1, 1:-1, :-2]
        & support[1:-1, 1:-1, 2:]
    )
    return support & ~interior, interior


def temporal_phase_qc(
    phase: np.ndarray, support: np.ndarray
) -> dict[str, Any]:
    boundary, interior = one_voxel_boundary(support)
    result: dict[str, Any] = {
        "boundary_voxels": int(boundary.sum()),
        "interior_voxels": int(interior.sum()),
        "components": {},
    }
    for component_index, component in enumerate(COMPONENTS):
        edges = []
        for index in range(EXPECTED_BINS):
            following = (index + 1) % EXPECTED_BINS
            delta = np.abs(
                np.angle(
                    np.exp(
                        1j
                        * (
                            phase[component_index, following]
                            - phase[component_index, index]
                        )
                    )
                )
            )
            boundary_q95 = float(np.quantile(delta[boundary], 0.95))
            interior_q95 = float(np.quantile(delta[interior], 0.95))
            edges.append(
                {
                    "edge_zero_based": index,
                    "is_cycle_seam": index == EXPECTED_BINS - 1,
                    "boundary_q95_rad": boundary_q95,
                    "interior_q95_rad": interior_q95,
                    "boundary_over_interior_q95": (
                        boundary_q95 / interior_q95
                        if interior_q95 > 0
                        else None
                    ),
                }
            )
        result["components"][component] = {"edges": edges}
    return result


def _write_video_manifest(
    paths: list[Path], output: Path, period: Fraction
) -> None:
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


def build_ffmpeg_command(
    ffmpeg: str,
    manifest: Path,
    temporary: Path,
    plan: Any,
    title: str,
) -> list[str]:
    """Build the portable H.264 command without optional drawtext support."""
    fps_text = f"{plan.fps.numerator}/{plan.fps.denominator}"
    timescale = plan.fps.numerator * max(1, math.ceil(1000 / plan.fps.numerator))
    return [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-vf",
        "format=yuv420p",
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


def encode_video(
    unique_frames: list[Path], output: Path, plan: Any, title: str
) -> dict[str, Any]:
    """Encode frames while keeping playback labels in metadata and filenames."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the requested cines")
    playlist = [unique_frames[index] for index in playback_indices(plan)]
    manifest = output.with_suffix(".ffconcat")
    _write_video_manifest(playlist, manifest, plan.frame_period_s)
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    command = build_ffmpeg_command(ffmpeg, manifest, temporary, plan, title)
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
        "playback_label_overlay": False,
        "playback_label_location": "filename and video title metadata",
        "fps_exact": f"{plan.fps.numerator}/{plan.fps.denominator}",
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
        **probe_video(output),
    }


def _rewrite_manifest_paths(
    videos: dict[str, dict[str, Any]], staging: Path, final: Path
) -> None:
    staging_text = staging.resolve().as_posix()
    final_text = final.resolve().as_posix()
    for variants in videos.values():
        for record in variants.values():
            manifest = Path(record["manifest"])
            content = manifest.read_text(encoding="ascii")
            temporary = manifest.with_name(f".{manifest.name}.partial")
            temporary.write_text(
                content.replace(staging_text, final_text), encoding="ascii"
            )
            os.replace(temporary, manifest)
            record["path"] = str(final / Path(record["path"]).relative_to(staging))
            record["manifest"] = str(final / manifest.relative_to(staging))


def main() -> int:
    args = parse_args()
    reconstruction_path = args.reconstruction.resolve()
    reconstruction_report_path = args.reconstruction_report.resolve()
    final_output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else reconstruction_path.parent / "grayscale_magnitude_phase_cines"
    )
    staging = final_output_dir.with_name(f".{final_output_dir.name}.partial")
    if final_output_dir.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite final or partial render: {final_output_dir} / {staging}"
        )
    with reconstruction_report_path.open("r", encoding="utf-8") as source:
        reconstruction_report = json.load(source)
    if reconstruction_report.get("status") != "PASS_RECONSTRUCTION_DERIVATIVE":
        raise RuntimeError("Reconstruction report is not PASS_RECONSTRUCTION_DERIVATIVE")
    expected_output_hash = (
        reconstruction_report.get("outputs", {})
        .get("reconstruction_h5", {})
        .get("sha256")
    )
    actual_output_hash = sha256_file(reconstruction_path)
    reconstruction_report_hash = sha256_file(reconstruction_report_path)
    if not expected_output_hash or expected_output_hash != actual_output_hash:
        raise RuntimeError("Reconstruction report does not hash-bind the HDF5 input")

    data = load_corrected_reconstruction(reconstruction_path)
    particle_label, _particle_slug = resolve_particle_label(
        reconstruction_path, args.particle_label
    )
    fov_m = np.asarray(args.fov_m, np.float64)
    matrix = np.asarray(args.matrix, np.int64)
    expected_spatial = tuple(int(value) for value in matrix[::-1])
    if data["magnitude"].shape[1:] != expected_spatial:
        raise ValueError("Reconstruction shape does not match matrix")
    definitions = plane_definitions(expected_spatial, 1000.0 * fov_m)
    support = data["support"]
    magnitude = data["magnitude"]
    phase = data["phase"]
    valid = np.broadcast_to(support, magnitude.shape)

    magnitude_positive = magnitude[magnitude > 0]
    magnitude_limit = max(float(np.quantile(magnitude_positive, 0.999)), 1e-6)
    phase_limits: dict[str, float] = {}
    phase_statistics: dict[str, Any] = {}
    for component_index, component in enumerate(COMPONENTS):
        selected = np.abs(phase[component_index, :, support])
        phase_limits[component] = max(float(np.quantile(selected, 0.995)), 1e-6)
        phase_statistics[component] = {
            "median_absolute_rad": float(np.quantile(selected, 0.50)),
            "p95_absolute_rad": float(np.quantile(selected, 0.95)),
            "p99_5_absolute_rad": phase_limits[component],
            "maximum_absolute_rad": float(selected.max()),
            "display_clipped_fraction": float(
                np.mean(selected > phase_limits[component])
            ),
        }
    p95_speed = np.asarray(
        [np.quantile(data["speed"][index, support], 0.95) for index in range(15)]
    )
    selected_index = int(np.argmax(p95_speed))
    plans = physical_and_slow_plans(
        data["RR_s"], EXPECTED_BINS, args.cycles, args.slow_hold_ms
    )

    staging.mkdir(parents=True)
    frame_paths = {name: [] for name in ("magnitude", *COMPONENTS)}
    for name in frame_paths:
        (staging / f"{name.lower()}_frames").mkdir()
    for index in range(EXPECTED_BINS):
        magnitude_path = staging / "magnitude_frames" / f"frame_{index + 1:03d}.png"
        render_six_panels(
            magnitude_panels(magnitude[index], definitions),
            definitions,
            magnitude_path,
            (
                f"dp150 seam-safe {particle_label} KomaMRI reference magnitude | "
                f"cardiac bin {index + 1}/15"
            ),
            "magnitude",
            magnitude_limit,
        )
        frame_paths["magnitude"].append(magnitude_path)
        for component_index, component in enumerate(COMPONENTS):
            output = (
                staging
                / f"{component.lower()}_frames"
                / f"frame_{index + 1:03d}.png"
            )
            render_six_panels(
                phase_panels(
                    phase[component_index, index],
                    magnitude[index],
                    valid[index],
                    definitions,
                ),
                definitions,
                output,
                (
                    f"dp150 seam-safe {particle_label} support-constrained phase "
                    f"{component} vs REF | "
                    f"cardiac bin {index + 1}/15 | +/-{phase_limits[component]:.3f} rad"
                ),
                "phase",
                phase_limits[component],
            )
            frame_paths[component].append(output)

    videos: dict[str, dict[str, Any]] = {}
    for name, frames in frame_paths.items():
        videos[name] = {}
        label = "reference magnitude" if name == "magnitude" else f"phase {name} vs REF"
        for plan in plans:
            output = staging / f"dp150_{name.lower()}_{plan.slug}_15_bins.mp4"
            videos[name][plan.slug] = encode_video(
                frames,
                output,
                plan,
                f"dp150 seam-safe {particle_label} {label}",
            )
    _rewrite_manifest_paths(videos, staging, final_output_dir)

    selected_stills = {
        name: {
            **image_record(paths[selected_index]),
            "path": str(final_output_dir / paths[selected_index].relative_to(staging)),
        }
        for name, paths in frame_paths.items()
    }
    input_hashes_unchanged = (
        sha256_file(reconstruction_path) == actual_output_hash
        and sha256_file(reconstruction_report_path) == reconstruction_report_hash
    )
    if not input_hashes_unchanged:
        raise RuntimeError("A scientific input changed while rendering")
    report = {
        "schema_version": 1,
        "status": "PASS_PRESENTATION_DERIVATIVE_WITH_QUALITATIVE_LIMITATIONS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "corrected_reconstruction": {
                "path": str(reconstruction_path),
                "sha256": actual_output_hash,
            },
            "reconstruction_report": {
                "path": str(reconstruction_report_path),
                "sha256": reconstruction_report_hash,
            },
            "input_hashes_unchanged_after_render": input_hashes_unchanged,
        },
        "selected_still": {
            "selection": "maximum full-support p95 corrected speed",
            "saved_phs_one_based": selected_index + 1,
            "p95_speed_m_per_s": float(p95_speed[selected_index]),
            "outputs": selected_stills,
        },
        "display": {
            "particle_count_label": particle_label,
            "layout": "top orthogonal projections; bottom geometric center slices",
            "magnitude_projection": "maximum intensity of acquired REF reconstruction",
            "phase_projection": (
                "support-constrained phase at the maximum REF-magnitude support "
                "voxel along each projection ray"
            ),
            "phase_mask": "one exact fixed support for every bin and component",
            "phase_colormap": (
                "signed grayscale: dark gray negative, mid-gray zero, white positive; "
                "pure black reserved for outside-support/no-data"
            ),
            "magnitude_limit_p99_9": magnitude_limit,
            "phase_limits_p99_5_absolute_rad": phase_limits,
            "phase_statistics": phase_statistics,
            "frame_dimensions_px": [18 * FRAME_DPI, 9 * FRAME_DPI],
            "spatial_interpolation": "nearest",
            "temporal_interpolation": False,
        },
        "mask_stability": {
            "support_voxels_every_bin_component": int(support.sum()),
            "support_transition_count_across_bins": 0,
            "black_holes_from_per_bin_threshold_mask": 0,
            "regularized_voxels_every_bin_component": int(data["regularized"].sum()),
            "direct_voxels_every_bin_component": int(data["direct"].sum()),
        },
        "temporal_phase_qc": temporal_phase_qc(phase, support),
        "videos": videos,
        "limitations": [
            "These are presentation derivatives of wrapped phase, not external velocity validation.",
            "The phase support is seed-derived lumen support, not measured wall material.",
            "Regularized voxels are spatial estimates; fixed support removes mask flicker but cannot create missing acquired signal.",
            "The phantom has no stationary wall/background spins, so wall realism cannot be established from these cines.",
            "No cardiac frames are interpolated; repeated cycles affect playback only.",
        ],
    }
    report_path = staging / "dp150_seam_safe_magnitude_phase_report.json"
    temporary_report = report_path.with_name(f".{report_path.name}.partial")
    temporary_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    os.replace(staging, final_output_dir)
    print(
        "PASS_PRESENTATION_DERIVATIVE_WITH_QUALITATIVE_LIMITATIONS\n"
        f"Output: {final_output_dir}\n"
        f"Selected bin: {selected_index + 1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
